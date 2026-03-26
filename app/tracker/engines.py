"""Engine wrappers with unified dict-in / list-out interface."""

import numpy as np
import torch
import tensorrt as trt
import onnxruntime as ort

from .constants import _NP_TO_TORCH

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    """TRT engine with pre-allocated GPU output buffers. No host copies.

    Usage mirrors ort.InferenceSession:
        engine = TRTEngine("model.engine", stream)
        out1, out2 = engine({"input_name": tensor, ...})
    """

    def __init__(self, engine_path: str, stream: torch.cuda.Stream = None):
        # Think of it like a 3-layer setup:
        #
        # 1. trt.Runtime — the "factory". It knows how to read .engine files.
        #    It doesn't hold any model — it's just the tool that can load one.
        #
        # 2. engine (ICudaEngine) — the "blueprint". Created by deserializing the .engine file.
        #    Contains the model weights, the optimized graph, and the layer definitions.
        #    But it can't run inference by itself — it's read-only, like a compiled program on disk.
        #
        # 3. context (IExecutionContext) — the "workspace". Created from the engine.
        #    This is where actual inference happens. It holds mutable state:
        #    input/output memory addresses, current dynamic shapes, internal scratch memory.
        #    You can create multiple contexts from one engine to run inference in parallel,
        #    each with its own shapes and buffers (we only need one).
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # A CUDA stream is a queue of GPU operations that execute in order.
        # The GPU can have many streams running in parallel — operations on different
        # streams can overlap, but operations on the SAME stream always run sequentially.
        #
        # We pass the same stream to all 4 engines (enc, dec, menc, matt) so that:
        #   enc runs first → matt waits for enc to finish → dec waits for matt → etc.
        # If each engine had its own stream, they could run at the same time and read
        # each other's output buffers before they're ready (garbage data).
        #
        # The stream also lets the CPU keep working while the GPU is busy:
        #   execute_async_v3(stream) returns immediately — the GPU works in the background.
        #   We only call stream.synchronize() when we actually need the results (e.g. before
        #   copying a mask to CPU, or before passing data to ORT which doesn't know about our stream).
        self.stream = stream

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.dynamic_inputs: set[str] = set()

        # we extract the input and output tensor names and shapes from the engine.
        # we find out which inputs are dynamic (have -1 in their shape).
        # dynamic outputs will be allocated on the fly in __call__.
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                if -1 in self.engine.get_tensor_shape(name):
                    self.dynamic_inputs.add(name)
            else:
                self.output_names.append(name)

        # Pre-allocate output buffers for static-shape outputs
        self._static_bufs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if -1 not in shape:
                # dt is the torch dtype corresponding to the engine's tensor dtype for this output.
                dt = _NP_TO_TORCH[np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))]
                # we create an empty tensor on the GPU with the appropriate shape and dtype and store it in _static_bufs under the output name.
                self._static_bufs[name] = torch.empty(shape, dtype=dt, device="cuda")

    def __call__(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        ctx = self.context

        for name in self.input_names:
            # make sure the input tensor is contiguous in memory, which is required by TensorRT which reads raw GPU addresses.
            t = inputs[name].contiguous()
            # if this input is dynamic, we need to tell the context the actual shape for this run.
            # if it's not dynamic, the context already knows the shape from engine build time so we don't need to set it again.
            if name in self.dynamic_inputs:
                ctx.set_input_shape(name, tuple(t.shape))
            # we need to tell the context the memory address of this input tensor on the GPU so it can read from it during execution.
            ctx.set_tensor_address(name, t.data_ptr())

        outputs: list[torch.Tensor] = []
        for name in self.output_names:
            if name in self._static_bufs:
                # we already pre-allocated a buffer for this output in __init__, so we can just reuse it.
                buf = self._static_bufs[name]
            else:
                # this is a dynamic output, so we need to allocate a buffer for it on the fly. we get the shape from
                # the context (not the engine — the engine only has -1 for dynamic dims, but the context knows
                # the actual shape after we set the input shapes above) and the dtype from the engine.
                shape = tuple(ctx.get_tensor_shape(name))
                dt = _NP_TO_TORCH[np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))]
                buf = torch.empty(shape, dtype=dt, device="cuda")
            # tell the context the memory address of this output tensor on the GPU so it can write to it during execution.
            ctx.set_tensor_address(name, buf.data_ptr())
            outputs.append(buf)
        # now that we've set up all the input and output addresses, we can execute the engine.
        # this will run the engine on the GPU, reading from the input addresses and writing to the output addresses we provided.
        # we use execute_async_v3. the "async" means it return immediately - the gpu works in the background.
        # the self.stream.cuda_stream is the CUDA stream we want this execution to run on. by using the same stream for all our engines, we ensure they
        # run in order and don't have to worry about synchronization between them.
        ctx.execute_async_v3(self.stream.cuda_stream)
        return outputs

    def print_io(self, label: str):
        print(f"  {label}:")
        for name in self.input_names:
            s = self.engine.get_tensor_shape(name)
            tag = " *dynamic*" if name in self.dynamic_inputs else ""
            print(f"    IN  {name:30s} {tuple(s)}{tag}")
        for name in self.output_names:
            s = self.engine.get_tensor_shape(name)
            tag = " [pre-alloc]" if name in self._static_bufs else ""
            print(f"    OUT {name:30s} {tuple(s)}{tag}")


class ORTEngine:
    """ONNX Runtime session with same dict interface as TRTEngine.

    Usage:
        engine = ORTEngine("model.onnx")
        out1, out2 = engine({"input_name": tensor, ...})
    """

    def __init__(self, onnx_path: str):
        providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def __call__(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        np_inputs = {}
        for name in self.input_names:
            t = inputs[name]
            np_inputs[name] = t.cpu().numpy() if t.is_cuda else t.numpy()
        np_outputs = self.sess.run(self.output_names, np_inputs)
        return [torch.from_numpy(o).cuda() for o in np_outputs]

    def print_io(self, label: str):
        print(f"  {label}: (ONNX Runtime)")
        for i in self.sess.get_inputs():
            print(f"    IN  {i.name:30s} {i.shape}")
        for o in self.sess.get_outputs():
            print(f"    OUT {o.name:30s} {o.shape}")
