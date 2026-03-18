# SAM2-TRT-Tracker

Real-time video object segmentation using **SAM2.1 Hiera Tiny** accelerated with **TensorRT**.

Give it a video and a prompt (bounding box or point) on the first frame — it segments and tracks the object through every subsequent frame, entirely on the GPU.

```
Frame 0 (prompted)          Frame N (propagated)
┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │
│    ┌───────┐        │     │   ╭───────╮         │
│    │ bbox  │        │     │   │ mask  │         │
│    │prompt │        │     │   │(auto) │         │
│    └───────┘        │     │   ╰───────╯         │
│                     │     │                     │
└─────────────────────┘     └─────────────────────┘
```

## Pipeline

```
                    GPU
 Video ──▶ Preprocess ──▶ Image Encoder ──┬──▶ Mask Decoder ──▶ Overlay ──▶ Video
                                          │        ▲                        Output
                                          │        │
                                          ▼        │
                                     Memory Bank ──┘
                                    (sliding window)
```

Four TensorRT FP16 engines run on a single CUDA stream:
- **Image Encoder** — Hiera Tiny backbone
- **Memory Attention** — fuses current frame with memory bank
- **Mask Decoder** — predicts segmentation mask
- **Memory Encoder** — stores frame features for future tracking

## Project Structure

```
sam2-trt-tracker/
├── app/
│   ├── sam2_trt_video_tracker.py      # TensorRT tracker (main)
│   └── sam2_onnx_video_tracker.py     # ONNX Runtime tracker (reference)
├── docker/
│   ├── Dockerfile                     # Based on nvcr.io/nvidia/tensorrt:26.02-py3
│   ├── launch.sh                      # Build / Run / Dev entry point
│   ├── entrypoint.sh                  # Container entrypoint
│   ├── requirements.txt               # Python dependencies
│   └── scripts/
│       ├── setup_models.sh            # Clone SAM2 repo, export ONNX
│       ├── build_trt_engines.sh       # Convert ONNX to TRT engines
│       ├── run_trt_tracker.sh         # Run TRT tracker
│       └── run_onnx_tracker.sh        # Run ONNX tracker
├── models/sam2/tiny/
│   ├── onnx/                          # Exported ONNX models (generated)
│   └── trt/                           # TensorRT engines (generated)
├── streams/                           # Input videos
│   └── sample.mp4
├── output/                            # Output videos (gitignored)
└── docs/                              # Architecture documentation
```

## Requirements

- NVIDIA GPU (tested on Jetson Orin / desktop GPUs with TensorRT)
- Docker with NVIDIA Container Toolkit
- ~4 GB GPU memory

## Quick Start

### 1. Build the Docker image and export models

```bash
./docker/launch.sh -b
```

This will:
1. Build the Docker image from `nvcr.io/nvidia/tensorrt:26.02-py3`
2. Clone NVIDIA's SAM2 ONNX exporter, download the SAM2.1 Tiny checkpoint
3. Export 4 ONNX models
4. Convert ONNX to TensorRT FP16 engines

The TRT engines are saved to `models/sam2/tiny/trt/` (persisted on host).

### 2. Run the tracker

```bash
# Track with a point prompt (default: center of frame)
./docker/launch.sh -r

# Custom point
POINT=200,150 ./docker/launch.sh -r

# Bounding box prompt
BBOX=100,50,300,250 ./docker/launch.sh -r

# Custom input/output
VIDEO=/streams/my_video.mp4 OUTPUT=/output/result.mp4 ./docker/launch.sh -r
```

### 3. Development shell

```bash
./docker/launch.sh -d
```

### ONNX Runtime (alternative backend)

```bash
./docker/launch.sh -r --onnx
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO` | `/streams/sample.mp4` | Input video path (inside container) |
| `POINT` | `640,360` | Point prompt `x,y` on frame 0 |
| `BBOX` | — | Bounding box `x1,y1,x2,y2` (overrides `POINT`) |
| `OUTPUT` | `/output/output_tracked_trt.mp4` | Output video path |
| `MODEL_DIR` | `/models/sam2/tiny/trt/` | TensorRT engines directory |

## How It Works

- **Frame 0**: User prompt (bbox/point) is passed to the mask decoder to get the initial segmentation
- **Frame 1+**: A memory bank (sliding window of recent frames + the conditioning frame) feeds into memory attention, which fuses past context with the current frame's features to propagate the mask
- All tensor operations stay on GPU — only the raw frame and final binary mask cross PCIe
- Threaded video I/O overlaps disk reads/writes with GPU compute

See [`docs/sam2_trt_video_tracker_explained.md`](docs/sam2_trt_video_tracker_explained.md) for a detailed architecture walkthrough with diagrams.

## License

This project uses [SAM2](https://github.com/facebookresearch/sam2) by Meta Research and [NVIDIA's SAM2 ONNX/TensorRT exporter](https://github.com/NVIDIA-AI-IOT/deepstream_tools).
