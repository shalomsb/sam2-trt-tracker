# SAM2-TRT-Tracker

Real-time video object segmentation using **SAM2.1 Hiera Tiny** accelerated with **TensorRT FP16**.

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

## Performance

| Metric | Value |
|--------|-------|
| GPU | NVIDIA RTX 5070 Ti |
| Backend | TensorRT FP16 |
| Throughput | ~65 fps (1280x720 input) |
| GPU Memory | ~4 GB |
| Accuracy | Mean IoU 0.983 vs PyTorch FP32 reference (360 frames, zero below 0.95) |

## Pipeline

```
                    GPU (zero-copy)
 Video ──▶ Preprocess ──▶ Image Encoder ──┬──▶ Mask Decoder ──▶ Overlay ──▶ Video
  (threaded I/O)                          │        ▲                        Output
                                          │        │
                                          ▼        │
                                     Memory Bank ──┘
                                    (sliding window)
```

Four TensorRT FP16 engines run on a single CUDA stream:
- **Image Encoder** — Hiera Tiny backbone (1024x1024 input)
- **Memory Attention** — fuses current frame with memory bank
- **Mask Decoder** — predicts segmentation mask from prompts or memory
- **Memory Encoder** — encodes mask features for future frame tracking

All intermediate tensors stay on GPU. Only the raw video frame (in) and binary mask (out) cross PCIe.

## Project Structure

```
sam2-trt-tracker/
├── app/
│   ├── sam2_trt_video_tracker.py      # TensorRT tracker (main)
│   ├── sam2_onnx_video_tracker.py     # ONNX Runtime tracker
│   ├── sam2_pytorch_video_tracker.py  # PyTorch reference tracker
│   └── compare_masks.py              # IoU comparison tool
├── docker/
│   ├── Dockerfile                     # Based on nvcr.io/nvidia/tensorrt:26.02-py3
│   ├── launch.sh                      # Host-side entry point
│   ├── entrypoint.sh                  # Container entrypoint
│   ├── requirements.txt
│   └── scripts/
│       ├── setup_models.sh            # Clone SAM2 repo, export ONNX
│       ├── build_trt_engines.sh       # Convert ONNX → TRT engines
│       ├── run_trt_tracker.sh
│       ├── run_onnx_tracker.sh
│       └── run_comparison.sh          # TRT vs PyTorch IoU comparison
├── models/sam2/tiny/
│   ├── onnx/                          # Exported ONNX models (generated)
│   └── trt/                           # TensorRT engines (generated)
├── streams/                           # Input videos
│   └── sample.mp4
└── output/                            # Output videos & masks (generated)
```

## Requirements

- NVIDIA GPU with TensorRT support
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~4 GB GPU memory

## Quick Start

### 1. Build

```bash
cd docker
./launch.sh -b
```

This will:
1. Build the Docker image from `nvcr.io/nvidia/tensorrt:26.02-py3`
2. Clone NVIDIA's SAM2 ONNX exporter and download the SAM2.1 Tiny checkpoint
3. Export 4 ONNX models
4. Convert ONNX to TensorRT FP16 engines (saved to `models/sam2/tiny/trt/`)

### 2. Run

```bash
# Track with default point prompt (center of frame)
./launch.sh -r

# Custom point prompt
POINT=200,150 ./launch.sh -r

# Bounding box prompt
BBOX=100,50,300,250 ./launch.sh -r

# Custom video
VIDEO=/streams/my_video.mp4 ./launch.sh -r

# ONNX Runtime backend (slower, no TRT needed)
./launch.sh -r --onnx
```

### 3. Compare TRT vs PyTorch accuracy

```bash
./launch.sh -c
```

Runs both TRT and PyTorch trackers on the same 15-second clip, then computes per-frame IoU. Outputs:
- `output/masks_trt/` — TRT binary masks
- `output/masks_pytorch/` — PyTorch binary masks
- `output/iou_trt_vs_pytorch.csv` — per-frame IoU values
- `output/iou_overlay_trt_vs_pt/` — side-by-side mask visualizations

### 4. Development shell

```bash
./launch.sh -d
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO` | `/streams/sample.mp4` | Input video path (inside container) |
| `POINT` | `640,360` | Point prompt `x,y` on frame 0 |
| `BBOX` | *(none)* | Bounding box `x1,y1,x2,y2` (overrides `POINT`) |
| `OUTPUT` | `/output/output_tracked_trt.mp4` | Output video path |
| `MODEL_DIR` | `/models/sam2/tiny/trt/` | TensorRT engines directory |
| `DURATION` | `15` | Clip duration for comparison mode (`-c`) |

## How It Works

- **Frame 0:** User prompt (bbox or point) is scaled to 1024x1024 and passed to the mask decoder for initial segmentation
- **Frame 1+:** A memory bank (sliding window of recent frames + the conditioning frame) feeds into memory attention, which fuses past context with the current frame to propagate the mask without further prompts
- Threaded video I/O overlaps disk reads/writes with GPU compute

The TRT engines use `NUM_MASKMEM=3` (1 conditioning + 2 recent frames) for memory efficiency. Despite this and FP16 quantization, accuracy remains high (IoU 0.983 vs PyTorch FP32 with `NUM_MASKMEM=7`).

See [`docs/sam2_trt_video_tracker_explained.md`](docs/sam2_trt_video_tracker_explained.md) for a detailed architecture walkthrough.

## License

This project uses [SAM2](https://github.com/facebookresearch/sam2) by Meta Research and [NVIDIA's SAM2 ONNX/TensorRT tools](https://github.com/NVIDIA-AI-IOT/deepstream_tools).
