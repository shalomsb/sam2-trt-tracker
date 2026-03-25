# SAM2-TRT-Tracker

Real-time video object segmentation using **SAM2.1 Hiera Tiny** accelerated with **TensorRT** on **Jetson Orin**.

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

Three backends are available:

- **TRT** (default) — 4 TensorRT FP16 engines on a single CUDA stream
- **Hybrid** — 3 TRT engines + ONNX Runtime CUDA EP for memory attention (TRT 10.3 Myelin bug workaround)
- **ONNX** — pure ONNX Runtime with CUDA EP (reference implementation)

Engines:
- **Image Encoder** — Hiera Tiny backbone
- **Memory Attention** — fuses current frame with memory bank
- **Mask Decoder** — predicts segmentation mask
- **Memory Encoder** — stores frame features for future tracking

## Project Structure

```
sam2-trt-tracker/
├── app/
│   ├── sam2_trt_video_tracker.py      # TensorRT tracker
│   ├── sam2_hybrid_video_tracker.py   # Hybrid TRT + ONNX Runtime tracker
│   ├── sam2_onnx_video_tracker.py     # ONNX Runtime tracker (reference)
│   ├── compare_masks.py              # IoU comparison between mask sets
│   └── overlay_compare_video.py      # Side-by-side overlay video generator
├── docker/
│   ├── Dockerfile                     # Based on nvcr.io/nvidia/l4t-jetpack:r36.4.0
│   ├── launch.sh                      # Build / Run / Compare / Dev entry point
│   ├── entrypoint.sh                  # Container entrypoint
│   ├── requirements.txt               # Python dependencies
│   └── scripts/
│       ├── setup_models.sh            # Clone SAM2 repo, export ONNX
│       ├── build_trt_engines.sh       # Convert ONNX to TRT engines
│       ├── run_trt_tracker.sh         # Run TRT tracker
│       ├── run_onnx_tracker.sh        # Run ONNX tracker
│       ├── run_hybrid_tracker.sh      # Run hybrid tracker
│       └── run_compare.sh            # Run comparison pipeline
├── models/sam2/tiny/
│   ├── onnx/                          # Exported ONNX models (generated)
│   └── trt/                           # TensorRT engines (generated)
├── streams/                           # Input videos
│   └── sample.mp4
├── output/                            # Output videos (gitignored)
└── docs/                              # Architecture documentation
```

## Requirements

- NVIDIA Jetson Orin (or desktop GPU with TensorRT)
- Docker with NVIDIA Container Toolkit
- ~4 GB GPU memory

## Quick Start

### 1. Build the Docker image and export models

```bash
./docker/launch.sh -b
```

This will:
1. Build the Docker image from `nvcr.io/nvidia/l4t-jetpack:r36.4.0`
2. Clone NVIDIA's SAM2 ONNX exporter, download the SAM2.1 Tiny checkpoint
3. Export 4 ONNX models
4. Convert ONNX to TensorRT FP16 engines

The TRT engines are saved to `models/sam2/tiny/trt/` (persisted on host).

### 2. Run the tracker

```bash
# TensorRT backend (default) — point prompt at center of frame
./docker/launch.sh -r

# Custom point prompt
POINT=200,150 ./docker/launch.sh -r

# Bounding box prompt
BBOX=100,50,300,250 ./docker/launch.sh -r

# Hybrid backend (TRT + ONNX Runtime for memory attention)
./docker/launch.sh -r --hybrid

# Hybrid with bounding box output instead of mask overlay
./docker/launch.sh -r --hybrid --bbox

# ONNX Runtime backend
./docker/launch.sh -r --onnx

# Custom input/output
VIDEO=/streams/my_video.mp4 OUTPUT=/output/result.mp4 ./docker/launch.sh -r
```

### 3. Compare hybrid vs PyTorch (mask IoU)

```bash
# Mask comparison — runs hybrid + PyTorch trackers, computes per-frame IoU
./docker/launch.sh -c

# Bounding box comparison
./docker/launch.sh -c --bbox
```

This runs a 4-step pipeline:
1. Run the hybrid tracker (saves per-frame masks/bboxes)
2. Run the PyTorch reference tracker (downloads checkpoint automatically)
3. Compare outputs via IoU, write CSV report
4. Generate a side-by-side overlay video

Results are written to `output/`:
- `compare_iou.csv` — per-frame IoU scores
- `compare_overlay_video.mp4` — side-by-side visualization
- `compare_overlays/` — per-frame overlay images (mask mode)

### 4. Development shell

```bash
./docker/launch.sh -d
```

## launch.sh Reference

```
usage: ./docker/launch.sh [-b/-d/-r/-c] [--onnx/--hybrid] [--bbox]

Actions:
  -b          Build Docker image + export ONNX models + build TRT engines
  -r          Run tracker
  -c          Compare hybrid vs PyTorch (mask/bbox IoU)
  -d          Development shell inside container

Backend (used with -r):
  (default)   TensorRT (all 4 engines)
  --hybrid    TRT + ONNX Runtime for memory attention
  --onnx      ONNX Runtime only

Options:
  --bbox      Output bounding boxes instead of mask overlays (used with -r --hybrid or -c)
  -h          Show usage
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO` | `/streams/sample.mp4` | Input video path (inside container) |
| `POINT` | `640,360` | Point prompt `x,y` on frame 0 |
| `BBOX` | — | Bounding box `x1,y1,x2,y2` (overrides `POINT`) |
| `OUTPUT` | `/output/output_tracked_trt.mp4` | Output video path |
| `MODEL_DIR` | `/models/sam2/tiny/trt/` | TensorRT engines directory |
| `ONNX_DIR` | `/models/sam2/tiny/onnx/` | ONNX models directory (hybrid/onnx) |
| `CHECKPOINT` | `/models/sam2/tiny/sam2.1_hiera_tiny.pt` | PyTorch checkpoint (compare mode) |
| `MAX_FRAMES` | `350` | Max frames for comparison runs |
| `SAVE_BBOXES` | `/output/bboxes.csv` | Bounding box CSV output (hybrid --bbox) |

## How It Works

- **Frame 0**: User prompt (bbox/point) is passed to the mask decoder to get the initial segmentation
- **Frame 1+**: A memory bank (sliding window of recent frames + the conditioning frame) feeds into memory attention, which fuses past context with the current frame's features to propagate the mask
- All tensor operations stay on GPU — only the raw frame and final binary mask cross PCIe
- Threaded video I/O overlaps disk reads/writes with GPU compute

See [`docs/sam2_trt_video_tracker_explained.md`](docs/sam2_trt_video_tracker_explained.md) for a detailed architecture walkthrough with diagrams.

## License

This project uses [SAM2](https://github.com/facebookresearch/sam2) by Meta Research and [NVIDIA's SAM2 ONNX/TensorRT exporter](https://github.com/NVIDIA-AI-IOT/deepstream_tools).
