# SAM2.1 TensorRT Issues on Jetson Orin (TRT 10.3 / SM_87)

## Summary

Running SAM2.1 Hiera Tiny on Jetson Orin AGX 64GB with JetPack 6 (TensorRT 10.3.0, SM_87 compute capability) revealed two critical TRT compilation bugs. Both are fixed in TRT 10.9+ but JetPack 6 ships TRT 10.3 with no upgrade path.

**Final working config: 3 TRT engines + 1 ONNX Runtime (mask_decoder)**

## The Two Bugs

### Bug 1: memory_attention — Myelin Assertion Failure

**Symptom:** `trtexec` crashes during engine build with:
```
bb.cpp:138: CHECK(op->parent() == this || op->parent() == nullptr)
```

**Root cause:** TRT 10.3 Myelin compiler fails on Cast(int64→float32) + Transpose + Reshape fusion patterns in RoPE attention layers.

**Fix found:** `--builderOptimizationLevel=5` forces Myelin to find alternative compilation paths. Engine builds and produces **correct results** (cosine similarity 1.000 vs ORT).

### Bug 2: mask_decoder — Silent Incorrect Results

**Symptom:** Engine builds successfully at any optimization level but produces completely wrong outputs.

**Root cause:** Identified via binary search of 690 ONNX nodes. The first divergence occurs at **node 241**: `cross_attn_token_to_image/k_proj/MatMul` — a simple linear projection.

```
Node 240: q_proj/Add  — OK  (cosine = 1.000000)
Node 241: k_proj/MatMul — BAD (cosine = 0.623724)
Node 243: v_proj/MatMul — BAD (cosine = -0.006517)
```

TRT 10.3 computes the k/v projection MatMuls incorrectly within the cross-attention transformer layers on SM_87. This is a kernel-level bug, not a fusion issue.

**Comparison across all optimization levels:**

| Level | obj_ptr cosine | pred_mask cosine | IoU ORT | IoU TRT |
|-------|---------------|-----------------|---------|---------|
| 0     | 0.387         | 0.977           | 0.31    | 0.80    |
| 1     | 0.387         | 0.977           | 0.31    | 0.80    |
| 2     | 0.387         | 0.977           | 0.31    | 0.80    |
| 3     | 0.387         | 0.977           | 0.31    | 0.80    |
| 4     | 0.387         | 0.977           | 0.31    | 0.80    |
| 5     | 0.387         | 0.977           | 0.31    | 0.80    |

All levels produce identical bad results.

## What We Tried (mask_decoder)

### TRT Build Flags — All Failed
- `--fp16` — bad results
- No `--fp16` (pure FP32) — bad results
- `--stronglyTyped` — bad results
- `--noTF32` — bad results
- `--builderOptimizationLevel=0..5` — all identical bad results
- `--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT` — bad results
- `onnxsim` (graph simplification) — bad results

### ONNX Graph Surgery — All Failed
- **Add(+0) MHA breaker**: Insert `Add(zero)` after QKV projections to break Myelin's MHA pattern recognition. TRT constant-folds it away.
- **Cast FP32→FP16→FP32 breaker**: Insert Cast ops after QKV projections. TRT still produces wrong results — bug is at kernel level, not fusion.
- **MatMul→Gemm replacement**: Replace QKV MatMul ops with Gemm (different kernel path). Failed: Gemm requires 2D inputs, mask_decoder uses 3D.

### ONNX Export Patches
- Removed `torch.repeat_interleave` → identity (eliminated OneHot/Tile pattern)
- Made all shapes static (batch=1, num_points=1, removed dynamic_axes)
- These fixed TRT build errors but didn't fix the incorrect results

## Evidence It's Fixed in TRT 10.9+

- Same ONNX files (md5 identical) produce correct TRT engines on PC with TRT >=10.9
- PC achieves 0.95+ mean IoU with all 4 TRT engines
- Jetson with TRT 10.3 gets incorrect results from mask_decoder TRT regardless of flags

## Final Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  image_encoder   │     │  memory_encoder   │
│    (TRT FP16)    │     │    (TRT FP16)     │
│    ~34ms GPU     │     │    ~2ms GPU       │
└─────────────────┘     └──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│ memory_attention  │     │  mask_decoder     │
│ (TRT FP16, opt5) │     │ (ONNX Runtime)    │
│   ~15ms GPU      │     │   ~19ms           │
└──────────────────┘     └──────────────────┘
```

**Total pipeline: ~70-80ms/frame ≈ 12-14 FPS on Orin AGX 64GB**

The mask_decoder ORT overhead (~19ms vs ~3ms TRT) costs about 1-2 FPS. This is the best achievable performance on JetPack 6 / TRT 10.3 without upgrading TensorRT.

## Diagnostic Tools Used

- `app/compare_mask_decoder.py` — compares ORT vs TRT outputs per-tensor (cosine similarity, max/mean diff, binary mask diff)
- `app/break_mha.py` — ONNX graph surgery tool (Add/Cast/Gemm MHA breakers)
- Binary search script — cuts ONNX graph at intermediate nodes to find exact divergence point
