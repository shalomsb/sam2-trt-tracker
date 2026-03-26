import numpy as np
import torch

NUM_MASKMEM = 3
IMG_SIZE = 1024

_NP_TO_TORCH = {
    np.dtype("float32"): torch.float32,
    np.dtype("float16"): torch.float16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
}
