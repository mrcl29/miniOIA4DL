import numpy as np
from modules.layer import Layer

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class GlobalAvgPool2D(Layer):

    def __init__(self, use_gpu=False):
        self.input = None
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward(self, x, training=True):  # shape: [batch, channels, h, w]
        if self.use_gpu:
            x_gpu = cp.asarray(x, dtype=np.float32)
            self.input_shape = x_gpu.shape
            return cp.mean(x_gpu, axis=(2, 3), keepdims=False).astype(cp.float32)
        else:
            self.input_shape = x.shape
            return np.mean(x, axis=(2, 3), keepdims=False).astype(np.float32)

    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, h, w = self.input_shape
        grad = grad_output[:, :, None, None] / (h * w)  # shape: [batch, channels, 1, 1]
        return np.ones((batch_size, channels, h, w), dtype=np.float32) * grad