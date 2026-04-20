from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class Flatten(Layer):
    def __init__(self, use_gpu=False):
        self.input_shape = None
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        if self.use_gpu:
            x = cp.asarray(input, dtype=np.float32)
            self.input_shape = x.shape
            return x.reshape(x.shape[0], -1)
        else:
            self.input_shape = input.shape
            return input.reshape(input.shape[0], -1)

    def backward(self, grad_output, learning_rate=None):
        return grad_output.reshape(self.input_shape)