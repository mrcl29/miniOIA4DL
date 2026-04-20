from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class Dropout(Layer):
    def __init__(self, p=0.5, use_gpu=False):
        self.p = p
        self.mask = None
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward(self, x, training=True):
        xp = cp if self.use_gpu else np
        x = xp.asarray(x, dtype=np.float32)
        if training:
            # Create dropout mask: keep rate is (1 - p)
            self.mask = (xp.random.rand(*x.shape) > self.p).astype(x.dtype)
            return x * self.mask / (1.0 - self.p)
        else:
            return x

    def backward(self, grad_output, learning_rate=None):
        return grad_output * self.mask / (1.0 - self.p)
#        else:
#            return grad_output

    