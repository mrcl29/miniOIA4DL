from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class ReLU(Layer):
    def __init__(self, use_gpu=False):
        self.input = None
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward(self, x, training=True):
        if self.use_gpu:
            # --- INICIO BLOQUE GENERADO CON IA ---
            self.input = cp.asarray(x, dtype=np.float32)
            return cp.maximum(0, self.input)
            # --- FIN BLOQUE GENERADO CON IA ---
        else:
            self.input = np.asarray(x, dtype=np.float32)
            return np.maximum(0, self.input)

    def backward(self, grad_output, learning_rate=None):
        return grad_output * (self.input > 0)
