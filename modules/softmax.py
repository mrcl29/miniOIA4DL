from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class Softmax(Layer):
    def __init__(self, use_gpu=False):
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)
    
    def forward(self, input, training=True):  # input: [batch_size x num_classes]
        if self.use_gpu:
            input_gpu = cp.asarray(input, dtype=cp.float32)
            shifted = input_gpu - cp.max(input_gpu, axis=1, keepdims=True)
            exps = cp.exp(shifted)
            self.output = exps / cp.sum(exps, axis=1, keepdims=True)
            return self.output.astype(cp.float32, copy=False)
        else:
            input = np.asarray(input, dtype=np.float32)
            shifted = input - np.max(input, axis=1, keepdims=True)
            exps = np.exp(shifted)
            self.output = exps / np.sum(exps, axis=1, keepdims=True)
            return self.output.astype(np.float32, copy=False)

    def backward(self, grad_output, learning_rate=None):
        # Assuming softmax used with cross-entropy loss, so gradient is simplified
        return grad_output