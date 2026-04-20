from modules.utils import *
from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class Dense(Layer):
    def __init__(self, in_features, out_features, weight_init="he", use_gpu=False):
        self.in_features = in_features
        self.out_features = out_features
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

        if weight_init == "he":
            std = np.sqrt(2.0 / in_features)
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        elif weight_init == "custom":
            self.weights = np.zeros((in_features, out_features), dtype=np.float32)
        else:
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * (1 / in_features**0.5)

        self.biases = np.zeros(out_features, dtype=np.float32)

        self.input = None

    def forward(self, input, training=True):  # input: [batch_size x in_features]
        if self.use_gpu and isinstance(input, cp.ndarray):
            self.input = cp.asnumpy(input).astype(np.float32, copy=False)
        else:
            self.input = np.asarray(input, dtype=np.float32)
        xp = cp if self.use_gpu else np
        x = xp.asarray(input, dtype=np.float32)
        weights = xp.asarray(self.weights, dtype=np.float32)
        biases = xp.asarray(self.biases, dtype=np.float32)
        output = x @ weights + biases
        return output.astype(np.float32, copy=False)

    def backward(self, grad_output, learning_rate):
        grad_output = np.asarray(grad_output, dtype=np.float32)

        # Gradient w.r.t. weights: dL/dW = X^T * dL/dY
        grad_weights = self.input.T @ grad_output
        # Gradient w.r.t. biases: dL/db = sum_b dL/dY
        grad_biases = grad_output.sum(axis=0)

        # Gradient w.r.t. input: dL/dX = dL/dY * W^T
        grad_input = grad_output @ self.weights.T

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input.astype(np.float32, copy=False)
    
    def get_weights(self):
        return {'weights': self.weights, 'biases': self.biases}

    def set_weights(self, weights):
        self.weights = weights['weights']
        self.biases = weights['biases']
        