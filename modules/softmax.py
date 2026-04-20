from modules.layer import Layer

import numpy as np

class Softmax(Layer):
    def forward(self, input, training=True):  # input: [batch_size x num_classes]
        input = np.asarray(input, dtype=np.float32)
        shifted = input - np.max(input, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output.astype(np.float32, copy=False)

    def backward(self, grad_output, learning_rate=None):
        # Assuming softmax used with cross-entropy loss, so gradient is simplified
        return grad_output