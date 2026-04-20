from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class BatchNorm2D(Layer):
    def __init__(self, num_channels, momentum=0.9, eps=1e-5, use_gpu=False):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)



        self.gamma = np.ones((1, num_channels, 1, 1), dtype=np.float32)  # scale
        self.beta = np.zeros((1, num_channels, 1, 1), dtype=np.float32)  # shift

        # Running stats for inference
        self.running_mean = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_channels, 1, 1), dtype=np.float32)


    # --- INICIO BLOQUE GENERADO CON IA ---
    def forward(self, x, training=True):
        xp = cp if self.use_gpu else np
        inp = xp.asarray(x, dtype=np.float32)
        gamma = xp.asarray(self.gamma, dtype=np.float32)
        beta = xp.asarray(self.beta, dtype=np.float32)

        if training:
            mean = inp.mean(axis=(0, 2, 3), keepdims=True)
            var = inp.var(axis=(0, 2, 3), keepdims=True)
            norm = (inp - mean) / ((var + self.eps) ** 0.5)
            out = gamma * norm + beta

            mean_np = cp.asnumpy(mean) if self.use_gpu else mean
            var_np = cp.asnumpy(var) if self.use_gpu else var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_np
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var_np
            # store numpy copies for backward
            if self.use_gpu and isinstance(x, cp.ndarray):
                self.input = cp.asnumpy(x).astype(np.float32, copy=False)
            else:
                self.input = np.asarray(x, dtype=np.float32)
            self.mean = mean_np
            self.var = var_np
            self.norm = cp.asnumpy(norm) if self.use_gpu else norm
        else:
            running_mean = xp.asarray(self.running_mean, dtype=np.float32)
            running_var = xp.asarray(self.running_var, dtype=np.float32)
            norm = (inp - running_mean) / ((running_var + self.eps) ** 0.5)
            out = gamma * norm + beta

        return out.astype(np.float32, copy=False)
    # --- FIN BLOQUE GENERADO CON IA ---

    def backward(self, grad_output, learning_rate):
        B, C, H, W = grad_output.shape
        N = B * H * W

        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        # Gradients of parameters
        grad_norm = grad_output * self.gamma
        grad_var = np.sum(grad_norm * (self.input - self.mean) * -0.5 * std_inv**3, axis=(0, 2, 3), keepdims=True)
        grad_mean = np.sum(grad_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + \
                    grad_var * np.mean(-2.0 * (self.input - self.mean), axis=(0, 2, 3), keepdims=True)

        # Gradient of input
        grad_input = grad_norm * std_inv + grad_var * 2.0 * (self.input - self.mean) / N + grad_mean / N

        # Update gamma and beta (SGD)
        grad_gamma = np.sum(grad_output * self.norm, axis=(0, 2, 3), keepdims=True)
        grad_beta = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)

        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta

        return grad_input

    def get_weights(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def set_weights(self, weights):
        self.gamma = weights["gamma"]
        self.beta = weights["beta"]
        self.running_mean = weights["running_mean"]
        self.running_var = weights["running_var"]
