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

    
    
    def forward_debug(self, x, training=True):
        print("=== BATCHNORM2D FORWARD DEBUG ===")
        print(
            f"num_channels={self.num_channels}, momentum={self.momentum}, "
            f"eps={self.eps}, use_gpu={self.use_gpu}, training={training}"
        )

        xp = cp if self.use_gpu else np
        inp = xp.asarray(x, dtype=xp.float32 if self.use_gpu else np.float32)
        gamma = xp.asarray(self.gamma, dtype=xp.float32 if self.use_gpu else np.float32)
        beta = xp.asarray(self.beta, dtype=xp.float32 if self.use_gpu else np.float32)

        print("\n[1] Input")
        print("shape:", inp.shape)
        print(cp.asnumpy(inp) if self.use_gpu else inp)

        print("\n[2] Parámetros")
        print("gamma.shape:", gamma.shape)
        print(cp.asnumpy(gamma) if self.use_gpu else gamma)
        print("beta.shape:", beta.shape)
        print(cp.asnumpy(beta) if self.use_gpu else beta)

        print("\n[3] Running stats antes del forward")
        print("running_mean.shape:", self.running_mean.shape)
        print(self.running_mean)
        print("running_var.shape:", self.running_var.shape)
        print(self.running_var)

        if training:
            print("\n[4] Modo training: calcular mean y var por canal")
            print("Ejes usados: axis=(0, 2, 3)  -> batch y dimensiones espaciales")

            mean = inp.mean(axis=(0, 2, 3), keepdims=True)
            var = inp.var(axis=(0, 2, 3), keepdims=True)

            print("mean.shape:", mean.shape)
            print(cp.asnumpy(mean) if self.use_gpu else mean)

            print("var.shape:", var.shape)
            print(cp.asnumpy(var) if self.use_gpu else var)

            print("\n[5] Normalización")
            centered = inp - mean
            std = (var + self.eps) ** 0.5
            norm = centered / std

            print("inp - mean:")
            print(cp.asnumpy(centered) if self.use_gpu else centered)
            print("sqrt(var + eps):")
            print(cp.asnumpy(std) if self.use_gpu else std)
            print("norm:")
            print(cp.asnumpy(norm) if self.use_gpu else norm)

            print("\n[6] Escalado y desplazamiento")
            out = gamma * norm + beta
            print("out = gamma * norm + beta")
            print(cp.asnumpy(out) if self.use_gpu else out)

            mean_np = cp.asnumpy(mean) if self.use_gpu else mean
            var_np = cp.asnumpy(var) if self.use_gpu else var

            print("\n[7] Actualización de running stats")
            print("Fórmulas:")
            print("running_mean = momentum * running_mean + (1 - momentum) * mean")
            print("running_var  = momentum * running_var  + (1 - momentum) * var")

            old_running_mean = self.running_mean.copy()
            old_running_var = self.running_var.copy()

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_np
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var_np

            print("running_mean antes:")
            print(old_running_mean)
            print("running_mean después:")
            print(self.running_mean)

            print("running_var antes:")
            print(old_running_var)
            print("running_var después:")
            print(self.running_var)

            if self.use_gpu and isinstance(x, cp.ndarray):
                self.input = cp.asnumpy(x).astype(np.float32, copy=False)
            else:
                self.input = np.asarray(x, dtype=np.float32)

            self.mean = mean_np
            self.var = var_np
            self.norm = cp.asnumpy(norm) if self.use_gpu else norm

            print("\n[8] Estado guardado para backward")
            print("self.input.shape:", self.input.shape)
            print("self.mean.shape:", self.mean.shape)
            print("self.var.shape:", self.var.shape)
            print("self.norm.shape:", self.norm.shape)

            return out.astype(np.float32, copy=False)

        else:
            print("\n[4] Modo inferencia: usar running_mean y running_var")
            running_mean = xp.asarray(self.running_mean, dtype=xp.float32 if self.use_gpu else np.float32)
            running_var = xp.asarray(self.running_var, dtype=xp.float32 if self.use_gpu else np.float32)

            print("running_mean:")
            print(cp.asnumpy(running_mean) if self.use_gpu else running_mean)
            print("running_var:")
            print(cp.asnumpy(running_var) if self.use_gpu else running_var)

            norm = (inp - running_mean) / ((running_var + self.eps) ** 0.5)
            out = gamma * norm + beta

            print("\n[5] Output en inferencia")
            print(cp.asnumpy(out) if self.use_gpu else out)

            return out.astype(np.float32, copy=False)


    def backward_debug(self, grad_output, learning_rate):
        print("\n=== BATCHNORM2D BACKWARD DEBUG ===")
        print("learning_rate:", learning_rate)

        grad_output = np.asarray(grad_output, dtype=np.float32)

        print("\n[1] grad_output")
        print("shape:", grad_output.shape)
        print(grad_output)

        print("\n[2] Estado guardado del forward")
        print("input.shape:", self.input.shape)
        print(self.input)
        print("mean.shape:", self.mean.shape)
        print(self.mean)
        print("var.shape:", self.var.shape)
        print(self.var)
        print("norm.shape:", self.norm.shape)
        print(self.norm)

        print("\n[3] Parámetros actuales")
        print("gamma.shape:", self.gamma.shape)
        print(self.gamma)
        print("beta.shape:", self.beta.shape)
        print(self.beta)

        B, C, H, W = grad_output.shape
        N = B * H * W

        print("\n[4] Tamaños")
        print(f"B={B}, C={C}, H={H}, W={W}, N={N}")

        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        print("\n[5] std_inv = 1 / sqrt(var + eps)")
        print(std_inv)

        print("\n[6] grad_norm = grad_output * gamma")
        grad_norm = grad_output * self.gamma
        print("grad_norm.shape:", grad_norm.shape)
        print(grad_norm)

        print("\n[7] grad_var")
        print("Fórmula: sum(grad_norm * (input - mean) * -0.5 * std_inv**3, axis=(0,2,3), keepdims=True)")
        grad_var = np.sum(
            grad_norm * (self.input - self.mean) * -0.5 * std_inv**3,
            axis=(0, 2, 3),
            keepdims=True
        )
        print("grad_var.shape:", grad_var.shape)
        print(grad_var)

        print("\n[8] grad_mean")
        grad_mean = (
            np.sum(grad_norm * -std_inv, axis=(0, 2, 3), keepdims=True) +
            grad_var * np.mean(-2.0 * (self.input - self.mean), axis=(0, 2, 3), keepdims=True)
        )
        print("grad_mean.shape:", grad_mean.shape)
        print(grad_mean)

        print("\n[9] grad_input")
        grad_input = (
            grad_norm * std_inv +
            grad_var * 2.0 * (self.input - self.mean) / N +
            grad_mean / N
        )
        print("grad_input.shape:", grad_input.shape)
        print(grad_input)

        print("\n[10] Gradientes de parámetros")
        grad_gamma = np.sum(grad_output * self.norm, axis=(0, 2, 3), keepdims=True)
        grad_beta = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)

        print("grad_gamma.shape:", grad_gamma.shape)
        print(grad_gamma)
        print("grad_beta.shape:", grad_beta.shape)
        print(grad_beta)

        print("\n[11] Actualización de gamma y beta")
        old_gamma = self.gamma.copy()
        old_beta = self.beta.copy()

        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta

        print("gamma antes:")
        print(old_gamma)
        print("gamma después:")
        print(self.gamma)

        print("beta antes:")
        print(old_beta)
        print("beta después:")
        print(self.beta)

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


bn = BatchNorm2D(num_channels=2, momentum=0.9, eps=1e-5, use_gpu=False)

x = np.array([
    [
        [[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]
    ],
    [
        [[2, 3],
         [4, 5]],

        [[6, 7],
         [8, 9]]
    ]
], dtype=np.float32)  # shape (2, 2, 2, 2)

out = bn.forward_debug(x, training=True)

grad_output = np.ones_like(x, dtype=np.float32)
grad_input = bn.backward_debug(grad_output, learning_rate=0.01)