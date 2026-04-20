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

    def forward_debug(self, x, training=True):
        print("=== DROPOUT FORWARD DEBUG ===")
        print(f"p={self.p}, use_gpu={self.use_gpu}, training={training}")

        xp = cp if self.use_gpu else np
        x_arr = xp.asarray(x, dtype=xp.float32)

        print("\n[1] Input")
        print("shape:", x_arr.shape)
        print(cp.asnumpy(x_arr) if self.use_gpu else x_arr)

        if training:
            print("\n[2] Generando máscara de dropout")
            print(f"Probabilidad de apagar neurona (p): {self.p}")
            print(f"Probabilidad de mantener (1-p): {1.0 - self.p}")

            rand_vals = xp.random.rand(*x_arr.shape)
            mask = (rand_vals > self.p).astype(x_arr.dtype)
            self.mask = mask

            print("\nValores aleatorios:")
            print(cp.asnumpy(rand_vals) if self.use_gpu else rand_vals)

            print("\nMáscara (1 = activo, 0 = apagado):")
            print(cp.asnumpy(mask) if self.use_gpu else mask)

            kept = mask.sum()
            total = mask.size
            print(f"\nNeuronas activas: {kept}/{total} ({kept/total:.2%})")

            print("\n[3] Aplicando máscara y escalado")
            scale = 1.0 / (1.0 - self.p)
            print("Factor de escala:", scale)

            out = x_arr * mask * scale

            print("\nOutput:")
            print(cp.asnumpy(out) if self.use_gpu else out)

            return out
        else:
            print("\n[2] Modo inferencia → no se aplica dropout")
            print("Output = input")
            return x_arr


    def backward_debug(self, grad_output, learning_rate=None):
        print("\n=== DROPOUT BACKWARD DEBUG ===")

        xp = cp if self.use_gpu else np
        grad_output = xp.asarray(grad_output, dtype=xp.float32)

        print("\n[1] grad_output recibido")
        print("shape:", grad_output.shape)
        print(cp.asnumpy(grad_output) if self.use_gpu else grad_output)

        print("\n[2] Máscara usada en forward")
        print(cp.asnumpy(self.mask) if self.use_gpu else self.mask)

        scale = 1.0 / (1.0 - self.p)
        print("\nFactor de escala:", scale)

        grad_input = grad_output * self.mask * scale

        print("\n[3] grad_input resultante")
        print(cp.asnumpy(grad_input) if self.use_gpu else grad_input)

        print("\n[4] Explicación")
        print("Solo las neuronas activas reciben gradiente")
        print("Las apagadas (mask=0) tienen gradiente 0")

        return grad_input

x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=np.float32)

layer = Dropout(p=0.5)

out = layer.forward_debug(x, training=True)

grad_output = np.ones_like(x)
grad_input = layer.backward_debug(grad_output)