from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


class ReLUDebug(Layer):
    def __init__(self, use_gpu=False):
        self.input = None
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward_debug(self, x):
        print("=== REUL DEBUG FORWARD ===")

        if self.use_gpu:
            xp = cp
            print("Modo GPU")
        else:
            xp = np
            print("Modo CPU")

        x_arr = xp.asarray(x, dtype=xp.float32)

        print("\nInput:")
        print(x_arr)
        print("shape:", x_arr.shape)
        print("dtype:", x_arr.dtype)

        # Guardamos input como en forward normal
        self.input = x_arr

        print("\nPaso 1: máscara (x > 0)")
        mask = x_arr > 0
        print(mask)

        print("\nPaso 2: aplicar ReLU")
        output = xp.maximum(0, x_arr)
        print(output)

        print("\nCheck: valores negativos eliminados")
        print("min(output):", output.min())

        return output

    def backward_debug(self, grad_output):
        print("\n=== RELU DEBUG BACKWARD ===")

        if self.use_gpu:
            xp = cp
            print("Modo GPU")
        else:
            xp = np
            print("Modo CPU")

        grad_output = xp.asarray(grad_output, dtype=xp.float32)

        print("\ngrad_output:")
        print(grad_output)

        print("\nMáscara (input > 0):")
        mask = self.input > 0
        print(mask)

        grad_input = grad_output * mask

        print("\ngrad_input:")
        print(grad_input)

        return grad_input

layer = ReLUDebug(use_gpu=False)

X = np.array([
    [-1.0, 2.0, -0.5],
    [3.0, -4.0, 0.0]
])

# Forward debug
out = layer.forward_debug(X)

# Backward debug
grad_output = np.ones_like(X)
grad = layer.backward_debug(grad_output)