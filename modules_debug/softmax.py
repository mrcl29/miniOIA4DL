from modules_debug.layer import Layer
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
    
    import numpy as np

    def softmax_forward_debug(self, input):
        print("=== INPUT ===")
        print(input)
        print("shape:", input.shape)

        input = np.asarray(input, dtype=np.float32)

        print("\n=== PASO 1: SHIFT (restar max por fila) ===")
        max_per_row = np.max(input, axis=1, keepdims=True)
        print("max_per_row:")
        print(max_per_row)

        shifted = input - max_per_row
        print("shifted:")
        print(shifted)

        print("\n=== PASO 2: EXP ===")
        exps = np.exp(shifted)
        print("exps:")
        print(exps)

        print("\n=== PASO 3: SUMA POR FILA ===")
        sums = np.sum(exps, axis=1, keepdims=True)
        print("sums:")
        print(sums)

        print("\n=== PASO 4: NORMALIZACIÓN ===")
        output = exps / sums
        print("output (softmax):")
        print(output)

        print("\n=== CHECK: cada fila suma 1 ===")
        print(np.sum(output, axis=1))

        return output
    def backward(self, grad_output, learning_rate=None):
        # Assuming softmax used with cross-entropy loss, so gradient is simplified
        return grad_output

layer = Softmax(use_gpu=False)

X = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 0.2]
])

out = layer.softmax_forward_debug(X)

print("\n=== OUTPUT FINAL ===")
print(out)

print("\nSuma por fila:")
print(np.sum(out, axis=1))