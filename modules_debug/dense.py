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

    def forward_debug(self, input, training=True):
        print("=== DENSE FORWARD DEBUG ===")
        print(f"in_features={self.in_features}, out_features={self.out_features}, use_gpu={self.use_gpu}")

        if self.use_gpu and isinstance(input, cp.ndarray):
            self.input = cp.asnumpy(input).astype(np.float32, copy=False)
        else:
            self.input = np.asarray(input, dtype=np.float32)

        xp = cp if self.use_gpu else np
        x = xp.asarray(input, dtype=np.float32)
        weights = xp.asarray(self.weights, dtype=np.float32)
        biases = xp.asarray(self.biases, dtype=np.float32)

        print("\n[1] Input")
        print("input.shape:", x.shape)
        print(cp.asnumpy(x) if self.use_gpu else x)

        print("\n[2] Weights")
        print("weights.shape:", weights.shape)
        print(cp.asnumpy(weights) if self.use_gpu else weights)

        print("\n[3] Biases")
        print("biases.shape:", biases.shape)
        print(cp.asnumpy(biases) if self.use_gpu else biases)

        print("\n[4] Multiplicación matricial: x @ weights")
        matmul = x @ weights
        print("matmul.shape:", matmul.shape)
        print(cp.asnumpy(matmul) if self.use_gpu else matmul)

        print("\n[5] Suma del bias: (x @ weights) + biases")
        output = matmul + biases
        print("output.shape:", output.shape)
        print(cp.asnumpy(output) if self.use_gpu else output)

        print("\n[6] Explicación")
        print("Cada fila del input se proyecta al espacio de salida")
        print("Fórmula: output = input @ weights + biases")

        return output.astype(np.float32, copy=False)


    def backward_debug(self, grad_output, learning_rate):
        print("\n=== DENSE BACKWARD DEBUG ===")

        grad_output = np.asarray(grad_output, dtype=np.float32)

        print("\n[1] grad_output recibido")
        print("grad_output.shape:", grad_output.shape)
        print(grad_output)

        print("\n[2] Input guardado del forward")
        print("self.input.shape:", self.input.shape)
        print(self.input)

        print("\n[3] Pesos actuales")
        print("self.weights.shape:", self.weights.shape)
        print(self.weights)

        print("\n[4] Gradiente respecto a los pesos")
        print("Fórmula: grad_weights = self.input.T @ grad_output")
        grad_weights = self.input.T @ grad_output
        print("grad_weights.shape:", grad_weights.shape)
        print(grad_weights)

        print("\n[5] Gradiente respecto a los biases")
        print("Fórmula: grad_biases = sum(grad_output, axis=0)")
        grad_biases = grad_output.sum(axis=0)
        print("grad_biases.shape:", grad_biases.shape)
        print(grad_biases)

        print("\n[6] Gradiente respecto al input")
        print("Fórmula: grad_input = grad_output @ self.weights.T")
        grad_input = grad_output @ self.weights.T
        print("grad_input.shape:", grad_input.shape)
        print(grad_input)

        print("\n[7] Actualización de parámetros")
        print("learning_rate:", learning_rate)

        old_weights = self.weights.copy()
        old_biases = self.biases.copy()

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        print("\nPesos antes:")
        print(old_weights)
        print("Pesos después:")
        print(self.weights)

        print("\nBiases antes:")
        print(old_biases)
        print("Biases después:")
        print(self.biases)

        print("\n[8] Explicación")
        print("grad_weights indica cómo cambia la loss respecto a cada peso")
        print("grad_biases indica cómo cambia la loss respecto a cada bias")
        print("grad_input es lo que se propaga a la capa anterior")

        return grad_input.astype(np.float32, copy=False)
        
    def get_weights(self):
        return {'weights': self.weights, 'biases': self.biases}

    def set_weights(self, weights):
        self.weights = weights['weights']
        self.biases = weights['biases']

layer = Dense(in_features=3, out_features=2, weight_init="custom")

layer.weights = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], dtype=np.float32)

layer.biases = np.array([0.01, 0.02], dtype=np.float32)

x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=np.float32)

out = layer.forward_debug(x)

grad_output = np.array([
    [1.0, 0.5],
    [0.2, 0.1]
], dtype=np.float32)

grad_input = layer.backward_debug(grad_output, learning_rate=0.01)