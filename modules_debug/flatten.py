from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class Flatten(Layer):
    def __init__(self, use_gpu=False):
        self.input_shape = None
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward_debug(self, input, training=True):
        print("=== FLATTEN FORWARD DEBUG ===")
        print(f"use_gpu={self.use_gpu}")

        if self.use_gpu:
            xp = cp
            x = xp.asarray(input, dtype=xp.float32)
            x_np = cp.asnumpy(x)
        else:
            xp = np
            x = np.asarray(input, dtype=np.float32)
            x_np = x

        print("\n[1] Input original")
        print("shape:", x.shape)
        print(x_np)

        self.input_shape = x.shape

        print("\n[2] Guardamos input_shape para backward")
        print("input_shape:", self.input_shape)

        output = x.reshape(x.shape[0], -1)

        print("\n[3] Output después de flatten")
        print("shape:", output.shape)
        print(cp.asnumpy(output) if self.use_gpu else output)

        print("\n[4] Explicación")
        print("Se mantiene batch_size =", x.shape[0])
        print("El resto de dimensiones se colapsan en una sola")

        return output


    def backward_debug(self, grad_output, learning_rate=None):
        print("\n=== FLATTEN BACKWARD DEBUG ===")

        grad_output = np.asarray(grad_output, dtype=np.float32)

        print("\n[1] grad_output recibido")
        print("shape:", grad_output.shape)
        print(grad_output)

        print("\n[2] input_shape guardado del forward")
        print(self.input_shape)

        grad_input = grad_output.reshape(self.input_shape)

        print("\n[3] grad_input después de reshape")
        print("shape:", grad_input.shape)
        print(grad_input)

        print("\n[4] Explicación")
        print("Se reconstruye la forma original del tensor")
        print("No se pierde información, solo cambia la vista")

        return grad_input
    
x = np.array([[
    [[1, 2],
     [3, 4]],

    [[5, 6],
     [7, 8]]
]], dtype=np.float32)  # shape (1, 2, 2, 2)

layer = Flatten()

out = layer.forward_debug(x)

grad_output = np.ones_like(out)
grad_input = layer.backward_debug(grad_output)