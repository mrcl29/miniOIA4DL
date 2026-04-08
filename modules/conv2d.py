from modules.layer import Layer
from modules.utils import *
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

try:
    from cython_modules.im2col import im2col_forward_cython
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


def _sliding_window_view_2d(input_tensor, kernel_size, stride, xp=np):
    windows = xp.lib.stride_tricks.sliding_window_view(
        input_tensor,
        (kernel_size, kernel_size),
        axis=(2, 3)
    )
    return windows[:, :, ::stride, ::stride, :, :]


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he", use_gpu=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

        # MODIFICAR: Añadir nuevo if-else para otros algoritmos de convolución
        if conv_algo == 0:
            self.mode = 'direct'
        elif conv_algo == 1:
            self.mode = 'im2col'
        elif conv_algo == 2:
            if _CYTHON_AVAILABLE:
                self.mode = 'im2col_cython'
            else:
                print("Cython no disponible, usando im2col NumPy como fallback")
                self.mode = 'im2col'
        else:
            print(f"Algoritmo {conv_algo} no soportado aún")
            self.mode = 'direct'

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(
                -0.1,
                0.1,
                (out_channels, in_channels, kernel_size, kernel_size)
            ).astype(np.float32)

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # Caché de pesos en GPU (se inicializan la primera vez que se usa use_gpu)
        self._kernels_gpu = None
        self._biases_gpu = None

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)

    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
        # Invalidar caché GPU al cambiar pesos
        self._kernels_gpu = None
        self._biases_gpu = None

    def forward(self, input, training=True):
        xp = cp if self.use_gpu else np
        if training:
            # Solo guardamos la entrada para el backward
            self.input = cp.asnumpy(input).astype(np.float32, copy=False) if (self.use_gpu and isinstance(input, cp.ndarray)) else np.asarray(input, dtype=np.float32)
        x = xp.asarray(input, dtype=xp.float32)

        if self.mode == 'direct':
            return self._forward_direct(x)

        if self.mode == 'im2col':
            return self._forward_im2col(x)
            
        if self.mode == 'im2col_cython':
            return self._forward_im2col_cython(x)
        
        raise ValueError("Mode must be 'direct', 'im2col' or 'im2col_cython'")

    def backward(self, grad_output, learning_rate):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        if self.mode in ('direct', 'im2col', 'im2col_cython'):
            return self._backward_direct(grad_output, learning_rate)
        raise ValueError("Mode must be 'direct', 'im2col' or 'im2col_cython'")

    # --- DIRECT IMPLEMENTATION ---

    def _forward_direct(self, input):
        xp = cp if self.use_gpu else np
        if self.use_gpu:
            input = cp.asarray(input, dtype=np.float32)
        batch_size, _, _, _ = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = xp.pad(
                input,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            ).astype(xp.float32)

        # calculo de dimensiones de salida (aplicar formula)
        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = xp.zeros((batch_size, self.out_channels, out_h, out_w), dtype=xp.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[
                                b,
                                in_c,
                                i * self.stride:i * self.stride + k_h,
                                j * self.stride:j * self.stride + k_w,
                            ]
                            kernel = xp.asarray(self.kernels[out_c, in_c], dtype=np.float32)
                            output[b, out_c, i, j] += xp.sum(region * kernel)
                output[b, out_c] += self.biases[out_c]

        return output.astype(np.float32, copy=False)

    # --- IM2COL NUMPY IMPLEMENTATION (conv_algo=1) ---

    def _forward_im2col(self, input):
        xp = cp if self.use_gpu else np
        input = xp.asarray(input, dtype=np.float32)
        batch_size = input.shape[0]

        if self.padding > 0:
            input = xp.pad(
                input,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )

        out_h = (input.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (input.shape[3] - self.kernel_size) // self.stride + 1

        windows = _sliding_window_view_2d(input, self.kernel_size, self.stride, xp)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size, out_h * out_w, -1)

        if self.use_gpu:
            # Subir pesos a GPU solo la primera vez (caché)
            if self._kernels_gpu is None:
                self._kernels_gpu = cp.asarray(self.kernels, dtype=cp.float32)
            if self._biases_gpu is None:
                self._biases_gpu = cp.asarray(self.biases, dtype=cp.float32)
            kernels = self._kernels_gpu.reshape(self.out_channels, -1)
            biases = self._biases_gpu
        else:
            kernels = np.asarray(self.kernels, dtype=np.float32).reshape(self.out_channels, -1)
            biases = np.asarray(self.biases, dtype=np.float32)

        output = cols @ kernels.T
        output += biases.reshape(1, 1, self.out_channels)

        output = output.transpose(0, 2, 1).reshape(batch_size, self.out_channels, out_h, out_w)
        return output.astype(np.float32, copy=False)

    # --- IM2COL CYTHON IMPLEMENTATION (conv_algo=2) ---

    def _forward_im2col_cython(self, input):
        input = np.ascontiguousarray(input, dtype=np.float32)
        batch_size = input.shape[0]

        if self.padding > 0:
            input = np.ascontiguousarray(
                np.pad(
                    input,
                    ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant'
                ),
                dtype=np.float32
            )

        out_h = (input.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (input.shape[3] - self.kernel_size) // self.stride + 1

        cols = im2col_forward_cython(input, self.kernel_size, self.stride, out_h, out_w)
        kernels = self.kernels.reshape(self.out_channels, -1)

        output = cols @ kernels.T
        output += self.biases.reshape(1, 1, self.out_channels)

        return output.transpose(0, 2, 1).reshape(batch_size, self.out_channels, out_h, out_w)

    # --- BACKWARD (shared by all modes) ---

    def _backward_direct(self, grad_output, learning_rate):
        grad_output = np.asarray(grad_output, dtype=np.float32)
        batch_size, _, out_h, out_w = grad_output.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(
                self.input,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            ).astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += (
                                self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                            )
                grad_biases[out_c] += grad_output[b, out_c].sum()

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input.astype(np.float32, copy=False)

    # PISTA: Se te ocurren otros algoritmos de convolución?
