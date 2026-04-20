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

    def forward_debug(self, input, training=True):
        print("=== CONV2D FORWARD DEBUG ===")
        print(
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"mode={self.mode}, use_gpu={self.use_gpu}"
        )

        xp = cp if self.use_gpu else np
        self.input = xp.asarray(input, dtype=xp.float32)

        print("\n[1] Input")
        print("shape:", self.input.shape)
        print(cp.asnumpy(self.input) if self.use_gpu else self.input)

        print("\n[2] Kernels")
        print("shape:", self.kernels.shape)
        print(self.kernels)

        print("\n[3] Biases")
        print("shape:", self.biases.shape)
        print(self.biases)

        if self.mode == 'direct':
            return self._forward_direct_debug(self.input)

        if self.mode == 'im2col':
            return self._forward_im2col_debug(self.input)

        if self.mode == 'im2col_cython':
            print("\n[4] Modo im2col_cython")
            print("Para debug detallado, se usará la lógica im2col Python/NumPy equivalente")
            x_np = cp.asnumpy(self.input) if self.use_gpu else np.asarray(self.input, dtype=np.float32)
            return self._forward_im2col_debug(x_np)

        raise ValueError("Mode must be 'direct', 'im2col' or 'im2col_cython'")


    def _forward_direct_debug(self, input):
        print("\n=== CONV2D DIRECT FORWARD DEBUG ===")

        xp = cp if self.use_gpu else np
        x = xp.asarray(input, dtype=xp.float32)

        batch_size, _, H, W = x.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        print("\n[1] Shape original")
        print(f"batch_size={batch_size}, in_channels={self.in_channels}, H={H}, W={W}")

        if self.padding > 0:
            print("\n[2] Aplicando padding")
            x = xp.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            ).astype(xp.float32)
            print("shape padded:", x.shape)
            print(cp.asnumpy(x) if self.use_gpu else x)
        else:
            print("\n[2] Sin padding")

        out_h = (x.shape[2] - k_h) // self.stride + 1
        out_w = (x.shape[3] - k_w) // self.stride + 1

        print("\n[3] Dimensiones de salida")
        print(f"out_h={out_h}, out_w={out_w}")

        output = xp.zeros((batch_size, self.out_channels, out_h, out_w), dtype=xp.float32)

        x_np = cp.asnumpy(x) if self.use_gpu else x

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                print(f"\n--- batch={b}, out_channel={out_c} ---")
                for in_c in range(self.in_channels):
                    kernel = xp.asarray(self.kernels[out_c, in_c], dtype=xp.float32)
                    kernel_np = cp.asnumpy(kernel) if self.use_gpu else kernel

                    print(f"\nKernel[out_c={out_c}, in_c={in_c}]")
                    print(kernel_np)

                    for i in range(out_h):
                        for j in range(out_w):
                            r0 = i * self.stride
                            c0 = j * self.stride

                            region = x[b, in_c, r0:r0 + k_h, c0:c0 + k_w]
                            region_np = cp.asnumpy(region) if self.use_gpu else region

                            prod = region * kernel
                            prod_np = cp.asnumpy(prod) if self.use_gpu else prod
                            s = xp.sum(prod)

                            print(f"\nPosición salida [{b},{out_c},{i},{j}] usando in_c={in_c}")
                            print(f"region input[{b},{in_c},{r0}:{r0+k_h},{c0}:{c0+k_w}] =")
                            print(region_np)
                            print("region * kernel =")
                            print(prod_np)
                            print("sum(region * kernel) =", float(cp.asnumpy(s) if self.use_gpu else s))

                            output[b, out_c, i, j] += s

                print("\nMapa antes de bias:")
                print(cp.asnumpy(output[b, out_c]) if self.use_gpu else output[b, out_c])

                output[b, out_c] += self.biases[out_c]

                print(f"\nBias añadido para out_c={out_c}: {self.biases[out_c]}")
                print("Mapa final:")
                print(cp.asnumpy(output[b, out_c]) if self.use_gpu else output[b, out_c])

        print("\n[4] Output final")
        print("shape:", output.shape)
        print(cp.asnumpy(output) if self.use_gpu else output)

        return output.astype(np.float32, copy=False)


    def _forward_im2col_debug(self, input):
        print("\n=== CONV2D IM2COL FORWARD DEBUG ===")

        xp = cp if self.use_gpu and isinstance(input, cp.ndarray) else np
        x = xp.asarray(input, dtype=np.float32 if xp is np else xp.float32)

        batch_size = x.shape[0]

        print("\n[1] Input")
        print("shape:", x.shape)
        print(cp.asnumpy(x) if xp is cp else x)

        if self.padding > 0:
            print("\n[2] Aplicando padding")
            x = xp.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
            print("shape padded:", x.shape)
            print(cp.asnumpy(x) if xp is cp else x)
        else:
            print("\n[2] Sin padding")

        out_h = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] - self.kernel_size) // self.stride + 1

        print("\n[3] Dimensiones de salida")
        print(f"out_h={out_h}, out_w={out_w}")

        windows = _sliding_window_view_2d(x, self.kernel_size, self.stride, xp)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size, out_h * out_w, -1)
        kernels = xp.asarray(self.kernels, dtype=np.float32 if xp is np else xp.float32).reshape(self.out_channels, -1)

        windows_np = cp.asnumpy(windows) if xp is cp else windows
        cols_np = cp.asnumpy(cols) if xp is cp else cols
        kernels_np = cp.asnumpy(kernels) if xp is cp else kernels

        print("\n[4] Ventanas extraídas")
        print("windows.shape:", windows.shape)

        for b in range(batch_size):
            print(f"\n--- batch={b} ---")
            for i in range(out_h):
                for j in range(out_w):
                    print(f"Ventana[{b},{i},{j}] =")
                    print(windows_np[b, :, i, j])

        print("\n[5] Matriz cols")
        print("cols.shape:", cols.shape)
        print(cols_np)

        print("\n[6] Kernels aplanados")
        print("kernels.shape:", kernels.shape)
        print(kernels_np)

        print("\n[7] Multiplicación cols @ kernels.T")
        output_lin = cols @ kernels.T
        output_lin_np = cp.asnumpy(output_lin) if xp is cp else output_lin
        print("shape:", output_lin.shape)
        print(output_lin_np)

        print("\n[8] Suma de biases")
        biases = xp.asarray(self.biases, dtype=np.float32 if xp is np else xp.float32).reshape(1, 1, self.out_channels)
        print("biases reshaped:", cp.asnumpy(biases) if xp is cp else biases)

        output_lin = output_lin + biases
        print("Después de bias:")
        print(cp.asnumpy(output_lin) if xp is cp else output_lin)

        output = output_lin.transpose(0, 2, 1).reshape(batch_size, self.out_channels, out_h, out_w)

        print("\n[9] Output final reordenado")
        print("shape:", output.shape)
        print(cp.asnumpy(output) if xp is cp else output)

        return output.astype(np.float32, copy=False)


    def backward_debug(self, grad_output, learning_rate):
        print("\n=== CONV2D BACKWARD DEBUG ===")
        print(f"mode={self.mode}, learning_rate={learning_rate}")

        if self.mode in ('direct', 'im2col', 'im2col_cython'):
            return self._backward_direct_debug(grad_output, learning_rate)

        raise ValueError("Mode must be 'direct', 'im2col' or 'im2col_cython'")


    def _backward_direct_debug(self, grad_output, learning_rate):
        print("\n=== CONV2D DIRECT BACKWARD DEBUG ===")

        grad_output = np.asarray(grad_output, dtype=np.float32)

        print("\n[1] grad_output")
        print("shape:", grad_output.shape)
        print(grad_output)

        input_np = cp.asnumpy(self.input) if self.use_gpu else np.asarray(self.input, dtype=np.float32)

        print("\n[2] Input guardado del forward")
        print("shape:", input_np.shape)
        print(input_np)

        batch_size, _, out_h, out_w = grad_output.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(
                input_np,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            ).astype(np.float32)
            print("\n[3] Input con padding")
            print("shape:", input_padded.shape)
            print(input_padded)
        else:
            input_padded = input_np
            print("\n[3] Sin padding")

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        print("\n[4] Acumulación de gradientes")

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                print(f"\n--- batch={b}, out_channel={out_c} ---")
                for in_c in range(self.in_channels):
                    print(f"\nCanal de entrada {in_c}")
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride

                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            go = grad_output[b, out_c, i, j]

                            print(f"\ngrad_output[{b},{out_c},{i},{j}] = {go}")
                            print(f"region input_padded[{b},{in_c},{r}:{r+k_h},{c}:{c+k_w}] =")
                            print(region)

                            grad_kernels[out_c, in_c] += go * region
                            print("Aporte a grad_kernels[out_c, in_c]:")
                            print(go * region)
                            print("grad_kernels acumulado:")
                            print(grad_kernels[out_c, in_c])

                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * go
                            print("Aporte a grad_input_padded:")
                            print(self.kernels[out_c, in_c] * go)
                            print("grad_input_padded acumulado en esa región:")
                            print(grad_input_padded[b, in_c, r:r + k_h, c:c + k_w])

                grad_biases[out_c] += grad_output[b, out_c].sum()
                print(f"\nAporte a grad_biases[{out_c}] =", grad_output[b, out_c].sum())
                print("grad_biases acumulado:", grad_biases)

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            print("\n[5] Quitando padding de grad_input")
        else:
            grad_input = grad_input_padded
            print("\n[5] grad_input sin recorte")

        print("grad_input.shape:", grad_input.shape)
        print(grad_input)

        print("\n[6] grad_kernels final")
        print("shape:", grad_kernels.shape)
        print(grad_kernels)

        print("\n[7] grad_biases final")
        print("shape:", grad_biases.shape)
        print(grad_biases)

        old_kernels = self.kernels.copy()
        old_biases = self.biases.copy()

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        print("\n[8] Actualización de parámetros")
        print("Kernels antes:")
        print(old_kernels)
        print("Kernels después:")
        print(self.kernels)

        print("\nBiases antes:")
        print(old_biases)
        print("Biases después:")
        print(self.biases)

        return grad_input.astype(np.float32, copy=False)

conv = Conv2D(
    in_channels=1,
    out_channels=1,
    kernel_size=2,
    stride=1,
    padding=0,
    conv_algo=0,   # direct
    weight_init="custom",
    use_gpu=False
)

conv.kernels = np.array([[[[1, 0],
                           [0, -1]]]], dtype=np.float32)

conv.biases = np.array([0.5], dtype=np.float32)

x = np.array([[[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]]], dtype=np.float32)

out = conv.forward_debug(x)

grad_output = np.array([[[[1, 2],
                          [3, 4]]]], dtype=np.float32)

grad_input = conv.backward_debug(grad_output, learning_rate=0.01)