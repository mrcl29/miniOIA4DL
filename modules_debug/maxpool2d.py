from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride, use_gpu=False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

    def forward_debug(self, input, training=True):
        print("=== MAXPOOL2D FORWARD DEBUG ===")
        print(f"kernel_size={self.kernel_size}, stride={self.stride}, use_gpu={self.use_gpu}")

        xp = cp if self.use_gpu else np
        inp = xp.asarray(input, dtype=np.float32)

        print("\n[1] Input")
        print("shape:", inp.shape)
        print(cp.asnumpy(inp) if self.use_gpu else inp)

        B, C, H, W = inp.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        print("\n[2] Dimensiones calculadas")
        print(f"B={B}, C={C}, H={H}, W={W}")
        print(f"KH={KH}, KW={KW}, SH={SH}, SW={SW}")
        print(f"out_h={out_h}, out_w={out_w}")

        windows = xp.lib.stride_tricks.sliding_window_view(inp, (KH, KW), axis=(2, 3))[:, :, ::SH, ::SW]
        flat = windows.reshape(B, C, out_h, out_w, KH * KW)

        print("\n[3] Ventanas extraídas")
        print("windows.shape:", windows.shape)
        print("flat.shape:", flat.shape)

        windows_np = cp.asnumpy(windows) if self.use_gpu else windows
        flat_np = cp.asnumpy(flat) if self.use_gpu else flat

        for b in range(B):
            for c in range(C):
                print(f"\n--- Ventanas para batch={b}, canal={c} ---")
                for i in range(out_h):
                    for j in range(out_w):
                        print(f"Ventana[{b},{c},{i},{j}] =")
                        print(windows_np[b, c, i, j])
                        print("Aplanada:", flat_np[b, c, i, j])

        argmax = flat.argmax(axis=-1)

        print("\n[4] Índices planos de los máximos (argmax)")
        print("argmax.shape:", argmax.shape)
        print(cp.asnumpy(argmax) if self.use_gpu else argmax)

        h_off = argmax // KW
        w_off = argmax % KW
        h_base = (xp.arange(out_h) * SH).reshape(1, 1, out_h, 1)
        w_base = (xp.arange(out_w) * SW).reshape(1, 1, 1, out_w)

        max_indices_xp = xp.empty((B, C, out_h, out_w, 2), dtype=np.intp)
        max_indices_xp[..., 0] = h_base + h_off
        max_indices_xp[..., 1] = w_base + w_off

        print("\n[5] Coordenadas absolutas de los máximos")
        max_indices_np = cp.asnumpy(max_indices_xp) if self.use_gpu else max_indices_xp
        print("max_indices.shape:", max_indices_np.shape)
        print(max_indices_np)

        output = xp.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]

        print("\n[6] Output final")
        print("output.shape:", output.shape)
        print(cp.asnumpy(output) if self.use_gpu else output)

        if self.use_gpu and isinstance(input, cp.ndarray):
            self.input = cp.asnumpy(input).astype(np.float32, copy=False)
        else:
            self.input = np.asarray(input, dtype=np.float32)

        self.max_indices = cp.asnumpy(max_indices_xp) if self.use_gpu else max_indices_xp.astype(np.intp)

        return output.astype(np.float32, copy=False)


    def backward_debug(self, grad_output, learning_rate=None):
        print("\n=== MAXPOOL2D BACKWARD DEBUG ===")

        grad_output = np.asarray(grad_output, dtype=np.float32)

        print("\n[1] grad_output recibido")
        print("shape:", grad_output.shape)
        print(grad_output)

        B, C, H, W = self.input.shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        print("\n[2] Estado guardado del forward")
        print("input.shape:", self.input.shape)
        print(self.input)
        print("max_indices.shape:", self.max_indices.shape)
        print(self.max_indices)

        grad_input = np.zeros((B, C, H, W), dtype=np.float32)

        print("\n[3] Propagación del gradiente a las posiciones máximas")

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_idx = self.max_indices[b, c, i, j, 0]
                        w_idx = self.max_indices[b, c, i, j, 1]
                        g = grad_output[b, c, i, j]

                        print(
                            f"grad_output[{b},{c},{i},{j}] = {g} "
                            f"-> grad_input[{b},{c},{h_idx},{w_idx}] += {g}"
                        )

                        grad_input[b, c, h_idx, w_idx] += g

        print("\n[4] grad_input final")
        print("shape:", grad_input.shape)
        print(grad_input)

        return grad_input.astype(np.float32, copy=False)


x = np.array([[
    [[1, 3, 2, 0],
     [4, 6, 5, 1],
     [7, 2, 8, 3],
     [0, 1, 2, 4]]
]], dtype=np.float32)  # shape (1,1,4,4)

pool = MaxPool2D(kernel_size=2, stride=2)

out = pool.forward_debug(x)

grad_output = np.array([[
    [[10, 20],
     [30, 40]]
]], dtype=np.float32)

grad_input = pool.backward_debug(grad_output)