from modules.layer import Layer
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

try:
    from cython_modules.maxpool2d import maxpool_forward_cython
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride, pool_algo=0, use_gpu=False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

        if pool_algo == 0:
            self.mode = 'numpy'
        elif pool_algo == 1:
            if _CYTHON_AVAILABLE:
                self.mode = 'cython'
            else:
                print("Cython no disponible, usando maxpool NumPy como fallback")
                self.mode = 'numpy'
        else:
            print(f"pool_algo {pool_algo} no soportado aún")
            self.mode = 'numpy'

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        if self.mode == 'cython':
            return self._forward_cython(input)
        return self._forward_numpy(input)

    # --- INICIO BLOQUE GENERADO CON IA ---
    def _forward_numpy(self, input):
        xp = cp if self.use_gpu else np
        inp = xp.asarray(input, dtype=np.float32)
        B, C, H, W = inp.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # Extraer todas las ventanas de una vez: (B, C, out_h, out_w, KH, KW)
        windows = xp.lib.stride_tricks.sliding_window_view(inp, (KH, KW), axis=(2, 3))[:, :, ::SH, ::SW]
        flat = windows.reshape(B, C, out_h, out_w, KH * KW)

        # Índice plano del máximo en cada ventana
        argmax = flat.argmax(axis=-1)          # (B, C, out_h, out_w)

        # Convertir índice plano a coordenadas absolutas
        h_off = argmax // KW
        w_off = argmax % KW
        h_base = (xp.arange(out_h) * SH).reshape(1, 1, out_h, 1)
        w_base = (xp.arange(out_w) * SW).reshape(1, 1, 1, out_w)

        max_indices_xp = xp.empty((B, C, out_h, out_w, 2), dtype=np.intp)
        max_indices_xp[..., 0] = h_base + h_off
        max_indices_xp[..., 1] = w_base + w_off

        output = xp.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]

        # Keep NumPy state for backward compatibility
        if self.use_gpu and isinstance(input, cp.ndarray):
            self.input = cp.asnumpy(input).astype(np.float32, copy=False)
        else:
            self.input = np.asarray(input, dtype=np.float32)
        self.max_indices = cp.asnumpy(max_indices_xp) if self.use_gpu else max_indices_xp.astype(np.intp)
        return output.astype(np.float32, copy=False)
    # --- FIN BLOQUE GENERADO CON IA ---

    # --- INICIO BLOQUE GENERADO CON IA ---
    def _forward_cython(self, input):
        self.input = np.ascontiguousarray(input, dtype=np.float32)
        return maxpool_forward_cython(self.input, self.kernel_size, self.stride)
    # --- FIN BLOQUE GENERADO CON IA ---

    # --- INICIO BLOQUE GENERADO CON IA ---
    def backward(self, grad_output, learning_rate=None):
        grad_output = np.asarray(grad_output, dtype=np.float32)
        B, C, H, W = self.input.shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        grad_input = np.zeros((B, C, H, W), dtype=np.float32)

        # Scatter-add vectorizado: propaga gradiente solo a la posición del máximo
        b_idx = np.broadcast_to(np.arange(B)[:, None, None, None], (B, C, out_h, out_w))
        c_idx = np.broadcast_to(np.arange(C)[None, :, None, None], (B, C, out_h, out_w))
        np.add.at(
            grad_input,
            (b_idx, c_idx, self.max_indices[..., 0], self.max_indices[..., 1]),
            grad_output,
        )

        return grad_input.astype(np.float32, copy=False)
    # --- FIN BLOQUE GENERADO CON IA ---
