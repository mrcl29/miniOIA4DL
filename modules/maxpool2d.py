from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = np.asarray(input, dtype=np.float32)
        B, C, H, W = self.input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # Extraer todas las ventanas de una vez: (B, C, out_h, out_w, KH, KW)
        windows = sliding_window_view(self.input, (KH, KW), axis=(2, 3))[:, :, ::SH, ::SW]
        flat = windows.reshape(B, C, out_h, out_w, KH * KW)

        # Índice plano del máximo en cada ventana
        argmax = np.argmax(flat, axis=-1)          # (B, C, out_h, out_w)

        # Convertir índice plano a coordenadas absolutas
        h_off = argmax // KW
        w_off = argmax % KW
        h_base = (np.arange(out_h) * SH).reshape(1, 1, out_h, 1)
        w_base = (np.arange(out_w) * SW).reshape(1, 1, 1, out_w)

        self.max_indices = np.empty((B, C, out_h, out_w, 2), dtype=np.intp)
        self.max_indices[..., 0] = h_base + h_off
        self.max_indices[..., 1] = w_base + w_off

        output = np.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]
        return output.astype(np.float32, copy=False)

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

        return grad_input