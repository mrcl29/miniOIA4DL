# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp

def maxpool_forward_cython(
    cnp.ndarray[cnp.float32_t, ndim=4] input not None,
    int kernel_size,
    int stride
):
    """
    Parámetros
    ----------
    input       : float32 array (B, C, H, W), contiguo en C
    kernel_size : tamaño de la ventana cuadrada K
    stride      : paso S

    Retorna
    -------
    output : float32 array (B, C, out_h, out_w)
        out_h = (H - K) // S + 1,  out_w = (W - K) // S + 1
    """
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]
    cdef int K = kernel_size
    cdef int S = stride
    cdef int out_h = (H - K) // S + 1
    cdef int out_w = (W - K) // S + 1

    # Output contiguo en C: el orden de escritura [b][c][i][j] recorre memoria linealmente.
    cdef cnp.ndarray[cnp.float32_t, ndim=4] output = np.empty(
        (B, C, out_h, out_w), dtype=np.float32
    )

    cdef cnp.float32_t[:, :, :, :] x = input
    cdef cnp.float32_t[:, :, :, :] y = output

    cdef int b, c, i, j, ki, kj
    cdef int h_base, w_base
    cdef cnp.float32_t m, v

    for b in range(B):
        for c in range(C):
            for i in range(out_h):
                h_base = i * S
                for j in range(out_w):
                    w_base = j * S
                    m = x[b, c, h_base, w_base]
                    for ki in range(K):
                        for kj in range(K):
                            v = x[b, c, h_base + ki, w_base + kj]
                            if v > m:
                                m = v
                    y[b, c, i, j] = m

    return output
