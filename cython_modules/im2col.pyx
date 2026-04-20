# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp

def im2col_forward_cython(
    cnp.ndarray[cnp.float32_t, ndim=4] input not None,
    int kernel_size,
    int stride,
    int out_h,
    int out_w
):
    """
    Construye la matriz de columnas im2col a partir del input ya padded.

    Parámetros
    ----------
    input      : float32 array (B, C, H_padded, W_padded)
    kernel_size: tamaño del kernel cuadrado K
    stride     : paso de la convolución
    out_h      : altura del mapa de salida
    out_w      : anchura del mapa de salida

    Retorna
    -------
    cols : float32 array (B, out_h*out_w, C*K*K)
        Cada fila es el patch aplanado listo para el GEMM.
    """
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int K = kernel_size
    cdef int col_len   = C * K * K
    cdef int n_patches = out_h * out_w

    # Array de salida contiguo en C — escritura secuencial, caché-friendly
    cdef cnp.ndarray[cnp.float32_t, ndim=3] cols = np.empty(
        (B, n_patches, col_len), dtype=np.float32
    )

    # Typed memoryviews: acceso directo sin overhead del objeto Python
    cdef cnp.float32_t[:, :, :, :] x = input
    cdef cnp.float32_t[:, :, :]    c = cols

    cdef int b, patch_idx, col_idx
    cdef int i, j, ci, ki, kj

    for b in range(B):
        patch_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                col_idx = 0
                for ci in range(C):
                    for ki in range(K):
                        for kj in range(K):
                            c[b, patch_idx, col_idx] = x[b, ci,
                                                          i * stride + ki,
                                                          j * stride + kj]
                            col_idx += 1
                patch_idx += 1

    return cols
