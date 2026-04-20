import numpy as np
import cupy as cp

# optimizar multiplicacion de matrices usando numpy (que está hecho en C)
def matmul_biasses(A, B, C, bias, use_gpu = False):
    if use_gpu:
        A_gpu = cp.asarray(A, dtype=cp.float32)
        B_gpu = cp.asarray(B, dtype=cp.float32)
        bias_gpu = cp.asarray(bias, dtype=cp.float32)

        result = (A_gpu @ B_gpu) + bias_gpu

        if C is not None:
            C[...] = cp.asnumpy(result)
            return C

        return cp.asnumpy(result)
    else:
        result = (np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32)) + np.asarray(bias, dtype=np.float32)
        if C is not None:
            C[...] = result
    return result