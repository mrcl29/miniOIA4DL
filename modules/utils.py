import numpy as np

# optimizar multiplicacion de matrices usando numpy (que está hecho en C)
def matmul_biasses(A, B, C, bias):
    result = np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32)
    result += np.asarray(bias, dtype=np.float32)

    if C is not None:
        C= result
    return result

