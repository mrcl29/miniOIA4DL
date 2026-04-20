import numpy as np
import cupy as cp

def matmul_biasses_debug(A, B, C, bias, use_gpu=False):
    print("=== ENTRADA A matmul_biasses ===")
    print("use_gpu:", use_gpu)
    print("A original:")
    print(A)
    print("B original:")
    print(B)
    print("bias original:")
    print(bias)
    print("C original:")
    print(C)

    if use_gpu:
        print("\n--- RAMA GPU ---")

        A_gpu = cp.asarray(A, dtype=cp.float32)
        B_gpu = cp.asarray(B, dtype=cp.float32)
        bias_gpu = cp.asarray(bias, dtype=cp.float32)

        print("A_gpu.shape:", A_gpu.shape, "dtype:", A_gpu.dtype)
        print("B_gpu.shape:", B_gpu.shape, "dtype:", B_gpu.dtype)
        print("bias_gpu.shape:", bias_gpu.shape, "dtype:", bias_gpu.dtype)

        mult = A_gpu @ B_gpu
        print("\nResultado de A_gpu @ B_gpu:")
        print(cp.asnumpy(mult))

        result = mult + bias_gpu
        print("\nResultado final (A_gpu @ B_gpu) + bias_gpu:")
        print(cp.asnumpy(result))

        if C is not None:
            print("\nC no es None, copiando resultado a C...")
            C[...] = cp.asnumpy(result)
            print("C después de copiar:")
            print(C)
            return C

        print("\nC es None, devolviendo resultado convertido a NumPy")
        return cp.asnumpy(result)

    else:
        print("\n--- RAMA CPU ---")

        A_np = np.asarray(A, dtype=np.float32)
        B_np = np.asarray(B, dtype=np.float32)
        bias_np = np.asarray(bias, dtype=np.float32)

        print("A_np.shape:", A_np.shape, "dtype:", A_np.dtype)
        print("B_np.shape:", B_np.shape, "dtype:", B_np.dtype)
        print("bias_np.shape:", bias_np.shape, "dtype:", bias_np.dtype)

        mult = A_np @ B_np
        print("\nResultado de A_np @ B_np:")
        print(mult)

        result = mult + bias_np
        print("\nResultado final (A_np @ B_np) + bias_np:")
        print(result)

        if C is not None:
            print("\nC no es None, copiando resultado a C...")
            # C[...] --> NO crea una nueva matriz, modifica la existente
            C[...] = result
            print("C después de copiar:")
            print(C)

        print("\nDevolviendo result")
        return result

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

bias = np.array([
    [10, 20],
    [30, 40]
])

C = np.zeros((2, 2), dtype=np.float32)

out = matmul_biasses_debug(A, B, C, bias, use_gpu=False)

print("\n=== RESULTADO DEVUELTO ===")
print(out)

print("\n=== C FINAL ===")
print(C)