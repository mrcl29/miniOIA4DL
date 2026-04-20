# Guía de Cython: desde cero hasta el proyecto

---

## PARTE 1 — Conceptos fundamentales desde cero

---

### ¿Qué es Cython?

Cython es un **superconjunto de Python** que se compila a código C/C++. El fichero resultante (`.so` en Unix, `.pyd` en Windows) se importa exactamente igual que un módulo Python normal, pero ejecuta el código a velocidad de C.

```
fichero.pyx  →  [compilador Cython]  →  fichero.c  →  [compilador C]  →  fichero.so
                                                                              ↑
                                               se importa con: import fichero
```

La idea clave: Python es lento porque cada operación pasa por el intérprete. Cython elimina ese overhead cuando declaras tipos explícitos.

---

### Flujo de trabajo mínimo

**1. Escribir el `.pyx`:**
```cython
# saluda.pyx
def saluda(str nombre):
    return f"Hola, {nombre}"
```

**2. `setup.py`:**
```python
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("saluda.pyx", language_level=3))
```

**3. Compilar:**
```bash
python setup.py build_ext --inplace
```

**4. Usar desde Python:**
```python
from saluda import saluda
print(saluda("mundo"))   # "Hola, mundo"
```

---

### Variables con `cdef` — el cambio más importante

Sin `cdef`, Cython trata las variables como objetos Python. Con `cdef`, las compila a variables C nativas (enteros de 64 bits, floats reales, etc.).

```cython
# LENTO — Python puro
def suma_lenta(n):
    total = 0
    for i in range(n):
        total += i
    return total

# RÁPIDO — tipos C explícitos
def suma_rapida(int n):
    cdef int i
    cdef long total = 0
    for i in range(n):
        total += i
    return total
```

El bucle de la versión rápida se compila a un `for` de C sin ningún objeto Python involucrado.

**Tipos C más usados:**

| Tipo Cython | Equivalente C | Uso típico |
|---|---|---|
| `int` | `int` (32 bits) | índices, contadores |
| `long` | `long` (64 bits) | acumuladores |
| `float` | `float` (32 bits) | aritmética simple |
| `double` | `double` (64 bits) | precisión alta |
| `bint` | `int` (bool) | flags |

---

### Directivas de compilador

Se ponen en la primera línea del `.pyx` como comentario especial:

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
```

| Directiva | Qué hace | Por defecto |
|---|---|---|
| `boundscheck=False` | No comprueba que `x[i]` esté dentro de rango | `True` |
| `wraparound=False` | Desactiva índices negativos (`x[-1]`) | `True` |
| `cdivision=True` | División entera C (sin check de divisor 0) | `False` |
| `nonecheck=False` | No comprueba si memoryviews son `None` | `False` |

**Importante**: `boundscheck=False` y `wraparound=False` juntos son los que mayor speedup dan. Sin ellos, el compilador inserta comprobaciones extra en cada acceso a array.

También se pueden activar solo en un bloque:
```cython
with cython.boundscheck(False), cython.wraparound(False):
    for i in range(n):
        c[i] = a[i] + b[i]
```

---

### Typed memoryviews — acceso directo a arrays

Son la forma estándar de trabajar con arrays NumPy en Cython. Dan acceso directo a memoria sin intermediarios.

```cython
import numpy as np
cimport numpy as cnp

def suma_arrays(cnp.float32_t[:] a, cnp.float32_t[:] b):
    cdef int n = a.shape[0]
    cdef int i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] resultado = np.empty(n, dtype=np.float32)
    cdef cnp.float32_t[:] r = resultado

    for i in range(n):
        r[i] = a[i] + b[i]

    return resultado
```

**Sintaxis de memoryviews:**

```cython
cnp.float32_t[:]          # 1D
cnp.float32_t[:, :]       # 2D
cnp.float32_t[:, :, :]    # 3D
cnp.float32_t[:, :, :, :] # 4D (imágenes: batch, canal, H, W)
cnp.float32_t[:, :, ::1]  # 2D contiguo en fila (C-order) — más rápido
```

La diferencia con `cnp.ndarray`:
- `cnp.ndarray` — el objeto Python array completo
- `cnp.float32_t[:, :]` — vista de memoria, más rápida en acceso indexado dentro de bucles

En la práctica, se combinan: se crea el array con `np.empty(...)` y se accede con el memoryview.

---

### `import` vs `cimport`

```cython
import numpy as np       # accede a la API Python de NumPy (np.empty, np.sum...)
cimport numpy as cnp     # accede a los tipos C de NumPy (cnp.float32_t, cnp.ndarray...)
```

Ambos son necesarios. `cimport` no ejecuta ningún código en tiempo de ejecución; solo informa al compilador de los tipos disponibles.

---

### Tipos de funciones: `def`, `cdef`, `cpdef`

```cython
def     funcion_python(x):   # llamable desde Python y Cython
    ...

cdef    funcion_c(int x):    # solo llamable desde Cython/C (NO desde Python)
    ...

cpdef   funcion_mixta(int x): # llamable desde ambos (genera dos versiones)
    ...
```

- `def` — para funciones que llamas desde Python (la función "pública")
- `cdef` — para funciones auxiliares internas que quieres que sean llamadas a velocidad C
- `cpdef` — compromiso, útil para funciones que llamas tanto de Python como de otros `.pyx`

---

### Paralelismo con `prange`

```cython
# cython: boundscheck=False, wraparound=False
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

def suma_paralela(cnp.float32_t[:] a, cnp.float32_t[:] b):
    cdef int n = a.shape[0]
    cdef int i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] r = np.empty(n, dtype=np.float32)
    cdef cnp.float32_t[:] rv = r

    for i in prange(n, nogil=True):   # libera el GIL, ejecuta en paralelo
        rv[i] = a[i] + b[i]

    return r
```

**Restricciones dentro de `nogil`:**
- No puedes crear objetos Python
- No puedes llamar a funciones Python
- Solo puedes operar sobre tipos C y memoryviews
- `prange` requiere OpenMP instalado y flags en `setup.py`

---

### `setup.py` completo

```python
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

HERE = Path(__file__).resolve().parent
pyx_files = sorted(HERE.glob("*.pyx"))

extensions = [
    Extension(
        f"mi_paquete.{p.stem}",   # nombre del módulo importable
        [str(p)],                   # fichero fuente
        include_dirs=[numpy.get_include()],  # headers de NumPy
        # Para OpenMP (solo Linux/macOS con libomp):
        # extra_compile_args=["-fopenmp"],
        # extra_link_args=["-fopenmp"],
    )
    for p in pyx_files
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
```

---

### Qué genera Cython — cómo leer el HTML de anotaciones

```bash
cython -a im2col.pyx    # genera im2col.html
```

El HTML colorea cada línea:
- **Amarillo intenso** — sigue siendo Python lento
- **Amarillo claro / blanco** — código C puro

El objetivo es que los bucles críticos estén en blanco. Si el bucle interno es amarillo intenso, falta un `cdef` o un tipo explícito.

---

---

## PARTE 2 — Aplicación al proyecto miniOIA4DL

---

### Dónde están los hot paths

El cuello de botella real son los bucles de convolución. La tabla de speedups ya lo confirma:

| Operación | Direct (s) | im2col NumPy (s) | im2col Cython (s) |
|---|---|---|---|
| Conv2D 3×3, 64→128 | ~6.9 | ~0.0009 | < 0.0009 |

Cython interviene en la construcción de la **matriz de columnas** (`cols`), que en NumPy requiere un `transpose + reshape` que fuerza una copia de memoria.

---

### La función `im2col_forward_cython` — análisis línea por línea

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
```
→ Desactiva todas las comprobaciones de seguridad. Safe porque `_forward_im2col_cython` ya garantiza que el input es `float32` y contiguo antes de llamar aquí.

```cython
import numpy as np
cimport numpy as cnp
```
→ `import` para crear arrays con `np.empty`; `cimport` para los tipos `cnp.float32_t`.

```cython
def im2col_forward_cython(
    cnp.ndarray[cnp.float32_t, ndim=4] input not None,
    int kernel_size, int stride, int out_h, int out_w
):
```
→ `cnp.ndarray[cnp.float32_t, ndim=4]` declara el tipo completo en la firma: Cython no necesita inspeccionarlo en runtime. `not None` evita el check de `nonecheck`.

```cython
cdef int B = input.shape[0]
cdef int C = input.shape[1]
cdef int K = kernel_size
cdef int col_len   = C * K * K
cdef int n_patches = out_h * out_w
```
→ Todas las dimensiones como `cdef int`. Si no, cada acceso a `.shape[i]` invoca Python.

```cython
cdef cnp.ndarray[cnp.float32_t, ndim=3] cols = np.empty(
    (B, n_patches, col_len), dtype=np.float32
)
cdef cnp.float32_t[:, :, :, :] x = input
cdef cnp.float32_t[:, :, :]    c = cols
```
→ Se crea el array de salida una sola vez fuera del bucle. Los memoryviews `x` y `c` darán acceso directo sin pasar por Python en los bucles.

```cython
for b in range(B):
    patch_idx = 0
    for i in range(out_h):
        for j in range(out_w):
            col_idx = 0
            for ci in range(C):
                for ki in range(K):
                    for kj in range(K):
                        c[b, patch_idx, col_idx] = x[b, ci, i*stride+ki, j*stride+kj]
                        col_idx += 1
            patch_idx += 1
```
→ 6 bucles anidados, pero todos sobre `cdef int` y accediendo a memoryviews. El compilador genera exactamente esto en C:
```c
for (int b = 0; b < B; b++) {
    for (int i = 0; i < out_h; i++) {
        ...
        c_ptr[b][patch_idx][col_idx] = x_ptr[b][ci][i*stride+ki][j*stride+kj];
```
Sin ninguna llamada al intérprete Python en el cuerpo del bucle.

---

### Por qué Cython mejora respecto a im2col NumPy

| Paso | im2col NumPy | im2col Cython |
|---|---|---|
| Obtener patches | `sliding_window_view` (vista, sin copia) | Bucle C directo |
| Construir `cols` | `transpose + reshape` → **copia forzada** | Escritura directa en orden de memoria correcto |
| Fallo de caché | Posible (acceso no contiguo tras transpose) | Mínimo (escritura secuencial) |
| Overhead Python | Una llamada por método | Cero dentro del bucle |

La diferencia clave es que `windows.transpose(0,2,3,1,4,5)` produce un array no-contiguo, y el `.reshape()` siguiente tiene que materializar una copia. Cython escribe directamente en el array de salida en el orden correcto.

---

### Parámetros `mc`, `nc`, `kc`, `mr`, `nr` en Conv2D — para qué sirven

```python
self.mc = 480    # filas del panel de A (input)
self.nc = 3072   # columnas del panel de B (kernels)
self.kc = 384    # dimensión de reducción por bloque
self.mr = 32     # micro-tile de A (por registro)
self.nr = 12     # micro-tile de B (por registro)
self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)
```

Son los parámetros del algoritmo **GEMM bloqueado** (Goto BLAS), el modelo que maximiza el uso de caché L1/L2/L3:

```
Para C = A × B, en lugar de recorrer A y B enteras:
  por paneles de kc columnas de A y kc filas de B:
    copiar panel de A en Ac (cabe en L3)
    por bloques de mc filas de Ac:
      por bloques de nc columnas de B:
        micro-kernel de (mr × nr) que cabe en registros
```

El objetivo es implementar `conv_algo=3` donde el GEMM del paso `cols @ kernels.T` se hace con este algoritmo en Cython en lugar de llamar a BLAS de NumPy. En la práctica solo vale la pena si la dimensión de la matriz supera los tamaños de caché — para capas grandes de ResNet sí puede haber ganancia.

---

### Próximos pasos de optimización Cython en el proyecto

#### a) `col2im` para backward (próximo paso lógico)

El backward necesita la operación inversa a `im2col`: distribuir gradientes de `cols` de vuelta a `grad_input`. En NumPy hay solapamiento porque múltiples patches comparten píxeles de la entrada, lo que obliga a sumar con bucles o `np.add.at`.

En Cython:
```cython
# Esquema de col2im
for b in range(B):
    for patch_idx in range(n_patches):
        i = patch_idx // out_w
        j = patch_idx  % out_w
        col_idx = 0
        for ci in range(C):
            for ki in range(K):
                for kj in range(K):
                    grad_input[b, ci, i*stride+ki, j*stride+kj] += grad_cols[b, patch_idx, col_idx]
                    col_idx += 1
```

#### b) GEMM bloqueado con `mc/nc/kc`

Implementar en Cython el micro-kernel de multiplicación matricial usando los parámetros ya definidos en `Conv2D`. Se sustituiría el `cols @ kernels.T` de NumPy por una llamada Cython que satura L1/L2 en lugar de depender del scheduler de BLAS.

#### c) Padding inline en `im2col_forward_cython`

Actualmente el padding se hace en Python antes de llamar al kernel Cython (con `np.pad`, que crea un array nuevo). Se puede fusionar en el bucle Cython:

```cython
# En lugar de leer x[b, ci, i*stride+ki, j*stride+kj],
# comprobar si el índice cae en la zona de padding y devolver 0.0:
cdef int hi = i * stride + ki - pad
cdef int wj = j * stride + kj - pad
c[b, patch_idx, col_idx] = x[b, ci, hi, wj] if (0 <= hi < H and 0 <= wj < W) else 0.0
```

Esto elimina la creación del array padded y reduce el pico de memoria.
