# Explicación de `_sliding_window_view_2d` y `_forward_im2col`

## Ejemplo de entrada

Usamos el caso más sencillo posible:

- **Batch**: 1 imagen
- **Canales de entrada**: 1
- **Imagen**: 4×4
- **Kernel**: 2×2, 1 filtro de salida
- **Stride**: 1, **Padding**: 0

```
Input (4×4):         Kernel (2×2):
 1  2  3  4           1  0
 5  6  7  8           0  1
 9 10 11 12
13 14 15 16
```

Las dimensiones de salida son:

$$H_{out} = \frac{4 - 2}{1} + 1 = 3 \qquad W_{out} = 3$$

---

## Paso 1 — `_sliding_window_view_2d`

```python
def _sliding_window_view_2d(input_tensor, kernel_size, stride):
    windows = np.lib.stride_tricks.sliding_window_view(
        input_tensor, (kernel_size, kernel_size), axis=(2, 3)
    )
    return windows[:, :, ::stride, ::stride, :, :]
```

`sliding_window_view` genera una **vista** (sin copiar memoria) del tensor de entrada donde cada posición `[b, c, i, j]` contiene el patch de tamaño K×K centrado en `(i, j)`.

Resultado tras la llamada, con shape `(1, 1, 3, 3, 2, 2)`:

```
windows[0, 0, i, j] → patch 2×2 que empieza en la fila i, columna j:

  (i=0,j=0)    (i=0,j=1)    (i=0,j=2)
   1  2          2  3          3  4
   5  6          6  7          7  8

  (i=1,j=0)    (i=1,j=1)    (i=1,j=2)
   5  6          6  7          7  8
   9 10         10 11         11 12

  (i=2,j=0)    (i=2,j=1)    (i=2,j=2)
   9 10         10 11         11 12
  13 14         14 15         15 16
```

El `::stride` del return selecciona solo las posiciones de inicio válidas según el stride (aquí stride=1, así que se toman todas).

---

## Paso 2 — Construcción de la matriz de columnas (`cols`)

Dentro de `_forward_im2col`:

```python
cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size, out_h * out_w, -1)
```

El `transpose(0, 2, 3, 1, 4, 5)` reordena a `(B, out_h, out_w, C, K, K)` y el `reshape` aplana cada patch a un vector de longitud `C·K·K = 1·2·2 = 4`.

Resultado `cols` con shape `(1, 9, 4)` — **9 patches**, cada uno como fila:

```
Fila 0  → patch(0,0) aplanado:  [ 1,  2,  5,  6]
Fila 1  → patch(0,1) aplanado:  [ 2,  3,  6,  7]
Fila 2  → patch(0,2) aplanado:  [ 3,  4,  7,  8]
Fila 3  → patch(1,0) aplanado:  [ 5,  6,  9, 10]
Fila 4  → patch(1,1) aplanado:  [ 6,  7, 10, 11]
Fila 5  → patch(1,2) aplanado:  [ 7,  8, 11, 12]
Fila 6  → patch(2,0) aplanado:  [ 9, 10, 13, 14]
Fila 7  → patch(2,1) aplanado:  [10, 11, 14, 15]
Fila 8  → patch(2,2) aplanado:  [11, 12, 15, 16]
```

---

## Paso 3 — El kernel como fila

```python
kernels = self.kernels.reshape(self.out_channels, -1)
```

El kernel `(1, 1, 2, 2)` se aplana a shape `(1, 4)`:

```
kernels = [[1, 0, 0, 1]]
```

---

## Paso 4 — Multiplicación matricial (GEMM)

```python
output = cols @ kernels.T   # (1, 9, 4) @ (4, 1) → (1, 9, 1)
```

Cada fila de `cols` se multiplica por el vector del kernel — equivale a hacer el producto escalar entre el patch aplanado y el kernel aplanado:

```
patch(0,0): [1, 2, 5, 6] · [1, 0, 0, 1] = 1+0+0+6  =  7
patch(0,1): [2, 3, 6, 7] · [1, 0, 0, 1] = 2+0+0+7  =  9
patch(0,2): [3, 4, 7, 8] · [1, 0, 0, 1] = 3+0+0+8  = 11
patch(1,0): [5, 6, 9,10] · [1, 0, 0, 1] = 5+0+0+10 = 15
patch(1,1): [6, 7,10,11] · [1, 0, 0, 1] = 6+0+0+11 = 17
patch(1,2): [7, 8,11,12] · [1, 0, 0, 1] = 7+0+0+12 = 19
patch(2,0): [9,10,13,14] · [1, 0, 0, 1] = 9+0+0+14 = 23
patch(2,1):[10,11,14,15] · [1, 0, 0, 1] =10+0+0+15 = 25
patch(2,2):[11,12,15,16] · [1, 0, 0, 1] =11+0+0+16 = 27
```

---

## Paso 5 — Reshape al formato de salida

```python
output.transpose(0, 2, 1).reshape(batch_size, self.out_channels, out_h, out_w)
```

El vector `[7, 9, 11, 15, 17, 19, 23, 25, 27]` se convierte en:

```
Output (1×1×3×3):
 7  9 11
15 17 19
23 25 27
```

---

## Resumen visual del flujo completo

```
Input (B, C, H, W)
       │
       ▼
_sliding_window_view_2d
       │  shape: (B, C, out_h, out_w, K, K)
       ▼
  transpose + reshape  →  cols  (B, out_h·out_w, C·K·K)
       │
       ▼
  cols @ kernels.T          kernels: (out_C, C·K·K)
       │  shape: (B, out_h·out_w, out_C)
       ▼
  + biases
       │
       ▼
  transpose + reshape  →  Output (B, out_C, out_h, out_w)
```

La clave es que **5 bucles anidados** se sustituyen por una única llamada GEMM (`@`), que NumPy ejecuta con rutinas BLAS en C/Fortran altamente optimizadas.
