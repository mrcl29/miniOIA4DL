# Plan de acción de optimización para miniOIA4DL

Este plan sustituye al anterior y está alineado con los requisitos de optimización por niveles:

- **Nivel Básico**: algoritmos y vectorización NumPy.
- **Nivel Medio**: bibliotecas específicas y Cython.
- **Nivel Alto**: uso de frameworks de optimización vistos en clase.
- **Nivel HPC**: multiproceso y aceleración por hardware (GPU/TPU, si aplica).

## 1) Objetivo y alcance

**Objetivo principal**: maximizar rendimiento de entrenamiento/inferencia en `modules/` sin romper API pública (`forward/backward`, serialización y `conv_algo`).

**Alcance**:
- Incluye: `modules/*.py` y benchmarking ligado a esas capas.
- No incluye (por ahora): rediseño de arquitectura de modelos, cambio de dataset o cambio de función de pérdida.

## 2) KPIs de rendimiento y calidad (obligatorios)

- **Correctitud numérica**: error relativo/absoluto vs implementación base ≤ `1e-5` (float32).
- **Speedup mínimo objetivo**:
  - `Dense.forward/backward`: **x3–x15** (según tamaño de batch y dimensiones).
  - `Conv2D.forward` (`conv_algo=1/2`): **x1.5–x8** frente a `direct`.
  - `MaxPool2D.forward`: **x2–x6**.
- **Regresión funcional**: 100% tests unitarios en verde.
- **Memoria**: monitorizar uso pico para evitar que `im2col` degrade escenarios de batch grande.

## 3) Priorización por niveles

### Nivel Básico (prioridad máxima)

1. Eliminar bucles Python críticos en `Dense`, `Softmax`, `MaxPool2D` y partes de `BatchNorm2D`.
2. Homogeneizar `dtype=np.float32` y evitar copias innecesarias (`np.asarray` en lugar de `np.array` cuando no se quiera copiar).
3. Sustituir `matmul_biasses` por operaciones BLAS (`@`, `np.dot`) conservando API.
4. Añadir micro-benchmarks por capa para medir antes/después.

### Nivel Medio (prioridad alta)

1. Implementar `Conv2D` con `im2col + GEMM` (`conv_algo=1`).
2. Implementar variante `conv_algo=2` optimizada para memoria/caché (im2col parcial, blocked, o ruta `einsum` optimizada).
3. Integrar Cython opcional para hot paths:
   - `im2col`
   - `col2im`
   - kernels de pooling
4. Mantener fallback puro NumPy cuando Cython no esté compilado.

### Nivel Alto (prioridad media)

1. Aplicar framework(s) de optimización vistos en clase sobre kernels clave (p. ej. Numba/JAX/CuPy/Pythran, según temario real).
2. Diseñar backend seleccionable por flag/env (`backend=numpy|cython|numba|...`) sin romper interfaz.
3. Añadir comparación sistemática de backend en `benchmarks/`.

### Nivel HPC (prioridad media-baja)

1. Multiproceso en preparación de lotes/augmentación (pipeline CPU).
2. Evaluar GPU backend (si hay soporte) para inferencia y conv (p. ej. CuPy/JAX).
3. Diseñar pruebas de escalado: throughput vs batch size vs nº procesos.
4. Definir criterio de activación HPC: solo si mejora neta > coste de transferencia/sincronización.

## 4) Mejoras concretas por archivo en modules/

### `modules/utils.py`
- Reemplazar triple bucle de `matmul_biasses` por `A @ B + bias`.
- Mantener firma actual para compatibilidad, pero desviar al camino vectorizado.

### `modules/dense.py`
- `forward`: usar producto matricial vectorizado.
- `backward`: vectorizar `grad_weights`, `grad_biases`, `grad_input`.
- Evitar conversiones/copies repetidas de input y gradientes.

### `modules/conv2d.py` (máximo impacto)
- Mantener `conv_algo=0` (direct) como baseline.
- Implementar `conv_algo=1` (`im2col + GEMM`).
- Implementar `conv_algo=2` (variante memory-aware / blocked / fused).
- Añadir ruta backward coherente con cada algoritmo (siempre que entrenamiento de conv esté habilitado).

### `modules/maxpool2d.py`
- `forward` vectorizado con ventanas (`stride_tricks`) o reshape inteligente cuando sea posible.
- `backward` con índices de máximos evitando bucles anidados en Python.

### `modules/avgpool2d.py`
- Mantener reducción vectorizada.
- Optimizar backward con broadcasting sin copias grandes evitables.

### `modules/batchnorm.py`
- Reducir recomputaciones (`std_inv`, medias/varianzas).
- Garantizar estabilidad numérica y coherencia `float32`.
- Revisar updates de running stats para entrenamiento/inferencia.

### `modules/softmax.py`
- Sustituir bucle por versión batch vectorizada y estable (`x - max(x)`).

### `modules/relu.py`
- Ya vectorizado: revisar que no haga copias innecesarias.

### `modules/dropout.py`
- Mantener máscara vectorizada.
- Validar escalado en train/inference y coste de generación de máscara.

### `modules/flatten.py` y `modules/layer.py`
- Sin hot spots relevantes; solo validar coherencia de shapes y evitar trabajo extra.

## 5) Plan de ejecución por fases

### Fase A — Base reproducible (día 1)
- Entorno + dependencias + tests actuales.
- Medición baseline por capa y por modelo.

### Fase B — Nivel Básico completo (día 1-2)
- Vectorización de `Dense`, `Softmax`, `MaxPool2D`, `utils`.
- Ajustes de `float32` y reducción de copias.
- Validación funcional total.

### Fase C — Nivel Medio (día 2-4)
- `Conv2D` `im2col` + segunda variante.
- Benchmarks comparativos `conv_algo=0/1/2`.
- Integración Cython opcional.

### Fase D — Nivel Alto/HPC (día 4+)
- Prototipo backend de optimización visto en clase.
- Multiproceso en pipeline.
- (Opcional) ruta GPU/TPU y análisis coste-beneficio.

## 6) Validación técnica (DoD)

- [ ] `pytest unit_tests/` pasa sin regresiones.
- [ ] Benchmarks guardados con baseline y versión optimizada.
- [ ] `Dense` y `Softmax` sin bucles Python por muestra.
- [ ] `Conv2D` soporta `conv_algo=0/1/2` y muestra mejora medible.
- [ ] Documentación de uso y flags de optimización actualizada.
- [ ] Compatibilidad de guardado/carga de pesos verificada.

## 7) Riesgos y mitigaciones

- **Riesgo**: `im2col` consume mucha memoria.  
  **Mitigación**: algoritmo alternativo `conv_algo=2` orientado a memoria + tests con batch grande.

- **Riesgo**: diferencias numéricas al vectorizar.  
  **Mitigación**: tolerancias explícitas en tests y comparación con baseline.

- **Riesgo**: Cython complica portabilidad.  
  **Mitigación**: Cython opcional y fallback NumPy siempre disponible.

## 8) Entrega mínima (MVP de optimización)

- [ ] Nivel Básico completado y validado.
- [ ] `Conv2D` con al menos `conv_algo=1` funcional y más rápido.
- [ ] Informe corto de speedups por capa/modelo.
- [ ] Plan de evolución a Nivel Alto/HPC documentado.
