# Resumen del repositorio miniOIA4DL

Este documento describe la estructura, componentes y flujo de ejecución del proyecto miniOIA4DL (implementación didáctica de CNNs con NumPy).

**Propósito**: Framework educativo para experimentar con arquitecturas CNN (AlexNet, TinyCNN, ResNet18, OIANet) sobre CIFAR-100. Implementaciones en NumPy pensadas para que el alumno entienda la ejecución de forward/backward y optimización.

**Estructura principal**
- `main.py`: Punto de entrada; carga datos, construye el modelo y lanza entrenamiento/medición.
- `train.py`: Bucle de entrenamiento, cálculo de loss, backprop y guardado de pesos.
- `eval.py`: Evaluación por sample sobre el set de test.
- `performance.py`: Medición simple de rendimiento (IPS) para forward.
- `data/`: Carga y preprocesado de CIFAR-100 y augmentor.
- `modules/`: Implementación de capas (Conv2D, Dense, ReLU, BatchNorm, MaxPool2D, etc.).
- `models/`: Definición de arquitecturas (AlexNet, ResNet18, OIANet, TinyCNN).
- `models/basemodel.py`: Clase `BaseModel` que coordina forward/backward y persistencia de pesos.
- `cython_modules/`: Marcos para optimizaciones en Cython (presente pero no usado por defecto).

**Flujo de ejecución (alto nivel)**
1. `main.py` llama a `data.load_cifar100()` y normaliza imágenes.
2. Construye un modelo según `--model` (p. ej. `OIANet`, `AlexNet`, `ResNet18`).
3. Si `--performance` se habilita, usa `performance.perf()` para medir IPS del forward.
4. Si no, `train.train()` ejecuta el bucle de entrenamiento:
   - mezcla los datos por época
   - realiza forward con `model.forward(batch)`
   - calcula la pérdida y gradiente con `compute_loss_and_gradient`
   - aplica `model.backward(grad, lr)` para actualizar pesos
   - evalúa en test con `eval.evaluate()` y guarda pesos con `BaseModel.save_weights()`

**Descripción de componentes clave**

**modules/conv2d.py (Conv2D)**
- Implementación 'direct' de la convolución (loops explícitos sobre batch, canales y spatial).
- Soporta inicialización `he`, `xavier` y placeholders para otros algoritmos (`conv_algo`).
- Métodos principales: `forward(input)` y `_forward_direct`, además de `_backward_direct` (actualiza kernels y biases).
- Comentario: es simple y didáctica, pero poco eficiente; hay pistas en el código para implementar `im2col` o fusiones.

**modules/dense.py (Dense)**
- Producto matricial implementado mediante la función `matmul_biasses` en `modules/utils.py` (triple loop, añade bias).
- `forward` guarda entrada; `backward` calcula gradientes y actualiza pesos/biases (SGD simple).

**modules/batchnorm.py (BatchNorm2D)**
- Mantiene `gamma` y `beta`, estadísticas en `running_mean`/`running_var`.
- `forward` en modo entrenamiento actualiza running stats; `backward` calcula gradientes y actualiza `gamma`/`beta`.

**modules/relu.py, softmax.py, maxpool2d.py, flatten.py, dropout.py**
- Implementaciones estándar: ReLU (elementwise), Softmax (establecida por fila, backward simplificado), MaxPool2D (guarda índices máximos para backward), Flatten (reshape), Dropout (máscara en forward/backward).

**modules/utils.py**
- `matmul_biasses`: multiplicación matricial básica con triple loop (no optimizada). Ideal para explicar, no para rendimiento.

**models/basemodel.py**
- `BaseModel` contiene la lista de capas y hace `forward` y `backward` encadenados.
- `save_weights(path)` y `load_weights(path)` guardan/cargan pesos por capa usando `numpy.savez` y `np.load` (ficheros `layer_{i}.npz`).

**models/**
- `OIANET_CIFAR100`: arquitectura secuencial de 3 bloques Conv2D→BatchNorm→ReLU→MaxPool, seguido de Flatten, Dense, Dropout y Softmax. Adecuada para CIFAR-100.
- `AlexNet_CIFAR100`: versión adaptada de AlexNet para imágenes 32×32.
- `ResNet18_CIFAR100`: implementación basada en `BasicBlock` con proyecciones cuando cambian dimensiones; incluye `GlobalAvgPool2D` y Dense final.

**data/**
- `cifar100.py`: Descarga (si falta) y extracción del tar, carga de batches, reshape a [N, 3, 32, 32], normalización por media/var calculadas del trainset.
- `one_hot_encode`: devuelve una lista de listas con codificación one-hot (100 clases).
- `cifar100_augmentator.py`: `CIFAR100Augmentor` con `random_crop`, `random_flip` y `add_noise`; método `augment_batch` aplica las transformaciones a un batch.

**Entrenamiento y evaluación**
- `train.py` contiene el bucle: shuffle por época, forward, cálculo de loss por muestra con `compute_loss_and_gradient`, backward via `model.backward` y guardado de mejores pesos (early stopping si no hay mejora).
- `eval.py` recorre las imágenes de test una a una (uso por muestra) y calcula accuracy e IPS.

**Salvado/recuperación de pesos**
- Los pesos por capa se guardan en `saved_models/<model_name>/layer_{i}.npz` mediante `BaseModel.save_weights`.

**Cómo ejecutar**
Ejemplo básico desde la raíz del repo:
```bash
python main.py --model OIANet --batch_size 8 --epochs 10 --learning_rate 0.01
```
- Flags útiles: `--performance` (mide IPS), `--eval_only` (evaluación solamente), `--conv_algo` (placeholder para distintos algoritmos de convolución).

**Limitaciones y notas importantes**
- Implementaciones en NumPy, con loops explícitos: correctas pero lentas para datasets grandes.
- Algunos módulos y optimizaciones (Cython, im2col) están apuntados en el código como extensiones pendientes.
- `matmul_biasses` y la convolución directa son ejemplos didácticos; para producción habría que usar `numpy.dot`, BLAS o implementaciones en C/Cython.
- `one_hot_encode` devuelve listas; en algunos puntos se convierten a `np.array`, revisar coherencia si se lanza profiling.

**Pruebas unitarias**
- Hay tests en `unit_tests/` (por ejemplo `test_conv2d.py`, `test_dense.py`, etc.). Ejecutarlos ayuda a validar implementaciones básicas.

**Siguientes pasos recomendados**
- Ejecutar los unit tests: `./unit_tests/run.sh` (o ejecutar pytest sobre `unit_tests/`).
- Si se desea medir rendimiento, habilitar `--performance` en `main.py`.
- Implementar/improvisar `im2col` y optimizar `matmul_biasses` o usar `np.dot` para mejorar velocidad.

---
Generado automáticamente: revisión del código fuente para documentar arquitectura y uso.
