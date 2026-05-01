# DETR (DEtection TRansformer)

---

## 🔧 Detección de objetos — DETR

### Descripción General
Este módulo implementa una capa de orquestación y personalización sobre la arquitectura **DETR**, diseñada específicamente para el trabajo de memoria de título: *"Implementación de algoritmos de reconocimiento de objetos para la identificación de fallas en correas transportadoras"* (Departamento de Ingeniería Mecánica, Universidad de Chile).

El sistema garantiza un flujo de trabajo reproducible y modular, integrando soporte experimental para entornos **AMD ROCm en Windows** mediante inyección de dependencias y mitigación de errores actuales (MIOpen).

Nota: estos parches temporales por incompatibilidad de software serán resueltos en el transcurso del desarrollo del trabajo de memoria de título.

### ⚖️ Licencia y Atribución

Este proyecto utiliza como núcleo el código fuente de **DETR** desarrollado por Facebook Research.

> **Reconocimiento de Autoría:**
> La arquitectura base, los pesos pre-entrenados y la lógica de inferencia subyacente pertenecen al repositorio oficial: [DETR](https://github.com/facebookresearch/detr).

---

## 🧩 Estructura del Proyecto

La jerarquía separa la lógica de orquestación (`engine`), la configuración (`configs`) y las herramientas de mantenimiento (`utility`), manteniendo el núcleo (`detr`) aislado.

```text
DETR/
│
├── configs/                  # Centro de Control de Configuraciones
│   ├── dataset.yaml          ← Rutas y definición de clases (Proyecto/Dataset)
│   ├── model_variants.yaml   ← Definición técnica de las variantes de DETR (r50, r101, dc5)
│   ├── train.yaml            ← Maestro de entrenamiento (hiperparámetros, presets, hardware)
│   └── valid.yaml            ← Maestro de validación (métricas, NMS/Thresholds, presets)
│
├── detr/                     # Submódulo estático: Repositorio Oficial DETR (Facebook Research)
│
├── engine/                   # Motor de Orquestación Personalizado
│   ├── bn2gn_patch.py        ← Mitigación ROCm: Conversión dinámica BatchNorm -> GroupNorm
│   ├── bootstrap_miopen.py   ← Inicialización y configuración de entorno MIOpen
│   ├── Trainer.py            ← Clase envolvente para el ciclo de entrenamiento
│   ├── Validator.py          ← Clase envolvente para el ciclo de validación
│   └── warnings.py           ← Filtros de advertencias del sistema
│
├── metrics/                  # Almacenamiento de reportes y métricas procesadas
├── runs/                     # Salida de experimentos (Logs, Gráficos, Checkpoints intermedios)
│
├── utility/                  # Scripts de Mantenimiento y Procesamiento
│   ├── data_loader.py        ← Adaptador Dataset YOLOv11 a DETR y auto-descarga COCO128
│   └── metrics.py            ← Motor gráfico para reportes y CLI interactiva (Modo Merge)
│
├── weights/                  # Almacenamiento de Pesos Consolidados (.pt)
│   └── base/                 ← Directorio para pesos base descargados automáticamente
│
├── train.py                  # CLI: Orquestador de Entrenamiento
├── valid.py                  # CLI: Orquestador de Validación
└── test.py                   # CLI: Visor Interactivo de Inferencia
```

### Descripción de Scripts Principales

  * **`train.py`**: Punto de entrada para el entrenamiento. Gestiona la inicialización segura del entorno MIOpen, aplica parches de normalización (si se requiere) e instancia el `engine.Trainer`. Permite el uso de *presets* definidos en `train.yaml`.
  * **`valid.py`**: Ejecuta la validación del modelo (cálculo de mAP, Precisión, Recall, F1) sobre un conjunto de datos específico. Utiliza `engine.Validator` para asegurar consistencia en la carga del modelo y generar reportes visuales.
  * **`test.py`**: Herramienta de visualización interactiva (basada en OpenCV). Permite inspeccionar cualitativamente el desempeño del modelo en el conjunto de prueba, dibujando las cajas de verdad (Ground Truth) frente a las predicciones generadas por el Transformer, e incluye una leyenda dinámica de métricas por imagen.

---

## ⚙️ Configuración y Presets

El sistema utiliza *presets* en los archivos YAML para reproducir experimentos complejos con un solo argumento.

### Presets de Entrenamiento (`configs/train.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `coco_r50_test` / `coco_r50_dc5_test` | **Pruebas de Humo**: Valida el pipeline completo usando el dataset mini COCO128 (200 épocas). |
| `detr_r50` / `detr_r101` | **Producción Base**: Entrenamiento completo (300 épocas, Batch 16 o 4) sobre el dataset de correas con arquitecturas estándar. |
| `detr_r50_dc5` / `detr_r101_dc5` | **Producción Alta Resolución**: Entrenamiento con convoluciones dilatadas (DC5) para detección fina (300 épocas, Batch 4). |
| `detr_r50_b4` | **Producción Comparativa**: Entrenamiento R50 forzado a Batch 4 para comparativas justas de hardware contra variantes DC5. |

### Presets de Validación (`configs/valid.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `coco_r50_test` / `coco_r50_dc5_test` | **Pruebas de Humo**: Valida el motor de inferencia/métricas con COCO128. |
| `detr_{variante}` | **Producción**: Generación de métricas (mAP, Matriz de Confusión, Curvas PR) para el modelo entrenado correspondiente (ej. `detr_r50`, `detr_r50_dc5`). |

---

## 🚀 Guía de Uso

Todos los comandos deben ejecutarse desde la raíz del proyecto (nivel superior a `DETR/`).

### 1. Entrenamiento

```bash
# Ejecutar prueba rápida de integración con COCO128
python DETR/train.py --preset coco_r50_test

# Ejecutar entrenamiento final (Variante ResNet-50)
python DETR/train.py --preset detr_r50

# Ejecutar entrenamiento final (Variante ResNet-50 con Convoluciones Dilatadas)
python DETR/train.py --preset detr_r50_dc5
```

### 2. Validación

```bash
# Validar usando el preset del proyecto (requiere haber ejecutado el entrenamiento respectivo)
python DETR/valid.py --preset detr_r50

# Validación manual especificando pesos
python DETR/valid.py --weights DETR/weights/r50/detr_r50_belt_best.pt
```

### 3. Inferencia Visual (Test)

```bash
# Abrir visor interactivo (Ajustar ruta y variante según el modelo entrenado)
python DETR/test.py --weights DETR/weights/r50/detr_r50_belt_best.pt --variant r50 --conf-thres 0.5
```

**Controles del Visor:**

  * `d` / `->`: Siguiente imagen
  * `a` / `<-`: Imagen anterior
  * `h` / `p`: Ocultar/Mostrar predicciones
  * `ESC`: Salir

---

## 🖥️ Notas sobre Hardware (AMD ROCm)

Esta implementación incluye mitigaciones automáticas para GPUs AMD en Windows:

1.  **Bootstrap MIOpen**: Inicialización forzada antes de PyTorch.
2.  **Caché Deshabilitada**: `MIOPEN_DISABLE_CACHE=1` por defecto.
3.  **Patch BN2GN**: Sustitución dinámica de `BatchNorm2d` por `GroupNorm` si se detecta inestabilidad.
4.  **Parches menores dentro del modelo aislado DETR**: Esto es para que el modelo sea compatible con versiones modernas de PyTorch.
5.  **Gestión de Memoria**: Uso de `expandable_segments` y Automatic Mixed Precision (AMP) para evitar errores OOM (Out Of Memory) debido a la complejidad cuadrática del Transformer.