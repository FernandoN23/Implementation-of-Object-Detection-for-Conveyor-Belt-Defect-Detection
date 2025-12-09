# SSD (Single Shot MultiBox Detector)

-----

## 🔧 Detección de objetos — SSD

### Descripción General

Este módulo implementa una arquitectura **SSD (Single Shot MultiBox Detector)** modularizada y adaptada para el trabajo de memoria de título: *"Implementación de algoritmos de reconocimiento de objetos para la identificación de fallas en correas transportadoras"* (Departamento de Ingeniería Mecánica, Universidad de Chile).

El diseño prioriza la legibilidad, la reproducibilidad de experimentos y la integración con flujos de trabajo modernos de PyTorch. Este sistema incluye soporte experimental para entornos **AMD ROCm en Windows** mediante inyección de dependencias y mitigación de errores (bootstrap MIOpen).

### ⚖️ Licencia y Atribución

Este proyecto es una implementación personalizada de la arquitectura SSD, adaptada para el contexto de esta memoria de título.

> **Reconocimiento de Autoría:**
> La arquitectura conceptual se fundamenta en el trabajo original:
> * *Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016). SSD: Single Shot MultiBox Detector. ECCV.*
> * **Repositorio Oficial (Caffe):** [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
-----



## 🧩 Estructura del Proyecto

La jerarquía separa la lógica de orquestación (`engine`), la definición del modelo (`ssd`), la configuración (`configs`) y las herramientas de mantenimiento (`utility`).

```text
SSD/
│
├── configs/                  # Centro de Control de Configuraciones
│   ├── dataset.yaml          ← Rutas y definición de clases del dataset (formato proyecto)
│   ├── model_variants.yaml   ← Parámetros específicos de variantes (vgg, resnet, input size)
│   ├── train.yaml            ← Maestro de entrenamiento (hiperparámetros, presets, optimizador)
│   └── valid.yaml            ← Maestro de validación (métricas, NMS, presets)
│
├── engine/                   # Motor de Orquestación
│   ├── bn2gn_patch.py        ← Mitigación ROCm: Conversión dinámica BatchNorm -> GroupNorm
│   ├── bootstrap_miopen.py   ← Inicialización y configuración de entorno MIOpen
│   ├── Trainer.py            ← Clase envolvente para el ciclo de entrenamiento
│   ├── Validator.py          ← Clase envolvente para el ciclo de validación
│   └── warnings.py           ← Gestión de advertencias del sistema
│
├── metrics/                  # Almacenamiento de reportes y métricas generadas
├── runs/                     # Salida de experimentos (Logs, Checkpoints, TensorBoard)
│
├── ssd/                      # Núcleo del Modelo
│   ├── ssd.py                ← Definición de la arquitectura y ensamblaje del detector
│   └── ...                   # Módulos auxiliares del modelo (layers, box_utils)
│
├── utility/                  # Scripts de Mantenimiento y Procesamiento
│   ├── __init__.py           
│   ├── clean_metrics.py      ← Limpieza de reportes generados
│   ├── clean_runs.py         ← Limpieza de experimentos (logs/checkpoints)
│   ├── clean_weights.py      ← Gestión de espacio (eliminación de pesos redundantes)
│   ├── data_loader.py        ← Adaptador de formato: Convierte Dataset YOLO -> Formato SSD
│   ├── metrics.py            ← Utilidades de cálculo numérico y post-procesado
│   └── ssd_check_params.py   ← Verificación de integridad de parámetros del modelo (pesos/sesgos)
│
├── weights/                  # Almacenamiento de Pesos (Base y Entrenados)
│
├── train.py                  # CLI: Orquestador de Entrenamiento
├── valid.py                  # CLI: Orquestador de Validación
└── test.py                   # CLI: Visor Interactivo de Inferencia
```

### Descripción de Scripts Principales

  * **`train.py`**: Punto de entrada para el entrenamiento. Gestiona la inicialización segura del entorno MIOpen, carga la configuración YAML y ejecuta el `engine.Trainer`. Permite seleccionar variantes mediante *presets* (ej. `ssd300`, `ssd512`).
  * **`valid.py`**: Ejecuta la validación del modelo calculando métricas estándar (mAP, Precisión, Recall). Utiliza `engine.Validator` para asegurar consistencia en el post-procesamiento (NMS).
  * **`test.py`**: Herramienta de visualización interactiva basada en OpenCV. Permite inspeccionar cualitativamente el desempeño del modelo sobre el conjunto de prueba, dibujando las cajas de predicción frente a las etiquetas reales.
  * **`utility/data_loader.py`**: Módulo crítico que actúa como puente entre el formato de etiquetas del proyecto (YOLO) y las estructuras de datos requeridas por SSD, aplicando las transformaciones geométricas necesarias.
  * **`utility/ssd_check_params.py`**: Herramienta de diagnóstico para inspeccionar la estructura interna del modelo, validando la carga correcta de pesos y la coherencia de las capas antes del entrenamiento o inferencia.

-----

## ⚙️ Configuración y Presets

El sistema utiliza *presets* definidos en `configs/train.yaml` y `configs/valid.yaml` para encapsular hiperparámetros y rutas.

### Presets de Entrenamiento (`configs/train.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `ssd300` | **Producción**: Entrenamiento estándar SSD con entrada 300x300. Optimizado para el dataset de fallas. |
| `ssd512` | **Alta Resolución**: Entrenamiento SSD con entrada 512x512 para mejorar la detección de objetos pequeños. |
| `ssd300_voc_debug` | **Depuración**: Configuración para pruebas rápidas de integridad sobre el dataset PASCAL VOC. |

### Presets de Validación (`configs/valid.yaml`)

| Preset | Descripción                                                                               |
| :--- |:------------------------------------------------------------------------------------------|
| `ssd300` | **Estándar**: Validación alineada con el entrenamiento `ssd300` (entrada 300x300).        |
| `ssd512` | **Alta Resolución**: Validación alineada con el entrenamiento `ssd512` (entrada 512x512). |

-----

## 🚀 Guía de Uso

Todos los comandos deben ejecutarse desde la raíz del proyecto (nivel superior a `SSD/`) para garantizar la resolución correcta de rutas.

### 1\. Entrenamiento

```bash
# Ejecutar entrenamiento estándar (SSD300)
python SSD/train.py --preset ssd300

# Ejecutar entrenamiento de alta resolución (SSD512) con nombre personalizado
python SSD/train.py --preset ssd512 --run-name ssd512_run_production

# Forzar uso de CPU (para depuración sin GPU)
python SSD/train.py --preset ssd300 --device cpu
```

### 2\. Validación

```bash
# Validar utilizando el preset por defecto (busca automáticamente el mejor checkpoint)
python SSD/valid.py --preset ssd300

# Validar un archivo de pesos específico
python SSD/valid.py --preset ssd300 --weights SSD/runs/detect/ssd300/train/run_01/best.pth
```

### 3\. Inferencia Visual (Test)

```bash
# Abrir visor interactivo con umbral de confianza ajustado
python SSD/test.py --weights SSD/runs/detect/ssd300/train/best.pth --conf-thres 0.3 --img-dim 300
```

**Controles del Visor:**

  * `d` / `->`: Siguiente imagen
  * `a` / `<-`: Imagen anterior
  * `h`: Ocultar/Mostrar predicciones
  * `ESC`: Salir

-----
## 🖥️ Notas sobre Hardware (AMD ROCm)

Esta implementación incluye mitigaciones automáticas para GPUs AMD en Windows, gestionadas por `engine/bootstrap_miopen.py` y `train.py`:

1.  **Bootstrap MIOpen**: Inicialización forzada del entorno antes de cargar PyTorch para evitar conflictos de DLL.
2.  **Patch BN2GN**: Opción configurable en YAML para sustituir `BatchNorm2d` por `GroupNorm` en caso de inestabilidad numérica en ROCm.
3.  **Variables de Entorno**: Configuración automática de `MIOPEN_FIND_MODE` y desactivación de logs verbosos de MIOpen.

Para desactivar el bootstrap en entornos NVIDIA estándar o CPU, utilice el flag `--no-bootstrap-miopen` en `train.py`.
}