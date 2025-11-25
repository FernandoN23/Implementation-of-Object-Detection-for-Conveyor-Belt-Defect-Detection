# YOLOv

---

## 🔧 Detección de objetos — YOLO

### Descripción General
El modelo de detección de objetos **YOLO** fue diseñado bajo una estructura modular, garantizando legibilidad, escalabilidad y control experimental por variantes (n, s, m, l, xl).  
El flujo integra los componentes principales del proyecto: *modelos, configuraciones, utilidades, métricas y gestión de checkpoints.*

---

### 🧩 Estructura principal del proyecto

```text
YOLO/
│
├── configs/
│   ├── dataset.yaml                ← Rutas y clases del dataset
│   ├── model_variants.yaml         ← Parámetros de escalado (variantes)
│   ├── train.yaml                  ← Hiperparámetros de entrenamiento
│   └── valid.yaml                  ← Hiperparámetros de validación
│
├── engine/
│   ├── bn2gn_patch.py              ← Parche para normalización por lotes
│   ├── bootstrap_miopen.py         ← Interfaz de mitigación de MIOpen
│   ├── Tester.py                   ← Módulo de pruebas finales
│   ├── Trainer.py                  ← Módulo de entrenamiento
│   ├── Validator.py                ← Módulo de validación interna/externa
│   └── warnings.py                 ← Módulo de advertencias y errores menores
│
├── metrics/                        ← Carpeta de almacenamiento de métricas
├── runs/                           ← Carpeta de almacenamiento de ejecuciones
│
├── utility/
│   ├── check_dataset.py            ← Revisa formato dataset para modelo a entrenar                   
│   ├── clean_logs_runs.py          ← Script de limpieza de logs y runs
│   ├── clean_metrics.py            ← Script de limpieza de métricas
│   ├── clean_weights.py            ← Script de limpieza de pesos registrados
│   ├── metrics.py                  ← Métricas a utilizar en el modelo
│   └── visualization.py            ← Visualización de entrenamiento en Tensorboard
│
├── weights/
│
├── yolov5/                         ← Carpeta con el modelo oficial YOLOv5
│
├── train.py                        ← Script principal de entrenamiento
├── valid.py                        ← Script principal de validación
└── test.py                         ← Script principal de pruebas


```

### ⚙️ Componentes Principales

### 🚀 Flujo del Script train.py


### 📊 Salidas para pruebas



### 📊 Salidas para entrenamiento final

