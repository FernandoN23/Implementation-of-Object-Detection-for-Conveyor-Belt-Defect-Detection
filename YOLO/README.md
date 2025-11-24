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
│   ├── model_variants.yaml         ← Parámetros de escalado (depth/width/channels)
│   ├── parser.yaml                 ← Rutas y configuraciones globales
│   ├── train.yaml                  ← Hiperparámetros de entrenamiento
│   ├── valid.yaml                  ← Hiperparámetros de validación
│   └── yolo11.yaml                 ← Definición estructural del modelo
│
├── engine/
│   ├── amp.py                      ← Precisión Mixta Automática (Automatic Mixed Precision)
│   ├── bn2gn_patch.py              ← Parche para normalización por lotes
│   ├── bootstrap_miopen.py         ← Interfaz de mitigación de MIOpen
│   ├── callbacks.py                ← Módulo de llamadas y gestión de eventos
│   ├── CLI.py                      ← Interfaz de Línea de Comandos
│   ├── ema.py                      ← Media Móvil Exponencial (Exponential Moving Average)
│   ├── hud.py                      ← Barra de estado (Head-up Display) 
│   ├── optim.py                    ← Optimizador y parámetros de entrenamiento
│   ├── utils.py                    ← Utilidades (seeds, helpers y varios)
│   ├── validator.py                ← Módulo de validación interna/externa
│   └── warmup_sanity.py            ← Módulo de calentamiento del entrenamiento
│
├── logs/                           ← Logs de entrenamiento y ejecución del modelo
├── metrics/                        ← Carpeta de almacenamiento de métricas
│
├── models/
│   ├── nn/         
│   │   ├── activation.py           ← Funciones de activación
│   │   ├── block.py                ← Bloques convolucionales
│   │   └── conv.py                 ← Redes neuronales convolucionales                                   
│   ├── backbone.py                 ← Extracción de características multiescala
│   ├── head.py                     ← Predicción y clasificación         
│   ├── neck.py                     ← Fusión de características multiescala
│   ├── parser_yaml.py              ← Módulo auxiliar de configuración y parámetros
│   └── yolo11.py                   ← Integración de la arquitectura de YOLOv11
│
├── runs/                           ← Carpeta de almacenamiento de ejecuciones
│
├── utility/
│   ├── check_dataset.py            ← Revisa formato dataset para modelo a entrenar                   
│   ├── clean_logs_runs.py          ← Script de limpieza de logs y runs
│   ├── clean_metrics.py            ← Script de limpieza de métricas
│   ├── clean_weights.py            ← Script de limpieza de pesos registrados
│   ├── data_loader.py              ← Cargador de datos para YOLOv11
│   ├── logger.py                   ← Script para registrar eventos
│   ├── losses.py                   ← Función de pérdida de YOLOv11
│   ├── metrics.py                  ← Métricas a utilizar en el modelo
│   ├── test_metrics.py             ← Script para probar el modelo antes de entrenar
│   ├── test_model.py               ← Script para probar el modelo antes de entrenar
│   ├── visualization.py            ← Visualización de entrenamiento en Tensorboard
│   └── weights.py                  ← Script para el manejo de checkpoints y pesos.
│
├── train.py                        ← Script principal de entrenamiento
├── valid.py                        ← Script principal de validación
└── test.py                         ← Script principal de pruebas


```

### ⚙️ Componentes Principales

### 🚀 Flujo del Script train.py


### 📊 Salidas para pruebas



### 📊 Salidas para entrenamiento final

