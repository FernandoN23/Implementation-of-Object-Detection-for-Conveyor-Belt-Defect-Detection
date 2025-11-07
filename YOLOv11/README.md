# YOLOv11

---

## 🔧 Detección de objetos — YOLOv11

### Descripción General
El modelo de detección de objetos **YOLOv11** fue diseñado bajo una estructura modular, garantizando legibilidad, escalabilidad y control experimental por variantes (n, s, m, l, xl).  
El flujo integra los componentes principales del proyecto: *modelos, configuraciones, utilidades, métricas y gestión de checkpoints.*

---

### 🧩 Estructura principal del proyecto

```text
YOLOv11/
│
├── configs/
│   ├── dataset.yaml                ← Rutas y clases del dataset
│   ├── model_variants.yaml         ← Parámetros de escalado (depth/width/channels)
│   ├── parser.yaml                 ← Rutas y configuraciones globales
│   ├── train.yaml                  ← Hiperparámetros de entrenamiento
│   ├── valid.yaml                  ← Hiperparámetros de validación
│   └── yolo11.yaml                 ← Definición estructural del modelo
│
├── logs/                           ← Logs de entrenamiento y ejecución del modelo
├── metrics/                        ← Carpeta de almacenamiento de métricas
├── runs/                           ← Carpeta de almacenamiento de ejecuciones
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
├── metrics/                        ← Carpeta de almacenamiento de los pesos de cada modelo entrenado
│
├── train.py                        ← Script principal de entrenamiento
├── valid.py                        ← Script principal de validación
└── test.py                         ← Script principal de pruebas


```

### ⚙️ Componentes Principales
1. Parser YAML (parser_yaml.py)

    Lee las configuraciones del modelo, dataset y entrenamiento desde archivos .yaml.
Permite parametrizar rutas, hiperparámetros y variantes sin modificar el código fuente.

2. Modelo YOLOv11 (yolo11.py)

    Integra los tres módulos principales:

    Backbone → extracción jerárquica de características (bloques Conv y C3k2).

    Neck → fusión FPN + PAN para combinación multi-escala.

    Head → predicción de cajas, clases y confianza por nivel de resolución.

3. Carga de Datos (data_loader.py)

    Implementa CustomDataset y DataLoader para imágenes y etiquetas YOLO.
Soporta lectura de data.yaml, cacheo en RAM y collate_fn para lotes variables.

4. Función de Pérdida (losses.py)

    Define la clase YoloLoss, que calcula:

    Pérdida de cajas (λ_box)

    Pérdida de objeto (λ_obj)

    Pérdida de clasificación (λ_cls)
Usando funciones SmoothL1, BCE y MSE respectivamente.

5. Registro y Visualización

    logger.py → registra mensajes de entrenamiento y validación.

    visualization.py → activa TensorBoard con subcarpetas por variante (n, s, m...).
Permite monitorear loss, overlay de bboxes c/r al real y métricas en tiempo real.

6. Gestión de Pesos (weights.py)

    Guarda y carga checkpoints:

    save_checkpoint() → crear last.pt y best.pt (mejor ó último)

    load_checkpoint() → permite reanudar entrenamientos interrumpidos.

7. Métricas y Resultados (metrics.py)

    Calcula y guarda métricas clave al entrenar mediante validación interna.

    Ej: Precision, Recall, AP, mAP, F-beta, IoU.

### 🚀 Flujo del Script train.py

1. Inicialización mediante CLI ingresando argumentos: variante, configs, etc.

### 📊 Salidas para pruebas

- **Checkpoints:**  
  `YOLOv11/weights/{variant}/{phase}/tests/{run_name}/yolo11_{variant}_epoch_X.pt`

- **Logs:**  
  `YOLOv11/logs/{variant}/{phase}/tests/{run_name}/yolo11_{variant}_{phase}_{date}.log`

- **TensorBoard:**  
  `YOLOv11/runs/{variant}/{phase}/tests/{run_name}/`

- **Métricas finales:**  
  `YOLOv11/metrics/{variant}/{phase}/tests/{run_name}/`

---

### 📊 Salidas para entrenamiento final

- **Checkpoints:**  
  `YOLOv11/weights/{variant}/{phase}/final/yolo11_{variant}_epoch_X.pt`

- **Logs:**  
  `YOLOv11/logs/{variant}/{phase}/final/yolo11_{variant}_{phase}.log`

- **TensorBoard:**  
  `YOLOv11/runs/{variant}/{phase}/final/`

- **Métricas finales:**  
  `YOLOv11/metrics/{variant}/{phase}/final/`

