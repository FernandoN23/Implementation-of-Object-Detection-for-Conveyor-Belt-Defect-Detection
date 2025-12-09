# YOLO (You Only Look Once)

---

## 🔧 Detección de objetos — YOLOv5

### Descripción General
Este módulo implementa una capa de orquestación y personalización sobre la arquitectura **YOLOv5**, diseñada específicamente para el trabajo de memoria de título: *"Implementación de algoritmos de reconocimiento de objetos para la identificación de fallas en correas transportadoras"* (Departamento de Ingeniería Mecánica, Universidad de Chile).

El sistema garantiza un flujo de trabajo reproducible y modular, integrando soporte experimental para entornos **AMD ROCm en Windows** mediante inyección de dependencias y mitigación de errores actuales (MIOpen).

Nota: estos parches temporales por incompatibilidad de software serán resueltos en el transcurso del desarrollo del trabajo de memoria de título.

### ⚖️ Licencia y Atribución

Este proyecto utiliza como núcleo el código fuente de **YOLOv5** desarrollado por Ultralytics.

> **Reconocimiento de Autoría:**
> La arquitectura base, los pesos pre-entrenados y la lógica de inferencia subyacente pertenecen al repositorio oficial: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).

---

## 🧩 Estructura del Proyecto

La jerarquía separa la lógica de orquestación (`engine`), la configuración (`configs`) y las herramientas de mantenimiento (`utility`), manteniendo el núcleo (`yolov5`) aislado.

```text
YOLO/
│
├── configs/                  # Centro de Control de Configuraciones
│   ├── dataset.yaml          ← Rutas y definición de clases (Proyecto/Dataset)
│   ├── model_variants.yaml   ← Parámetros de escalado de arquitectura (n, s, m, l, x)
│   ├── train.yaml            ← Maestro de entrenamiento (hiperparámetros, presets, hardware)
│   └── valid.yaml            ← Maestro de validación (métricas, NMS, presets)
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
│   ├── clean_metrics.py      ← Limpieza de métricas generadas
│   ├── clean_runs.py         ← Limpieza de experimentos antiguos
│   ├── clean_weights.py      ← Gestión de espacio (eliminación de pesos redundantes) 
│   ├── clean_weights.py      ← Gestión de espacio (eliminación de pesos redundantes)
│   └── metrics.py            ← Script de procesamiento de datos y generación de reportes
│
├── weights/                  # Almacenamiento de Pesos Consolidados (.pt)
│
├── yolov5/                   # Submódulo: Repositorio Oficial YOLOv5 (Ultralytics)
│
├── train.py                  # CLI: Orquestador de Entrenamiento
├── valid.py                  # CLI: Orquestador de Validación
└── test.py                   # CLI: Visor Interactivo de Inferencia
````

### Descripción de Scripts Principales

  * **`train.py`**: Punto de entrada para el entrenamiento. Gestiona la inicialización segura del entorno MIOpen, aplica parches de normalización (si se requiere) e instancia el `engine.Trainer`. Permite el uso de *presets* definidos en `train.yaml`.
  * **`valid.py`**: Ejecuta la validación del modelo (cálculo de mAP, Precisión, Recall) sobre un conjunto de datos específico. Utiliza `engine.Validator` para asegurar consistencia en la carga del modelo y los parámetros de NMS.
  * **`test.py`**: Herramienta de visualización interactiva (basada en OpenCV). Permite inspeccionar cualitativamente el desempeño del modelo en el conjunto de prueba, dibujando las cajas de verdad (Ground Truth) frente a las predicciones.

-----

## ⚙️ Configuración y Presets

El sistema utiliza *presets* en los archivos YAML para reproducir experimentos complejos con un solo argumento.

### Presets de Entrenamiento (`configs/train.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `smoke_coco128_yolov5s` | **Prueba de Humo**: Valida el pipeline completo (MIOpen, BN2GN, logging) usando el dataset mini COCO128. |
| `yolov5_{n,s,m,l,x}` | **Producción**: Entrenamiento completo (200 épocas, Batch 16) sobre el dataset de correas. Reemplace `{var}` por la variante deseada (ej: `yolov5_s`). |

### Presets de Validación (`configs/valid.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `smoke_coco128_val` | **Prueba de Humo**: Valida el motor de inferencia/métricas con COCO128. |
| `yolov5_{n,s,m,l,x}` | **Producción**: Generación de métricas (mAP, Matriz de Confusión) para el modelo entrenado correspondiente. |

-----

## 🚀 Guía de Uso

Todos los comandos deben ejecutarse desde la raíz del proyecto (nivel superior a `YOLO/`).

### 1\. Entrenamiento

```bash
# Ejecutar prueba rápida de integración
python YOLO/train.py --preset smoke_coco128_yolov5s

# Ejecutar entrenamiento final (Variante Small)
python YOLO/train.py --preset yolov5_s

# Ejecutar entrenamiento final (Variante Nano)
python YOLO/train.py --preset yolov5_n
```

### 2\. Validación

```bash
# Validar usando el preset del proyecto (requiere haber ejecutado el entrenamiento respectivo)
python YOLO/valid.py --preset yolov5_s

# Validación manual
python YOLO/valid.py --weights YOLO/runs/train/belt_defects_yolov5_s/weights/best.pt --task-model detect --imgsz 640
```

### 3\. Inferencia Visual (Test)

```bash
# Abrir visor interactivo (Ajustar ruta según variante entrenada)
python YOLO/test.py --weights YOLO/runs/train/belt_defects_yolov5_s/weights/best.pt --conf-thres 0.4
```

**Controles del Visor:**

  * `d` / `->`: Siguiente imagen
  * `a` / `<-`: Imagen anterior
  * `h`: Ocultar/Mostrar predicciones
  * `ESC`: Salir
-----

## 🖥️ Notas sobre Hardware (AMD ROCm)

Esta implementación incluye mitigaciones automáticas para GPUs AMD en Windows:

1.  **Bootstrap MIOpen**: Inicialización forzada antes de PyTorch.
2.  **Caché Deshabilitada**: `MIOPEN_DISABLE_CACHE=1` por defecto.
3.  **Patch BN2GN**: Sustitución dinámica de `BatchNorm2d` por `GroupNorm` si se detecta inestabilidad.
4. **Parches menores dentro del modelo aislado YOLOv5** : Esto es para que el modelo sea compatible con versiones modernas de Pytorch.
-----