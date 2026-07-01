# DINO (DETR with Improved DeNoising Anchor Boxes) 🦖

---

## 🔧 Detección de objetos — DINO

---

### Descripción General
Este módulo implementa una capa de orquestación y personalización sobre la arquitectura **DINO**, el modelo State-of-the-Art basado en Transformers. Está diseñado específicamente para el trabajo de memoria de título: *"Implementación de algoritmos de reconocimiento de objetos para la identificación de fallas en correas transportadoras"* (Departamento de Ingeniería Mecánica, Universidad de Chile).

DINO mejora la convergencia y el rendimiento de los modelos tipo DETR mediante el uso de **Contrastive DeNoising Training (CDN)**, **Mixed Query Selection**, y **Look Forward Twice**. 

El sistema garantiza un flujo de trabajo reproducible y modular, integrando soporte experimental para entornos **AMD ROCm en Windows** mediante inyección de dependencias, mitigación de errores de memoria (OOM) y cirugía dinámica de pesos para Transfer Learning.

### ⚖️ Licencia y Atribución

Este proyecto utiliza como núcleo el código fuente de **DINO** desarrollado por **IDEA-Research** (International Digital Economy Academy).

> **Reconocimiento de Autoría:**
> La arquitectura base, los pesos pre-entrenados y la lógica de inferencia subyacente pertenecen al repositorio oficial: [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO).
> Autores originales: Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum.

---

## 🧩 Estructura del Proyecto

---

La jerarquía separa la lógica de orquestación (`engine`), la configuración (`configs`) y las herramientas de mantenimiento (`utility`), manteniendo el núcleo oficial (`dino`) aislado pero optimizado.

```text
DINO/
│
├── configs/                  # Centro de Control de Configuraciones
│   ├── dataset.yaml          ← Rutas y definición de clases (Proyecto/Dataset)
│   ├── model_variants.yaml   ← Definición técnica de variantes (r50_4scale, r50_5scale, swin_l)
│   ├── train.yaml            ← Maestro de entrenamiento (hiperparámetros, presets, hardware)
│   └── valid.yaml            ← Maestro de validación (métricas, NMS/Thresholds, presets)
│
├── dino/                     # Submódulo estático: Repositorio Oficial DINO (IDEA-Research)
│   └── models/dino/
│       └── transformer_deformable.py ← [Modificado] Inyección de Gradient Checkpointing
│
├── engine/                   # Motor de Orquestación Personalizado
│   ├── bn2gn_patch.py        ← Mitigación ROCm: Conversión dinámica BatchNorm -> GroupNorm
│   ├── bootstrap_miopen.py   ← Inicialización y configuración de entorno MIOpen
│   ├── Trainer.py            ← Ciclo de entrenamiento con Cirugía de Pesos (Size Mismatch)
│   ├── Validator.py          ← Ciclo de validación con métricas COCO
│   └── warnings.py           ← Filtros de advertencias del sistema
│
├── metrics/                  # Almacenamiento de reportes y métricas procesadas
├── runs/                     # Salida de experimentos (Logs, Gráficos, Checkpoints intermedios)
│
├── utility/                  # Scripts de Mantenimiento y Procesamiento
│   ├── clean_metrics.py      ← Limpieza de reportes generados
│   ├── clean_runs.py         ← Limpieza de experimentos (logs/checkpoints)
│   ├── clean_weights.py      ← Gestión de espacio (eliminación de pesos redundantes)
│   ├── data_loader.py        ← Adaptador Dataset YOLOv11 a DINO y auto-descarga COCO128
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

  * **`train.py`**: Punto de entrada para el entrenamiento. Gestiona la inicialización segura de MIOpen, aplica parches de normalización e instancia el `engine.Trainer`. Incluye lógica de **Cirugía de Pesos** para recortar dinámicamente tensores pre-entrenados al usar variantes "Lite" (ej. 100 queries vs 900 queries).
  * **`valid.py`**: Ejecuta la validación del modelo (cálculo de mAP, Precisión, Recall, F1) sobre un conjunto de datos específico. Genera reportes visuales y matrices de confusión.
  * **`test.py`**: Herramienta de visualización interactiva. Permite inspeccionar cualitativamente el desempeño del modelo en el conjunto de prueba, dibujando las cajas de verdad (Ground Truth) frente a las predicciones. Incluye **Auto-detección de arquitectura** desde el checkpoint para garantizar una inferencia perfecta.

---

## ⚙️ Configuración y Presets

---

El sistema utiliza *presets* en los archivos YAML para reproducir experimentos complejos con un solo argumento.

### Presets de Entrenamiento (`configs/train.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `coco_r50_4scale_test` | **Pruebas de Humo**: Valida el pipeline completo usando el dataset mini COCO128. |
| `dino_r50_4scale_b4` | **Producción Base**: Entrenamiento DINO ResNet-50 4-Scale (300 épocas, Batch 4). |
| `dino_r50_4scale_b4_mod` | **Producción Lite**: Entrenamiento R50 4-Scale con modificaciones de arquitectura Lite (100 queries, 2 puntos) para ahorro de VRAM. |
| `dino_r50_5scale_b2` | **Producción Alta Resolución**: Entrenamiento R50 5-Scale (Optimizado para objetos pequeños). Requiere Batch 2 por alto consumo de memoria. |
| `dino_swin_l_b2` | **Producción SOTA**: Entrenamiento con backbone Swin-Large (State-of-the-Art). Máxima precisión, requiere Batch 2. |

### Presets de Validación (`configs/valid.yaml`)

| Preset | Descripción |
| :--- | :--- |
| `coco_r50_4scale_test` | **Pruebas de Humo**: Valida el motor de inferencia/métricas con COCO128. |
| `dino_r50_4scale_b4` | Validación del modelo DINO R50 4-Scale entrenado. |
| `dino_r50_4scale_b4_mod` | Validación del modelo DINO R50 4-Scale (Lite) entrenado. |
| `dino_r50_5scale_b2` | Validación del modelo DINO R50 5-Scale entrenado. |
| `dino_swin_l_b2` | Validación del modelo DINO Swin-Large entrenado. |

---

## 📥 Carga de Pesos (Checkpoints)

---

Debido a las restricciones de tamaño de GitHub (límites de LFS para archivos mayores a 2GB), **los pesos base, los pesos finales entrenados y los checkpoints completos no están incluidos directamente en este repositorio (a excepción de algunas variantes)**. 

Se solicita al usuario descargar los archivos `.pt` necesarios desde el siguiente enlace y colocarlos manualmente en sus respectivas carpetas:

🔗 **[Descargar Pesos y Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1Fqc22K-zZ0VYC4McjaS9K6EIi-6SCbor?usp=sharing)**

El Drive está estructurado en tres categorías principales para cada variante:
1. **Pesos Base (`weights/base/`)**: Contiene los pesos pre-entrenados oficiales (ej. `36epochs_R50_4scale.pth`, `36epochs_Swin_L.pth`) necesarios para iniciar un entrenamiento desde cero aprovechando el Transfer Learning. Deben ubicarse en `DINO/weights/base/`.
2. **Pesos Finales (`weights/`)**: Contiene los modelos consolidados (versiones ligeras) ideales para ejecutar `valid.py` y `test.py`. Deben ubicarse en `DINO/weights/{variante}/`.
3. **Checkpoints de Entrenamiento (`runs/`)**: Contiene los estados completos del modelo (incluyendo el optimizador AdamW y el modelo EMA). Estos archivos son pesados (hasta 3.5 GB) y son necesarios si deseas **reanudar el entrenamiento** o aplicar optimizaciones adicionales a partir de las 300 épocas ya entrenadas. Deben ubicarse en `DINO/runs/{variante}/train/{run_name}/weights/`.

---

## 🚀 Guía de Uso

--- 

Todos los comandos deben ejecutarse desde la raíz del proyecto (nivel superior a `DINO/`).

### 1. Entrenamiento

```bash
# Ejecutar prueba rápida de integración con COCO128
python DINO/train.py --preset coco_r50_4scale_test

# Ejecutar entrenamiento final (Variante ResNet-50 4-Scale)
python DINO/train.py --preset dino_r50_4scale_b4

# Ejecutar entrenamiento final SOTA (Variante Swin-Large)
python DINO/train.py --preset dino_swin_l_b2
```

### 2. Validación

```bash
# Validar usando el preset del proyecto (requiere haber descargado/entrenado los pesos)
python DINO/valid.py --preset dino_r50_4scale_b4

# Validación manual especificando pesos
python DINO/valid.py --weights DINO/weights/r50_4scale/dino_r50_4s_belt_best.pt
```

### 3. Inferencia Visual (Test)

```bash
# Probar variante ResNet-50 4-Scale
python DINO/test.py --weights DINO/weights/r50_4scale/dino_r50_4s_belt_best.pt --variant r50_4scale --conf-thres 0.5

# Probar variante ResNet-50 5-Scale
python DINO/test.py --weights DINO/weights/r50_5scale/dino_r50_5s_belt_best.pt --variant r50_5scale --conf-thres 0.5

# Probar variante Swin-Large
python DINO/test.py --weights DINO/weights/swin_l/dino_swin_l_belt_best.pt --variant swin_l --conf-thres 0.25
```

**Controles del Visor:**

  * `d` / `->`: Siguiente imagen
  * `a` / `<-`: Imagen anterior
  * `h` / `p`: Ocultar/Mostrar predicciones (Útil para ver el Ground Truth subyacente)
  * `ESC`: Salir

---

## 🖥️ Notas sobre Hardware y Optimizaciones (AMD ROCm)

---

Debido a la complejidad cuadrática de la *Multi-Scale Deformable Attention* y las limitaciones de compilación C++ en Windows/ROCm, esta implementación incluye optimizaciones críticas de ingeniería:

1.  **Gradient Checkpointing Global**: Inyectado directamente en el `DeformableTransformer` (Encoder y Decoder) para aplanar los picos de consumo de VRAM (evitando OOM de 30GB+ en ResNet-50).
2.  **Cirugía Dinámica de Pesos (Size Mismatch Mitigation)**: Permite cargar pesos oficiales masivos (900 queries) en arquitecturas "Lite" (100 queries) recortando los tensores al vuelo durante el Transfer Learning.
3.  **Bootstrap MIOpen**: Inicialización forzada antes de PyTorch para evitar bloqueos del driver HIP.
4.  **Gestión de Memoria**: Uso de `expandable_segments` y Automatic Mixed Precision (AMP).
```