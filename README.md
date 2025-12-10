# Implementación de Algoritmos de Reconocimiento de Objetos para la Identificación de Fallas en Correas Transportadoras

Este repositorio contiene el desarrollo y los resultados de la investigación enfocada en desarrollar un sistema de detección de fallas mediante técnicas de visión computacional y aprendizaje profundo. [cite_start]El proyecto aborda la problemática de la inspección visual tradicional, proponiendo soluciones automatizadas y escalables basadas en Redes Neuronales Convolucionales (CNN) y arquitecturas Transformer[cite: 6701, 6702].

## Objetivo General

Implementar y evaluar algoritmos de detección de objetos, basados en inteligencia artificial, para la identificación automática de fallas en correas transportadoras.

## Objetivos Específicos

La investigación se estructura en torno al cumplimiento de las siguientes metas técnicas:
1.  **Implementación de Modelos CNN:** Despliegue de arquitecturas de una sola etapa basadas en redes convolucionales, específicamente **YOLO (You Only Look Once)** y sus variantes, y **SSD (Single Shot Multibox Detector)**.
2.  **Implementación de Modelos Transformer:** Integración de arquitecturas de detección basadas en mecanismos de atención, tales como **DETR (DEtection Transformer)** y **DINO**.
3.  **Entrenamiento Supervisado:** Entrenamiento de los algoritmos utilizando bases de datos especializadas con registros etiquetados de fallas (agujeros, desgarros, abrasión, daño por impacto y perforaciones) en correas transportadoras.
4.  **Evaluación Comparativa:** Análisis cuantitativo del desempeño de los modelos mediante métricas estándar de visión por computadora (mAP, Recall, F1-Score) para determinar su idoneidad en entornos operativos.

---
## Dataset

El conjunto de datos empleado para el entrenamiento de los algoritmos corresponde a una recopilación estructurada de registros visuales de fallas.

Para detalles técnicos sobre la distribución de clases, pre-procesamiento y aumentación de datos, consulte: [Dataset/README.md](Dataset/README.md).

### Particionamiento de Datos
El conjunto de datos sigue una estrategia de división estándar para garantizar la robustez de los resultados:

* **Entrenamiento (Train - 70%)**: Subconjunto mayoritario destinado a la optimización de pesos y sesgos del modelo.
* **Validación (Val - 20%)**: Subconjunto utilizado para el ajuste de hiperparámetros y monitoreo de métricas durante el entrenamiento para evitar el *overfitting*.
* **Pruebas (Test - 10%)**: Subconjunto ciego utilizado exclusivamente para la evaluación final del rendimiento y la generación de métricas de inferencia.

El formato de las etiquetas sigue el estándar de vectores normalizados de cajas delimitadoras `[clase, x_centro, y_centro, ancho, alto]`.

---

## Modelos de Detección de Objetos de Una Sola Etapa

Esta sección detalla la implementación de arquitecturas *one-stage*, caracterizadas por realizar la predicción de cajas y clasificación en una sola pasada de la red, priorizando la velocidad de inferencia.

### 1. YOLO (You Only Look Once)

Consulte la documentación técnica específica y guías de reproducción en: [YOLO/README.md](YOLO/README.md).

#### 1.1 Antecedentes y Arquitectura
El modelo implementado corresponde a **YOLOv5** en su variante **Small (s)**. Esta arquitectura opera como un detector de una sola etapa que integra la extracción de características y la predicción en un flujo unificado.

* **Arquitectura:**
    * **Backbone:** Variante de **CSPDarknet53** que incorpora bloques **C3** (Cross Stage Partial simplificado) y una capa **SPPF** (Spatial Pyramid Pooling Fast) al final para ampliar el campo receptivo. Utiliza funciones de activación **SiLU**.
    * **Neck:** Estructura **CSP-PAN** (Path Aggregation Network) para la fusión de características multiescala, mejorando la propagación de información semántica y espacial.
    * **Head:** Cabeza de detección convolucional multiescala acoplada, que predice simultáneamente coordenadas, *objectness* y probabilidades de clase en tres niveles de resolución.

* **Hiperparámetros (Entrenamiento):**
    * **Optimizador:** SGD (Stochastic Gradient Descent) con momentum de 0.937.
    * **Learning Rate (lr0):** 0.01 con decaimiento cíclico.
    * **Tamaño de imagen:** 640x640 píxeles.
    * **Batch size:** 8.
    * **Épocas:** 100.
    * **Aumentación:** Mosaic (probabilidad 1.0), HSV augmentations, Escalamiento y Traslación.

#### 1.2 Entrenamiento y Validación
El entrenamiento mostró una convergencia asintótica estable hacia la época 100. Se observó una rápida disminución de la pérdida de clasificación, indicando una fácil discriminación de las clases de fallas. No se detectaron signos de *overfitting* severo, manteniendo una correlación consistente entre las pérdidas de entrenamiento y validación.

#### 1.3 Resultados Experimentales
Desempeño del modelo **YOLOv5s** sobre el conjunto de prueba (230 imágenes).

| Modelo | Tamaño Img | Parámetros | mAP@0.5 | mAP@0.5:0.95 | F1-Score Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv5-s** | 640 | 7.2M | **0.99** | **0.52** | **0.87** |

> **Nota:** El modelo demostró un desempeño sobresaliente en fallas volumétricas (*Hole*, *Impact Damage*), pero presentó desafíos en la recuperación (*Recall*) de fallas pequeñas como *Puncture*.
### 2. SSD (Single Shot MultiBox Detector)

Consulte la documentación técnica específica y guías de reproducción en: [SSD/README.md](SSD/README.md).

#### 2.1 Antecedentes y Arquitectura
Se implementó la variante **SSD300**, un detector que discretiza el espacio de salida de cajas delimitadoras utilizando un conjunto de cajas por defecto (*default boxes*) con diferentes relaciones de aspecto.

* **Arquitectura:**
    * **Backbone:** **VGG-16** pre-entrenado (truncado en la capa `pool5`). Se eliminan las capas densas finales y se convierten `fc6` y `fc7` en capas convolucionales.
    * **Capas Adicionales:** Se incorporan capas convolucionales extra (`conv8_2`, `conv9_2`, `conv10_2`, `conv11_2`) para generar una pirámide de características que permite la detección en múltiples escalas.
    * **Cabezas de Predicción:** Filtros convolucionales de 3x3 aplicados densamente sobre los mapas de características para predecir puntuaciones de clase y offsets de localización.

* **Hiperparámetros (Entrenamiento):**
    * **Optimizador:** SGD con momentum de 0.9.
    * **Learning Rate (lr):** 0.001 con esquema de decaimiento *MultiStep*.
    * **Tamaño de entrada:** 300x300 píxeles.
    * **Batch size:** 32.
    * **Iteraciones/Épocas:** 100 épocas (aprox. 5000 iteraciones).
    * **Hard Negative Mining:** Ratio 3:1 (Negativos:Positivos).

#### 2.2 Entrenamiento y Validación
El modelo SSD300 requirió un mayor número de iteraciones para estabilizar sus pérdidas en comparación con YOLO. Aunque logró converger, se observó una brecha de generalización más pronunciada en la componente de clasificación. La baja resolución de entrada (300x300) limitó la capacidad del modelo para caracterizar geométricamente defectos pequeños.

#### 2.3 Resultados Experimentales
Evaluación del modelo SSD300 sobre el conjunto de prueba.

| Modelo | Backbone | Tamaño Entrada | mAP@0.5 | mAP@0.5:0.95 | F1-Score Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SSD300** | VGG16 | 300x300 | **~0.85** | **0.45** | **0.44** |

> **Nota:** SSD presentó dificultades significativas en la detección de la clase *Puncture* y una mayor tasa de falsos positivos en la clase *Tear* (desgarro) debido a la redundancia de cajas no suprimidas correctamente.
## Comparativa Global

Resumen comparativo entre las mejores variantes de cada arquitectura.

| Arquitectura | Mejor Modelo | mAP@0.5:0.95 | Latencia (ms) | Observaciones |
| :--- | :--- | :---: | :---: | :--- |
| **YOLO** | *[Modelo]* | - | - | *[Comentario breve]* |
| **SSD** | *[Modelo]* | - | - | *[Comentario breve]* |



