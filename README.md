# Implementación de Detección de Objetos para la Identificación de Fallas en Correas Transportadoras

Este repositorio contiene el desarrollo y los resultados de la investigación enfocada en desarrollar un sistema de detección de fallas mediante técnicas de visión computacional y aprendizaje profundo. El proyecto aborda la problemática de la inspección visual tradicional, proponiendo soluciones automatizadas y escalables basadas en Redes Neuronales Convolucionales (CNN) y arquitecturas Transformer.

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

Para detalles técnicos sobre la distribución de clases, pre-procesamiento y aumentación de datos, consulte:[Dataset/README.md](Dataset/README.md).

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

Consulte la documentación técnica específica y guías de reproducción en:[YOLO/README.md](YOLO/README.md).

#### 1.1 Antecedentes y Arquitectura
El modelo implementado corresponde a **YOLOv5**. Esta arquitectura opera como un detector de una sola etapa que integra la extracción de características y la predicción en un flujo unificado.

* **Arquitectura:**
    * **Backbone:** Variante de **CSPDarknet53** con bloques **C3** y capa **SPPF**.
    * **Neck:** Estructura **CSP-PAN** para fusión de características multiescala.
    * **Head:** Cabeza de detección convolucional multiescala (3 niveles).

#### 1.2 Configuración del Entrenamiento
Los experimentos se estandarizaron para todas las variantes (n, s, m, l, x) utilizando la siguiente configuración de hiperparámetros:

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| **Optimizador** | SGD | Momentum: 0.937, Weight Decay: 0.0005 |
| **Épocas** | 200 | Convergencia completa sin Early Stopping agresivo |
| **Batch Size** | 16 | Ajustado para estabilidad de gradiente y memoria VRAM |
| **Imagen** | 640x640 | Resolución de entrada estándar |
| **LR Inicial** | 0.01 | Decaimiento cíclico (Linear/Cosine) |

El entrenamiento mostró una convergencia asintótica estable. La pérdida de clasificación disminuyó rápidamente, indicando una alta separabilidad de las clases de fallas, sin signos severos de *overfitting*.

#### 1.3 Resultados obtenidos
Desempeño de las variantes del modelo **YOLOv5** en el conjunto de validación/test.

| Variante | Tamaño entrada | Parámetros | mAP@0.5  | mAP@0.5:0.95 | F1-Score Max |
|:---------|:--------------:|:----------:|:--------:|:------------:|:------------:|
| **n** |      640       |    1.9M    | **0.86** |   **0.50** |   **0.86** |
| **s** |      640       |    7.2M    | **0.89** |   **0.54** |   **0.88** |
| **m** |      640       |   21.2M    | **0.88** |   **0.55** |   **0.89** |
| **l** |      640       |   46.5M    | **0.88** |   **0.54** |   **0.89** |
| **x** |      640       |   86.7M    | **0.87** |   **0.54** |   **0.89** |

**Conclusión sobre YOLOv5**

La variante **YOLOv5s (Small)** se establece como la solución óptima de despliegue, descartando las arquitecturas de mayor complejidad (**m, l, x**) debido a un estricto criterio de eficiencia algorítmica. El análisis experimental demostró que el escalamiento del modelo hacia variantes más densas —incluyendo la **Medium**, que pese a ser más ligera que las versiones **Large** y **Extra-Large** sigue triplicando la carga computacional frente a la **Small**— incurre en un claro fenómeno de **rendimientos decrecientes**. 

---

### 2. SSD (Single Shot MultiBox Detector)

Consulte la documentación técnica específica y guías de reproducción en: [SSD/README.md](SSD/README.md).

#### 2.1 Antecedentes y Arquitectura
Se implementó **SSD** sobre un backbone **VGG-16** truncado, añadiendo capas convolucionales auxiliares para formar la pirámide de características y permitir la detección en múltiples escalas.

#### 2.2 Configuración del Entrenamiento
A diferencia de YOLO, SSD es más sensible al tamaño de entrada. Se evaluaron dos configuraciones principales (SSD300 y SSD512) bajo los siguientes parámetros comunes:

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| **Optimizador** | SGD | Momentum: 0.937, Weight Decay: 0.0005 |
| **Épocas** | 200 | Aprox. 20,000 iteraciones (dependiendo del split) |
| **Batch Size** | 16 | Unificado para comparativa justa |
| **LR Inicial** | 0.001 | Esquema MultiStep (decaimiento en iteraciones 14k y 17k) |
| **Ratio Neg/Pos**| 3:1 | Hard Negative Mining para balance de clases |

Aunque el modelo logró converger, SSD presentó una curva de aprendizaje más lenta que YOLO y una mayor dificultad para generalizar en la clasificación de defectos morfológicamente similares.

#### 2.3 Resultados obtenidos
Comparativa de desempeño según la resolución de entrada.

| Variante   | Tamaño entrada | Parámetros | mAP@0.5  | mAP@0.5:0.95 | F1-Score Max |
|:-----------|:--------------:|------------|:--------:|:------------:|:------------:|
| **SSD300** |    300x300     | 26.3M      | **0.67** |   **0.39** |   **0.69** |
| **SSD512** |    512x512     | 27.1M      | **0.72** |   **0.42** |   **0.74** |

**Conclusión sobre SSD**

La variante **SSD512** demostró un desempeño superior (+5% mAP@0.5) frente a SSD300. Si bien ambas configuraciones comparten la misma topología base (diferenciándose estrictamente en la dimensionalidad de entrada y la consecuente adaptación de las anclas), este escalamiento a 512x512 resultó determinante. El aumento de resolución permitió mitigar la pérdida de información espacial en los mapas de características, facilitando la detección de defectos de baja magnitud (como perforaciones o grietas finas) que, debido a su tamaño, quedaban suprimidos o indetectables bajo la resolución limitada de la variante SSD300.

---

## Modelos de Detección de Objetos basados en Transformers

Esta sección detalla la implementación de arquitecturas que prescinden de anclas (anchors) y algoritmos de supresión de no máximos (NMS), tratando la detección como un problema de predicción directa de conjuntos mediante mecanismos de atención.

### 3. DETR (DEtection TRansformer)

Consulte la documentación técnica específica y guías de reproducción en:[DETR/README.md](DETR/README.md).

#### 3.1 Antecedentes y Arquitectura
El modelo **DETR** revoluciona el paradigma clásico al utilizar una arquitectura Encoder-Decoder basada en Transformers. 

* **Arquitectura:**
    * **Backbone:** Red convolucional (ResNet-50 o ResNet-101) para la extracción de características 2D.
    * **Transformer:** Un Encoder que modela el contexto global de la imagen mediante auto-atención, y un Decoder que utiliza *Object Queries* aprendidos para predecir las cajas en paralelo.
    * **Loss:** Función de pérdida global basada en emparejamiento bipartito (Hungarian Algorithm) que fuerza predicciones únicas.

#### 3.2 Configuración del Entrenamiento
Debido a la complejidad cuadrática de memoria inherente al mecanismo de atención del Transformer, los hiperparámetros debieron ajustarse a las restricciones de hardware (VRAM), estandarizando el tamaño de lote para permitir una comparativa justa entre variantes.

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| **Optimizador** | AdamW | Weight Decay: 0.0001 |
| **Épocas** | 300 | Requiere mayor tiempo de convergencia que las CNNs |
| **Batch Size** | 4 | Reducido para evitar OOM, especialmente en variantes DC5 |
| **LR Inicial** | 1e-4 / 1e-5 | LR diferenciado: 1e-4 (Transformer) / 1e-5 (Backbone) |
| **LR Drop** | 150 y 250 | Caída del LR (gamma 0.1) en épocas clave |

#### 3.3 Resultados obtenidos
Desempeño de las variantes del modelo **DETR** en el conjunto de validación/test.

| Variante | Tamaño entrada | Parámetros | mAP@0.5  | mAP@0.5:0.95 | F1-Score Max |
|:---------|:--------------:|:----------:|:--------:|:------------:|:------------:|
| **R50** |    Dinámico    |   41.3M    | **0.81** |   **0.50** |   **0.70** |
| **R50_DC5** |    Dinámico    |   41.3M    | **0.78** |   **0.47** |   **0.68** |
| **R101** |    Dinámico    |   60.1M    | **0.80** |   **0.49** |   **0.69** |
| **R101_DC5** |    Dinámico    |   60.1M    | **0.80** |   **0.49** |   **0.69** |

**Conclusión sobre DETR**

Los resultados empíricos demuestran un fenómeno de **paradoja de capacidad**. La variante base **ResNet-50 (R50)** obtuvo las métricas más altas (mAP@0.5 de 81%), superando a la variante más profunda (R101). Esto se debe a que los Transformers carecen de sesgos inductivos y son extremadamente dependientes de grandes volúmenes de datos; al utilizar un dataset industrial de tamaño limitado, el modelo R101 tiende a sobreajustarse, mientras que el R50 posee la capacidad óptima para generalizar. 

Adicionalmente, el uso de convoluciones dilatadas (**DC5**), diseñadas para duplicar la resolución espacial y detectar objetos microscópicos, degradó el rendimiento. Esto sugiere que la escala de los defectos en las correas transportadoras es lo suficientemente grande como para ser capturada por la resolución estándar, y que el aumento de resolución del DC5 solo añadió ruido computacional, dificultando la convergencia del mecanismo de atención.

---

### Comparativa Global

Resumen comparativo seleccionando la mejor configuración de cada arquitectura evaluada para el despliegue final.

| Arquitectura | Mejor Modelo | Tamaño entrada | Parámetros | mAP@0.5:0.95 |
| :--- | :--- |----------------|:----------:| :---: |
| **YOLO** | **YOLOv5s** | 640x640        |    7.2M    | **0.54** |
| **DETR** | **DETR-R50** | Dinámico       |   41.3M    | **0.50** |
| **SSD** | **SSD512** | 512x512        |   27.1M    | **0.42** |

### Conclusiones Parciales

El análisis experimental integral concluye que la arquitectura **YOLOv5s** se mantiene como la solución más robusta y eficiente para el sistema de detección de fallas en correas transportadoras. 

En la comparativa directa, YOLOv5s supera a **DETR-R50** (+4% en mAP@0.5:0.95) utilizando casi **6 veces menos parámetros** (7.2M vs 41.3M) y convergiendo en menos épocas. Aunque DETR demuestra ser una arquitectura poderosa y conceptualmente elegante al eliminar la necesidad de anclas y NMS, su naturaleza basada en Transformers lo hace inherentemente dependiente de conjuntos de datos masivos. En el contexto de este proyecto, donde el dataset industrial es de tamaño moderado, las redes puramente convolucionales (CNN) como YOLO aprovechan mejor sus sesgos inductivos espaciales para extraer características finas.

Por su parte, **SSD512** queda relegado al último lugar de la comparativa, evidenciando que su backbone basado en VGG ha sido superado generacionalmente tanto por las estructuras CSPDarknet (YOLO) como por los mecanismos de atención global (DETR) para tareas de inspección industrial compleja.