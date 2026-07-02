# Implementación de Detección de Objetos para la Identificación de Fallas en Correas Transportadoras

> **📄 Nota Importante:** El documento completo con el detalle exhaustivo de esta investigación, marco teórico, metodología y análisis profundo de resultados se encuentra disponible en el archivo **[`Informe_Final_Memoria_FN.pdf`](Informe_Final_Memoria_FN.pdf)**.

Este repositorio contiene el desarrollo y los resultados de la investigación enfocada en desarrollar un sistema de detección de fallas mediante técnicas de visión computacional y aprendizaje profundo. El proyecto aborda la problemática de la inspección visual tradicional, proponiendo soluciones automatizadas y escalables basadas en Redes Neuronales Convolucionales (CNN) y arquitecturas Transformer.

## Estructura y Navegación del Repositorio

Este proyecto está organizado de manera modular. En la rama `main` se encuentra el trabajo consolidado y el script maestro de evaluación, pero existen **ramas dedicadas** para aislar el desarrollo y experimentación de cada modelo de forma independiente.

Para facilitar la reproducción y comprensión del proyecto, se han dispuesto archivos `README.md` específicos en los directorios clave:

* ⚙️ **[`Environment/README.md`](Environment/README.md):** Contiene las instrucciones detalladas, dependencias y la configuración utilizada para replicar el entorno virtual de Python en este equipo.
* 📊 **[`Dataset/README.md`](Dataset/README.md):** Detalles técnicos sobre la distribución de clases, pre-procesamiento, particionamiento y aumentación de datos.
* 🧠 **Modelos:** Cada arquitectura cuenta con su propia documentación para consultar su funcionamiento, entrenamiento e inferencia por separado:
  * [`YOLO/README.md`](YOLO/README.md)
  * [`SSD/README.md`](SSD/README.md)
  * [`DETR/README.md`](DETR/README.md)
  * [`DINO/README.md`](DINO/README.md)

---

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

### Particionamiento de Datos
El conjunto de datos sigue una estrategia de división estándar para garantizar la robustez de los resultados:
* **Entrenamiento (Train - 70%)**: Subconjunto mayoritario destinado a la optimización de pesos y sesgos del modelo.
* **Validación (Val - 20%)**: Subconjunto utilizado para el ajuste de hiperparámetros y monitoreo de métricas durante el entrenamiento para evitar el *overfitting*.
* **Pruebas (Test - 10%)**: Subconjunto ciego utilizado exclusivamente para la evaluación final del rendimiento y la generación de métricas de inferencia.

---

## Resumen de Hiperparámetros

Para mantener un régimen estandarizado y permitir una comparativa justa, los modelos fueron entrenados bajo configuraciones controladas. A continuación, se presenta un resumen de los hiperparámetros clave utilizados:

| Modelo | Épocas | Batch Size | Optimizador | Tasa de Aprendizaje (LR) | Regularización (Weight Decay) |
| :--- | :---: |:----------:| :---: | :---: | :---: |
| **YOLOv5** | 300 |     16     | SGD (Mom: 0.937) | Inicial: 0.01 / Final: 0.01 | 0.0005 |
| **SSD** | 300 |     16     | SGD (Mom: 0.9) | 0.001 (Decaimiento por pasos) | 0.001 |
| **DETR** | 300 |     4      | AdamW | 1e-4 (Backbone: 1e-5) | 0.0001 |
| **DINO** | 300 |   2 a 4    | AdamW | 1e-4 (Backbone: 1e-5) | 0.0001 |

*Nota: Las arquitecturas Transformer (DETR y DINO) requirieron tamaños de lote (batch size) más reducidos debido a su alto consumo de memoria VRAM.*

---

## Tiempos y Rendimiento del Entrenamiento

A continuación, se presenta un resumen del rendimiento computacional extraído durante la fase de entrenamiento. La siguiente tabla detalla la tasa de procesamiento por hora para cada variante arquitectónica y proyecta el tiempo total estimado para completar el régimen estandarizado de 300 épocas.

| Arquitectura | Variante | Rend. [épocas/h] | Tiempo Estimado [h] |
| :--- | :--- | :---: | :---: |
| **YOLOv5** | Nano (n) | 80 | 3.75 |
| | Small (s) | 65 | 4.62 |
| | Medium (m) | 55 | 5.45 |
| | Large (l) | 40 | 7.50 |
| | Extra Large (x) | 25 | 12.00 |
| **SSD** | SSD300 | 36 | 8.33 |
| | SSD512 | 15 | 20.00 |
| **DETR** | R50 | 22 | 13.64 |
| | R50-DC5 | 18 | 16.67 |
| | R101 | 12 | 25.00 |
| | R101-DC5 | 7 | 42.86 |
| **DINO** | R50-4scale | 10 | 30.00 |
| | R50-5scale | 1 | 300.00 |
| | Swin_L | 2 | 150.00 |

Cabe destacar que las estimaciones de tiempo total presentadas no obedecen a un comportamiento estrictamente lineal. Esta escalada en los tiempos de procesamiento representa el comportamiento teóricamente esperado: la transición desde arquitecturas convolucionales livianas de una sola etapa (*One-Stage*) hacia modelos basados en *Transformers* demanda un incremento sustancial en el consumo secuencial de GFLOPs, inherente a la complejidad de los mecanismos de atención. 

A partir de los resultados expuestos, se evidencia una degradación drástica en el rendimiento computacional al transitar desde las variantes de DETR hacia las de DINO. Esta caída se atribuye a la saturación de la memoria de video dedicada (VRAM), lo que forzó al hardware a recurrir a la memoria RAM compartida del sistema (mecanismo de paginación o *swapping*).

---

## Resumen Global de Resultados

A continuación, se muestra el resumen exacto de métricas de rendimiento, complejidad paramétrica y costo computacional para todos los modelos implementados en el conjunto de validación/test.

| Variante | mAP@0.5 | mAP@0.5:0.95 | F1-Score | Parám. (M) | GFLOPs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv5-n** | 0.8735 | 0.5089 | 0.8744 | 1.9 | 4.2 |
| **YOLOv5-s** | **0.8978** | **0.5517** | **0.8897** | **7.2** | **15.8** |
| **YOLOv5-m** | 0.8984 | 0.5480 | 0.8937 | 21.2 | 48.0 |
| **YOLOv5-l** | 0.8958 | 0.5359 | 0.8937 | 46.5 | 108.3 |
| **YOLOv5-x** | 0.9012 | 0.5435 | 0.8910 | 86.7 | 204.7 |
| | | | | | |
| **SSD300** | 0.6856 | 0.3995 | 0.6954 | 26.3 | 31.4 |
| **SSD512** | **0.7394** | **0.4277** | **0.7424** | **27.1** | **90.8** |
| | | | | | |
| **DETR-R50** | **0.8067** | **0.5033** | **0.6959** | **41.0** | **86.0** |
| **DETR-R50-DC5** | 0.7784 | 0.4713 | 0.6845 | 41.0 | 187.0 |
| **DETR-R101** | 0.7984 | 0.4914 | 0.6887 | 60.0 | 152.0 |
| **DETR-R101-DC5** | 0.7960 | 0.4878 | 0.6922 | 60.0 | 253.0 |
| | | | | | |
| **DINO-R50-4scale** | **0.8809** | **0.5298** | **0.7550** | **47.0** | **279.0** |
| **DINO-R50-5scale** | 0.0123 | 0.0064 | 0.0225 | 47.0 | 860.0 |
| **DINO-Swin-L** | 0.8710 | 0.5240 | 0.7554 | 218.0 | 1040.0 |

### Análisis por Arquitectura

*   **YOLOv5:** La variante **Small (s)** se establece como la solución óptima. El escalamiento hacia variantes más densas (m, l, x) incurre en un claro fenómeno de rendimientos decrecientes, triplicando la carga computacional sin mejoras significativas en las métricas.
*   **SSD:** La variante **SSD512** demostró un desempeño superior. El aumento de resolución permitió mitigar la pérdida de información espacial, facilitando la detección de defectos de baja magnitud que quedaban suprimidos en la resolución de 300x300.
*   **DETR:** Se observa una **paradoja de capacidad**. La variante base **ResNet-50 (R50)** obtuvo las métricas más altas. Al utilizar un dataset industrial de tamaño limitado, los modelos más profundos (R101) o con convoluciones dilatadas (DC5) tienden a sobreajustarse o añadir ruido computacional.
*   **DINO:** La variante **R50_4scale** logró el mejor desempeño entre los Transformers, destacando su capacidad superior para predecir defectos difíciles como las perforaciones (*Puncture*). La variante *5scale* sufrió un colapso de gradiente debido a las severas restricciones de tamaño de lote impuestas por el hardware.

---

## Conclusiones y Trabajo Futuro

En el presente trabajo de memoria se cumplió con el objetivo general de implementar y evaluar algoritmos de detección de objetos, basados en inteligencia artificial, para la identificación automática de fallas en correas transportadoras. Se implementaron y evaluaron dos arquitecturas convolucionales de una sola etapa (YOLOv5 y SSD) junto con modelos basados en mecanismos de atención global (*Transformers*: DETR y DINO). Todo el proceso se realizó utilizando una base de datos pública debidamente estructurada, considerando cinco clases de falla.

Respecto a los resultados, la variante **small (s) de YOLOv5 se consolidó como el modelo superior** al maximizar la eficiencia y precisión, mientras que SSD512 obtuvo el rendimiento más deficiente. El mayor desafío recayó en la detección de la clase perforación (*Puncture*); no obstante, **DINO-R50-4scale demostró una ligera superioridad prediciendo este defecto** gracias a su atención global y su mecanismo mejorado de eliminación de ruido contrastivo.

El desempeño general de las arquitecturas *Transformers* se vio penalizado por la escala de los datos, la brecha de dominio y las restricciones físicas. Su extremo consumo de memoria forzó a reducir los tamaños de lote, ralentizando la convergencia de los gradientes y aumentando los tiempos de entrenamiento. Al mantener un régimen estandarizado sin optimización profunda de hiperparámetros, **los detectores convolucionales *One-Stage* (YOLO) se ratifican como la alternativa de mayor viabilidad práctica e industrial en la actualidad.**

**Trabajo Futuro:**
* Optimización exhaustiva de hiperparámetros para cada modelo específico.
* Ejecución de entrenamientos en servidores o clústeres dedicados para mitigar las restricciones de tamaño de lote experimentadas en equipos portátiles.
* Implementación de arquitecturas híbridas (como RT-DETR), orientadas a fusionar la eficiencia de extracción local de las CNN con el mecanismo de atención global de los *Transformers*.