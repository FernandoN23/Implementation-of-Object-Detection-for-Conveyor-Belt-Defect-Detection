# Implementación de Algoritmos de Reconocimiento de Objetos para la Detección de Fallas en Correas Transportadoras

Trabajo orientado al entrenamiento, validación y evaluación de modelos de detección de objetos aplicados a fallas comunes en correas transportadoras. Se implementan y evalúan cuatro algoritmos representativos de distintos enfoques en visión por computadora (YOLOv11, SSD300, DETR y DINO), analizando su desempeño en términos de precisión y velocidad de inferencia mediante métricas estándar.

## Dataset

El conjunto de datos empleado para el entrenamiento de los algoritmos de reconocimiento o detección de objetos, corresponden a una recopilación de registros de fallas, en formato imágenes, sobre fallas en correas transportadoras.

Ver estructura del dataset: [Dataset/README.md](Dataset/README.md)

Este conjunto de datos considera el uso de tres particiones: entrenamiento, validación y pruebas:
- Entrenamiento (train): partición más grande (70%) enfocada en el aprendizaje y calibración de los parámetros de los modelos.
- Validación (valid): partición de tamaño reducida (20%) enfocada en comprobar el entrenamiento antes de las pruebas finales.
- Pruebas (test): partición más pequeña (10%) para realizar pruebas y obtener métricas de desempeño de los modelos.

Cada partición posee imágenes (images) y etiquetas (labels), las cuáles corresponden a los registros de fallas de correas transportadoras con sus respectivas etiquetas (vectores normalizados de cajas delimitadoras).

## Modelos de detección de objetos de una sola etapa

### YOLOv11



### SSD 300