# Implementación de Algoritmos de Reconocimiento de Objetos para la Detección de Fallas en Correas Transportadoras

Link Repositorio: https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git

Trabajo orientado al entrenamiento, validación y evaluación de modelos de detección de objetos aplicados a fallas comunes en correas transportadoras. Se implementan y evalúan cuatro algoritmos representativos de distintos enfoques en visión por computadora (YOLOv11, SSD300, DETR y DINO), analizando su desempeño en términos de precisión y velocidad de inferencia mediante métricas estándar.

El dataset de fallas se utiliza como base para entrenamiento, validación y prueba, pero el énfasis principal del trabajo está en la programación, entrenamiento y evaluación experimental de los modelos de detección.

## 📂 Dataset
Descripción breve del dataset utilizado:  
- Origen: público
- Carpeta de imágenes y etiquetas basados en formato YOLO.
- Estructura de carpetas (train, val, test)  

📂 Estructura de datos (basado en YOLO)

El dataset sigue el formato estándar de YOLO, organizado en dos carpetas principales:

```bash
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
📂 Etiquetado de datos (labels)

Las clases a utilizar se describen en el siguiente vector de clases:
```yaml
Standard_Classes: ['Hole', 'Puncture', 'Tear', 'Wear', 'Impact Damage', 'Good']
```

‼️Los formatos de datos antes mencionados serán adaptados según el modelo a utilizar mediante un script de modificación de formato de datos.

## ⚡ Object Detection Models - One Stage
Modelos de una etapa utilizados en el proyecto, con foco en velocidad y simplicidad:  
- YOLOv11: Arquitectura de YOLO en su versión 11.
- SSD300: Arquitectura versión 300. 

---

## 🤖 Object Detection Models - Transformers
Modelos basados en transformers, orientados a precisión y capacidades avanzadas:  
- DETR: DEtection TRansformer creado por Facebook.
- DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

---

## 🚀 Métricas de evaluación cuantitativas
Se utilizaron métricas de evaluación usadas comúmnente para evaluar el rendimiento de los modelos y realizar una comparación:
- Precision
- Recall
- IoU: Intersection over Union
- FPS

## 📌 Referencias
Enlaces a papers, repositorios oficiales o documentación relevante.  
