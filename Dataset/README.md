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
Null == Good (considerando que la ausencia de fallas es estado sano)
```

‼️Los formatos de datos antes mencionados serán adaptados según el modelo a utilizar mediante un script de modificación de formato de datos.

