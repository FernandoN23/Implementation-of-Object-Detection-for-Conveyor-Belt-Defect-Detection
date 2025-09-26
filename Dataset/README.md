# 📂 Dataset 
Descripción breve del dataset (conjunto de datos) utilizado:  
- Origen: público
- Cantidad de imágenes totales: 2312
- Carpeta de imágenes y etiquetas basados en formato YOLO v11.
- Estructura de carpetas (train, val, test)  

🔍Para visualizar el dataset mediante una interfaz de opencv, ejecutar el siguiente script en la terminal:

```bash
python Dataset/view_dataset.py
```
## 📂 Estructura de datos (basado en YOLO v11)

El dataset sigue el formato estándar de YOLO v11, organizado en dos carpetas principales:

```bash
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```
## 📂 Etiquetado de datos (labels)

Las clases a utilizar (fallas) se describen en el siguiente vector de clases:
```yaml
Standard_Classes: ['Hole', 'Impact Damage', 'Puncture', 'Tear', 'Wear']
Null == Good (considerando que la ausencia de fallas es estado sano)
```
Estas se encuentran dentro del archivo `data.yaml`.

## 📦 Formato de Etiquetado (YOLO v11)

Este proyecto utiliza etiquetado en formato YOLO v11 para detección de objetos.
Las etiquetas de cada imagen se almacenan en archivos .txt con el mismo nombre que la imagen y contienen, por línea, la información de cada objeto detectado.

### 📝 Vector de etiquetado  
Cada línea representa **un objeto** con el vector:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id` → Entero (0,1,2,3,4). Cada `class_id` corresponde a la línea (index) en `Standard_classes` (empezando en 0).  
- `x_center`, `y_center`, `width`, `height` → Valores normalizados en [0,1]. `x_center` e `y_center` indican las coordenadas del centro del cuadro delimitador en el plano X-Y, mientras que `width` y `height` representan su ancho y alto relativos al tamaño total de la imagen respecto al punto central (`x_center`, `y_center`).


---
<p align="center">
  <img src="yolo_v11_label.png" alt="label_format" width="400">
</p>


‼️Los formatos de datos antes mencionados serán adaptados según el modelo a utilizar mediante un script de modificación de formato de datos.


## 🖼️ Ejemplos de fallas que contiene el dataset

El dataset filtrado y post-procesado, contiene 6 tipos de clases: 5 fallas y una sana. A continuación se muestran ejemplos de cada falla que contiene el dataset.

<table align="center">
  <tr>
    <td align="center">
      <b>Hole (Agujero)</b><br>
      <img src="train/images/000769_jpg.rf.7870606bb8cfd8dddf852f2d578dfb35.jpg" alt="Hole" width="150">
    </td>
    <td align="center">
      <b>Impact Damage (Daño por impacto)</b><br>
      <img src="train/images/000370_jpg.rf.464d26a480da7cf0a973559d6be927a6.jpg" alt="Impact Damage" width="150">
    </td>
    <td align="center">
      <b>Puncture (Perforación(es))</b><br>
      <img src="train/images/000422_jpg.rf.d4dbd9f20c88220e34afaec9222b214b.jpg" alt="Puncture" width="150">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Tear (Desgarro)</b><br>
      <img src="train/images/001099_jpg.rf.db618932b57d448a3617451f88bc6ee1.jpg" alt="Tear" width="150">
    </td>
    <td align="center">
      <b>Wear (Abrasión)</b><br>
      <img src="train/images/001463_jpg.rf.38912063e9bc84eb8a98c7af85a02954.jpg" alt="Wear" width="150">
    </td>
    <td align="center">
      <b>Sin fallas (Saludable)</b><br>
      <img src="train/images/000804_jpg.rf.0028c0162f4875eca697aecfbd8a6daf.jpg" alt="Sin fallas" width="150">
    </td>
  </tr>
</table>

