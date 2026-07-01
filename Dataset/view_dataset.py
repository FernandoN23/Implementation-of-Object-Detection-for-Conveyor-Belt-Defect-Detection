# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: Dataset/view_dataset.py
# Descripción: Visualizador interactivo del dataset anotado en
#  formato YOLO. Permite recorrer imágenes por split (train/valid/test),
#  dibujar bounding boxes a partir de labels y mostrar un panel lateral
#  con conteo por clase e instrucciones de uso.
#==============================================================

import os
import cv2
import yaml
import glob
import numpy as np
from collections import defaultdict

# === CONFIGURACIÓN ===
dataset_dir = "./Dataset"  # carpeta Dataset con train/, valid/, test/ y data.yaml
image_exts = (".jpg", ".jpeg", ".png")
split = "train"  # split inicial

# === 1. LEER CLASES DESDE data.yaml ===
data_yaml_path = os.path.join(dataset_dir, "data.yaml")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"No se encontró 'data.yaml' en {dataset_dir}")

with open(data_yaml_path, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml.get("names") or data_yaml.get("classes")
if not class_names:
    raise ValueError("No se encontraron 'names' ni 'classes' en data.yaml")

# === 2. FUNCIÓN PARA CARGAR IMÁGENES DE UN SPLIT ===

def load_images(split):
    images_path = os.path.join(dataset_dir, split, "images")
    labels_path = os.path.join(dataset_dir, split, "labels")

    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(images_path, f"*{ext}")))

    if not image_files:
        raise ValueError(f"No se encontraron imágenes en {images_path}")

    return image_files, labels_path


image_files, labels_path = load_images(split)

# === 3. GENERAR PALETA DE COLORES PARA BBOX ===
np.random.seed(45)
colors = [tuple(int(c) for c in np.random.randint(0, 255, size=3)) for _ in range(len(class_names))]


# === 4. FUNCIÓN PARA DIBUJAR BOXES Y RETORNAR CONTADOR POR CLASE ===

def draw_boxes(img, label_file):
    h, w = img.shape[:2]
    class_counts = defaultdict(int)

    if not os.path.exists(label_file):
        return img, class_counts

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
            cls_id = int(cls_id)

            class_counts[cls_id] += 1

            x_c *= w
            y_c *= h
            bw *= w
            bh *= h

            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # bbox más delgado

            label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Intentar colocar arriba del bbox
            text_x1 = x1
            text_y1 = y1 - text_h - baseline - 3
            text_x2 = x1 + text_w + 4
            text_y2 = y1

            # Si se sale arriba, colocarlo debajo del bbox
            if text_y1 < 0:
                text_y1 = y2 + 3
                text_y2 = y2 + text_h + baseline + 3

            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1)
            cv2.putText(
                img,
                label_text,
                (text_x1 + 2, text_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return img, class_counts


# === 5. FUNCIÓN PARA DIBUJAR LEYENDA ===

def draw_legend(split, idx, num_images, class_names, class_counts):
    legend_width = 300
    legend_height = 600  # ajusta según tu ventana
    canvas = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    canvas[:] = (50, 60, 90)  # azul frío

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    y_start = 30
    line_height = 25

    total_objects = sum(class_counts.values())
    info_lines = [
        f"Split: {split}",
        f"Imagen: {idx+1}/{num_images}",
        f"Total objetos: {total_objects}",
        "",
        "Comandos:",
        "'d': siguiente",
        "'a': anterior",
        "ESC: salir",
        "Train (t), Valid (v), Test (p)",
        "",
        "Conteo por clase:",
    ]

    y = y_start
    for line in info_lines:
        color = (255, 255, 0) if "Split" in line or "Total objetos" in line else (255, 255, 255)
        cv2.putText(canvas, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

    # Mostrar cantidad por clase con círculo del color de la clase
    for i, cls_name in enumerate(class_names):
        count = class_counts.get(i, 0)
        txt = f"{cls_name}: {count}"
        color = colors[i % len(colors)]

        cv2.circle(canvas, (20, y - 8), 6, color, -1)  # círculo
        cv2.putText(
            canvas,
            txt,
            (40, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return canvas


# === 6. BUCLE PRINCIPAL DE VISUALIZACIÓN ===

idx = 0
num_images = len(image_files)

while True:
    if idx < 0:
        idx = 0
    if idx >= num_images:
        idx = num_images - 1

    img_path = image_files[idx]
    filename = os.path.splitext(os.path.basename(img_path))[0]
    label_file = os.path.join(labels_path, filename + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"No se pudo leer la imagen: {img_path}")
        idx += 1
        continue

    img, class_counts = draw_boxes(img, label_file)
    legend = draw_legend(split, idx, num_images, class_names, class_counts)

    if legend.shape[0] != img.shape[0]:
        legend = cv2.resize(legend, (legend.shape[1], img.shape[0]))

    combined = np.hstack((img, legend))
    cv2.imshow("Visualizador Dataset", combined)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord("d"):  # siguiente
        idx = min(num_images - 1, idx + 1)
    elif key == ord("a"):  # anterior
        idx = max(0, idx - 1)
    elif key == ord("t"):
        split = "train"
        image_files, labels_path = load_images(split)
        num_images = len(image_files)
        idx = 0
    elif key == ord("v"):
        split = "valid"
        image_files, labels_path = load_images(split)
        num_images = len(image_files)
        idx = 0
    elif key == ord("p"):
        split = "test"
        image_files, labels_path = load_images(split)
        num_images = len(image_files)
        idx = 0

cv2.destroyAllWindows()
