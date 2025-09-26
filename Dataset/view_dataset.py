import os
import cv2
import yaml
import glob
import numpy as np

# === CONFIGURACIÓN ===
dataset_dir = "./Dataset"
splits = ["train", "valid", "test"]
split = "train"
image_exts = (".jpg", ".jpeg", ".png")

# === LEER CLASES ===
data_yaml_path = os.path.join(dataset_dir, "data.yaml")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"No se encontró 'data.yaml' en {dataset_dir}")

with open(data_yaml_path, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml.get("names") or data_yaml.get("classes")
if not class_names:
    raise ValueError("No se encontraron 'names' ni 'classes' en data.yaml")

# Paleta de colores para bounding boxes (puedes modificarla a gusto)
np.random.seed(42)
colors = [tuple(int(c) for c in np.random.randint(0, 255, size=3)) for _ in range(len(class_names))]

# === FUNCIONES ===
def load_images(split_name):
    images_path = os.path.join(dataset_dir, split_name, "images")
    labels_path = os.path.join(dataset_dir, split_name, "labels")
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(images_path, f"*{ext}")))
    return image_files, labels_path

def draw_boxes(img, label_file):
    h, w = img.shape[:2]
    if not os.path.exists(label_file):
        return img
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
            cls_id = int(cls_id)
            x_c *= w; y_c *= h; bw *= w; bh *= h
            x1 = int(max(0, x_c - bw / 2))
            y1 = int(max(0, y_c - bh / 2))
            x2 = int(min(w - 1, x_c + bw / 2))
            y2 = int(min(h - 1, y_c + bh / 2))
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_x1 = x1
            text_y1 = max(0, y1 - text_h - baseline - 3)
            text_x2 = min(w - 1, x1 + text_w + 4)
            text_y2 = max(text_y1 + text_h + baseline, 0)
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1)
            text_pos_y = text_y2 - baseline - 1
            text_pos_y = max(text_h, text_pos_y)
            cv2.putText(img, label_text, (text_x1 + 2, text_pos_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def add_legend(img, split, idx, total):
    h, w = img.shape[:2]
    legend_width = 250
    # Paleta nueva: fondo azul verdoso
    canvas = np.full((h, legend_width, 3), (50, 120, 140), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    y_start = 30
    line_height = 30

    # Información general
    info_lines = [
        f"Imagen: {idx+1}/{total}",
        "",
        "Comandos:",
        "'a': anterior",
        "'d': siguiente",
        "ESC: salir",
        "",
        "Cambiar Split:"
    ]
    for i, line in enumerate(info_lines):
        y = y_start + i*line_height
        cv2.putText(canvas, line, (10, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # Splits verticales como texto simple
    button_y_start = y_start + len(info_lines)*line_height + 10
    button_spacing = 35
    split_shortcuts = {"train": "t", "valid": "v", "test": "s"}
    for i, s in enumerate(splits):
        y = button_y_start + i*button_spacing
        text_color = (255,255,0) if s == split else (255,255,255)  # resaltar split actual
        text = f"{s.capitalize()} ({split_shortcuts[s]})"
        cv2.putText(canvas, text, (15, y), font, 0.7, text_color, 1, cv2.LINE_AA)

    # Mostrar clases detectadas al final
    classes_y_start = button_y_start + len(splits)*button_spacing + 20
    cv2.putText(canvas, "Standard_classes:", (10, classes_y_start), font, 0.6, (255,200,200), 1, cv2.LINE_AA)
    for i, cname in enumerate(class_names):
        y = classes_y_start + (i+1)*25
        cv2.putText(canvas, f"{i}: {cname}", (15, y), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    combined = np.hstack((img, canvas))
    return combined

# === VISUALIZADOR INTERACTIVO ===
image_files, labels_path = load_images(split)
idx = 0

while True:
    if not image_files:
        print(f"No hay imágenes en el split '{split}'.")
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        continue

    img_path = image_files[idx]
    filename = os.path.splitext(os.path.basename(img_path))[0]
    label_file = os.path.join(labels_path, filename + ".txt")
    img = cv2.imread(img_path)
    if img is None:
        print(f"No se pudo leer {img_path}")
        idx = min(len(image_files)-1, idx+1)
        continue

    img = draw_boxes(img, label_file)
    img_with_legend = add_legend(img, split, idx, len(image_files))
    cv2.imshow("Dataset Viewer", img_with_legend)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('d'):
        idx = min(len(image_files)-1, idx+1)
    elif key == ord('a'):
        idx = max(0, idx-1)
    elif key == ord('t'):
        split = "train"; image_files, labels_path = load_images(split); idx = 0
    elif key == ord('v'):
        split = "valid"; image_files, labels_path = load_images(split); idx = 0
    elif key == ord('s'):
        split = "test"; image_files, labels_path = load_images(split); idx = 0

cv2.destroyAllWindows()
