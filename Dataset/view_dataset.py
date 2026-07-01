# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: Dataset/view_dataset.py
# Descripción: Visualizador interactivo del dataset anotado en
#  formato YOLO. Permite recorrer imágenes, guardar samples y
#  NUEVO: Sistema de MULTIFILTRO (AND) interactivo con teclado,
#  incluyendo detección de imágenes Saludables (Background).
# ==============================================================

import os
import cv2
import yaml
import glob
import numpy as np
from collections import defaultdict

# === CONFIGURACIÓN ===
dataset_dir = "./Dataset"  # carpeta Dataset con train/, valid/, test/ y data.yaml
samples_dir = os.path.join(dataset_dir, "samples")
image_exts = (".jpg", ".jpeg", ".png")
split = "train"  # split inicial

# Variables de estado para el filtro
HEALTHY_ID = -1
active_filters = set()  # Conjunto para guardar los filtros activos simultáneamente

os.makedirs(samples_dir, exist_ok=True)

# === 1. LEER CLASES DESDE data.yaml ===
data_yaml_path = os.path.join(dataset_dir, "data.yaml")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"No se encontró 'data.yaml' en {dataset_dir}")

with open(data_yaml_path, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml.get("names") or data_yaml.get("classes")
if not class_names:
    raise ValueError("No se encontraron 'names' ni 'classes' en data.yaml")


# === 2. FUNCIONES DE CARGA Y FILTRADO ===

def load_images(split):
    images_path = os.path.join(dataset_dir, split, "images")
    labels_path = os.path.join(dataset_dir, split, "labels")

    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(images_path, f"*{ext}")))

    if not image_files:
        print(f"[ADVERTENCIA] No se encontraron imágenes en {images_path}")

    return sorted(image_files), labels_path


def apply_filters(img_files, lbl_path, filters_set):
    # Si no hay filtros activos, mostrar todo
    if not filters_set:
        return img_files

    filtered_files = []
    for img_path in img_files:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(lbl_path, filename + ".txt")

        present_classes = set()

        # Verificar si es saludable (no existe txt o está vacío)
        if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
            present_classes.add(HEALTHY_ID)
        else:
            # Leer las clases presentes en la imagen
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        present_classes.add(int(parts[0]))

        # Condición AND: Todos los filtros activos deben estar en la imagen
        if filters_set.issubset(present_classes):
            filtered_files.append(img_path)

    return filtered_files


# Carga inicial
all_image_files, labels_path = load_images(split)
image_files = apply_filters(all_image_files, labels_path, active_filters)

# === 3. GENERAR PALETA DE COLORES PARA BBOX ===
np.random.seed(45)
colors = [tuple(int(c) for c in np.random.randint(0, 255, size=3)) for _ in range(len(class_names))]


# === 4. FUNCIÓN PARA DIBUJAR BOXES ===

def draw_boxes(img, label_file):
    h, w = img.shape[:2]
    class_counts = defaultdict(int)
    bboxes = []

    if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
        return img, class_counts, bboxes

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

            bboxes.append((x1, y1, x2, y2))

            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            text_x1 = x1
            text_y1 = y1 - text_h - baseline - 3
            text_x2 = x1 + text_w + 4
            text_y2 = y1

            if text_y1 < 0:
                text_y1 = y2 + 3
                text_y2 = y2 + text_h + baseline + 3

            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1)
            cv2.putText(img, label_text, (text_x1 + 2, text_y2 - 4), font, font_scale, (255, 255, 255), thickness,
                        cv2.LINE_AA)

    return img, class_counts, bboxes


# === 5. FUNCIÓN PARA DIBUJAR LEYENDA ===

def draw_legend(split, idx, num_images, class_names, class_counts, active_filters):
    legend_width = 350
    legend_height = 750
    canvas = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    canvas[:] = (40, 45, 60)  # Fondo oscuro elegante

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    y = 30
    line_height = 25

    # --- SECCIÓN 1: INFO GENERAL ---
    cv2.putText(canvas, f"Split: {split.upper()}", (10, y), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    y += line_height
    cv2.putText(canvas, f"Imagen: {idx + 1}/{num_images}" if num_images > 0 else "Imagen: 0/0", (10, y), font,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    y += line_height * 2

    # --- SECCIÓN 2: MULTIFILTROS ---
    cv2.putText(canvas, "MULTIFILTROS (Presiona numero):", (10, y), font, font_scale, (0, 255, 255), thickness,
                cv2.LINE_AA)
    y += line_height

    # Filtro Saludable
    box = "[X]" if HEALTHY_ID in active_filters else "[ ]"
    color = (0, 255, 0) if HEALTHY_ID in active_filters else (150, 150, 150)
    cv2.putText(canvas, f"0: {box} Saludable (Background)", (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
    y += line_height

    # Filtros de Clases
    for i, cls_name in enumerate(class_names):
        box = "[X]" if i in active_filters else "[ ]"
        color = colors[i % len(colors)] if i in active_filters else (150, 150, 150)
        cv2.putText(canvas, f"{i + 1}: {box} {cls_name}", (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

    y += 5
    cv2.putText(canvas, "'c': Limpiar todos los filtros", (10, y), font, 0.5, (200, 200, 200), thickness, cv2.LINE_AA)
    y += line_height * 2

    # --- SECCIÓN 3: COMANDOS ---
    cv2.putText(canvas, "COMANDOS:", (10, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    y += line_height
    cmds = ["'d': Siguiente | 'a': Anterior", "'s': Guardar sample", "ESC: Salir",
            "Splits: Train(t), Valid(v), Test(p)"]
    for cmd in cmds:
        cv2.putText(canvas, cmd, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height
    y += line_height

    # --- SECCIÓN 4: CONTEO ACTUAL ---
    cv2.putText(canvas, "CONTEO EN IMAGEN ACTUAL:", (10, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    y += line_height

    if not class_counts:
        cv2.putText(canvas, "Saludable (0 fallas)", (40, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    else:
        for i, cls_name in enumerate(class_names):
            count = class_counts.get(i, 0)
            if count > 0:
                txt = f"{cls_name}: {count}"
                color = colors[i % len(colors)]
                cv2.circle(canvas, (20, y - 5), 6, color, -1)
                cv2.putText(canvas, txt, (40, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                y += line_height

    return canvas


# === 6. FUNCIÓN PARA AÑADIR CONTEO A LA IMAGEN GUARDADA ===

def add_counts_overlay(img, class_counts, class_names, colors, bboxes):
    img_out = img.copy()
    img_h, img_w = img_out.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 25
    margin = 20

    texts_and_colors = [("Conteo de fallas:", (255, 255, 255))]
    if not class_counts:
        texts_and_colors.append(("Saludable (Sin fallas)", (0, 255, 0)))
    else:
        for cls_id, count in class_counts.items():
            texts_and_colors.append((f"{class_names[cls_id]}: {count}", colors[cls_id % len(colors)]))

    max_text_w = 0
    for text, _ in texts_and_colors:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if tw > max_text_w:
            max_text_w = tw
    text_block_h = len(texts_and_colors) * line_height

    candidates = [
        (margin, margin),
        (img_w - max_text_w - margin, margin),
        (margin, img_h - text_block_h - margin),
        (img_w - max_text_w - margin, img_h - text_block_h - margin)
    ]

    best_pos = candidates[0]
    min_overlap = float('inf')

    for cx, cy in candidates:
        cx2 = cx + max_text_w
        cy2 = cy + text_block_h
        overlap_area = 0

        for bx1, by1, bx2, by2 in bboxes:
            ix1 = max(cx, bx1)
            iy1 = max(cy, by1)
            ix2 = min(cx2, bx2)
            iy2 = min(cy2, by2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            overlap_area += (iw * ih)

        if overlap_area < min_overlap:
            min_overlap = overlap_area
            best_pos = (cx, cy)

        if overlap_area == 0:
            break

    start_x, start_y = best_pos
    y = start_y + 20

    for text, color in texts_and_colors:
        cv2.putText(img_out, text, (start_x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_out, text, (start_x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

    return img_out


# === 7. BUCLE PRINCIPAL DE VISUALIZACIÓN ===

idx = 0

while True:
    num_images = len(image_files)

    if num_images == 0:
        img_boxed = np.zeros((600, 800, 3), dtype=np.uint8)
        img_boxed[:] = (30, 30, 30)
        cv2.putText(img_boxed, "Ninguna imagen cumple con TODOS los filtros activos.",
                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        class_counts = {}
        bboxes = []
    else:
        if idx < 0: idx = 0
        if idx >= num_images: idx = num_images - 1

        img_path = image_files[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(labels_path, filename + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            idx += 1
            continue

        img_boxed, class_counts, bboxes = draw_boxes(img.copy(), label_file)

    legend = draw_legend(split, idx, num_images, class_names, class_counts, active_filters)

    if legend.shape[0] != img_boxed.shape[0]:
        legend = cv2.resize(legend, (legend.shape[1], img_boxed.shape[0]))

    combined = np.hstack((img_boxed, legend))
    cv2.imshow("Visualizador Dataset", combined)

    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord("d") and num_images > 0:  # siguiente
        idx = min(num_images - 1, idx + 1)
    elif key == ord("a") and num_images > 0:  # anterior
        idx = max(0, idx - 1)
    elif key == ord("s"):  # GUARDAR SAMPLE
        if num_images > 0:
            img_to_save = add_counts_overlay(img_boxed, class_counts, class_names, colors, bboxes)
            save_path = os.path.join(samples_dir, f"{filename}_sample.jpg")
            cv2.imwrite(save_path, img_to_save)
            print(f"[INFO] Sample guardado exitosamente en: {save_path}")
    elif key == ord("c"):  # LIMPIAR FILTROS
        active_filters.clear()
        image_files = apply_filters(all_image_files, labels_path, active_filters)
        idx = 0
    elif ord("0") <= key <= ord("9"):  # TOGGLE FILTROS (0 = Saludable, 1-5 = Fallas)
        num_pressed = key - ord("0")

        if num_pressed == 0:
            filter_id = HEALTHY_ID
        elif 1 <= num_pressed <= len(class_names):
            filter_id = num_pressed - 1
        else:
            continue  # Tecla numérica fuera de rango

        # Lógica de Toggle (Activar/Desactivar)
        if filter_id in active_filters:
            active_filters.remove(filter_id)
        else:
            # Si se activa Saludable, se desactivan las fallas (son excluyentes)
            if filter_id == HEALTHY_ID:
                active_filters.clear()
                active_filters.add(HEALTHY_ID)
            else:
                # Si se activa una falla, se desactiva Saludable
                if HEALTHY_ID in active_filters:
                    active_filters.remove(HEALTHY_ID)
                active_filters.add(filter_id)

        image_files = apply_filters(all_image_files, labels_path, active_filters)
        idx = 0

    elif key in [ord("t"), ord("v"), ord("p")]:  # CAMBIAR SPLIT
        if key == ord("t"):
            split = "train"
        elif key == ord("v"):
            split = "valid"
        elif key == ord("p"):
            split = "test"

        all_image_files, labels_path = load_images(split)
        image_files = apply_filters(all_image_files, labels_path, active_filters)
        idx = 0

cv2.destroyAllWindows()