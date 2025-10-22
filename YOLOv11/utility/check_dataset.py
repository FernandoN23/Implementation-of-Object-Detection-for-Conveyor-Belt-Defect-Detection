"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: check_dataset.py
Prepara el dataset local verificando las carpetas train/, valid/,
test/ y data.yaml del directorio principal Dataset/.
-------------------------------------------------------------
"""

import os
import yaml

# Formatos válidos de imagen
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def check_dataset_structure(dataset_path: str):
    print(f"\n📂 Verificando dataset en: {dataset_path}\n")

    expected_splits = ['train', 'valid', 'test']
    all_ok = True

    # --- 1. Verificación de carpetas principales ---
    for split in expected_splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Falta carpeta: {split_path}")
            all_ok = False
            continue

        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')

        if not os.path.exists(images_path):
            print(f"❌ Falta carpeta de imágenes: {images_path}")
            all_ok = False
        if not os.path.exists(labels_path):
            print(f"❌ Falta carpeta de etiquetas: {labels_path}")
            all_ok = False

        # --- 2. Verificación de correspondencia imagen-etiqueta ---
        if os.path.exists(images_path) and os.path.exists(labels_path):
            img_files = [f for f in os.listdir(images_path) if f.lower().endswith(IMG_EXT)]
            lbl_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

            img_basenames = {os.path.splitext(f)[0] for f in img_files}
            lbl_basenames = {os.path.splitext(f)[0] for f in lbl_files}

            missing_labels = img_basenames - lbl_basenames
            missing_images = lbl_basenames - img_basenames

            if missing_labels:
                print(f"⚠️ {len(missing_labels)} imágenes sin etiqueta en {split}/: {list(missing_labels)[:5]}")
                all_ok = False
            if missing_images:
                print(f"⚠️ {len(missing_images)} etiquetas sin imagen en {split}/: {list(missing_images)[:5]}")
                all_ok = False

            # --- 3. Verificación de formato de etiquetas ---
            for lbl_name in lbl_files:
                lbl_path = os.path.join(labels_path, lbl_name)
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines, start=1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"❌ Formato inválido en {lbl_path}, línea {i}: {line.strip()}")
                        all_ok = False
                    else:
                        try:
                            int(parts[0])
                            [float(x) for x in parts[1:]]
                        except ValueError:
                            print(f"❌ Valores no numéricos en {lbl_path}, línea {i}: {line.strip()}")
                            all_ok = False

    # --- 4. Verificación del archivo data.yaml ---
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"\n❌ No se encontró el archivo data.yaml en {dataset_path}")
        all_ok = False
    else:
        print(f"\n📘 Verificando archivo: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        expected_classes = ['Hole', 'Impact Damage', 'Puncture', 'Tear', 'Wear']

        if data.get('nc', None) != 5:
            print(f"❌ nc incorrecto ({data.get('nc')}) → se esperaba 5")
            all_ok = False
        if data.get('names', None) != expected_classes:
            print(f"❌ Nombres de clases incorrectos: {data.get('names')}")
            print(f"✅ Esperado: {expected_classes}")
            all_ok = False

        for split in expected_splits:
            if split not in data:
                print(f"⚠️ Falta la ruta para '{split}' en data.yaml")
                all_ok = False
            elif not os.path.exists(os.path.join(dataset_path, data[split])):
                print(f"⚠️ Ruta de {split} inválida: {data[split]}")

    # --- Resultado final ---
    print("\n" + ("✅ Dataset verificado correctamente." if all_ok else "❌ Errores detectados en la estructura del dataset."))
    return all_ok


if __name__ == "__main__":
    # Ruta de tu dataset (ajustar según memoria guardada)
    dataset_path = r"C:\Users\memorista\Desktop\Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection\Dataset"
    check_dataset_structure(dataset_path)
