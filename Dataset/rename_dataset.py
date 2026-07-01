import os
import glob

# === CONFIGURACIÓN ===
dataset_dir = "./Dataset"  # ruta raíz
splits = ["train", "valid", "test"]  # orden
image_exts = (".jpg", ".jpeg", ".png")

# === FUNCIONES ===
def renombrar_split(split, start_index):
    """
    Renombra imágenes y labels en un split (train/valid/test)
    desde start_index. Devuelve el último índice usado.
    """
    images_path = os.path.join(dataset_dir, split, "images")
    labels_path = os.path.join(dataset_dir, split, "labels")

    # lista de imágenes
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(images_path, f"*{ext}")))

    image_files.sort()  # orden para reproducibilidad

    index = start_index
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # buscar label
        label_path = os.path.join(labels_path, basename + ".txt")

        new_name = f"{index:04d}"  # 0001, 0002, etc.

        # extensión original de imagen
        ext_img = os.path.splitext(img_path)[1].lower()
        new_img_path = os.path.join(images_path, new_name + ext_img)
        new_label_path = os.path.join(labels_path, new_name + ".txt")

        # renombrar
        os.rename(img_path, new_img_path)
        if os.path.exists(label_path):
            os.rename(label_path, new_label_path)

        print(f"{basename} -> {new_name}")
        index += 1

    return index

# === PROCESO ===
index = 1
for split in splits:
    print(f"\nRenombrando {split}...")
    index = renombrar_split(split, index)
