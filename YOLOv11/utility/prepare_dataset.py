"""
prepare_dataset.py
---------------------------------
Copia solo las carpetas relevantes del dataset (train/, valid/, test/)
y el archivo data.yaml hacia la carpeta local 'data/' del proyecto YOLOv11.
También copia el script 'view_dataset.py' si existe en el dataset original.
"""

import os
import shutil
import yaml

def load_dataset_path(config_path="configs/dataset.yaml"):
    """Lee la ruta del dataset desde el archivo YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ No se encontró {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    dataset_path = config.get("path", None)
    if dataset_path is None:
        raise ValueError("⚠️ El archivo dataset.yaml no contiene la clave 'path'")
    return dataset_path


def copy_selected_items(src_dir, dest_dir="data"):
    """Copia solo train/, valid/, test/ y data.yaml."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"🚀 Copiando dataset filtrado desde:\n   {src_dir}\na:\n   {os.path.abspath(dest_dir)}")

    folders_to_copy = ["train", "valid", "test"]
    files_to_copy = ["data.yaml", "view_dataset.py"]

    for item in folders_to_copy + files_to_copy:
        src_item = os.path.join(src_dir, item)
        dest_item = os.path.join(dest_dir, item)

        if os.path.exists(src_item):
            try:
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
                    print(f"📁 Carpeta copiada: {item}")
                else:
                    shutil.copy2(src_item, dest_item)
                    print(f"📄 Archivo copiado: {item}")
            except Exception as e:
                print(f"⚠️ No se pudo copiar {item}: {e}")
        else:
            print(f"⚠️ {item} no encontrado en el dataset original.")

    print("✅ Copia finalizada correctamente.")


def main():
    dataset_path = load_dataset_path("configs/dataset.yaml")
    copy_selected_items(dataset_path, "data")


if __name__ == "__main__":
    main()
