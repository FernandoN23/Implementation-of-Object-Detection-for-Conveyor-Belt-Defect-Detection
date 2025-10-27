"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: check_dataset.py
Verifica la estructura del dataset local (train/, valid/, test/)
y el archivo data.yaml. Compatible con YOLOv8/YOLOv11.
-------------------------------------------------------------
"""

from pathlib import Path
import yaml

# Extensiones válidas
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def check_dataset_structure(dataset_path: str) -> bool:
    """Verifica la integridad del dataset y su data.yaml asociado."""
    dataset_path = Path(dataset_path)
    print(f"\n📂 Verificando dataset en: {dataset_path}\n")

    expected_splits = ['train', 'valid', 'test']
    all_ok = True

    # =============================================================
    # 1. Verificación de data.yaml
    # =============================================================
    yaml_path = dataset_path / 'data.yaml'
    if not yaml_path.exists():
        print(f"❌ No se encontró el archivo data.yaml en {dataset_path}")
        return False

    print(f"📘 Verificando archivo: {yaml_path}")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ Error al leer data.yaml: {e}")
        return False

    nc = data.get('nc', None)
    names = data.get('names', [])
    if nc != len(names):
        print(f"⚠️ Inconsistencia: nc={nc} pero se definieron {len(names)} nombres de clase.")
        all_ok = False

    # =============================================================
    # 2. Verificación de carpetas principales
    # =============================================================
    for split in expected_splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"❌ Falta carpeta: {split_path}")
            all_ok = False
            continue

        images_path = split_path / 'images'
        labels_path = split_path / 'labels'

        if not images_path.exists():
            print(f"❌ Falta carpeta de imágenes: {images_path}")
            all_ok = False
        if not labels_path.exists():
            print(f"❌ Falta carpeta de etiquetas: {labels_path}")
            all_ok = False

        # ---------------------------------------------------------
        # 3. Correspondencia imagen ↔ etiqueta
        # ---------------------------------------------------------
        if images_path.exists() and labels_path.exists():
            img_files = [f for f in images_path.iterdir() if f.suffix.lower() in IMG_EXT]
            lbl_files = [f for f in labels_path.iterdir() if f.suffix == '.txt']

            img_basenames = {f.stem for f in img_files}
            lbl_basenames = {f.stem for f in lbl_files}

            missing_labels = img_basenames - lbl_basenames
            missing_images = lbl_basenames - img_basenames

            if missing_labels:
                print(f"⚠️ {len(missing_labels)} imágenes sin etiqueta en {split}/: {list(missing_labels)[:5]}")
                all_ok = False
            if missing_images:
                print(f"⚠️ {len(missing_images)} etiquetas sin imagen en {split}/: {list(missing_images)[:5]}")
                all_ok = False

            # -----------------------------------------------------
            # 4. Verificación del formato de etiquetas
            # -----------------------------------------------------
            for lbl_path in lbl_files:
                try:
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    print(f"❌ No se pudo leer {lbl_path.name}: {e}")
                    all_ok = False
                    continue

                for i, line in enumerate(lines, start=1):
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"❌ Formato inválido en {lbl_path.name}, línea {i}: {line}")
                        all_ok = False
                        continue
                    try:
                        int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if not all(0 <= c <= 1 for c in coords):
                            print(f"⚠️ Coordenadas fuera de rango [0,1] en {lbl_path.name}, línea {i}: {coords}")
                            all_ok = False
                    except ValueError:
                        print(f"❌ Valores no numéricos en {lbl_path.name}, línea {i}: {line}")
                        all_ok = False

    # =============================================================
    # 5. Verificación de rutas en data.yaml
    # =============================================================
    for split in expected_splits:
        if split not in data:
            print(f"⚠️ Falta la clave '{split}' en data.yaml")
            all_ok = False
            continue
        path_rel = Path(data[split])
        abs_path = (dataset_path / path_rel).resolve()
        if not abs_path.exists():
            print(f"⚠️ Ruta de {split} inválida o inexistente: {path_rel}")
            all_ok = False

    # =============================================================
    # Resultado Final
    # =============================================================
    print("\n" + ("✅ Dataset verificado correctamente." if all_ok else "❌ Errores detectados en la estructura del dataset."))
    return all_ok


# =============================================================
# Ejecución directa
# =============================================================
if __name__ == "__main__":
    dataset_path = Path(
        r"C:\Users\memorista\Desktop\Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection\Dataset"
    )
    check_dataset_structure(dataset_path)
