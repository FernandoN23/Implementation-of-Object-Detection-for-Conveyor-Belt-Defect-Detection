# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/data_loader.py
# Descripción: Adaptador de Dataset YOLOv11 a formato DETR.
#              Gestiona la carga de imágenes, conversión de
#              coordenadas y empaquetado en NestedTensors.
# ==============================================================

import os
import sys
import yaml
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURACIÓN DE RUTAS ---
FILE = Path(__file__).resolve()
UTILITY_ROOT = FILE.parent
DETR_ROOT = UTILITY_ROOT.parent
PROJECT_ROOT = DETR_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DETR_SUBMODULE = DETR_ROOT / "detr"

# Asegurar que el submódulo sea visible para imports
if str(DETR_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(DETR_SUBMODULE))

try:
    from datasets.coco import make_coco_transforms
    from util.misc import nested_tensor_from_tensor_list
except ImportError as e:
    print(f"[data_loader] ERROR: No se pudo importar desde el submódulo DETR: {e}")
    sys.exit(1)


class YoloToDetrDataset(Dataset):
    """
    Dataset personalizado que lee formato YOLO (.txt normalizado)
    y lo adapta al formato de diccionarios que espera DETR.
    """

    def __init__(self, dataset_path, image_set="train", transforms=None):
        self.dataset_path = Path(dataset_path)
        self.image_set = image_set
        self.transforms = transforms

        # Determinar carpetas según el split (train/valid/test)
        # Ajustado a la estructura: Dataset/train/images y Dataset/train/labels
        split_dir = self.dataset_path / image_set
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {self.images_dir}")

        # Listar archivos de imagen válidos
        self.img_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # 1. Cargar imagen y convertir a RGB (Requerido por DETR/PIL)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 2. Leer etiquetas YOLO
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.split())

                    # YOLO (cx, cy, w, h) normalizado -> DETR (xmin, ymin, xmax, ymax) ABSOLUTO
                    # Se convierte a absoluto para que las transformaciones espaciales
                    # de DETR (crops, flips) se apliquen correctamente sobre píxeles.
                    xmin = (cx - bw / 2) * w
                    ymin = (cy - bh / 2) * h
                    xmax = (cx + bw / 2) * w
                    ymax = (cy + bh / 2) * h

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(cls))

        # 3. Convertir a tensores
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 4. Crear el diccionario 'target' que espera DETR
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor(
                [0.0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)])
        }

        # 5. Aplicar Transformaciones (Data Augmentation oficial de DETR)
        # Ojo: make_coco_transforms al final usa 'Normalize' que vuelve a
        # dejar las cajas en formato (cx, cy, w, h) normalizado para la red.
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def detr_collate_fn(batch):
    """
    Función de empaquetado vital para DETR.
    Convierte una lista de (imagen, target) en:
    - NestedTensor: Imágenes con sus máscaras de padding para el Transformer.
    - List[Dict]: Lista de targets.
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def build_dataloader(image_set, batch_size, num_workers=4):
    """
    Orquestador de alto nivel para obtener el DataLoader listo para entrenar.
    """
    # 1. Cargar transformaciones oficiales (train o val)
    # DETR requiere aumentos específicos para converger (visto en transforms.py)
    transforms = make_coco_transforms(image_set)

    # 2. Instanciar Dataset
    # Usamos DATASET_ROOT (Proyecto/Dataset)
    dataset = YoloToDetrDataset(
        dataset_path=DATASET_ROOT,
        image_set=image_set,
        transforms=transforms
    )

    # 3. Crear DataLoader
    # Importante: usar detr_collate_fn para las máscaras del Transformer
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set == "train"),
        num_workers=num_workers,
        collate_fn=detr_collate_fn,
        pin_memory=True
    )

    return loader


# --- PRUEBA UNITARIA RÁPIDA ---
if __name__ == "__main__":
    print(f"[data_loader] Iniciando prueba con split 'valid'...")
    try:
        test_loader = build_dataloader(image_set="valid", batch_size=2, num_workers=0)
        images, targets = next(iter(test_loader))

        print(f"✓ Éxito: Batch de imágenes cargado. Tipo: {type(images)}")
        print(f"✓ Dimensiones tensores (con padding): {images.tensors.shape}")
        print(f"✓ Cantidad de etiquetas en imagen 1: {len(targets[0]['labels'])}")
        print(f"✓ Ejemplo cajas (formato normalizado por transforms): \n{targets[0]['boxes'][0]}")
    except Exception as e:
        print(f"✗ Error durante la prueba: {e}")