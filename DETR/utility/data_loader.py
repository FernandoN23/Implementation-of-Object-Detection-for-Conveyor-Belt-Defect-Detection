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

# [CORRECCIÓN]: Usamos append para evitar colisiones con el paquete engine/ local
if str(DETR_SUBMODULE) not in sys.path:
    sys.path.append(str(DETR_SUBMODULE))

try:
    from datasets.coco import make_coco_transforms
    from util.misc import nested_tensor_from_tensor_list
except ImportError as e:
    print(f"[data_loader] ERROR: No se pudo importar desde el submódulo DETR: {e}")
    sys.exit(1)


class YoloToDetrDataset(Dataset):
    def __init__(self, dataset_path, image_set="train", transforms=None):
        self.dataset_path = Path(dataset_path)
        self.image_set = image_set
        self.transforms = transforms

        split_dir = self.dataset_path / image_set
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {self.images_dir}")

        self.img_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.split())
                    xmin = (cx - bw / 2) * w
                    ymin = (cy - bh / 2) * h
                    xmax = (cx + bw / 2) * w
                    ymax = (cy + bh / 2) * h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(cls))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([0.0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)])
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def detr_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def build_dataloader(image_set, batch_size, num_workers=4):
    transform_set = "val" if image_set == "valid" else image_set
    transforms = make_coco_transforms(transform_set)

    dataset = YoloToDetrDataset(
        dataset_path=DATASET_ROOT,
        image_set=image_set,
        transforms=transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set == "train"),
        num_workers=num_workers,
        collate_fn=detr_collate_fn,
        pin_memory=True
    )

    return loader