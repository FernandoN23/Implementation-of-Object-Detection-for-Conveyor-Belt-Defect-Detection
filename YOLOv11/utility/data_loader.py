"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: data_loader.py
Cargador de datos (Dataset y DataLoader) para YOLOv11.
Compatible con datasets tipo YOLOv8/YOLOv11.
-------------------------------------------------------------
"""

import os
import warnings
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Redimensiona manteniendo relación de aspecto (como YOLOv8)."""
    shape = im.size[::-1]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * r), int(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im = im.resize(new_unpad, Image.Resampling.BILINEAR)
    new_im = Image.new("RGB", new_shape, color)
    new_im.paste(im, (int(dw), int(dh)))
    return new_im


# =============================================================
# DATASET PERSONALIZADO
# =============================================================
class CustomDataset(Dataset):
    """
    Dataset compatible con detección YOLO (formato txt: class x_center y_center w h).
    """
    def __init__(self, root_dir, img_size=640, cache_images=False, transform=None):
        self.img_dir = Path(root_dir)
        self.label_dir = self.img_dir.parent / "labels"
        self.img_size = img_size
        self.cache_images = cache_images

        if not self.img_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio de imágenes: {self.img_dir}")

        # Cargar nombres de clases desde data.yaml si existe
        data_yaml = self.img_dir.parents[1] / "data.yaml"
        self.class_names = []
        if data_yaml.exists():
            with open(data_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.class_names = data.get("names", []) or data.get("classes", [])

        self.image_paths = sorted([
            p for p in self.img_dir.glob("*.*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

        self.cached = {}
        if self.cache_images:
            print(f"[INFO] Precaching {len(self.image_paths)} imágenes en RAM...")
            for path in self.image_paths:
                img = Image.open(path).convert("RGB")
                img = self.transform(letterbox(img, img_size))
                self.cached[str(path)] = img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        if self.cache_images and str(path) in self.cached:
            img = self.cached[str(path)]
        else:
            img = Image.open(path).convert("RGB")
            img = self.transform(letterbox(img, self.img_size))

        label_path = self.label_dir / (path.stem + ".txt")
        boxes, classes = [], []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, x_c, y_c, w, h = map(float, parts[:5])
                        if not all(0.0 <= v <= 1.0 for v in (x_c, y_c, w, h)):
                            warnings.warn(f"Etiqueta fuera de rango [0,1] descartada: {parts}")
                            continue
                        classes.append(int(cls_id))
                        boxes.append([x_c, y_c, w, h])
        else:
            warnings.warn(f"[WARN] No se encontró etiqueta para {path.name}")

        targets = torch.zeros((len(boxes), 5), dtype=torch.float32)
        if boxes:
            targets[:, 0] = torch.tensor(classes)
            targets[:, 1:] = torch.tensor(boxes)

        return img, targets


# =============================================================
# COLLATE FUNCTION
# =============================================================
def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, targets


# =============================================================
# FACTORY
# =============================================================
def create_dataloader(cfg, phase="train"):
    """Crea el DataLoader según la fase indicada."""
    phase = phase.lower()
    if phase not in ["train", "valid", "test"]:
        raise ValueError("phase debe ser 'train', 'valid' o 'test'")

    path = _resolve_dataset_path(cfg, phase)
    dataset = CustomDataset(path, img_size=getattr(cfg, "img_size", 640),
                            cache_images=getattr(cfg, "cache_images", False))

    loader = DataLoader(
        dataset,
        batch_size=getattr(cfg, "batch_size", 8),
        shuffle=(phase == "train"),
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"[INFO] DataLoader ({phase}) → {len(dataset)} imágenes | Batch: {getattr(cfg, 'batch_size', 8)}")
    return loader


# =============================================================
# RUTA DEL DATASET
# =============================================================
def _resolve_dataset_path(cfg, phase):
    dataset_yaml = getattr(cfg, "data", None)
    base_path = getattr(cfg, "dataset_path", None)

    if dataset_yaml and os.path.isfile(dataset_yaml):
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        base = data.get("path", os.path.dirname(dataset_yaml))
        key_map = {"train": ["train"], "valid": ["val", "valid"], "test": ["test"]}
        for key in key_map[phase]:
            if data.get(key):
                return os.path.join(base, data[key], "images")

    if base_path and os.path.isdir(base_path):
        for variant in ["images", "train/images", "valid/images", "test/images"]:
            candidate = os.path.join(base_path, variant)
            if os.path.isdir(candidate):
                return candidate

    raise FileNotFoundError("No se pudo resolver la ruta del dataset.")


# =============================================================
# TEST LOCAL
# =============================================================
if __name__ == "__main__":
    class DummyCfg:
        data = "C:/Users/memorista/Desktop/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection/Dataset/data.yaml"
        img_size = 640
        batch_size = 8

    loader = create_dataloader(DummyCfg(), phase="train")
    imgs, targets = next(iter(loader))
    print(f"Batch imgs: {imgs.shape}")
    print(f"Targets lens: {[t.shape for t in targets]}")
