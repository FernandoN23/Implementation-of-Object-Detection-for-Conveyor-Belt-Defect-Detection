"""
utility/data_loader.py

Trabajo de Memoria de Título:
IMPLEMENTACIÓN DE ALGORITMOS DE RECONOCIMIENTO DE OBJETOS
PARA LA IDENTIFICACIÓN DE FALLAS EN CORREAS TRANSPORTADORAS

Memorista: Fernando Navarrete
Modelo actual: YOLOv11
Código actual: data_loader.py
------------------------------------------------------------
Cargador de datos para detección de objetos multiclase (formato YOLO).
Compatible con Dataset/data.yaml y estructura de carpetas train/valid/test.
"""

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
import yaml


class CustomDataset(Dataset):
    """
    Dataset compatible con detección YOLO.
    Lee imágenes y etiquetas en formato:
        class x_center y_center width height
    Los valores están normalizados entre [0, 1].
    """

    def __init__(self, root_dir, img_size=640, cache_images=False, transform=None):
        self.root_dir = root_dir
        self.img_dir = root_dir
        self.label_dir = root_dir.replace("images", "labels")

        # Leer data.yaml si existe en la raíz del dataset
        data_yaml = os.path.join(os.path.dirname(os.path.dirname(root_dir)), "data.yaml")
        self.class_names = []
        if os.path.exists(data_yaml):
            with open(data_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.class_names = data.get("names", []) or data.get("classes", [])

        # Cargar rutas de imágenes
        self.image_paths = [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths.sort()

        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        self.cache_images = cache_images
        self.cached = {}

        if self.cache_images:
            print(f"[INFO] Precaching {len(self.image_paths)} images into RAM...")
            for path in self.image_paths:
                img = Image.open(path).convert("RGB")
                self.cached[path] = self.transform(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # Cargar imagen
        if self.cache_images and path in self.cached:
            img = self.cached[path]
        else:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)

        # Buscar etiqueta correspondiente
        label_path = path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        boxes, classes = [], []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id, x_c, y_c, w, h = map(float, parts[:5])
                        classes.append(int(cls_id))
                        boxes.append([x_c, y_c, w, h])
                    else:
                        print(f"[WARN] Etiqueta inválida en {label_path}: {line}")
        else:
            print(f"[WARN] No label found for: {path}")

        n = len(boxes)
        targets = torch.zeros((n, 5))
        if n > 0:
            targets[:, 0] = torch.tensor(classes, dtype=torch.float32)
            targets[:, 1:] = torch.tensor(boxes, dtype=torch.float32)

        return img, targets


def collate_fn(batch):
    """
    collate_fn personalizado para detección multiclase.
    Mantiene listas variables de targets por imagen.
    """
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs, 0)
    return imgs, targets


def create_dataloader(cfg):
    """
    Crea un DataLoader a partir del bloque dataloader en train.yaml.
    """
    params = cfg.dataloader

    dataset = CustomDataset(
        root_dir=params.path,
        img_size=params.img_size,
        cache_images=params.cache_images
    )

    loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        persistent_workers=params.persistent_workers,
        collate_fn=collate_fn
    )

    print(f"[INFO] DataLoader initialized -> {len(dataset)} images | Batch size: {params.batch_size}")
    return loader


if __name__ == "__main__":
    # Prueba rápida del DataLoader
    class DummyCfg:
        class dataloader:
            path = "C:/Users/memorista/Desktop/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection/Dataset/train/images"
            img_size = 640
            batch_size = 8
            shuffle = True
            num_workers = 2
            pin_memory = True
            persistent_workers = False
            cache_images = False

    loader = create_dataloader(DummyCfg())
    imgs, targets = next(iter(loader))
    print(f"Batch imgs: {imgs.shape}")
    print(f"Targets sample lens: {[t.shape for t in targets]}")
