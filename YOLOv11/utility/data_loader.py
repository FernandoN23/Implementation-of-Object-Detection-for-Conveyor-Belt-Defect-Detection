"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: data_loader.py
Cargador de datos (Dataset y DataLoader) para YOLOv11.
Compatible con estructuras de dataset tipo YOLOv8
(train/valid/test + data.yaml).
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Estructura principal:
#   • CustomDataset: lee imágenes y etiquetas YOLO (x_c, y_c, w, h)
#   • collate_fn: apila lotes con targets de tamaño variable
#   • create_dataloader(): crea DataLoader configurable
#
# Compatibilidad:
#   - Permite cacheo en RAM (opcional)
#   - Admite lectura directa de "data.yaml" para nombres de clases
#
# Conexión:
#   Usado por train.py y valid.py para construir los loaders
#   según las rutas definidas en train.yaml.
# -------------------------------------------------------------

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
import yaml


# =============================================================
# DATASET PERSONALIZADO
# =============================================================
class CustomDataset(Dataset):
    """
    Dataset compatible con detección YOLO.
    Lee imágenes y etiquetas en formato:
        class x_center y_center width height
    """
    def __init__(self, root_dir, img_size=640, cache_images=False, transform=None):
        self.root_dir = root_dir
        self.img_dir = root_dir
        self.label_dir = root_dir.replace("images", "labels")

        # Leer data.yaml si existe
        data_yaml = os.path.join(os.path.dirname(os.path.dirname(root_dir)), "data.yaml")
        self.class_names = []
        if os.path.exists(data_yaml):
            with open(data_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.class_names = data.get("names", []) or data.get("classes", [])

        # Cargar imágenes
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

        if self.cache_images and path in self.cached:
            img = self.cached[path]
        else:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)

        label_path = path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        boxes, classes = [], []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, x_c, y_c, w, h = map(float, parts[:5])
                        classes.append(int(cls_id))
                        boxes.append([x_c, y_c, w, h])
        else:
            print(f"[WARN] No label found for: {path}")

        n = len(boxes)
        targets = torch.zeros((n, 5))
        if n > 0:
            targets[:, 0] = torch.tensor(classes, dtype=torch.float32)
            targets[:, 1:] = torch.tensor(boxes, dtype=torch.float32)

        return img, targets


# =============================================================
# COLLATE FUNCTION
# =============================================================
def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs, 0)
    return imgs, targets


# =============================================================
# DATALOADER FACTORY
# =============================================================
def create_dataloader(cfg, phase="train"):
    """
    Crea un DataLoader según la fase: 'train', 'valid' o 'test'.
    Si no se indica explícitamente, usa 'train' por defecto.
    """

    base_path = "C:/Users/memorista/Desktop/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection/Dataset"

    # Selecciona el subdirectorio correspondiente
    phase = phase.lower()
    if phase not in ["train", "valid", "test"]:
        raise ValueError("phase debe ser 'train', 'valid' o 'test'")

    path = getattr(cfg, "dataset_path", None)
    if path is None:
        path = os.path.join(base_path, phase, "images")

    img_size = getattr(cfg, "img_size", 640)
    batch_size = getattr(cfg, "batch_size", 8)
    shuffle = (phase == "train")  # solo barajar en entrenamiento
    num_workers = 2
    pin_memory = True
    cache_images = False

    dataset = CustomDataset(root_dir=path, img_size=img_size, cache_images=cache_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory,
                        collate_fn=collate_fn)

    print(f"[INFO] DataLoader ({phase}) → {len(dataset)} images | Batch size: {batch_size}")
    return loader



# =============================================================
# TEST LOCAL
# =============================================================
if __name__ == "__main__":
    class DummyCfg:
        dataset_path = "C:/Users/memorista/Desktop/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection/Dataset/train/images"
        img_size = 640
        batch_size = 8

    loader = create_dataloader(DummyCfg())
    imgs, targets = next(iter(loader))
    print(f"Batch imgs: {imgs.shape}")
    print(f"Targets lens: {[t.shape for t in targets]}")
