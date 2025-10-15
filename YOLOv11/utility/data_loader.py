"""
utility/data_loader.py

Este módulo define un DataLoader optimizado para AMD ROCm y sistemas con alta RAM.
Compatible con configuraciones definidas en configs/train.yaml.
"""

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch


class CustomDataset(Dataset):
    """
    Dataset genérico para imágenes.
    Espera una estructura estándar: data/train/images/, data/valid/images/, etc.
    """

    def __init__(self, root_dir, img_size=640, cache_images=False, transform=None):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
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

        # Si está cacheado, lo toma desde memoria
        if self.cache_images and path in self.cached:
            img = self.cached[path]
        else:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)

        # Si existieran etiquetas, aquí se cargarían
        # e.g., label = self.get_label(path)
        label = torch.tensor(0)  # placeholder

        return img, label


def create_dataloader(cfg):
    """
    Crea un DataLoader a partir del bloque dataloader en train.yaml
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
        persistent_workers=params.persistent_workers
    )

    print(f"[INFO] DataLoader initialized -> {len(dataset)} images | Batch size: {params.batch_size}")
    return loader
