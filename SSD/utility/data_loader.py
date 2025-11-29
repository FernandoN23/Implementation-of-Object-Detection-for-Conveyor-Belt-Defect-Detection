# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/data_loader.py
# Descripción: Utilidades de carga de datos para SSD.
#              Adaptación de dataset en formato YOLO (txt normalizado)
#              a formato interno SSD (cajas xyxy normalizadas) con
#              pipeline SSDAugmentation y DataLoader para train/val.
# ==============================================================

from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import torch.utils.data as data
import yaml
import numpy as np

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parents[1]  # .../SSD
PROJECT_ROOT = SSD_ROOT.parent  # raíz del proyecto
CONFIGS_ROOT = SSD_ROOT / "configs"  # SSD/configs
DEFAULT_DATASET_CONFIG = CONFIGS_ROOT / "dataset.yaml"

# ---------------------------------------------------------------------------
# Carga dinámica de SSDAugmentation desde SSD/ssd/utils/Augmentations.py
# ---------------------------------------------------------------------------

AUGMENTATIONS_PATH = SSD_ROOT / "ssd" / "utils" / "augmentations.py"
# Nota: el archivo original suele ser augmentations.py (minúscula) o Augmentations.py
# Probamos ambos por robustez
if not AUGMENTATIONS_PATH.is_file():
    AUGMENTATIONS_PATH = SSD_ROOT / "ssd" / "utils" / "Augmentations.py"

if not AUGMENTATIONS_PATH.is_file():
    raise ImportError(
        f"No se encontró augmentations.py en la ruta esperada: {SSD_ROOT / 'ssd' / 'utils'}"
    )


# FIX (2025-05): Multiprocessing Pickle Fix (Robust Version)
# En Windows (spawn), los procesos hijos no heredan sys.modules del padre.
# Pickle intenta importar clases por nombre. Si registramos un módulo 'fake'
# en el padre, pickle le dice al hijo "importa ssd.utils.augmentations".
# El hijo intenta y falla porque no sabe dónde está ese archivo.

# Solución: El Proxy es una clase real en ESTE archivo (data_loader.py).
# Pickle serializa el Proxy sin problemas.
# Cuando el hijo ejecuta proxy(img), llama a _get_aug_instance_lazy.
# Esta función carga el módulo manualmente en el hijo.

def _load_aug_module_isolated():
    """Carga el módulo de augmentations desde la ruta física."""
    # Nombre único para evitar colisiones, pero consistente
    module_name = "ssd_augmentations_dynamic"

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, AUGMENTATIONS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para {AUGMENTATIONS_PATH}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # Registrar para que imports internos funcionen

    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[module_name]
        raise

    return mod


class SSDAugmentationProxy:
    """Proxy serializable para SSDAugmentation.

    Permite que DataLoader use multiprocessing en Windows sin error de pickle.
    En lugar de guardar la clase dinámica, guardamos los parámetros de init
    y reconstruimos el objeto real al llamarse (__call__) en el worker.
    """

    def __init__(self, size=300, mean=(104, 117, 123)):
        self.size = size
        self.mean = mean
        # NO guardamos self._aug en __init__ para asegurar que sea None al pickling
        self._aug = None

    def __call__(self, img, boxes, labels):
        # Lazy initialization en el proceso trabajador (o main si num_workers=0)
        if self._aug is None:
            mod = _load_aug_module_isolated()
            # Asumimos que la clase se llama SSDAugmentation
            AugClass = getattr(mod, "SSDAugmentation", None)
            if AugClass is None:
                raise ImportError(f"La clase 'SSDAugmentation' no se encontró en {AUGMENTATIONS_PATH}")

            self._aug = AugClass(self.size, self.mean)

        return self._aug(img, boxes, labels)

    def __getstate__(self):
        """Control explícito de lo que se serializa."""
        return {'size': self.size, 'mean': self.mean, '_aug': None}

    def __setstate__(self, state):
        self.size = state['size']
        self.mean = state['mean']
        self._aug = None  # Forzar recarga en el nuevo proceso


# ---------------------------------------------------------------------------
# Helpers de configuración
# ---------------------------------------------------------------------------

def load_dataset_config(path: Path = DEFAULT_DATASET_CONFIG) -> Dict[str, Any]:
    """Carga el YAML de configuración del dataset para SSD."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset config en: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Dataset YOLO → SSD
# ---------------------------------------------------------------------------

@dataclass
class YoloSample:
    """Estructura interna para mapear imágenes y labels en disco."""
    image_path: Path
    label_path: Path


class YoloDetectionDataset(data.Dataset):
    """Dataset que lee etiquetas en formato YOLO txt y las adapta a SSD.

    Formato en disco (layout tipo YOLO):
        - Raíz: <dataset_root>
        - train:
            - images: train/images/*.jpg
            - labels: train/labels/*.txt
        - val:
            - images: valid/images/*.jpg
            - labels: valid/labels/*.txt

    Formato etiqueta (.txt):
        <class_id> <cx> <cy> <w> <h>   (normalizado en [0,1])

    Salida:
        - image: Tensor [3, H, W] (float32, BGR con medias restadas)
        - target: Tensor [N, 5] con:
            [x_min, y_min, x_max, y_max, class_id] (normalizado)
    """

    def __init__(
            self,
            root: Path,
            images_rel: str,
            labels_rel: str,
            img_dim: int = 300,
            transform: Optional[Any] = None,
            skip_empty: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / images_rel
        self.labels_dir = self.root / labels_rel
        self.img_dim = int(img_dim)
        self.transform = transform
        self.skip_empty = bool(skip_empty)

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Directorio de imágenes no encontrado: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Directorio de etiquetas no encontrado: {self.labels_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

        self.samples: List[YoloSample] = []
        for img_path in image_paths:
            lbl_path = self.labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                if self.skip_empty:
                    continue
                self.samples.append(YoloSample(img_path, lbl_path))
                continue

            if self.skip_empty:
                with lbl_path.open("r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if not lines:
                    continue

            self.samples.append(YoloSample(img_path, lbl_path))

        if not self.samples:
            raise RuntimeError(
                f"No se encontraron muestras válidas en {self.images_dir} "
                f"(skip_empty={self.skip_empty})."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        img_path, lbl_path = sample.image_path, sample.label_path

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"No se pudo leer la imagen: {img_path}")

        # Leer etiquetas YOLO
        if lbl_path.is_file():
            with lbl_path.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        else:
            lines = []

        if not lines:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            parsed = []
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                parsed.append((cls_id, cx, cy, w, h))

            if not parsed:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                parsed_arr = np.array(parsed, dtype=np.float32)
                labels = parsed_arr[:, 0].astype(np.int64)
                cx = parsed_arr[:, 1]
                cy = parsed_arr[:, 2]
                w = parsed_arr[:, 3]
                h = parsed_arr[:, 4]

                # (cx, cy, w, h) → (x_min, y_min, x_max, y_max) normalizado
                x_min = cx - w / 2.0
                y_min = cy - h / 2.0
                x_max = cx + w / 2.0
                y_max = cy + h / 2.0

                boxes = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, 1.0)
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, 1.0)

        # Augmentations SSD (usando el Proxy)
        if self.transform is not None:
            # La llamada a self.transform(...) invocará al __call__ del proxy
            # que a su vez cargará dinámicamente SSDAugmentation si es necesario
            img, boxes, labels = self.transform(img, boxes, labels)

        if not isinstance(img, np.ndarray):
            img = np.asarray(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

        if boxes.size == 0:
            target_np = np.zeros((0, 5), dtype=np.float32)
        else:
            labels = labels.astype(np.float32)
            target_np = np.concatenate([boxes, labels[:, None]], axis=1).astype(np.float32)

        target_tensor = torch.from_numpy(target_np)
        return img_tensor, target_tensor

    def find_index_by_stem(self, stem: str) -> Optional[int]:
        """Devuelve el índice de la primera imagen cuyo nombre base coincide con `stem`."""
        for i, s in enumerate(self.samples):
            if s.image_path.stem == stem:
                return i
        return None


# ---------------------------------------------------------------------------
# Collate para detección
# ---------------------------------------------------------------------------

def detection_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Apila imágenes y mantiene lista de targets."""
    images: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return torch.stack(images, dim=0), targets


# ---------------------------------------------------------------------------
# DataLoaders de alto nivel
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: Any):
    """Construye DataLoaders de entrenamiento y validación para SSD.

    Se espera que `cfg` tenga:
        - cfg.data_config
        - cfg.img_dim
        - cfg.batch_size
        - cfg.num_workers
    """
    ds_cfg = load_dataset_config(Path(getattr(cfg, "data_config", DEFAULT_DATASET_CONFIG)))

    dataset_root = Path(ds_cfg["path"])
    train_images_rel = ds_cfg["train"]["images"]
    train_labels_rel = ds_cfg["train"]["labels"]
    val_images_rel = ds_cfg["val"]["images"]
    val_labels_rel = ds_cfg["val"]["labels"]

    img_dim = int(getattr(cfg, "img_dim", ds_cfg.get("img_dim_default", 300)))
    mean = ds_cfg.get("mean", [104, 117, 123])

    # FIX: Usar SSDAugmentationProxy en lugar de la clase dinámica directa
    train_transform = SSDAugmentationProxy(size=img_dim, mean=tuple(mean))
    val_transform = SSDAugmentationProxy(size=img_dim, mean=tuple(mean))

    train_dataset = YoloDetectionDataset(
        root=dataset_root,
        images_rel=train_images_rel,
        labels_rel=train_labels_rel,
        img_dim=img_dim,
        transform=train_transform,
        skip_empty=True,
    )

    val_dataset = YoloDetectionDataset(
        root=dataset_root,
        images_rel=val_images_rel,
        labels_rel=val_labels_rel,
        img_dim=img_dim,
        transform=val_transform,
        skip_empty=False,
    )

    batch_size = int(getattr(cfg, "batch_size", 32))
    num_workers = int(getattr(cfg, "num_workers", max(os.cpu_count() - 1, 1) if os.cpu_count() else 2))

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Test rápido con una imagen de ejemplo (debug manual)
# ---------------------------------------------------------------------------

def _debug_single_sample(stem: str = "0044") -> None:
    """Test rápido usando una imagen individual del dataset."""
    ds_cfg = load_dataset_config(DEFAULT_DATASET_CONFIG)
    dataset_root = Path(ds_cfg["path"])
    train_images_rel = ds_cfg["train"]["images"]
    train_labels_rel = ds_cfg["train"]["labels"]

    img_dim = int(ds_cfg.get("img_dim_default", 300))
    mean = ds_cfg.get("mean", [104, 117, 123])

    # Usar proxy también aquí
    transform = SSDAugmentationProxy(size=img_dim, mean=tuple(mean))

    dataset = YoloDetectionDataset(
        root=dataset_root,
        images_rel=train_images_rel,
        labels_rel=train_labels_rel,
        img_dim=img_dim,
        transform=transform,
        skip_empty=False,
    )

    idx = dataset.find_index_by_stem(stem)
    if idx is None:
        raise RuntimeError(
            f"No se encontró ninguna imagen con nombre base '{stem}' en {dataset.images_dir}"
        )

    img_tensor, target_tensor = dataset[idx]
    sample = dataset.samples[idx]

    # Salida mínima pedida
    print(str(sample.image_path))
    print(target_tensor)


if __name__ == "__main__":
    _debug_single_sample(stem="0044")