# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: data_loader.py
# Cargador de datos (Dataset + DataLoader) para YOLOv11.
# Lee rutas desde configs/dataset.yaml y parámetros desde configs/train.yaml.
# Soporta imágenes sin clases (negativas) y labels YOLO (cls x y w h, normalizados).
#==============================================================

import os
import math
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------- Constantes ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".ppm"}
LBL_EXT = ".txt"


# ---------------- Utilidades de ruta ----------------
def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_yaml(project_root: Path) -> Dict:
    data = load_yaml(project_root / "configs" / "dataset.yaml")
    # Normaliza claves
    if "val" in data and "valid" not in data:
        data["valid"] = data["val"]
    required = ["train", "valid", "test", "nc", "names"]
    missing = [k for k in required if k not in data]
    if missing:
        raise FileNotFoundError(f"dataset.yaml carece de claves requeridas: {missing}")
    return data


def load_train_yaml(project_root: Path) -> Dict:
    cfg = load_yaml(project_root / "configs" / "train.yaml")
    # defaults
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("batch", 16)
    # aug defaults
    cfg.setdefault("fliplr", 0.0)
    hsv = {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0}
    for k, v in hsv.items():
        cfg.setdefault(k, v)
    # dataloader defaults
    dl = cfg.setdefault("dataloader", {})
    dl.setdefault("workers", 4)
    dl.setdefault("pin_memory", True)
    dl.setdefault("persistent_workers", True)
    dl.setdefault("shuffle", True)
    return cfg


# ---------------- Conversión y preprocesamiento ----------------
def letterbox(im: np.ndarray, new_shape=640, stride=32, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Redimensiona manteniendo aspecto y aplica padding para que H y W sean múltiplos de 'stride'.
    Retorna: im_lb, r, (padw, padh) con im_lb.shape[:2] == (new_shape, new_shape)."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def augment_hsv(im: np.ndarray, hgain=0.0, sgain=0.0, vgain=0.0):
    if hgain == 0 and sgain == 0 and vgain == 0:
        return im
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1.0
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return im


def images_in(path: Path) -> List[Path]:
    if not path.exists():
        return []
    files = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def label_path_for(img_path: Path) -> Path:
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(LBL_EXT)
        return lbl
    except ValueError:
        if img_path.parent.name.lower() == "images":
            lbl_dir = img_path.parent.parent / "labels"
            return lbl_dir / (img_path.stem + LBL_EXT)
        return img_path.with_suffix(LBL_EXT)


def read_label_file(lbl_path: Path) -> np.ndarray:
    """Devuelve ndarray Nx5 (cls, x, y, w, h) con valores float32 en [0,1]. Si el archivo no existe o está vacío, retorna (0,5) vacío."""
    if not lbl_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    content = lbl_path.read_text(encoding="utf-8").strip()
    if content == "":
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            # línea inválida -> ignora
            continue
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        rows.append([c, x, y, w, h])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    # Clampea a [0,1] por seguridad
    arr[:, 1:] = np.clip(arr[:, 1:], 0.0, 1.0)
    return arr


def adjust_labels_letterbox(labels: np.ndarray,
                            orig_shape: Tuple[int, int],
                            resized_shape: Tuple[int, int],
                            r: float, padw: int, padh: int) -> np.ndarray:
    """Ajusta labels normalizados (cx,cy,w,h) del tamaño original a la imagen tras letterbox (escala r + padding)."""
    if labels.size == 0:
        return labels
    H0, W0 = orig_shape
    H1, W1 = resized_shape
    out = labels.copy()
    # Absolutos en original
    x_abs = out[:, 1] * W0
    y_abs = out[:, 2] * H0
    w_abs = out[:, 3] * W0
    h_abs = out[:, 4] * H0
    # Escala + padding -> normalizados en imagen letterboxed
    out[:, 1] = (x_abs * r + padw) / W1
    out[:, 2] = (y_abs * r + padh) / H1
    out[:, 3] = (w_abs * r) / W1
    out[:, 4] = (h_abs * r) / H1
    # Seguridad
    out[:, 1:] = np.clip(out[:, 1:], 0.0, 1.0)
    return out


# ---------------- Dataset YOLO ----------------
class YOLODataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        imgsz: int = 640,
        augment: bool = False,
        fliplr: float = 0.0,
        hsv_h: float = 0.0,
        hsv_s: float = 0.0,
        hsv_v: float = 0.0,
        stride: int = 32,
    ) -> None:
        self.images = images_in(images_dir)
        if len(self.images) == 0:
            raise FileNotFoundError(f"No se encontraron imágenes en: {images_dir}")
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.fliplr = float(fliplr)
        self.hsv_h, self.hsv_s, self.hsv_v = float(hsv_h), float(hsv_s), float(hsv_v)
        self.stride = int(stride)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, path: Path) -> np.ndarray:
        im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            # intento estándar
            im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        return im

    def __getitem__(self, i: int):
        im_path = self.images[i]
        lb_path = label_path_for(im_path)
        labels = read_label_file(lb_path)  # Nx5

        im = self._load_image(im_path)
        orig_shape = im.shape[:2]  # (H0, W0)

        # Augmentaciones ligeras (sólo train)
        if self.augment:
            im = augment_hsv(im, self.hsv_h, self.hsv_s, self.hsv_v)
            if self.fliplr > 0.0 and random.random() < self.fliplr:
                im = np.fliplr(im).copy()  # asegura strides positivos
                if labels.size:
                    labels[:, 1] = 1.0 - labels[:, 1]  # x -> 1-x (centro)

        # Letterbox + ajuste de labels
        im, r, (padw, padh) = letterbox(im, self.imgsz, stride=self.stride)
        labels = adjust_labels_letterbox(labels, orig_shape, im.shape[:2], r, padw, padh)

        # BGR->RGB sin strides negativos y a CHW [0,1]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.transpose(im, (2, 0, 1))  # CHW
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
        im_tensor = torch.from_numpy(im)  # float32

        sample = {
            "img": im_tensor,                    # (3,H,W)
            "targets": torch.from_numpy(labels), # (N,5) -> [cls, x, y, w, h] normalizado a letterbox
            "path": str(im_path),
            "label_path": str(lb_path),
            "orig_shape": orig_shape,            # (H0, W0)
            "resized_shape": im_tensor.shape[-2:],  # (H1, W1)
            "is_empty": labels.shape[0] == 0,
        }
        return sample


# ---------------- Collate para lotes con targets variables ----------------
def collate_yolo(batch: List[Dict]):
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    targets = []
    for i, b in enumerate(batch):
        t = b["targets"]
        if t.numel():
            # agrega índice de imagen como 1a columna para compatibilidad
            img_i = torch.full((t.shape[0], 1), i, dtype=t.dtype)
            targets.append(torch.cat([img_i, t], dim=1))  # (N,6) -> [i, cls, x, y, w, h]
    targets = torch.cat(targets, dim=0) if len(targets) else torch.zeros((0, 6), dtype=torch.float32)
    meta = {
        "paths": [b["path"] for b in batch],
        "label_paths": [b["label_path"] for b in batch],
        "is_empty": [bool(b["is_empty"]) for b in batch],
        "resized_shape": batch[0]["resized_shape"],
    }
    return imgs, targets, meta


# ---------------- Builder de DataLoader ----------------
def build_yolo_dataloader(
    split: str = "train",
    batch: Optional[int] = None,
    shuffle: Optional[bool] = None,
    workers: Optional[int] = None,
    imgsz: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    augment: Optional[bool] = None,
    project_root: Optional[Path] = None,
):
    root = project_root or find_project_root()
    data_cfg = load_dataset_yaml(root)
    train_cfg = load_train_yaml(root)

    # Parámetros por defecto desde train.yaml
    bz = batch if batch is not None else int(train_cfg.get("batch", 16))
    imgs = imgsz if imgsz is not None else int(train_cfg.get("imgsz", 640))
    dl_cfg = train_cfg.get("dataloader", {})
    num_workers = workers if workers is not None else int(dl_cfg.get("workers", 4))
    pin = bool(dl_cfg.get("pin_memory", True) if pin_memory is None else pin_memory)
    pw = bool(dl_cfg.get("persistent_workers", True) if persistent_workers is None else persistent_workers)
    do_shuffle = bool(dl_cfg.get("shuffle", True) if shuffle is None else shuffle)
    do_augment = bool(augment if augment is not None else (split == "train"))

    # Aug parámetros
    fliplr = float(train_cfg.get("fliplr", 0.0))
    hsv_h = float(train_cfg.get("hsv_h", 0.0))
    hsv_s = float(train_cfg.get("hsv_s", 0.0))
    hsv_v = float(train_cfg.get("hsv_v", 0.0))

    images_dir = Path(data_cfg[split])
    dataset = YOLODataset(
        images_dir=images_dir,
        imgsz=imgs,
        augment=do_augment,
        fliplr=fliplr,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        stride=32,
    )
    # Nota: shuffle solo para train; para valid/test suele ser False
    if split != "train":
        do_shuffle = False if shuffle is None else do_shuffle

    loader = DataLoader(
        dataset,
        batch_size=bz,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_yolo,
        persistent_workers=pw if num_workers > 0 else False,
        drop_last=False,
    )
    return loader


# ---------------- CLI de prueba ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prueba del DataLoader YOLOv11.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    root = find_project_root()
    print(f"[Info] Proyecto: {root}")
    print(f"[Info] Usando split: {args.split}")
    loader = build_yolo_dataloader(split=args.split, batch=args.batch, imgsz=args.imgsz, project_root=root)

    # Itera primer batch
    imgs, targets, meta = next(iter(loader))
    print(f"Batch imgs: {tuple(imgs.shape)}  dtype={imgs.dtype}  range=({imgs.min():.3f},{imgs.max():.3f})")
    print(f"Targets shape: {tuple(targets.shape)}  Ejemplo (hasta 5 filas):\n{targets[:5]}")
    print(f"Paths[0]: {meta['paths'][0]}")
    print(f"Resized shape: {meta['resized_shape']}  Negativos en batch: {sum(meta['is_empty'])}/{len(meta['is_empty'])}")
