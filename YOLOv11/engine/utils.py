# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/utils.py
# Descripción: Utilitarios de orquestación para el pipeline de
#              entrenamiento: seeds/dispositivos, save_dir,
#              timed_stop, nan_recovery, maybe_compile
#              y helpers varios (sin gestión de checkpoints aquí).
#==============================================================

from __future__ import annotations

import dataclasses
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from torchvision.ops import nms as tv_nms
except Exception:  # pragma: no cover
    tv_nms = None

import yaml
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "seed_everything",
    "select_device",
    "setup_save_dir",
    "maybe_compile",
    "timed_stop",
    "nan_recovery",
    "device_info",
    "auto_amp_mode",
    "build_model",
    "Validator_Utilities",

    "build_dataloaders",
]

# -------------------------------
# Seeds y dispositivo
# -------------------------------

def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def select_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
            return torch.device("hip")
        return torch.device("cpu")
    return torch.device(spec)


# -------------------------------
# save_dir y estructura de experimento
# -------------------------------


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def setup_save_dir(
    project: str = "runs/train",
    name: str = "exp",
    exist_ok: bool = False,
    *,
    variant: Optional[str] = None,
    phase: str = "train",
    is_test: bool = False,
    project_root: Optional[str] = None,
) -> str:
    """Construye el directorio de guardado para un experimento.

    Modo legacy (compatibilidad):
        - Usado cuando ``variant is None``.
        - Respeta el esquema clásico ``project/name`` (por defecto ``runs/train/exp``).

    Modo nuevo (recomendado):
        - Activado cuando se proporciona ``variant``.
        - Sigue el layout unificado del proyecto:

            runs/<variant>/<phase>/<slot>/

          donde ``slot`` depende del tipo de ejecución:

            - train + is_test=True  →  tests/<run_name>
            - train + is_test=False →  final
            - otras phases          →  <run_name> (o timestamp si no se da nombre)
    """

    # ---------------------------
    # Modo nuevo: layout por variant/phase/slot
    # ---------------------------
    if variant is not None:
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        runs_root = root / "runs"

        # Normalizamos el nombre de corrida. "auto"/"exp" → timestamp.
        run_name: Optional[str]
        if name in (None, "", "auto", "exp"):
            run_name = None
        else:
            run_name = str(name)

        if phase == "train" and not is_test:
            # Entrenamiento final: slot único "final".
            slot = "final"
        elif is_test:
            # Slots de pruebas de ensamblado / warmups cortos.
            if run_name is None:
                run_name = _timestamp()
            slot = f"tests/{run_name}"
        else:
            # Otras fases (valid/test externos, etc.) → nombre explícito o timestamp.
            if run_name is None:
                run_name = _timestamp()
            slot = run_name

        # slot puede contener subdirectorios (p.ej. "tests/<run_name>").
        save_dir = runs_root / variant / phase
        for part in slot.split("/"):
            if part:
                save_dir /= part

        save_dir.mkdir(parents=True, exist_ok=exist_ok)
        return str(save_dir.resolve())

    # ---------------------------
    # Modo legacy: mantiene compatibilidad con código existente
    # ---------------------------
    root = Path(project)
    if name == "auto" or name is None or name == "exp":
        base = root / "exp"
        if not base.exists():
            save_dir = base
        else:
            i = 2
            while (root / f"exp{i}").exists():
                i += 1
            save_dir = root / f"exp{i}"
    else:
        save_dir = root / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)
    return str(save_dir.resolve())


# -------------------------------
# torch.compile (opcional)
# -------------------------------


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    try:
        model = torch.compile(model)  # type: ignore[attr-defined]
        return model
    except Exception:
        return model


# -------------------------------
# Timed stop y NaN recovery
# -------------------------------


@dataclass
class _TimerStop:
    time_limit_s: Optional[float] = None
    start_s: float = dataclasses.field(default_factory=lambda: time.perf_counter())

    def expired(self) -> bool:
        if self.time_limit_s is None or self.time_limit_s <= 0:
            return False
        return (time.perf_counter() - self.start_s) >= self.time_limit_s


def timed_stop(limit) -> _TimerStop:
    if limit is None:
        return _TimerStop(None)
    if isinstance(limit, (int, float)):
        if float(limit) <= 0:
            return _TimerStop(None)
        return _TimerStop(float(limit))
    s = str(limit).strip().lower()
    if not s:
        return _TimerStop(None)
    total_s = None
    parts = s.split(":")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        h, mm = int(parts[0]), int(parts[1])
        total_s = float(h * 3600 + mm * 60)
    elif s.endswith("m") and s[:-1].isdigit():
        total_s = float(int(s[:-1]) * 60)
    elif s.endswith("s") and s[:-1].isdigit():
        total_s = float(int(s[:-1]))
    elif s.isdigit():
        total_s = float(int(s))
    return _TimerStop(total_s)


@dataclass
class _NaNPolicy:
    lower_lr_factor: float = 0.2
    disable_amp: bool = True



def nan_recovery(
    loss_value: float,
    optimizer: torch.optim.Optimizer,
    amp_enabled: bool,
    *,
    policy: Optional[_NaNPolicy] = None,
) -> Tuple[bool, bool]:
    if policy is None:
        policy = _NaNPolicy()
    if not (np.isnan(loss_value) or np.isinf(loss_value)):
        return False, amp_enabled
    for pg in optimizer.param_groups:
        old = pg.get("lr", 0.0)
        pg["lr"] = float(old) * float(policy.lower_lr_factor)
    if policy.disable_amp and amp_enabled:
        amp_enabled = False
    return True, amp_enabled


# -------------------------------
# Señales del SO para interrupción limpia
# -------------------------------


class _SignalCatcher:
    def __init__(self) -> None:
        self._stop = False
        try:
            signal.signal(signal.SIGINT, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        except Exception:
            pass

    def _handle(self, signum, frame):  # type: ignore[override]
        self._stop = True

    @property
    def stop(self) -> bool:
        return self._stop


SIGNALS = _SignalCatcher()

# -------------------------------
# Información de dispositivo y AMP auto
# -------------------------------


def device_info() -> str:
    try:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = []
            for i in range(n):
                try:
                    names.append(torch.cuda.get_device_name(i))
                except Exception:
                    names.append(f"cuda:{i}")
            runtime = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            return f"{runtime} {n}x [" + ", ".join(names) + "]"
        return "CPU"
    except Exception as e:
        return f"Unknown device ({e})"



def auto_amp_mode() -> str:
    try:
        if torch.cuda.is_available():
            bf16_ok = False
            if hasattr(torch.cuda, "is_bf16_supported"):
                try:
                    bf16_ok = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    bf16_ok = False
            if bf16_ok:
                return "bf16"
            return "fp16"
    except Exception:
        pass
    return "off"


# -------------------------------
# Utilidades específicas de validación (YOLOv11)
# -------------------------------


class Validator_Utilities:
    """Helpers estáticos para la validación/metricado YOLOv11.

    Se agrupan en una clase para no contaminar el namespace de módulo y
    facilitar su mantenimiento por lienzos.
    """

    @staticmethod
    def log(msg: str, cfg: Optional[Any] = None, level: int = 1) -> None:
        """Imprime mensajes de validator respetando cfg.verbose.

        Si cfg es None o no tiene atributo ``verbose``, se asume nivel 1.
        """
        v = 1 if cfg is None else int(getattr(cfg, "verbose", 1) or 1)
        if v >= level:
            print(f"[validator] {msg}")

    @staticmethod
    def select_device(spec: str = "auto") -> torch.device:
        """Delegado a ``select_device`` de este módulo.

        Se expone aquí para unificar la API usada por validator.
        """
        return select_device(spec)

    # ---------- IoU y NMS ----------

    @staticmethod
    def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calcula la matriz IoU entre dos conjuntos de cajas en formato xyxy.

        boxes1: [N, 4], boxes2: [M, 4] -> IoU [N, M]
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

        b1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        b2 = boxes2.unsqueeze(0)  # [1, M, 4]

        inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
        inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
        inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
        inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        return inter_area / (union + 1e-9)

    @staticmethod
    def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """Implementación NMS pura en PyTorch (fallback cuando no hay torchvision)."""
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)

        idxs = scores.argsort(descending=True)
        keep: List[int] = []
        while idxs.numel() > 0:
            i = int(idxs[0])
            keep.append(i)
            if idxs.numel() == 1:
                break
            ious = Validator_Utilities.box_iou_xyxy(boxes[i : i + 1], boxes[idxs[1:]])[0]
            idxs = idxs[1:][ious <= float(iou_thres)]
        return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """Wrapper NMS que usa torchvision.ops.nms si está disponible.

        Si no, cae a ``nms_pytorch``.
        """
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        if tv_nms is not None:
            try:
                return tv_nms(boxes, scores, float(iou_thres))
            except Exception:
                # Si por alguna razón falla torchvision, usamos fallback.
                pass
        return Validator_Utilities.nms_pytorch(boxes, scores, iou_thres)

    # ---------- Tags y targets ----------

    @staticmethod
    def sanitize_phase_tag(tag: str) -> str:
        """Normaliza el nombre de fase para uso en rutas de métricas.

        Mantiene letras, dígitos, "-" y "_"; el resto se sustituye por "_".
        """
        cleaned = []
        for ch in str(tag):
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        out = "".join(cleaned).strip("_")
        return out or "metrics"

    @staticmethod
    def targets_global_to_per_image(
        targets: torch.Tensor,
        bs: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Convierte un tensor global [N,6]/[N,5] en lista por imagen [B][Ni,5].

        Formato esperado típico YOLO:
            [img_idx, cls, x, y, w, h]  (6 columnas)
        Se convierte a:
            [cls, x, y, w, h] por imagen.
        """
        out: List[torch.Tensor] = []
        if targets is None or targets.numel() == 0:
            empty = torch.zeros(0, 5, device=device, dtype=torch.float32)
            return [empty.clone() for _ in range(bs)]

        if targets.dim() != 2 or targets.size(1) not in (5, 6):
            empty = torch.zeros(0, 5, device=device, dtype=torch.float32)
            return [empty.clone() for _ in range(bs)]

        t = targets.to(device)
        if t.size(1) == 6:
            img_idx = t[:, 0].long()
            cls = t[:, 1:2]
            xywh = t[:, 2:6]
        else:  # [N,5] → asumimos cls + xywh sin índice, se asigna a la imagen 0
            img_idx = torch.zeros(t.size(0), dtype=torch.long, device=device)
            cls = t[:, 0:1]
            xywh = t[:, 1:5]

        packed = torch.cat([cls, xywh], dim=1)
        for i in range(bs):
            m = img_idx == i
            if not torch.any(m):
                out.append(torch.zeros(0, 5, device=device, dtype=torch.float32))
            else:
                out.append(packed[m])
        return out

    @staticmethod
    def targets_to_list_per_image(
        targets_any: Any,
        bs: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Normaliza targets a lista [B][Ni,5] en formato [cls, x, y, w, h].

        Acepta:
            - tensor global [N,6]/[N,5] (collate estilo YOLO).
            - lista de tensores por imagen.
            - None → lista de vacíos.
        """
        # Caso tensor global
        if isinstance(targets_any, torch.Tensor):
            return Validator_Utilities.targets_global_to_per_image(targets_any, bs=bs, device=device)

        # Caso lista de tensores por imagen
        if isinstance(targets_any, (list, tuple)):
            out: List[torch.Tensor] = []
            for t in targets_any:
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    out.append(torch.zeros(0, 5, device=device, dtype=torch.float32))
                    continue
                tt = t.to(device)
                if tt.dim() != 2 or tt.size(1) not in (5, 6):
                    out.append(torch.zeros(0, 5, device=device, dtype=torch.float32))
                    continue
                if tt.size(1) == 6:
                    # [img_idx, cls, x, y, w, h] → [cls, x, y, w, h]
                    tt = tt[:, 1:6]
                out.append(tt[:, 0:5])
            # Ajuste de tamaño a batch size
            if len(out) < bs:
                empty = torch.zeros(0, 5, device=device, dtype=torch.float32)
                out.extend(empty.clone() for _ in range(bs - len(out)))
            elif len(out) > bs:
                out = out[:bs]
            return out

        # Cualquier otro formato → lista de targets vacíos
        empty = torch.zeros(0, 5, device=device, dtype=torch.float32)
        return [empty.clone() for _ in range(bs)]


# -------------------------------
# Construcción de modelo y dataloaders (prioriza parser_yaml)
# -------------------------------


class _TrainWrapper(torch.nn.Module):
    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, batch: Any):
        if isinstance(batch, dict):
            x = batch.get("img")
        else:
            x = batch
        out = self.core(x)
        cls_mean = 0.0
        reg_mean = 0.0
        for c in out.get("cls", []):
            cls_mean = cls_mean + c.float().mean()
        for r in out.get("reg", []):
            reg_mean = reg_mean + r.float().mean()
        loss = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        items = {
            "cls_mean": float(cls_mean.detach().cpu().item() if torch.is_tensor(cls_mean) else 0.0),
            "reg_mean": float(reg_mean.detach().cpu().item() if torch.is_tensor(reg_mean) else 0.0),
        }
        return loss, items



def _read_yaml_dict(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML no encontrado: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



def build_model(model_yaml_path: str, *, variant: str) -> torch.nn.Module:
    try:
        from YOLOv11.models.parser_yaml import ConfigParserYaml
    except Exception:
        from models.parser_yaml import ConfigParserYaml  # type: ignore

    root = Path(__file__).resolve().parents[1]
    cfg = ConfigParserYaml(project_root=str(root)).load()

    try:
        imgsz_for_strides = int(cfg.train_cfg.get("normalized", {}).get("data", {}).get("imgsz", 640))
        if not imgsz_for_strides:
            imgsz_for_strides = 640
    except Exception:
        imgsz_for_strides = 640

    core = cfg.build_model(variant=variant, imgsz_for_strides=imgsz_for_strides)
    return _TrainWrapper(core)


class _SyntheticDetectionDataset(Dataset):
    def __init__(self, length: int, imgsz: int, nc: int = 5) -> None:
        super().__init__()
        self.length = int(length)
        self.imgsz = int(imgsz)
        self.nc = int(nc)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = torch.zeros(3, self.imgsz, self.imgsz, dtype=torch.float32)
        return {"img": x, "cls": torch.tensor([], dtype=torch.long), "bboxes": torch.zeros(0, 4), "img_id": int(idx)}



def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    imgs = torch.stack([b["img"] for b in batch], 0)
    cls = [b["cls"] for b in batch]
    bboxes = [b["bboxes"] for b in batch]
    img_ids = [b.get("img_id", -1) for b in batch]
    return {"img": imgs, "cls": cls, "bboxes": bboxes, "img_id": img_ids}



def build_dataloaders(data_yaml_path: str, *, imgsz: int, batch: int, workers: int):
    try:
        from YOLOv11.models.parser_yaml import ConfigParserYaml
    except Exception:
        from models.parser_yaml import ConfigParserYaml  # type: ignore

    try:
        root = Path(__file__).resolve().parents[1]
        cfg = ConfigParserYaml(project_root=str(root)).load()
        ds_yaml_path = cfg.paths.dataset_yaml
        y = _read_yaml_dict(str(ds_yaml_path))
    except Exception:
        y = _read_yaml_dict(data_yaml_path)

    names = y.get("names", {}) if isinstance(y, dict) else {}
    nc = int(y.get("nc", len(names) if isinstance(names, dict) else 5))

    train_len = int(y.get("__synthetic_train_len__", 8))
    val_len = int(y.get("__synthetic_val_len__", 4))

    train_ds = _SyntheticDetectionDataset(train_len, imgsz, nc=nc)
    val_ds = _SyntheticDetectionDataset(val_len, imgsz, nc=nc)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch),
        shuffle=True,
        num_workers=int(workers),
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader, (names if isinstance(names, dict) else {i: str(i) for i in range(nc)})
