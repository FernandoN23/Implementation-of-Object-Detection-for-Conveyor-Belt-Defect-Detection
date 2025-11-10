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
#              ResultsCSV, CheckpointManager, resume, timed_stop,
#              nan_recovery, maybe_compile y helpers varios.
#==============================================================

from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "seed_everything",
    "select_device",
    "setup_save_dir",
    "ResultsCSV",
    "CheckpointManager",
    "TrainingState",
    "save_training_state",
    "load_training_state",
    "maybe_compile",
    "timed_stop",
    "nan_recovery",
    "device_info",
    "auto_amp_mode",
    "build_model",
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


def setup_save_dir(project: str = "runs/train", name: str = "exp", exist_ok: bool = False) -> str:
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
# CSV de resultados por época
# -------------------------------

class ResultsCSV:
    def __init__(self, save_dir: str, filename: str = "results.csv") -> None:
        self.path = Path(save_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = None

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            if self._fh.tell() == 0:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# -------------------------------
# Checkpoints y reanudar (resume)
# -------------------------------

@dataclass
class TrainingState:
    epoch: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    ema_state: Optional[Dict[str, Any]]
    metrics_val: Optional[Dict[str, float]]
    best_fitness: float


def _ckpt_paths(save_dir: str) -> Dict[str, Path]:
    d = Path(save_dir)
    return {
        "last": d / "last.pt",
        "best": d / "best.pt",
        "state": d / "last_state.pth",
        "meta": d / "meta.json",
    }


class CheckpointManager:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.paths = _ckpt_paths(save_dir)

    def save(self,
             epoch: int,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
             ema: Optional[Any],
             val_metrics: Dict[str, float],
             *,
             best: bool,
             best_fitness: float) -> Tuple[str, bool]:
        state = TrainingState(
            epoch=epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            ema_state=(ema.state_dict() if ema is not None and hasattr(ema, "state_dict") else None),
            metrics_val=val_metrics,
            best_fitness=float(best_fitness),
        )
        torch.save(state, self.paths["state"])
        torch.save(model.state_dict(), self.paths["last"])
        if best:
            shutil.copy2(self.paths["last"], self.paths["best"])
        meta = {
            "epoch": epoch,
            "best_fitness": float(best_fitness),
            "val": val_metrics,
            "last": str(self.paths["last"]),
            "best": str(self.paths["best"]),
        }
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return str(self.paths["last"]), best

    def load_last(self) -> Optional[TrainingState]:
        p = self.paths["state"]
        if not p.exists():
            return None
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict) and "epoch" in state and not isinstance(state, TrainingState):
            return TrainingState(**state)  # type: ignore[arg-type]
        return state  # type: ignore[return-value]


def save_training_state(path: str, state: TrainingState) -> None:
    torch.save(dataclasses.asdict(state), path)


def load_training_state(path: str) -> TrainingState:
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict) and "epoch" in d:
        return TrainingState(**d)
    raise RuntimeError("training state inválido/inesperado")


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


def nan_recovery(loss_value: float,
                 optimizer: torch.optim.Optimizer,
                 amp_enabled: bool,
                 *,
                 policy: Optional[_NaNPolicy] = None) -> Tuple[bool, bool]:
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
