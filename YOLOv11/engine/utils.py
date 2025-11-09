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
]

# -------------------------------
# Seeds y dispositivo
# -------------------------------

def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo (opcional)
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
        # similar a Ultralytics: exp, exp2, exp3…
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
        torch.save(state, self.paths["state"])  # estado detallado
        # Guardar pesos del modelo entrenado
        torch.save(model.state_dict(), self.paths["last"])  # last
        if best:
            shutil.copy2(self.paths["last"], self.paths["best"])  # best
        # Meta JSON (liviano)
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
        # Compatibilidad: si se guardó dict simple, adaptarlo
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


def timed_stop(limit_str: Optional[str]) -> _TimerStop:
    """Crea un temporizador de parada segura dado un string "HH:MM" o minutos.

    Ejemplos: "2:30" → 2h30m; "90m" → 90 min; "5400s" → 5400 s.
    """
    if not limit_str:
        return _TimerStop(None)
    s = limit_str.strip().lower()
    total_s: Optional[float] = None
    m = re.match(r"^(\d+):(\d{1,2})$", s)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
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
    """Aplica recuperación básica ante NaN/Inf.

    Retorna (recovered, amp_enabled_new).
    - Si loss es NaN/Inf: reduce LR y opcionalmente desactiva AMP.
    """
    if policy is None:
        policy = _NaNPolicy()
    if not (np.isnan(loss_value) or np.isinf(loss_value)):
        return False, amp_enabled

    # Reducir LR
    for pg in optimizer.param_groups:
        old = pg.get("lr", 0.0)
        pg["lr"] = float(old) * float(policy.lower_lr_factor)
    # Deshabilitar AMP si procede
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


# Exponer un singleton utilizable por el Trainer
SIGNALS = _SignalCatcher()
