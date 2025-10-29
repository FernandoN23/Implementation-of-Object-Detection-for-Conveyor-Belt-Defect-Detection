# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: weights.py
# Gestión de pesos/checkpoints para YOLOv11. Guarda por variante/fase/run con convenciones de nombre y retención.
#==============================================================

import os
import re
import time
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch

# ---------------- Utilidades de ruta y tiempo ----------------

def _find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# ---------------- Carga de opciones de guardado desde parser.yaml (opcional) ----------------

def _load_save_opts(root: Path) -> Dict[str, Any]:
    """Lee configs/parser.yaml si existe para extraer opciones de guardado."""
    cfg = {
        "save_best": True,
        "save_last": True,
        "save_period": 10,          # guardar cada N épocas
        "keep_checkpoint_max": 5,   # máximo de checkpoints intermedios a retener
    }
    try:
        import yaml  # lazy
        p = root / "configs" / "parser.yaml"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            save = data.get("save", {}) or {}
            for k in cfg.keys():
                if k in save:
                    cfg[k] = save[k]
    except Exception:
        pass
    return cfg

# ---------------- Clase principal ----------------

class WeightsManager:
    """
    Maneja guardado de checkpoints con estructura coherente con scripts de limpieza.

    Estructura:
        weights/<variant>/<phase>/<run_name>/
            ├─ N_Train_Epoch_001.pt   (ejemplo)
            ├─ last.pt
            ├─ best.pt
            └─ meta.json

    Convención de nombres por época:
        {VAR}_{{Train|Valid}}_Epoch_{epoch:03d}.pt
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        variant: str = "m",
        phase: str = "train",
        run_name: Optional[str] = None,
    ) -> None:
        self.root = project_root or _find_project_root()
        self.variant = str(variant).lower()
        self.phase = str(phase).lower()
        self.run_name = run_name or f"yolo11_{self.variant}_{self.phase}_{_timestamp()}"
        self.weights_dir = self.root / "weights" / self.variant / self.phase / self.run_name
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.save_opts = _load_save_opts(self.root)
        self.best_score: Optional[float] = None  # mayor es mejor por defecto

        # meta.json con info básica
        self._write_meta()

    # ---------------- Internos ----------------

    def _write_meta(self) -> None:
        meta = {
            "run_name": self.run_name,
            "variant": self.variant,
            "phase": self.phase,
            "created_at": _timestamp(),
            "save_options": self.save_opts,
        }
        with open(self.weights_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _epoch_filename(self, epoch: int) -> str:
        VAR = self.variant.upper()
        PH = "Train" if self.phase == "train" else "Valid"
        return f"{VAR}_{PH}_Epoch_{epoch:03d}.pt"

    def _list_epoch_files(self) -> List[Path]:
        patt = re.compile(rf"^{self.variant.upper()}_(Train|Valid)_Epoch_\d{{3}}\.pt$", re.IGNORECASE)
        return sorted([p for p in self.weights_dir.iterdir() if p.is_file() and patt.match(p.name)])

    def _apply_retention(self) -> None:
        """Mantiene como máximo keep_checkpoint_max archivos de época (excluye best.pt y last.pt)."""
        keep = int(self.save_opts.get("keep_checkpoint_max", 0) or 0)
        if keep <= 0:
            return
        epoch_files = self._list_epoch_files()
        if len(epoch_files) <= keep:
            return
        to_remove = epoch_files[:-keep]  # borrar los más antiguos
        for p in to_remove:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    # ---------------- API pública ----------------

    def save_epoch(
        self,
        model,
        epoch: int,
        score: Optional[float] = None,
        optimizer=None,
        scheduler=None,
        extra: Optional[Dict[str, Any]] = None,
        save_full_model: bool = False,
    ) -> Path:
        """
        Guarda un checkpoint para la época dada.
        - score: métrica para decidir 'best.pt' (mayor es mejor). Si None, no actualiza best.
        - save_full_model: si True, guarda el objeto modelo completo; de lo contrario state_dict (recomendado).
        Devuelve la ruta del archivo de época.
        """
        ckpt = {
            "epoch": int(epoch),
            "variant": self.variant,
            "phase": self.phase,
            "run_name": self.run_name,
            "timestamp": _timestamp(),
            "state_dict": None if save_full_model else model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "extra": extra or {},
        }
        if save_full_model:
            ckpt["model"] = model  # menos portable, pero útil para depuración

        # Guardar por época según política save_period
        period = int(self.save_opts.get("save_period", 1) or 1)
        save_epoch_file = (epoch % period == 0)
        epoch_path = self.weights_dir / self._epoch_filename(epoch)
        if save_epoch_file:
            torch.save(ckpt, epoch_path)
        else:
            # Aun si no guardamos el archivo numerado, igual actualizamos last.pt
            epoch_path = self.weights_dir / "last.pt"  # referencia para retorno

        # last.pt
        if self.save_opts.get("save_last", True):
            torch.save(ckpt, self.weights_dir / "last.pt")

        # best.pt
        if score is not None and self.save_opts.get("save_best", True):
            if (self.best_score is None) or (score > self.best_score):
                self.best_score = float(score)
                torch.save(ckpt, self.weights_dir / "best.pt")

        # Retención
        self._apply_retention()

        return epoch_path

    def load(self, path: Path) -> Dict[str, Any]:
        """Carga un checkpoint desde 'path' y lo retorna como dict."""
        return torch.load(path, map_location="cpu")

# ---------------- Ejemplo de uso (documentación) ----------------
EXAMPLE = """
from YOLOv11.utility.weights import WeightsManager

wm = WeightsManager(variant='n', phase='train')
for epoch in range(1, 151):
    # ... entrenamiento ...
    score = val_map50  # la métrica que desees para decidir 'best.pt'
    wm.save_epoch(model, epoch, score=score, optimizer=opt, scheduler=sched, extra={'imgsz':640})
# Para cargar:
# ckpt = wm.load(wm.weights_dir/'best.pt')
"""
