# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: logger.py
# Registrador de experimentos para YOLOv11. Crea estructuras en logs/ y runs/ (TensorBoard). Registra configuración, resumen de modelo y métricas por época para train/valid.
#==============================================================

import os
import csv
import time
import json
import socket
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from torch.utils.tensorboard import SummaryWriter  # opcional si TensorBoard está instalado
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# ---------------- Utilidades de ruta ----------------

def _find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# ---------------- Clase principal ----------------

class ExperimentLogger:
    """
    Logger científico para entrenamiento/validación.

    - Estructura de carpetas:
        logs/<variant>/<phase>/<run_name>/
        runs/<variant>/<phase>/<run_name>/   (archivos de TensorBoard)
    - Archivos clave:
        config_snapshot.json/yaml (opcional, vía save_config*)
        model_summary.txt
        train.csv / valid.csv  (métricas por época)
    - Soporta TensorBoard (si está disponible): agrega scalars con prefijo 'train/' o 'valid/'
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        variant: str = "m",
        phase: str = "train",
        run_name: Optional[str] = None,
        enable_tensorboard: bool = True,
        flush_secs: int = 10,
    ) -> None:
        self.root = project_root or _find_project_root()
        self.variant = str(variant).lower()
        self.phase = str(phase).lower()
        self.run_name = run_name or f"yolo11_{self.variant}_{self.phase}_{_timestamp()}"
        # Directorios
        self.logs_dir = self.root / "logs" / self.variant / self.phase / self.run_name
        self.runs_dir = self.root / "runs" / self.variant / self.phase / self.run_name
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # CSV writers por split
        self._csv_files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, csv.DictWriter] = {}
        self._headers: Dict[str, list] = {}

        # TensorBoard
        self.tb: Optional[SummaryWriter] = None
        if enable_tensorboard and SummaryWriter is not None:
            self.tb = SummaryWriter(log_dir=str(self.runs_dir), flush_secs=flush_secs)

        # Metadata base
        self.meta = {
            "run_name": self.run_name,
            "variant": self.variant,
            "phase": self.phase,
            "host": socket.gethostname(),
            "start_time": _timestamp(),
        }

    # ---------------- Configuración y resumen ----------------

    def save_config_json(self, cfg: Dict, fname: str = "config_snapshot.json") -> Path:
        path = self.logs_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return path

    def save_text(self, text: str, fname: str) -> Path:
        path = self.logs_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def save_model_summary(self, model, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Guarda conteo de parámetros y detalles relevantes."""
        import torch

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lines = [
            f"Run: {self.run_name}",
            f"Variant: {self.variant} | Phase: {self.phase}",
            f"Total params: {total:,}",
            f"Trainable params: {trainable:,}",
        ]
        if extra:
            for k, v in extra.items():
                lines.append(f"{k}: {v}")
        path = self.logs_dir / "model_summary.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # También deja un JSON para parsing
        jpath = self.logs_dir / "model_summary.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_params": total,
                    "trainable_params": trainable,
                    **(extra or {}),
                    **self.meta,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return path

    # ---------------- Métricas por época ----------------

    def _ensure_csv(self, split: str, columns: Optional[list] = None):
        split = split.lower()
        if split not in self._csv_writers:
            csv_path = self.logs_dir / f"{split}.csv"
            # Crear writer con encabezado flexible
            cols = ["epoch", "time"] + (columns or [])
            self._headers[split] = cols
            f = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=cols)
            if csv_path.stat().st_size == 0:
                writer.writeheader()
            self._csv_files[split] = f
            self._csv_writers[split] = writer

    def log_epoch(self, epoch: int, metrics: Dict[str, float], split: str = "train") -> None:
        split = split.lower()
        # Crear CSV si hace falta
        current_keys = sorted(metrics.keys())
        if split not in self._csv_writers:
            self._ensure_csv(split, current_keys)
        # Si aparecen nuevas métricas, extender encabezado
        header = self._headers[split]
        new_keys = [k for k in current_keys if k not in header]
        if new_keys:
            header.extend(new_keys)
            # reescribir CSV con nuevo encabezado
            csv_path = self.logs_dir / f"{split}.csv"
            self._csv_files[split].close()
            # Leer todo, reescribir con nuevo header
            rows = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            # reabrir para append
            f = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=header)
            self._csv_files[split] = f
            self._csv_writers[split] = writer

        row = {"epoch": int(epoch), "time": _timestamp()}
        row.update(metrics)
        self._csv_writers[split].writerow(row)
        self._csv_files[split].flush()

        # TensorBoard
        if self.tb is not None:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(f"{split}/{k}", float(v), epoch)
                except Exception:
                    pass

    # ---------------- Utilidades varias ----------------

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], epoch: int, split: str = "train"):
        if self.tb is not None:
            self.tb.add_scalars(f"{split}/{main_tag}", tag_scalar_dict, epoch)

    def add_text(self, tag: str, text: str, epoch: int = 0):
        if self.tb is not None:
            self.tb.add_text(tag, text, epoch)

    def close(self) -> None:
        for f in self._csv_files.values():
            try:
                f.close()
            except Exception:
                pass
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass

# ---------------- Ejemplo de uso (documentación) ----------------
EXAMPLE = """
from YOLOv11.utility.logger import ExperimentLogger

logger = ExperimentLogger(variant='m', phase='train')
logger.save_config_json({'imgsz':640, 'epochs':150, 'optimizer':'adamw'})
logger.save_model_summary(model, extra={'strides':[8,16,32]})
for epoch in range(1, 151):
    train_metrics = {'loss': 1.23, 'cls':0.1, 'box':0.9, 'map50':0.65}
    val_metrics   = {'loss': 1.10, 'map50':0.67, 'map50-95':0.42}
    logger.log_epoch(epoch, train_metrics, split='train')
    logger.log_epoch(epoch, val_metrics, split='valid')
logger.close()
"""
