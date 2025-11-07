# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: logger.py
# Registrador de experimentos para YOLOv11. Crea estructuras en logs/ y runs/ (TensorBoard).
# Registra configuración, resumen de modelo y métricas por época para train/valid con
# fecha completa, variante, número de época y volcado JSONL/LOG por época.
# Ahora soporta "slots" de ejecución: pruebas (tests/<run_name>) y final (final/),
# análogo a weights.py. El slot "final" es único y reanudable.
#==============================================================

from __future__ import annotations

import csv
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    from torch.utils.tensorboard import SummaryWriter  # opcional
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


# ---------------- Utilidades de ruta/tiempo ----------------

def _find_project_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()


def _timestamp_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _timestamp_iso() -> str:
    # Fecha completa en ISO‑8601 con zona horaria local
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------- Clase principal ----------------

class ExperimentLogger:
    """
    Logger científico para entrenamiento/validación.

    Estructura de carpetas:
        logs/<variant>/<phase>/<slot>/
        runs/<variant>/<phase>/<slot>/
    donde <slot> es "tests/<run_name>" si is_test=True, y "final" si is_test=False.

    Archivos clave en logs/<...>/<slot>/ :
        - run_manifest.json (metadatos de la corrida)
        - model_summary.txt / model_summary.json
        - <split>.csv (train.csv, valid.csv) — cabecera dinámica
        - <split>_epochs.jsonl — 1 JSON por línea
        - <split>_epochs.log   — texto legible por humanos

    Soporte TensorBoard (si está disponible) en runs/<...>/<slot>/.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        variant: str = "m",
        phase: str = "train",
        run_name: Optional[str] = None,
        is_test: bool = False,
        reset_final: bool = False,
        enable_tensorboard: bool = True,
        flush_secs: int = 10,
    ) -> None:
        self.root = project_root or _find_project_root()
        self.variant = str(variant).lower()
        self.phase = str(phase).lower()
        self.is_test = bool(is_test)

        # Resolución de slot
        if self.is_test:
            # pruebas aisladas por run_name (no reescribibles)
            self.run_name = run_name or f"yolo11_{self.variant}_{self.phase}_{_timestamp_compact()}"
            slot = Path("tests") / self.run_name
        else:
            # final: slot único y reanudable (se puede limpiar con reset_final)
            self.run_name = run_name or "final"
            slot = Path("final")
            slot_dir = self.root / "logs" / self.variant / self.phase / slot
            if reset_final and slot_dir.exists():
                # borrar logs final previos (equivalente a empezar de cero)
                import shutil
                shutil.rmtree(slot_dir, ignore_errors=True)

        # Directorios base
        self.logs_dir = self.root / "logs" / self.variant / self.phase / slot
        self.runs_dir = self.root / "runs" / self.variant / self.phase / slot
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # CSV writers por split
        self._csv_files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, csv.DictWriter] = {}
        self._headers: Dict[str, List[str]] = {}

        # TensorBoard
        self.tb: Optional[SummaryWriter] = None
        if enable_tensorboard and SummaryWriter is not None:
            self.tb = SummaryWriter(log_dir=str(self.runs_dir), flush_secs=flush_secs)

        # Metadata base
        self.meta = {
            "run_name": self.run_name,
            "variant": self.variant,
            "phase": self.phase,
            "slot_type": "test" if self.is_test else "final",
            "host": socket.gethostname(),
            "start_time_iso": _timestamp_iso(),
            "start_time_compact": _timestamp_compact(),
        }

        # Manifest inicial
        self._write_run_manifest()

    # ---------------- Configuración y resumen ----------------

    def _write_run_manifest(self) -> None:
        man = self.meta.copy()
        (self.logs_dir / "run_manifest.json").write_text(
            json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lines = [
            f"Run: {self.run_name}",
            f"Variant: {self.variant} | Phase: {self.phase} | Slot: {'test' if self.is_test else 'final'}",
            f"Started at: {self.meta['start_time_iso']}",
            f"Total params: {total:,}",
            f"Trainable params: {trainable:,}",
        ]
        if extra:
            for k, v in extra.items():
                lines.append(f"{k}: {v}")
        path = self.logs_dir / "model_summary.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # JSON espejo
        jpath = self.logs_dir / "model_summary.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({
                "total_params": total,
                "trainable_params": trainable,
                **(extra or {}),
                **self.meta,
            }, f, indent=2, ensure_ascii=False)
        return path

    # ---------------- Métricas por época ----------------

    def _ensure_csv(self, split: str, columns: Optional[List[str]] = None):
        split = split.lower()
        base_cols = ["epoch", "date_iso", "variant", "phase", "run_name", "slot_type"]
        if split not in self._csv_writers:
            csv_path = self.logs_dir / f"{split}.csv"
            cols = base_cols + (columns or [])
            self._headers[split] = cols
            f = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=cols)
            if csv_path.stat().st_size == 0:
                writer.writeheader()
            self._csv_files[split] = f
            self._csv_writers[split] = writer
        else:
            # Garantiza que base_cols estén presentes (migración suave)
            header = self._headers[split]
            need = [c for c in base_cols if c not in header]
            if need:
                header.extend(need)
                self._rewrite_csv(split, header)

    def _rewrite_csv(self, split: str, header: List[str]) -> None:
        csv_path = self.logs_dir / f"{split}.csv"
        # cierra y reabre
        try:
            self._csv_files[split].close()
        except Exception:
            pass
        # lee todo
        rows: List[Dict[str, Any]] = []
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        # reescribe
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                # defaults para nuevas columnas base
                r.setdefault("variant", self.variant)
                r.setdefault("phase", split)
                r.setdefault("date_iso", self.meta.get("start_time_iso", _timestamp_iso()))
                r.setdefault("run_name", self.run_name)
                r.setdefault("slot_type", "test" if self.is_test else "final")
                writer.writerow(r)
        # reabrir para append
        f = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=header)
        self._csv_files[split] = f
        self._csv_writers[split] = writer
        self._headers[split] = header

    def log_epoch(self, epoch: int, metrics: Dict[str, float], split: str = "train") -> None:
        split = split.lower()
        # Crear/actualizar CSV
        current_keys = sorted(metrics.keys())
        if split not in self._csv_writers:
            self._ensure_csv(split, current_keys)
        # Extiende cabecera si aparecen nuevas métricas o faltan base_cols
        header = self._headers[split]
        base_cols = ["epoch", "date_iso", "variant", "phase", "run_name", "slot_type"]
        new_keys = [k for k in (current_keys + base_cols) if k not in header]
        if new_keys:
            header.extend([k for k in new_keys if k not in header])
            self._rewrite_csv(split, header)

        date_iso = _timestamp_iso()
        row = {
            "epoch": int(epoch),
            "date_iso": date_iso,
            "variant": self.variant,
            "phase": split,
            "run_name": self.run_name,
            "slot_type": "test" if self.is_test else "final",
            **metrics,
        }
        self._csv_writers[split].writerow(row)
        self._csv_files[split].flush()

        # JSONL: una línea por época
        jsonl_path = self.logs_dir / f"{split}_epochs.jsonl"
        record = {
            "run_name": self.run_name,
            "variant": self.variant,
            "phase": split,
            "slot_type": "test" if self.is_test else "final",
            "epoch": int(epoch),
            "date_iso": date_iso,
            **metrics,
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # LOG texto humano-legible
        kv = " ".join([f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in sorted(metrics.items())])
        with open(self.logs_dir / f"{split}_epochs.log", "a", encoding="utf-8") as f:
            f.write(
                f"[{date_iso}] variant={self.variant} slot={'test' if self.is_test else 'final'} run={self.run_name} epoch={int(epoch)} {kv}\n"
            )

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

# Slot de PRUEBAS (separado por run_name)
logger_test = ExperimentLogger(variant='n', phase='train', is_test=True, run_name='ablation_lr0.001')
logger_test.log_epoch(1, {'loss': 1.2, 'mAP50': 0.62}, split='train')
logger_test.close()

# Slot FINAL (único y reanudable)
logger_final = ExperimentLogger(variant='m', phase='train', is_test=False, reset_final=False)
logger_final.log_epoch(1, {'loss': 1.1, 'mAP50': 0.67}, split='train')
logger_final.close()
"""
