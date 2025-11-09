# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/hud.py
# Descripción: HUD minimalista para consola con telemetría de
#              entrenamiento (época, iteración, LR, pérdidas,
#              tiempo/iter, VRAM reservada/máxima, etc.). Incluye
#              control de frecuencia (throttle) y compatibilidad con
#              DDP (solo rank 0).
#==============================================================

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

__all__ = ["HUDConfig", "HUD", "format_bytes"]


# -------------------------------
# Utilidades
# -------------------------------

def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} PB"


def _device_mem() -> Dict[str, int]:
    """Lee memoria de dispositivo (CUDA/HIP)."""
    out = {"reserved": 0, "allocated": 0, "max_allocated": 0}
    try:
        if torch.cuda.is_available():
            out["reserved"] = int(torch.cuda.memory_reserved())
            out["allocated"] = int(torch.cuda.memory_allocated())
            out["max_allocated"] = int(torch.cuda.max_memory_allocated())
        elif hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
            # PyTorch ROCm expone la misma API de cuda.*
            out["reserved"] = int(torch.cuda.memory_reserved())  # type: ignore[attr-defined]
            out["allocated"] = int(torch.cuda.memory_allocated())  # type: ignore[attr-defined]
            out["max_allocated"] = int(torch.cuda.max_memory_allocated())  # type: ignore[attr-defined]
    except Exception:
        pass
    return out


def _is_main_process() -> bool:
    # Torch DDP estándar: rank 0 escribe al stdout
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank == 0


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class HUDConfig:
    width: int = 80                  # ancho de barra
    interval: float = 0.05           # fracción de época entre updates (5%)
    min_update_s: float = 0.25       # límite temporal mínimo entre updates
    enable: bool = True
    show_vram: bool = True
    show_items: bool = True          # mostrar desglose de pérdidas (box/cls/dfl)
    precision: int = 4
    stream: object = sys.stdout


# -------------------------------
# HUD
# -------------------------------

class HUD:
    """HUD simple y estable para bucles de entrenamiento.

    Uso típico:
        hud = HUD(HUDConfig(interval=0.05))
        hud.on_epoch_start(epoch, epochs, iters_per_epoch)
        for i, batch in enumerate(loader):
            ...
            hud.update(epoch, i+1, iters_per_epoch, lr, loss, items, dt_ms)
        hud.on_epoch_end()
    """

    def __init__(self, cfg: Optional[HUDConfig] = None) -> None:
        self.cfg = cfg or HUDConfig()
        self._last_print_t = 0.0
        self._last_progress_bucket = -1
        self._epoch = 0
        self._epochs = 0
        self._iters_per_epoch = 0

    # Eventos
    def on_epoch_start(self, epoch: int, epochs: int, iters_per_epoch: int) -> None:
        if not self.cfg.enable or not _is_main_process():
            return
        self._epoch = epoch
        self._epochs = epochs
        self._iters_per_epoch = max(1, iters_per_epoch)
        self._last_print_t = 0.0
        self._last_progress_bucket = -1
        self._writeln(f"Epoch {epoch+1}/{epochs}")

    def on_epoch_end(self) -> None:
        if not self.cfg.enable or not _is_main_process():
            return
        self._writeln("")  # salto de línea para separar épocas

    # Núcleo
    def update(self,
               epoch: int,
               it: int,
               iters_per_epoch: int,
               lr: float,
               loss: float,
               items: Optional[Dict[str, float]],
               dt_ms: float) -> None:
        if not self.cfg.enable or not _is_main_process():
            return

        now = time.perf_counter()
        progress = min(0.9999, max(0.0, (it / float(max(1, iters_per_epoch)))))
        bucket = int(progress / max(1e-6, self.cfg.interval))
        if bucket == self._last_progress_bucket and (now - self._last_print_t) < self.cfg.min_update_s:
            return
        self._last_progress_bucket = bucket
        self._last_print_t = now

        bar = self._render_bar(progress)
        vram = _device_mem() if self.cfg.show_vram else {"reserved": 0, "allocated": 0, "max_allocated": 0}
        mem_txt = f"VRAM {format_bytes(vram['allocated'])} / {format_bytes(vram['reserved'])} (peak {format_bytes(vram['max_allocated'])})"
        base = (
            f"{bar}  it {it:>4}/{iters_per_epoch:<4}  lr {lr:.6f}  "
            f"loss {loss:.{self.cfg.precision}f}  {dt_ms:.1f} ms/it"
        )
        if self.cfg.show_items and items:
            parts = [f"{k}:{float(v):.{self.cfg.precision}f}" for k, v in sorted(items.items())]
            base += "  [" + ", ".join(parts) + "]"
        if self.cfg.show_vram:
            base += "  |  " + mem_txt

        self._write("\r" + base)
        if it == iters_per_epoch:
            self._writeln("")

    # Renderizado
    def _render_bar(self, progress: float) -> str:
        w = max(10, self.cfg.width)
        fill = int(progress * w)
        return "[" + "#" * fill + "-" * (w - fill) + f"] {progress*100:5.1f}%"

    # Salida
    def _write(self, s: str) -> None:
        try:
            self.cfg.stream.write(s)
            self.cfg.stream.flush()
        except Exception:
            pass

    def _writeln(self, s: str) -> None:
        self._write(s + "\n")


# -------------------------------
# Prueba mínima
# -------------------------------
if __name__ == "__main__":  # pragma: no cover
    import random

    hud = HUD(HUDConfig(width=40, interval=0.1, min_update_s=0.0))
    epochs, iters = 1, 50
    hud.on_epoch_start(0, epochs, iters)
    t0 = time.perf_counter()
    for i in range(1, iters + 1):
        time.sleep(0.02)
        lr = 0.002 * (1 - i / iters)
        loss = 1.0 / (i + 1)
        items = {"box": loss * 0.6, "cls": loss * 0.3, "dfl": loss * 0.1}
        dt_ms = (time.perf_counter() - t0) * 1000.0
        hud.update(0, i, iters, lr, loss, items, dt_ms)
        t0 = time.perf_counter()
    hud.on_epoch_end()
