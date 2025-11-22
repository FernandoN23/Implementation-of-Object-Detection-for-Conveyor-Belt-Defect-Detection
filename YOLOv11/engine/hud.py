# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/hud.py
# Descripción: HUD unificado para consola con telemetría de warmup y
#              entrenamiento: progreso, LR, pérdidas, tiempos (ms/it,
#              it/s, ETA), VRAM (alloc/reserved/peak y % global) y
#              cabeceras de fase. Incluye throttle y compatibilidad DDP.
#==============================================================

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Optional, List

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


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or not (seconds < 1e7):
        return "--:--"
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _device_mem() -> Dict[str, float]:
    """Lee memoria del dispositivo (CUDA/HIP) y % de uso global si está disponible.

    Retorna:
        dict con claves: reserved, allocated, max_allocated (bytes), total (bytes, 0 si no disponible),
        used_pct (0–100, -1 si no disponible).
    """
    out = {
        "reserved": 0.0,
        "allocated": 0.0,
        "max_allocated": 0.0,
        "total": 0.0,
        "used_pct": -1.0,
    }
    try:
        if torch.cuda.is_available():
            out["reserved"] = float(torch.cuda.memory_reserved())
            out["allocated"] = float(torch.cuda.memory_allocated())
            out["max_allocated"] = float(torch.cuda.max_memory_allocated())
            # mem_get_info existe en CUDA y ROCm modernos
            try:
                free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                out["total"] = float(total)
                used_pct = 100.0 * (1.0 - float(free) / float(total)) if total > 0 else -1.0
                out["used_pct"] = used_pct
            except Exception:
                pass
        elif hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
            # En ROCm Windows Preview, la API cuda.* apunta a HIP
            out["reserved"] = float(torch.cuda.memory_reserved())  # type: ignore[attr-defined]
            out["allocated"] = float(torch.cuda.memory_allocated())  # type: ignore[attr-defined]
            out["max_allocated"] = float(torch.cuda.max_memory_allocated())  # type: ignore[attr-defined]
            try:
                free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                out["total"] = float(total)
                used_pct = 100.0 * (1.0 - float(free) / float(total)) if total > 0 else -1.0
                out["used_pct"] = used_pct
            except Exception:
                pass
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
    # Visual
    width: int = 20                  # ancho de barra (caracteres internos del bloque)
    interval: float = 0.05           # fracción de época/iter entre updates
    min_update_s: float = 0.25       # límite temporal mínimo entre updates
    precision: int = 4
    stream: object = sys.stdout
    enable: bool = True

    # Qué mostrar
    show_vram: bool = True
    show_items: bool = True          # desglose de pérdidas
    show_rate: bool = True           # it/s y samples/s
    show_eta: bool = False           # ETA por época (train) → opcional para no ensanchar demasiado

    # Contexto opcional
    batch: Optional[int] = None      # para calcular samples/s
    phase_label_train: str = "TRAIN"
    phase_label_warmup: str = "WARMUP"


# -------------------------------
# HUD
# -------------------------------

class HUD:
    """HUD unificado para warmup y entrenamiento.

    Modo "full" (compacto dinámico) para consola, pensado para caber en una sola
    línea y actualizarse in-place con `\r`, por ejemplo:

        [TRN] ▕████████████░░░░░░▏ 405/406 (99.8%) | lr 3.1e-05 | L 20.96 |
        380ms | 2.8 it/s | b:1.71 c:0.23 d:5.36 | VRAM 286M|1.65G

    Uso típico (train):
        hud = HUD(HUDConfig(interval=0.05))
        hud.on_epoch_start(epoch, epochs, iters_per_epoch)
        for i, batch in enumerate(loader, 1):
            hud.update(epoch, i, iters_per_epoch, lr, loss, items, dt_ms)
        hud.on_epoch_end()

    Uso típico (warmup):
        hud.on_warmup_start(total_iters, dtype="fp16", compile=False, stride=32,
                            bn2gn="on", amp=True, find_mode="FAST", cache_disabled=True)
        for i in range(1, total_iters+1):
            hud.update_warmup(i, total_iters, dt_ms)
        hud.on_warmup_end()
    """

    def __init__(self, cfg: Optional[HUDConfig] = None) -> None:
        self.cfg = cfg or HUDConfig()
        self._last_print_t = 0.0
        self._last_progress_bucket = -1
        self._epoch = 0
        self._epochs = 0
        self._iters_per_epoch = 0
        # Ventana de tiempos (ms) para ETA y estadísticas
        self._dt_window: deque[float] = deque(maxlen=50)
        # Últimos valores observados en entrenamiento
        self._last_lr: float = 0.0
        self._last_loss: float = 0.0
        self._last_items: Dict[str, float] = {}
        self._last_it_per_s: float = 0.0
        self._last_smp_per_s: float = 0.0
        # Warmup context
        self._wu_total = 0
        self._wu_times: List[float] = []  # ms por iter
        self._wu_ctx: Dict[str, object] = {}

    # ---------------------------
    # Entrenamiento (épocas)
    # ---------------------------
    def on_epoch_start(self, epoch: int, epochs: int, iters_per_epoch: int) -> None:
        if not self.cfg.enable or not _is_main_process():
            return
        self._epoch = epoch
        self._epochs = epochs
        self._iters_per_epoch = max(1, iters_per_epoch)
        self._dt_window.clear()
        self._last_print_t = 0.0
        self._last_progress_bucket = -1
        self._last_lr = 0.0
        self._last_loss = 0.0
        self._last_items.clear()
        self._last_it_per_s = 0.0
        self._last_smp_per_s = 0.0
        self._writeln(f"[{self.cfg.phase_label_train:>5}] Epoch {epoch+1}/{epochs}")

    def on_epoch_end(self) -> None:
        if not self.cfg.enable or not _is_main_process():
            return
        self._writeln("")  # salto de línea para separar épocas

    def update(
        self,
        epoch: int,
        it: int,
        iters_per_epoch: int,
        lr: float,
        loss: float,
        items: Optional[Dict[str, float]],
        dt_ms: float,
    ) -> None:
        if not self.cfg.enable or not _is_main_process():
            return

        now = time.perf_counter()
        progress = min(0.9999, max(0.0, it / float(max(1, iters_per_epoch))))
        bucket = int(progress / max(1e-6, self.cfg.interval))
        if bucket == self._last_progress_bucket and (now - self._last_print_t) < self.cfg.min_update_s:
            return
        self._last_progress_bucket = bucket
        self._last_print_t = now

        # Estadísticas de tiempo
        self._dt_window.append(max(1e-6, float(dt_ms)))
        mean_dt = mean(self._dt_window) if self._dt_window else float(dt_ms)
        it_per_s = 1000.0 / mean_dt
        smp_per_s = it_per_s * (self.cfg.batch or 0)
        eta_s = (iters_per_epoch - it) * (mean_dt / 1000.0)

        # Guardar últimos valores observados para resumen de época
        self._last_lr = float(lr)
        self._last_loss = float(loss)
        self._last_items = dict(items or {})
        self._last_it_per_s = float(it_per_s)
        self._last_smp_per_s = float(smp_per_s)

        # Barra + progreso
        phase = self.cfg.phase_label_train.upper()
        bar = self._render_bar(progress)
        pct = progress * 100.0

        # VRAM (compacto: alloc MB | peak GB)
        vram = _device_mem() if self.cfg.show_vram else {
            "reserved": 0.0,
            "allocated": 0.0,
            "max_allocated": 0.0,
            "total": 0.0,
            "used_pct": -1.0,
        }
        alloc_mb = vram["allocated"] / (1024.0 ** 2)
        peak_gb = vram["max_allocated"] / (1024.0 ** 3)
        mem_txt = f"VRAM {alloc_mb:.0f}M|{peak_gb:.2f}G"

        # Métricas de items (compactas: b:1.71 c:0.23 d:5.36)
        items_txt = ""
        if self.cfg.show_items and items:
            alias_map = {"box": "b", "cls": "c", "dfl": "d"}
            parts: List[str] = []
            for k, v in sorted(items.items()):
                alias = alias_map.get(k, (k[:1] if k else "?"))
                parts.append(f"{alias}:{float(v):.2f}")
            items_txt = " ".join(parts)

        # Construcción de línea compacta
        line_parts: List[str] = []
        line_parts.append(f"[{phase}] {bar} {it}/{iters_per_epoch} ({pct:4.1f}%)")
        line_parts.append(f"lr {lr:.2e}")
        #line_parts.append(f"L {loss:.2f}")
        line_parts.append(f"{dt_ms:.0f}ms")

        if self.cfg.show_rate:
            line_parts.append(f"{it_per_s:.1f} it/s")
            if self.cfg.batch:
                line_parts.append(f"{smp_per_s:.1f} samp/s")

        if items_txt:
            line_parts.append(items_txt)

        if self.cfg.show_eta:
            line_parts.append(f"ETA {_format_eta(eta_s)}")

        if self.cfg.show_vram:
            line_parts.append(mem_txt)

        line = " | ".join(line_parts)
        self._write("\r" + line)
        if it == iters_per_epoch:
            self._writeln("")

    def update_epoch(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Resumen compacto de fin de época.

        Pensado para llamarse desde Trainer.fit() una vez calculadas las métricas
        agregadas de entrenamiento y validación. Si no se entregan métricas,
        utiliza los últimos valores observados durante `update()`.
        """
        if not self.cfg.enable or not _is_main_process():
            return

        # Preferimos métricas explícitas, si se entregan
        train_items = dict(train_metrics or {})
        val_items = dict(val_metrics or {})

        # Fallback: usar últimos valores observados en el loop de train
        if not train_items:
            train_items = dict(self._last_items)

        phase = self.cfg.phase_label_train.upper()
        epoch_txt = f"Epoch {epoch + 1}/{self._epochs or '?'}"

        parts: List[str] = [f"[{phase}] {epoch_txt} resumen"]

        # Loss/it/s de entrenamiento
        if self._last_loss > 0:
            parts.append(f"L={self._last_loss:.3f}")
        if self._last_lr > 0:
            parts.append(f"lr={self._last_lr:.2e}")
        if self.cfg.show_rate and self._last_it_per_s > 0:
            txt_rate = f"{self._last_it_per_s:.2f} it/s"
            if self.cfg.batch and self._last_smp_per_s > 0:
                txt_rate += f", {self._last_smp_per_s:.1f} samp/s"
            parts.append(txt_rate)

        # Desglose de pérdidas de entrenamiento
        if train_items and self.cfg.show_items:
            alias_map = {"box": "b", "cls": "c", "dfl": "d"}
            loss_parts: List[str] = []
            for k, v in sorted(train_items.items()):
                alias = alias_map.get(k, (k[:1] if k else "?"))
                try:
                    fv = float(v)
                except Exception:
                    continue
                loss_parts.append(f"{alias}:{fv:.3f}")
            if loss_parts:
                parts.append("train=" + " ".join(loss_parts))

        # Métricas de validación (mAP, etc.) si están disponibles
        if val_items:
            val_parts: List[str] = []
            # Intento de aliasado ligero para mAP
            alias_map_val = {
                "map50": "mAP50",
                "map5095": "mAP50-95",
                "map": "mAP",
            }
            for k, v in sorted(val_items.items()):
                name = alias_map_val.get(k, k)
                try:
                    fv = float(v)
                except Exception:
                    continue
                val_parts.append(f"{name}={fv:.3f}")
            if val_parts:
                parts.append("val=" + " ".join(val_parts))

        # Snapshot de VRAM en el cierre de época
        if self.cfg.show_vram:
            vram = _device_mem()
            alloc = format_bytes(int(vram.get("allocated", 0.0)))
            peak = format_bytes(int(vram.get("max_allocated", 0.0)))
            used_pct = vram.get("used_pct", -1.0)
            vram_txt = f"VRAM {alloc} (peak {peak}"
            if used_pct is not None and used_pct >= 0:
                vram_txt += f", {used_pct:.1f}%"
            vram_txt += ")"
            parts.append(vram_txt)

        # self._writeln(" | ".join(parts))  # epoch summary silenced (console output disabled)

    # ---------------------------
    # Warmup (iteraciones)
    # ---------------------------
    def on_warmup_start(
        self,
        total_iters: int,
        *,
        dtype: str,
        compile: bool,
        stride: int,
        bn2gn: str,
        amp: bool,
        find_mode: Optional[str] = None,
        cache_disabled: Optional[bool] = None,
    ) -> None:
        if not self.cfg.enable or not _is_main_process():
            return
        self._wu_total = max(1, int(total_iters))
        self._wu_times.clear()
        self._wu_ctx = {
            "dtype": str(dtype),
            "compile": bool(compile),
            "stride": int(stride),
            "bn2gn": str(bn2gn),
            "amp": bool(amp),
            "find_mode": (str(find_mode) if find_mode is not None else None),
            "cache_disabled": (bool(cache_disabled) if cache_disabled is not None else None),
        }
        head = (
            f"[{self.cfg.phase_label_warmup:>6}] iters={self._wu_total}  dtype={dtype}  "
            f"compile={bool(compile)}  stride={stride}  bn2gn={bn2gn}  amp={bool(amp)}"
        )
        if find_mode is not None:
            head += f"  miopen.find={find_mode}"
        if cache_disabled is not None:
            head += f"  miopen.cache={'OFF' if cache_disabled else 'ON'}"
        self._writeln(head)
        # Reset de throttle para warmup
        self._last_print_t = 0.0
        self._last_progress_bucket = -1

    def update_warmup(self, iter_idx: int, total_iters: int, dt_ms: float) -> None:
        if not self.cfg.enable or not _is_main_process():
            return

        now = time.perf_counter()
        progress = min(0.9999, max(0.0, iter_idx / float(max(1, total_iters))))
        bucket = int(progress / max(1e-6, self.cfg.interval))

        # Siempre guardamos tiempo para estadísticas finales
        self._wu_times.append(max(1e-6, float(dt_ms)))

        if bucket == self._last_progress_bucket and (now - self._last_print_t) < self.cfg.min_update_s:
            return
        self._last_progress_bucket = bucket
        self._last_print_t = now

        mean_dt = mean(self._wu_times) if self._wu_times else float(dt_ms)
        it_per_s = 1000.0 / (mean_dt if mean_dt > 0 else 1e-6)

        phase = self.cfg.phase_label_warmup.upper()
        bar = self._render_bar(progress)
        pct = progress * 100.0

        vram = _device_mem() if self.cfg.show_vram else {
            "reserved": 0.0,
            "allocated": 0.0,
            "max_allocated": 0.0,
            "total": 0.0,
            "used_pct": -1.0,
        }
        alloc_mb = vram["allocated"] / (1024.0 ** 2)
        peak_gb = vram["max_allocated"] / (1024.0 ** 3)
        mem_txt = f"VRAM {alloc_mb:.0f}M|{peak_gb:.2f}G"

        line_parts: List[str] = []
        line_parts.append(f"[{phase}] {bar} {iter_idx}/{total_iters} ({pct:4.1f}%)")
        line_parts.append(f"{dt_ms:.0f}ms")
        line_parts.append(f"{it_per_s:.1f} it/s")
        if self.cfg.batch:
            line_parts.append(f"{(it_per_s * self.cfg.batch):.1f} samp/s")
        if self.cfg.show_vram:
            line_parts.append(mem_txt)

        line = " | ".join(line_parts)
        self._write("\r" + line)
        if iter_idx == total_iters:
            self._writeln("")

    def on_warmup_end(self) -> None:
        if not self.cfg.enable or not _is_main_process():
            return
        if not self._wu_times:
            self._writeln(f"[{self.cfg.phase_label_warmup:>6}] sin métricas")
            return
        t_first_ms = self._wu_times[0]
        rest = self._wu_times[1:] if len(self._wu_times) > 1 else self._wu_times
        t_mean_ms = mean(rest)
        # p95 simple
        p95_ms = sorted(rest)[max(0, int(0.95 * (len(rest) - 1)))] if rest else t_mean_ms
        vram = _device_mem() if self.cfg.show_vram else {
            "reserved": 0.0,
            "allocated": 0.0,
            "max_allocated": 0.0,
            "total": 0.0,
            "used_pct": -1.0,
        }

        summary = (
            f"[{self.cfg.phase_label_warmup:>6}] t_first={t_first_ms/1000.0:.3f} s  "
            f"t_mean={t_mean_ms:.1f} ms  p95={p95_ms:.1f} ms"
        )
        if self.cfg.show_vram:
            summary += (
                f"  |  VRAM {format_bytes(int(vram['allocated']))} / {format_bytes(int(vram['reserved']))} "
                f"(peak {format_bytes(int(vram['max_allocated']))}"
            )
            if vram.get("used_pct", -1.0) >= 0:
                summary += f", {vram['used_pct']:.1f}%"
            summary += ")"
        self._writeln(summary)

    # ---------------------------
    # Renderizado / Salida
    # ---------------------------
    def _render_bar(self, progress: float) -> str:
        """Barra de progreso compacta con bloques Unicode.

        Ejemplo para width=16:
            ▕████████████░░░░░░▏
        """
        w = max(4, int(self.cfg.width))
        fill = int(progress * w + 0.5)
        fill = max(0, min(w, fill))
        empty = w - fill
        bar_inner = "█" * fill + "░" * empty
        return "▕" + bar_inner + "▏"

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

    hud = HUD(HUDConfig(width=16, interval=0.1, min_update_s=0.0, batch=4))

    # Warmup demo
    hud.on_warmup_start(3, dtype="fp16", compile=False, stride=32, bn2gn="on", amp=True, find_mode="FAST", cache_disabled=True)
    t = time.perf_counter()
    for i in range(1, 4):
        time.sleep(0.05)
        dt_ms = (time.perf_counter() - t) * 1000.0
        hud.update_warmup(i, 3, dt_ms)
        t = time.perf_counter()
    hud.on_warmup_end()

    # Train demo
    epochs, iters = 1, 20
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
    hud.update_epoch(0)
    hud.on_epoch_end()
