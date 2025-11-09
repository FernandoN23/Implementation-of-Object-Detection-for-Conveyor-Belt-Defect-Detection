# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/warmup_sanity.py
# Descripción: Rutinas de calentamiento (warm-up) y verificación
#              de forward para inicializar kernels HIP/ROCm,
#              validar shapes/targets y medir latencias/memoria.
#==============================================================

from __future__ import annotations

import argparse
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import torch.cuda.amp as amp
except Exception:  # CPU-only fallback
    class _Dummy:
        def __getattr__(self, name):
            raise AttributeError("AMP no disponible en esta plataforma")
    amp = _Dummy()  # type: ignore

__all__ = [
    "WarmupConfig",
    "make_dummy_batch",
    "adjust_imgsz_to_stride",
    "warmup_sanity",
]

# -------------------------------
# Configuración
# -------------------------------

@dataclass
class WarmupConfig:
    imgsz: int = 640
    bs: int = 4
    nc: int = 5  # número de clases
    amp: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda:0" | "hip:0"
    channels: int = 3
    stride: int = 32  # si el modelo expone .stride se ajusta automáticamente
    iters: int = 10   # número de iteraciones de warm-up (>=2 recomendado)
    compile: bool = False
    dtype: str = "fp16"  # "fp16" | "bf16" | "fp32"
    verbose: int = 1

    def autocast_dtype(self):
        if self.dtype == "bf16":
            return torch.bfloat16
        if self.dtype == "fp16":
            return torch.float16
        return torch.float32


# -------------------------------
# Utilidades
# -------------------------------

def _log(msg: str, cfg: Optional[WarmupConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[warmup] {msg}")


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
            return torch.device("hip")
        return torch.device("cpu")
    return torch.device(spec)


def _device_mem_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    if torch.cuda.is_available():
        res = torch.cuda.memory_reserved()
        alloc = torch.cuda.memory_allocated()
        max_alloc = torch.cuda.max_memory_allocated()
        snap.update({"reserved": res, "allocated": alloc, "max_allocated": max_alloc})
    return snap


@contextmanager
def _nvtx_range(name: str):
    # Placeholder para futuros marcadores; en ROCm/NVTX no siempre disponible
    yield


# -------------------------------
# Preparación de batch sintético
# -------------------------------

def adjust_imgsz_to_stride(imgsz: int, stride: int) -> int:
    """Ajusta imgsz al múltiplo superior de stride (estilo Ultralytics)."""
    if stride <= 0:
        return imgsz
    return int(math.ceil(imgsz / stride) * stride)


def _rand_boxes(n: int, w: int, h: int) -> torch.Tensor:
    """Genera cajas [cls, x, y, w, h] en formato normalizado (0-1)."""
    if n == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    cls = torch.randint(low=0, high=80, size=(n, 1), dtype=torch.int64)
    xywh = torch.rand((n, 4), dtype=torch.float32)
    # aseguremos cajas razonables (w,h > 0.01, <= 0.5)
    xywh[:, 2:] = 0.01 + 0.49 * xywh[:, 2:]
    return torch.cat([cls.to(torch.float32), xywh], dim=1)


def make_dummy_batch(cfg: WarmupConfig, device: torch.device) -> Dict[str, Any]:
    bs = cfg.bs
    imgsz = cfg.imgsz
    C = cfg.channels
    images = torch.rand((bs, C, imgsz, imgsz), dtype=torch.float32, device=device)
    # targets: lista por imagen o tensor agregado; mantenemos dict estilo Ultralytics
    targets: List[torch.Tensor] = []
    for _ in range(bs):
        n = int(torch.randint(0, 8, (1,)).item())  # 0..7 objetos
        t = _rand_boxes(n, imgsz, imgsz)
        targets.append(t.to(device))
    batch = {"img": images, "targets": targets}
    return batch


# -------------------------------
# Calentamiento principal
# -------------------------------

def warmup_sanity(model: torch.nn.Module,
                  device: torch.device | str = "auto",
                  cfg: Optional[WarmupConfig] = None) -> Dict[str, Any]:
    """Ejecuta warm-up y sanity forward.

    Retorna un dict con tiempos promedio, memoria y shape de salida si existe.
    El modelo debe seguir el contrato: model(batch) -> (loss, items_dict).
    """
    cfg = cfg or WarmupConfig()
    dev = _select_device(device if isinstance(device, str) else str(device))

    # Ajuste de imgsz según stride del modelo (si existe)
    stride = getattr(model, "stride", None)
    if isinstance(stride, (tuple, list)) and len(stride) > 0:
        stride = int(max(stride))
    elif isinstance(stride, int):
        stride = max(1, stride)
    else:
        stride = cfg.stride
    cfg.imgsz = adjust_imgsz_to_stride(cfg.imgsz, stride)
    _log(f"imgsz ajustado a múltiplo de stride={stride}: {cfg.imgsz}", cfg, 1)

    model.to(dev)
    model.train()  # durante warm-up queremos pasar por los caminos de train (BN/GN, Dropout, etc.)

    # torch.compile opcional
    if cfg.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            _log("Modelo compilado con torch.compile()", cfg, 1)
        except Exception as e:
            _log(f"torch.compile() no disponible/compatible: {e}", cfg, 1)

    # Preparar lote sintético
    batch = make_dummy_batch(cfg, dev)

    # Selección de autocast dtype
    ac_dtype = cfg.autocast_dtype()

    # Sincronización inicial y marcador de memoria
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Preforca caminos: 2 iteraciones con AMP si corresponde
    times: List[float] = []
    iters = max(int(cfg.iters), 2)

    _log(f"Iniciando warm-up por {iters} iteraciones | AMP={cfg.amp} dtype={ac_dtype}", cfg, 1)

    for i in range(iters):
        t0 = time.perf_counter()
        with _nvtx_range(f"warmup_iter_{i}"):
            if cfg.amp:
                with amp.autocast(enabled=True, dtype=ac_dtype):  # type: ignore
                    out = model(batch)
            else:
                out = model(batch)
        # Asegurar sincronización para medir correctamente la primera compilación HIP/JIT
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)

    # Medición final de memoria (si aplica)
    mem = _device_mem_snapshot()

    # Resumen
    avg_ms = sum(times[1:]) / max(1, (len(times) - 1))  # descartar la primera (compilación)
    first_ms = times[0]
    result = {
        "config": asdict(cfg),
        "timings_ms": {
            "first_iter": round(first_ms, 3),
            "avg_rest": round(avg_ms, 3),
            "all": [round(x, 3) for x in times],
        },
        "memory": mem,
        "imgsz": cfg.imgsz,
        "stride": stride,
    }

    _log(
        f"Warm-up OK | first={result['timings_ms']['first_iter']} ms | "
        f"avg_rest={result['timings_ms']['avg_rest']} ms | mem={mem}",
        cfg,
        1,
    )

    return result


# -------------------------------
# CLI mínima
# -------------------------------

class _Toy(torch.nn.Module):  # pragma: no cover
    def __init__(self, nc: int = 5):
        super().__init__()
        self.stride = 32
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.head = torch.nn.Conv2d(32, 16, 1)
        self.nc = nc

    def forward(self, batch: Dict[str, Any]):
        x = batch["img"]
        x = self.backbone(x)
        x = self.head(x)
        # Simular pérdida: media del tensor + pequeña penalización por boxes
        loss = x.mean()
        items = {"box": float(loss.item()), "cls": 0.0, "dfl": 0.0}
        return loss, items


def _build_argparser() -> argparse.ArgumentParser:  # pragma: no cover
    ap = argparse.ArgumentParser("engine.warmup_sanity")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--nc", type=int, default=5)
    ap.add_argument("--amp", type=str, default="true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--compile", type=str, default="false")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    return ap


def _parse_bool(v: str) -> bool:  # pragma: no cover
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    ap = _build_argparser()
    ns = ap.parse_args(argv)
    cfg = WarmupConfig(
        imgsz=int(ns.imgsz), bs=int(ns.bs), nc=int(ns.nc), amp=_parse_bool(ns.amp),
        device=str(ns.device), iters=int(ns.iters), compile=_parse_bool(ns.compile),
        dtype=str(ns.dtype), verbose=int(ns.verbose)
    )
    model = _Toy(nc=cfg.nc)
    out = warmup_sanity(model, cfg.device, cfg)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
