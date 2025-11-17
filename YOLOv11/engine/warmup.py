# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/warmup.py
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
from torch import amp as torch_amp

__all__ = [
    "WarmupConfig",
    "make_dummy_batch",
    "adjust_imgsz_to_stride",
    "warmup_sanity",
    "build_warmup_config_from_train",
    "run_trainer_warmup",
]

# -------------------------------
# Configuración
# -------------------------------


@dataclass
class WarmupConfig:
    """Configuración genérica de warm-up sintético.

    Esta estructura se pensó como bloque único de configuración para cualquier
    cliente (scripts standalone, pruebas y la clase Trainer). La idea es que
    Trainer no implemente lógica propia de warmup, sino que derive siempre en
    este módulo.
    """

    imgsz: int = 640
    bs: int = 4
    nc: int = 5  # número de clases
    amp: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda:0" | "hip:0"
    channels: int = 3
    stride: int = 32  # si el modelo expone .stride se ajusta automáticamente
    iters: int = 10  # número de iteraciones de warm-up (>=2 recomendado)
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
# Utilidades internas
# -------------------------------


def _log(msg: str, cfg: Optional[WarmupConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[WARMUP] {msg}")


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        # Orden preferente: ROCm/CUDA si está disponible, luego CPU
        if hasattr(torch, "hip") and getattr(torch, "hip", None) is not None:  # type: ignore[attr-defined]
            try:
                if torch.hip.is_available():  # type: ignore[attr-defined]
                    return torch.device("hip")
            except Exception:
                pass
        if torch.cuda.is_available():
            return torch.device("cuda")
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
    """Ajusta imgsz al múltiplo superior de stride (estilo Ultralytics).

    Equivalente a la lógica clásica: ceil(imgsz / stride) * stride.
    Si imgsz ya es múltiplo de stride (p.ej. 640 con stride=32), se mantiene.
    """

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
    """Construye un batch sintético compatible con el pipeline YOLOv11.

    Estructura devuelta: {"img": Tensor[B, C, H, W], "targets": List[Tensor[N_i, 5]]}
    donde cada fila es [cls, cx, cy, w, h] en coordenadas normalizadas (0-1).

    El modelo sólo consume `img` (Tensor); `targets` se mantienen para posibles
    extensiones futuras (pérdidas sintéticas, validaciones adicionales, etc.).
    """

    bs = cfg.bs
    imgsz = cfg.imgsz
    C = cfg.channels
    images = torch.rand((bs, C, imgsz, imgsz), dtype=torch.float32, device=device)
    targets: List[torch.Tensor] = []
    for _ in range(bs):
        n = int(torch.randint(0, 8, (1,)).item())  # 0..7 objetos
        t = _rand_boxes(n, imgsz, imgsz)
        targets.append(t.to(device))
    batch = {"img": images, "targets": targets}
    return batch


# -------------------------------
# Calentamiento principal genérico
# -------------------------------


def warmup_sanity(
    model: torch.nn.Module,
    device: torch.device | str = "auto",
    cfg: Optional[WarmupConfig] = None,
) -> Dict[str, Any]:
    """Ejecuta warm-up sintético y sanity forward.

    Retorna un dict con tiempos promedio, memoria y configuración efectiva.

    Contrato del modelo para este warmup:
    - Se asume que `model(x)` recibe un tensor de imágenes de forma
      `[B, C, H, W]`, coherente con el uso en `Trainer.fit()`.

    Esta función es agnóstica a HUD/Trainer: encapsula únicamente la lógica de
    batch sintético + medición. Trainer debería delegar aquí el trabajo pesado
    y sólo encargarse de la orquestación (mensajes [WARMUP], HUD, etc.).
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
    # Durante warm-up queremos pasar por los caminos de train (GN, Dropout, etc.)
    model.train()

    # torch.compile opcional
    if cfg.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            _log("Modelo compilado con torch.compile()", cfg, 1)
        except Exception as e:
            _log(f"torch.compile() no disponible/compatible: {e}", cfg, 1)

    # Preparar lote sintético
    batch = make_dummy_batch(cfg, dev)
    images = batch["img"]  # El modelo sólo consume el tensor de imágenes

    # Selección de autocast dtype
    ac_dtype = cfg.autocast_dtype()

    # Sincronización inicial y marcador de memoria
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Preforca caminos: >=2 iteraciones para capturar bien t_first vs resto
    times: List[float] = []
    iters = max(int(cfg.iters), 2)

    _log(
        f"Iniciando warm-up por {iters} iteraciones | AMP={cfg.amp} dtype={ac_dtype}",
        cfg,
        1,
    )

    # Selección de backend para autocast (cuda/rocm vs cpu)
    use_cuda_amp = torch.cuda.is_available()
    device_type = "cuda" if use_cuda_amp else "cpu"

    for i in range(iters):
        t0 = time.perf_counter()
        with _nvtx_range(f"warmup_iter_{i}"):
            if cfg.amp:
                # API moderna: torch.amp.autocast en lugar de torch.cuda.amp.autocast
                with torch_amp.autocast(device_type=device_type, dtype=ac_dtype):  # type: ignore[arg-type]
                    _ = model(images)
            else:
                _ = model(images)
        # Asegurar sincronización para medir correctamente la primera compilación HIP/JIT
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)

    # Medición final de memoria (si aplica)
    mem = _device_mem_snapshot()

    # Resumen (descartando la primera iteración, típicamente dominada por compilación)
    avg_ms = sum(times[1:]) / max(1, (len(times) - 1))
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
# Helpers para integración con Trainer
# -------------------------------


def build_warmup_config_from_train(
    train_cfg: Any,
    *,
    device: Optional[str] = None,
    iters: Optional[int] = None,
    verbose: Optional[int] = None,
) -> WarmupConfig:
    """Construye un ``WarmupConfig`` a partir de la configuración de entrenamiento.

    Este helper permite que ``Trainer`` derive toda la config de warmup desde su
    propio objeto de configuración (por ejemplo, campos leídos desde
    ``train.yaml`` e hydra/argparse), sin replicar lógica de warmup en
    ``Trainer``.
    """

    cfg = WarmupConfig()

    # Campos básicos: imgsz, batch, nc, amp, compile, dtype
    imgsz_attr = getattr(train_cfg, "imgsz", None)
    if imgsz_attr is not None:
        cfg.imgsz = int(imgsz_attr)

    batch_attr = getattr(train_cfg, "batch", None)
    if batch_attr is not None:
        cfg.bs = int(batch_attr)

    nc_attr = getattr(train_cfg, "nc", None)
    if nc_attr is not None:
        cfg.nc = int(nc_attr)

    amp_attr = getattr(train_cfg, "amp", None)
    if amp_attr is not None:
        cfg.amp = bool(amp_attr)

    compile_attr = getattr(train_cfg, "compile", None)
    if compile_attr is not None:
        cfg.compile = bool(compile_attr)

    amp_dtype = getattr(train_cfg, "amp_dtype", None)
    if isinstance(amp_dtype, str) and amp_dtype in {"fp16", "bf16", "fp32"}:
        cfg.dtype = amp_dtype

    # Dispositivo
    if device is not None:
        cfg.device = device
    else:
        dev_attr = getattr(train_cfg, "device", None)
        if isinstance(dev_attr, str) and dev_attr:
            cfg.device = dev_attr

    # Número de iteraciones (permite distinguir warmup corto vs intenso)
    if iters is not None:
        cfg.iters = max(int(iters), 2)
    else:
        warmup_iters = getattr(train_cfg, "warmup_iters", None)
        if isinstance(warmup_iters, int) and warmup_iters > 0:
            cfg.iters = max(warmup_iters, 2)

    # Verbosidad
    if verbose is not None:
        cfg.verbose = int(verbose)
    else:
        v_attr = getattr(train_cfg, "verbose", None)
        if isinstance(v_attr, int):
            cfg.verbose = v_attr

    return cfg


def run_trainer_warmup(
    model: torch.nn.Module,
    train_cfg: Any,
    *,
    device: torch.device | str = "auto",
    warmup_cfg: Optional[WarmupConfig] = None,
) -> Dict[str, Any]:
    """Entry-point pensado para ser llamado desde ``Trainer``.

    * No imprime mensajes estilo ``[WARMUP]`` ni maneja HUD.
    * Se limita a:
      1) Construir un ``WarmupConfig`` consistente con la config de entrenamiento.
      2) Ejecutar ``warmup_sanity`` y devolver su resumen.

    El formateo de mensajes (``[WARMUP] >>> Inicio warmup``, barras de progreso,
    integración con HUD, etc.) debe vivir en ``Trainer``. De este modo, todo el
    trabajo "real" del warmup queda contenido en este módulo.
    """

    if warmup_cfg is None:
        dev_str = device if isinstance(device, str) else str(device)
        warmup_cfg = build_warmup_config_from_train(train_cfg, device=dev_str)
    return warmup_sanity(model, device=device, cfg=warmup_cfg)


# -------------------------------
# CLI mínima para pruebas aisladas
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

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.head(x)
        # Simular pérdida: media del tensor + pequeña penalización genérica
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
    ap.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"]
    )
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    return ap


def _parse_bool(v: str) -> bool:  # pragma: no cover
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    ap = _build_argparser()
    ns = ap.parse_args(argv)
    cfg = WarmupConfig(
        imgsz=int(ns.imgsz),
        bs=int(ns.bs),
        nc=int(ns.nc),
        amp=_parse_bool(ns.amp),
        device=str(ns.device),
        iters=int(ns.iters),
        compile=_parse_bool(ns.compile),
        dtype=str(ns.dtype),
        verbose=int(ns.verbose),
    )
    model = _Toy(nc=cfg.nc)
    out = warmup_sanity(model, cfg.device, cfg)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
