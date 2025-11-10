# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/amp.py
# Descripción: Gestión de AMP (autocast + GradScaler) con utilidades
#              de backward/step seguros, detección de overflow y
#              fallback opcional. Diseñado para integrarse con Trainer.
#==============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = [
    "AmpConfig",
    "build_grad_scaler",
    "autocast_ctx",
    "AmpManager",
    "safe_backward_step",
]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class AmpConfig:
    enabled: bool = True            # habilitar AMP
    dtype: str = "fp16"             # "fp16" | "bf16"
    init_scale: float = 65536.0     # escala inicial del GradScaler
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    hysteresis: int = 2             # (no usado por torch.amp)
    detect_anomaly: bool = False    # habilitar detect_anomaly() de autograd (costoso)
    verbose: int = 1

    def torch_dtype(self):
        return torch.bfloat16 if self.dtype == "bf16" else torch.float16

    def device_type(self) -> str:
        # En ROCm (Windows Preview) torch.cuda.is_available() suele ser True
        return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# Logging local
# -------------------------------

def _log(msg: str, cfg: Optional[AmpConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[amp] {msg}")


# -------------------------------
# Construcción de GradScaler y autocast (API unificada torch.amp)
# -------------------------------

def build_grad_scaler(cfg: AmpConfig) -> Optional[Any]:
    """Crea GradScaler usando torch.amp. En bf16 no se crea scaler."""
    if not cfg.enabled:
        return None
    # En bf16 no es necesario GradScaler
    if str(cfg.dtype).lower() != "fp16":
        _log("GradScaler omitido (dtype != fp16)", cfg, 2)
        return None
    try:
        scaler = torch.amp.GradScaler(
            cfg.device_type(),
            init_scale=cfg.init_scale,
            growth_factor=cfg.growth_factor,
            backoff_factor=cfg.backoff_factor,
            growth_interval=cfg.growth_interval,
            enabled=True,
        )
        _log(f"GradScaler creado | init_scale={cfg.init_scale}", cfg, 1)
        return scaler
    except Exception as e:
        _log(f"GradScaler no disponible: {e}", cfg, 1)
        return None


class autocast_ctx:
    """Context manager unificado para autocast con dtype configurable."""

    def __init__(self, cfg: AmpConfig):
        self.cfg = cfg
        self._ctx = None

    def __enter__(self):
        if not self.cfg.enabled:
            return None
        try:
            self._ctx = torch.amp.autocast(
                self.cfg.device_type(),
                dtype=self.cfg.torch_dtype(),
                enabled=True,
            )
            return self._ctx.__enter__()
        except Exception:
            # fallback silencioso
            self._ctx = None
            return None

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


# -------------------------------
# Gestor de AMP para Trainer
# -------------------------------

class AmpManager:
    """Encapsula GradScaler + autocast y utilidades de step/overflow.

    Uso:
        amp = AmpManager(AmpConfig())
        with amp.autocast():
            loss, items = model(batch)
        amp.backward(loss)
        if need_step:
            amp.step(optimizer, model, clip_fn=lambda: clip_gradients(model, 10.0))
    """

    def __init__(self, cfg: AmpConfig) -> None:
        self.cfg = cfg
        if self.cfg.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        self.scaler = build_grad_scaler(cfg)
        self._last_overflow: bool = False

    # Context manager de autocast
    def autocast(self):
        return autocast_ctx(self.cfg)

    # Backward (con o sin scaler)
    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is None:
            loss.backward()
            return
        self.scaler.scale(loss).backward()

    # Unscale + clip + step + update
    @torch.no_grad()
    def step(self,
             optimizer: torch.optim.Optimizer,
             model: Optional[nn.Module] = None,
             *,
             clip_fn: Optional[Callable[[], float]] = None,
             zero_grad: bool = True,
             set_to_none: bool = True) -> Dict[str, Any]:
        info: Dict[str, Any] = {"overflow": False, "clipped": 0.0}

        if self.scaler is None:
            if clip_fn is not None:
                info["clipped"] = float(clip_fn())
            optimizer.step()
            if zero_grad:
                optimizer.zero_grad(set_to_none=set_to_none)
            return info

        self.scaler.unscale_(optimizer)
        if clip_fn is not None:
            info["clipped"] = float(clip_fn())

        try:
            self.scaler.step(optimizer)
            self.scaler.update()
        except Exception as e:
            _log(f"Excepción durante scaler.step(): {e}", self.cfg, 1)
            info["overflow"] = True
        finally:
            if zero_grad:
                optimizer.zero_grad(set_to_none=set_to_none)

        # Detección de overflow básica
        self._last_overflow = False
        if self.scaler is not None and hasattr(self.scaler, "get_scale"):
            try:
                scale_val = float(self.scaler.get_scale())
                info["scale"] = scale_val
                self._last_overflow = scale_val < 1.0
            except Exception:
                pass
        info["overflow"] = info.get("overflow", False) or self._last_overflow
        return info

    def last_overflow(self) -> bool:
        return bool(self._last_overflow)


# -------------------------------
# Función utilitaria compacta
# -------------------------------

def safe_backward_step(loss: torch.Tensor,
                       optimizer: torch.optim.Optimizer,
                       amp: Optional[AmpManager] = None,
                       *,
                       clip_fn: Optional[Callable[[], float]] = None,
                       zero_grad: bool = True,
                       set_to_none: bool = True) -> Dict[str, Any]:
    """Realiza backward y step con o sin AMP de forma uniforme.

    Devuelve dict con {"overflow": bool, "clipped": float, "scale": float?}.
    """
    if amp is None or amp.scaler is None:
        loss.backward()
        clipped = float(clip_fn()) if clip_fn is not None else 0.0
        optimizer.step()
        if zero_grad:
            optimizer.zero_grad(set_to_none=set_to_none)
        return {"overflow": False, "clipped": clipped}

    amp.backward(loss)
    return amp.step(optimizer, clip_fn=clip_fn, zero_grad=zero_grad, set_to_none=set_to_none)


# -------------------------------
# Prueba mínima
# -------------------------------
if __name__ == "__main__":  # pragma: no cover
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = nn.Linear(16, 4)
        def forward(self, x):
            y = self.m(x)
            return y.mean(), {"toy": float(y.mean().item())}

    cfg = AmpConfig(enabled=True, dtype="fp16", verbose=2)
    amp = AmpManager(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Toy().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    for i in range(5):
        x = torch.randn(32, 16, device=dev)
        with amp.autocast():
            loss, _ = net(x)
        stats = safe_backward_step(loss, opt, amp, clip_fn=None)
        print("step", i, stats)
