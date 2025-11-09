# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/ema.py
# Descripción: Implementación de Exponential Moving Average (EMA)
#              con sombras en float32, decaimiento dinámico (τ) y
#              utilidades de copia/sincronización para validación.
#==============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "EMAConfig",
    "ModelEMA",
]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class EMAConfig:
    """Parámetros de configuración para ModelEMA.

    Atributos
    ---------
    tau: int
        Constante de tiempo (en iteraciones) para decaimiento dinámico.
        decay = 1 - exp(-t/τ), implementado aquí como: decay = 1 - 1/(t/τ + 1).
    warmup_iters: int
        Iteraciones de warm-up donde se fuerza mayor arrastre del modelo (decay menor).
    force_float32: bool
        Mantener las sombras en float32 para estabilidad numérica.
    pin_buffers: bool
        Si True, copia buffers (running_mean/var, etc.) en copy_to(). Útil si BN está activo
        o si se sustituyó por GN pero se quiere portabilidad completa.
    device: Optional[str]
        Dispositivo para almacenar las sombras. Si None, usa el dispositivo de cada parámetro.
    verbose: int
        0 silencioso, 1 info, 2 detalle.
    """

    tau: int = 4000
    warmup_iters: int = 0
    force_float32: bool = True
    pin_buffers: bool = True
    device: Optional[str] = None
    verbose: int = 1


# -------------------------------
# Utilidades internas
# -------------------------------

def _log(msg: str, cfg: Optional[EMAConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[ema] {msg}")


def _iter_model_params(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for n, p in model.named_parameters():
        if p.requires_grad:
            yield n, p


def _iter_model_buffers(model: nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    for n, b in model.named_buffers():
        yield n, b


# -------------------------------
# Clase principal
# -------------------------------

class ModelEMA:
    """Mantenedor de sombras EMA para parámetros (y opcionalmente buffers).

    Uso típico
    ----------
    ema = ModelEMA(model, EMAConfig(tau=4000))
    # durante el entrenamiento, tras cada optimizer.step():
    ema.update(model)
    # para validar:
    ema.copy_to(model_eval)
    """

    def __init__(self, model: nn.Module, cfg: Optional[EMAConfig] = None) -> None:
        self.cfg = cfg or EMAConfig()
        self.decay: float = 0.0
        self.updates: int = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}

        self._register(model)
        _log(
            f"EMA inicializada | params={len(self.shadow)} | buffers={len(self.shadow_buffers)} | cfg={asdict(self.cfg)}",
            self.cfg,
            1,
        )

    # ---------------------------
    # Registro y estado
    # ---------------------------
    def _register(self, model: nn.Module) -> None:
        device = torch.device(self.cfg.device) if self.cfg.device else None
        dtype = torch.float32 if self.cfg.force_float32 else None

        # Parámetros
        for n, p in _iter_model_params(model):
            t = p.detach()
            if dtype is not None:
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            self.shadow[n] = t.clone()

        # Buffers (opcional)
        if self.cfg.pin_buffers:
            for n, b in _iter_model_buffers(model):
                tb = b.detach()
                if dtype is not None and tb.is_floating_point():
                    tb = tb.to(dtype)
                if device is not None:
                    tb = tb.to(device)
                self.shadow_buffers[n] = tb.clone()

    def state_dict(self) -> Dict[str, object]:
        return {
            "cfg": asdict(self.cfg),
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "shadow_buffers": {k: v.clone() for k, v in self.shadow_buffers.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state: Dict[str, object], *, strict: bool = False) -> None:
        self.decay = float(state.get("decay", 0.0))
        self.updates = int(state.get("updates", 0))
        shadow = state.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {k: v.clone() for k, v in shadow.items()}  # type: ignore[assignment]
        shadow_buffers = state.get("shadow_buffers", {})
        if isinstance(shadow_buffers, dict):
            self.shadow_buffers = {k: v.clone() for k, v in shadow_buffers.items()}  # type: ignore[assignment]
        # cfg puede no coincidir exactamente; no reemplazamos la actual salvo que strict
        if strict and "cfg" in state:
            cfg_dict = state["cfg"]  # type: ignore[index]
            if isinstance(cfg_dict, dict):
                self.cfg = EMAConfig(**cfg_dict)

    # ---------------------------
    # Decaimiento y actualización
    # ---------------------------
    def _compute_decay(self, updates: int) -> float:
        # decaimiento dinámico basado en τ (suavizado, crece con las iteraciones)
        # Forma: d = 1 - 1 / (updates/τ + 1)
        if self.cfg.tau <= 0:
            return 0.0
        return 1.0 - 1.0 / (updates / float(self.cfg.tau) + 1.0)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Actualiza sombras EMA con los parámetros actuales del modelo."""
        self.updates += 1
        self.decay = self._compute_decay(self.updates)

        # Warm-up EMA: hacer el decay más pequeño (más arrastre) al inicio
        if self.cfg.warmup_iters > 0 and self.updates <= self.cfg.warmup_iters:
            w = float(self.updates) / float(self.cfg.warmup_iters)
            self.decay *= w  # lineal; alternativa: cuadrática w**2

        for n, p in _iter_model_params(model):
            if n not in self.shadow:
                # nuevo parámetro (p. ej., si cambió la arquitectura on-the-fly)
                t = p.detach().clone()
                if self.cfg.force_float32 and t.is_floating_point():
                    t = t.to(torch.float32)
                if self.cfg.device is not None:
                    t = t.to(self.cfg.device)
                self.shadow[n] = t
                continue

            s = self.shadow[n]
            t = p.detach()
            if self.cfg.force_float32 and t.is_floating_point():
                t = t.to(torch.float32)
            # s = decay * s + (1 - decay) * t
            s.mul_(self.decay).add_(t, alpha=(1.0 - self.decay))

        if self.cfg.pin_buffers:
            for n, b in _iter_model_buffers(model):
                if n not in self.shadow_buffers:
                    tb = b.detach().clone()
                    if self.cfg.force_float32 and tb.is_floating_point():
                        tb = tb.to(torch.float32)
                    if self.cfg.device is not None:
                        tb = tb.to(self.cfg.device)
                    self.shadow_buffers[n] = tb
                    continue
                sb = self.shadow_buffers[n]
                tb = b.detach()
                if self.cfg.force_float32 and tb.is_floating_point():
                    tb = tb.to(torch.float32)
                sb.mul_(self.decay).add_(tb, alpha=(1.0 - self.decay))

    # ---------------------------
    # Copia hacia/desde el modelo
    # ---------------------------
    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copia parámetros/buffers sombreados al `model` (in-place)."""
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else None
        for n, p in model.named_parameters():
            if n in self.shadow:
                t = self.shadow[n]
                if device is not None and t.device != device:
                    t = t.to(device)
                p.copy_(t if t.dtype == p.dtype else t.to(p.dtype))
        if self.cfg.pin_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow_buffers:
                    tb = self.shadow_buffers[n]
                    if device is not None and tb.device != device:
                        tb = tb.to(device)
                    b.copy_(tb if tb.dtype == b.dtype else tb.to(b.dtype))

    @torch.no_grad()
    def to(self, device: str | torch.device) -> None:
        """Mueve las sombras a `device`."""
        dev = torch.device(device)
        for k in list(self.shadow.keys()):
            self.shadow[k] = self.shadow[k].to(dev)
        for k in list(self.shadow_buffers.keys()):
            self.shadow_buffers[k] = self.shadow_buffers[k].to(dev)
        self.cfg.device = str(dev)

    # ---------------------------
    # Utilidades adicionales
    # ---------------------------
    def __len__(self) -> int:
        return len(self.shadow)

    def summary(self) -> Dict[str, object]:
        return {
            "params": len(self.shadow),
            "buffers": len(self.shadow_buffers),
            "updates": self.updates,
            "decay": round(float(self.decay), 8),
            "cfg": asdict(self.cfg),
        }


# -------------------------------
# Prueba mínima
# -------------------------------
if __name__ == "__main__":  # pragma: no cover
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
        def forward(self, x):
            y = self.lin(x)
            return y.mean(), {"toy": float(y.mean().item())}

    model = Toy()
    ema = ModelEMA(model, EMAConfig(tau=10, warmup_iters=3, verbose=2))

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for i in range(5):
        loss, _ = model(torch.randn(2, 4))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model)
        print("step", i, ema.summary())

    # copiar a un clon para "validación"
    clone = Toy()
    ema.copy_to(clone)
    print("copiado a clon ok")
