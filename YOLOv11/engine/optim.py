# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/optim.py
# Descripción: Param groups, optimizador, scheduler (warm-up + cosine),
#              grad-accumulate y clipping. Inspirado en patrones Ultralytics
#              y adaptado a la arquitectura YOLOv11 del proyecto.
#==============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import math
import torch
import torch.nn as nn

try:
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
except Exception:  # pragma: no cover
    CosineAnnealingLR = object  # type: ignore
    LambdaLR = object  # type: ignore

__all__ = [
    "OptimConfig",
    "build_param_groups",
    "adjust_lr_by_effective_batch",
    "build_optimizer_and_scheduler",
    "compute_accumulate",
    "clip_gradients",
]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class OptimConfig:
    # Optimizador
    optimizer: str = "adamw"  # "adamw" | "sgd"
    lr: float = 2e-3          # LR base (para NBS=64)
    lr_head_mult: float = 1.0 # multiplicador LR para capas de la head
    weight_decay: float = 5e-4
    momentum: float = 0.9     # SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # AdamW
    eps: float = 1e-8

    # Programación LR
    epochs: int = 300
    iters_per_epoch: int = 1000
    warmup_epochs: int = 3    # si >0, prioridad sobre warmup_iters
    warmup_iters: int = 0
    cosine: bool = True
    min_lr_ratio: float = 0.01  # LR mínimo = ratio * LR inicial

    # Batch & acumulación
    nbs: int = 64             # nominal batch size Ultralytics style (global)
    batch_effective: int = 64 # world_size * batch_per_gpu * accumulate

    # Gradientes
    clip_norm: float = 10.0
    clip_mode: str = "norm"   # "norm" | "value"

    # Verbosidad
    verbose: int = 1


# -------------------------------
# Utilidades
# -------------------------------

def _log(msg: str, cfg: Optional[OptimConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[optim] {msg}")


def _named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for n, m in model.named_modules():
        yield n, m


def _is_norm_layer(m: nn.Module) -> bool:
    return isinstance(
        m,
        (
            nn.BatchNorm2d,
            nn.SyncBatchNorm,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


# -------------------------------
# Param groups
# -------------------------------

def build_param_groups(model: nn.Module, cfg: OptimConfig) -> List[Dict[str, object]]:
    """Construye grupos de parámetros: decay, no_decay y head.

    Reglas:
    - decay: pesos de Conv/Linear.
    - no_decay: bias y capas de norm.
    - head: (opcional) todo lo que matchee 'head' en el nombre, con LR * lr_head_mult.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    head: List[nn.Parameter] = []

    for name, module in _named_modules(model):
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            full = f"{name}.{pname}" if name else pname

            is_bias = pname.endswith("bias")
            if is_bias or _is_norm_layer(module):
                no_decay.append(p)
            else:
                # pesos de capas con weight param
                decay.append(p)

            if "head" in (name or ""):  # convención del proyecto
                head.append(p)

    # eliminar duplicados preservando el orden
    def _unique(seq: List[nn.Parameter]) -> List[nn.Parameter]:
        seen = set()
        out: List[nn.Parameter] = []
        for x in seq:
            if id(x) not in seen:
                out.append(x)
                seen.add(id(x))
        return out

    decay = _unique(decay)
    no_decay = _unique(no_decay)
    head = _unique(head)

    groups: List[Dict[str, object]] = []
    base_group = {
        "params": decay,
        "weight_decay": cfg.weight_decay,
    }
    groups.append(base_group)
    groups.append({"params": no_decay, "weight_decay": 0.0})

    if cfg.lr_head_mult != 1.0 and len(head) > 0:
        groups.append({"params": head, "lr": cfg.lr * cfg.lr_head_mult})

    _log(
        f"Param groups -> decay={len(decay)} | no_decay={len(no_decay)} | head={len(head)} | lr_head_mult={cfg.lr_head_mult}",
        cfg,
        1,
    )
    return groups


# -------------------------------
# Escalado de LR por batch efectivo
# -------------------------------

def adjust_lr_by_effective_batch(lr_base: float, batch_effective: int, nbs: int) -> float:
    """Escala LR linealmente con el batch efectivo (estilo Ultralytics)."""
    if batch_effective <= 0:
        return lr_base
    return lr_base * batch_effective / float(max(1, nbs))


# -------------------------------
# Accumulate y clipping
# -------------------------------

def compute_accumulate(batch_per_gpu: int, world_size: int, cfg: OptimConfig) -> int:
    effective = max(1, batch_per_gpu) * max(1, world_size)
    acc = max(1, round(cfg.nbs / float(effective)))
    cfg.batch_effective = effective * acc
    _log(f"accumulate={acc} | batch_effective={cfg.batch_effective}", cfg, 1)
    return acc


def clip_gradients(model: nn.Module, max_norm: float, mode: str = "norm") -> float:
    """Aplica clipping y retorna la norma/clamp aplicada."""
    if max_norm is None or max_norm <= 0:
        return 0.0
    if mode == "value":
        nn.utils.clip_grad_value_(model.parameters(), max_norm)
        return float(max_norm)
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # type: ignore[assignment]
    return float(total_norm)


# -------------------------------
# Optimizer + Scheduler
# -------------------------------

def _make_optimizer(param_groups: List[Dict[str, object]], cfg: OptimConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,  # weight_decay individual en groups tiene prioridad
            betas=cfg.betas,
            eps=cfg.eps,
        )
    elif cfg.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError("optimizer debe ser 'adamw' o 'sgd'")
    return opt


def _make_scheduler(opt: torch.optim.Optimizer, cfg: OptimConfig) -> torch.optim.lr_scheduler._LRScheduler:
    total_iters = cfg.epochs * cfg.iters_per_epoch

    if cfg.warmup_epochs > 0:
        warmup_iters = cfg.warmup_epochs * cfg.iters_per_epoch
    else:
        warmup_iters = max(0, int(cfg.warmup_iters))

    min_lr = cfg.lr * cfg.min_lr_ratio

    def lr_lambda(current_iter: int) -> float:
        # Warm-up lineal
        if warmup_iters > 0 and current_iter < warmup_iters:
            return max(1e-8, float(current_iter + 1) / float(warmup_iters))
        # Post-warmup
        remain = max(1, total_iters - max(warmup_iters, 0))
        t = current_iter - warmup_iters
        if not cfg.cosine:
            # step=const (mantener LR base)
            return 1.0
        # Cosine annealing de 1.0 a min_lr_ratio
        cos = 0.5 * (1 + math.cos(math.pi * t / float(remain)))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return sched


def build_optimizer_and_scheduler(model: nn.Module,
                                  cfg: OptimConfig,
                                  *,
                                  batch_per_gpu: int,
                                  world_size: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]:
    """Crea param groups, optimizador, scheduler y retorna accumulate.

    - Escala LR por batch efectivo (Ultralytics: LR ∝ batch_effective/NBS).
    - Calcula accumulate para alcanzar NBS objetivo.
    - Devuelve (optimizer, scheduler, accumulate).
    """
    # 1) Compute accumulate y ajustar LR
    accumulate = compute_accumulate(batch_per_gpu, world_size, cfg)
    scaled_lr = adjust_lr_by_effective_batch(cfg.lr, cfg.batch_effective, cfg.nbs)

    # 2) Param groups
    groups = build_param_groups(model, cfg)

    # 3) Crear optimizador con LR escalado
    lr_backup = cfg.lr
    cfg.lr = scaled_lr
    opt = _make_optimizer(groups, cfg)
    cfg.lr = lr_backup  # restaurar en cfg para consistencia del resumen

    # 4) Scheduler
    sched = _make_scheduler(opt, cfg)

    _log(
        f"Optimizer={cfg.optimizer} | lr(base)={lr_backup} -> lr(scaled)={scaled_lr:.6f} | wd={cfg.weight_decay} | accumulate={accumulate}",
        cfg,
        1,
    )
    _log(f"LR schedule: epochs={cfg.epochs}, iters/ep={cfg.iters_per_epoch}, warmup_ep={cfg.warmup_epochs}, cosine={cfg.cosine}", cfg, 2)

    return opt, sched, accumulate


# -------------------------------
# Prueba mínima
# -------------------------------
if __name__ == "__main__":  # pragma: no cover
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True)
            )
            self.head = nn.Conv2d(16, 8, 1)
        def forward(self, x):
            y = self.head(self.backbone(x))
            return y.mean()

    net = Toy()
    cfg = OptimConfig(optimizer="adamw", lr=0.002, weight_decay=0.0005, epochs=2, iters_per_epoch=5, warmup_epochs=1)
    opt, sch, acc = build_optimizer_and_scheduler(net, cfg, batch_per_gpu=8, world_size=1)
    print("accumulate=", acc)
    x = torch.randn(4, 3, 64, 64)
    for ep in range(cfg.epochs):
        for it in range(cfg.iters_per_epoch):
            loss = net(x)
            loss.backward()
            if (it + 1) % acc == 0:
                clip_gradients(net, cfg.clip_norm, cfg.clip_mode)
                opt.step(); opt.zero_grad(set_to_none=True)
            sch.step()
        print(f"epoch {ep} lr now:", sch.get_last_lr()[:3])
