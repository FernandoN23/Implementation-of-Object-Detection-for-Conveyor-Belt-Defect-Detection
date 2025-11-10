# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/optim.py
# Descripción: Param groups, optimizador, scheduler multi‑política
#              (warm‑up + cosine/linear/step/one_cycle), grad‑accumulate
#              y clipping. Incluye mapeador desde parser_yaml/train.yaml
#              para poblar OptimConfig.
#==============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import torch
import torch.nn as nn

try:
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LambdaLR,
        StepLR,
        OneCycleLR,
    )
except Exception:  # pragma: no cover
    CosineAnnealingLR = object  # type: ignore
    LambdaLR = object  # type: ignore
    StepLR = object  # type: ignore
    OneCycleLR = object  # type: ignore

__all__ = [
    "OptimConfig",
    "build_param_groups",
    "adjust_lr_by_effective_batch",
    "build_optimizer_and_scheduler",
    "compute_accumulate",
    "clip_gradients",
    # helpers de integración con parser_yaml
    "from_parser_to_optimcfg",
    "build_optim_from_parser",
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

    # Programación LR general
    epochs: int = 300
    iters_per_epoch: int = 1000
    # warmup (si warmup_epochs>0, tiene prioridad sobre warmup_iters)
    warmup_epochs: int = 0
    warmup_iters: int = 1000

    # Scheduler multi‑política
    scheduler: str = "cosine"  # "cosine" | "linear" | "step" | "one_cycle"
    lrf: float = 0.01          # razón LR_min = lrf * LR_inicial (cosine/linear)
    min_lr_ratio: float = 0.01 # alias legacy (se sincroniza con lrf)
    # parámetros step
    step_size: Optional[int] = None
    gamma: float = 0.1
    # parámetros one_cycle
    one_cycle_pct_start: float = 0.3
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: Optional[float] = None  # si None, = 1.0 / lrf

    # Batch & acumulación
    nbs: int = 64              # nominal batch size (global)
    batch_effective: int = 64  # world_size * batch_per_gpu * accumulate

    # Gradientes
    clip_norm: float = 10.0
    clip_mode: str = "norm"    # "norm" | "value"

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


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------------
# Param groups
# -------------------------------

def build_param_groups(model: nn.Module, cfg: OptimConfig) -> List[Dict[str, object]]:
    """Construye grupos de parámetros: decay, no_decay y head.

    Reglas:
    - decay: pesos de Conv/Linear.
    - no_decay: bias y capas de norm.
    - head: (opcional) todo lo que matchee 'head' en el nombre o atributo .head
      con LR * lr_head_mult.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    head: List[nn.Parameter] = []

    # 1) si el modelo expone atributo .head, preferirlo
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Module):
        for p in getattr(model, "head").parameters(recurse=True):
            if p.requires_grad:
                head.append(p)

    # 2) recorrer módulos por nombre
    for name, module in _named_modules(model):
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            is_bias = pname.endswith("bias")
            if is_bias or _is_norm_layer(module):
                no_decay.append(p)
            else:
                decay.append(p)

            if "head" in (name or ""):
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
    groups.append({"params": decay, "weight_decay": cfg.weight_decay})
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
    """Construye scheduler con warm‑up + política principal.

    Notas:
    - Para 'one_cycle' se usa OneCycleLR nativo y **se ignora** el warm‑up externo
      (OneCycle incorpora su propia rampa inicial mediante pct_start).
    - Para 'cosine' y 'linear' se usa LambdaLR con factor tras warm‑up.
    - Para 'step' se usa LambdaLR con saltos discretos (o StepLR si prefieres por época).
    """
    total_iters = max(1, int(cfg.epochs) * int(cfg.iters_per_epoch))

    # warmup iters
    if cfg.warmup_epochs and cfg.warmup_epochs > 0:
        warmup_iters = max(0, int(cfg.warmup_epochs) * int(cfg.iters_per_epoch))
    else:
        warmup_iters = max(0, int(cfg.warmup_iters))

    # sincronizar alias
    if cfg.min_lr_ratio != cfg.lrf:
        cfg.min_lr_ratio = cfg.lrf

    # --- Caso OneCycleLR (ignora warmup externo) ---
    if str(cfg.scheduler).lower() == "one_cycle":
        final_div = cfg.one_cycle_final_div_factor if cfg.one_cycle_final_div_factor is not None else (1.0 / max(1e-8, cfg.lrf))
        sched = OneCycleLR(
            opt,
            max_lr=[group.get("lr", cfg.lr) for group in opt.param_groups],
            total_steps=total_iters,
            pct_start=cfg.one_cycle_pct_start,
            div_factor=cfg.one_cycle_div_factor,
            final_div_factor=final_div,
            anneal_strategy="cos",
            three_phase=False,
        )
        return sched

    # --- Políticas Lambda (con warmup externo) ---
    min_ratio = max(1e-8, cfg.lrf)

    def after_warmup_ratio(t: int, remain: int) -> float:
        pol = str(cfg.scheduler).lower()
        if pol == "cosine":
            # Cosine annealing de 1.0 → min_ratio
            cos = 0.5 * (1.0 + math.cos(math.pi * t / float(max(1, remain))))
            return min_ratio + (1.0 - min_ratio) * cos
        elif pol == "linear":
            # Lineal 1.0 → min_ratio
            return max(min_ratio, 1.0 - (1.0 - min_ratio) * (t / float(max(1, remain))))
        elif pol == "step":
            # Escalones en iteraciones: si no hay step_size, usar 80% de las épocas
            step_iters = cfg.step_size * cfg.iters_per_epoch if cfg.step_size else int(0.8 * cfg.epochs * cfg.iters_per_epoch)
            k = t // max(1, step_iters)
            return max(min_ratio, (cfg.gamma ** k))
        else:
            # defecto: constante
            return 1.0

    def lr_lambda(current_iter: int) -> float:
        # Warm‑up lineal 0 → 1
        if warmup_iters > 0 and current_iter < warmup_iters:
            return max(1e-8, float(current_iter + 1) / float(warmup_iters))
        # Post‑warmup
        remain = max(1, total_iters - max(0, warmup_iters))
        t = current_iter - warmup_iters
        return after_warmup_ratio(t, remain)

    sched = LambdaLR(opt, lr_lambda)
    return sched


def build_optimizer_and_scheduler(model: nn.Module,
                                  cfg: OptimConfig,
                                  *,
                                  batch_per_gpu: int,
                                  world_size: int,
                                  force_accumulate: Optional[int] = None) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]:
    """Crea param groups, optimizador, scheduler y retorna accumulate.

    - Escala LR por batch efectivo (Ultralytics: LR ∝ batch_effective/NBS).
    - Calcula accumulate para alcanzar NBS objetivo (o usa force_accumulate si viene).
    - Devuelve (optimizer, scheduler, accumulate).
    """
    # 1) Compute accumulate y ajustar LR
    if force_accumulate is not None and force_accumulate > 0:
        effective = max(1, batch_per_gpu) * max(1, world_size)
        accumulate = int(force_accumulate)
        cfg.batch_effective = effective * accumulate
        _log(f"accumulate(forced)={accumulate} | batch_effective={cfg.batch_effective}", cfg, 1)
    else:
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
    _log(
        f"LR schedule: policy={cfg.scheduler}, epochs={cfg.epochs}, iters/ep={cfg.iters_per_epoch}, warmup_ep={cfg.warmup_epochs}, warmup_it={cfg.warmup_iters}, lrf={cfg.lrf}",
        cfg,
        2,
    )

    return opt, sched, accumulate


# -------------------------------
# Integración con parser_yaml.py (mapeador)
# -------------------------------

def _resolve_train_root(parser: Any) -> Dict[str, Any]:
    """Intenta obtener un dict raíz de configuración de entrenamiento desde parser_yaml.
    Soporta varias convenciones: parser.train_cfg, parser.cfg, parser.config, parser.data, parser.to_dict().
    """
    if parser is None:
        raise ValueError("parser=None")

    # Si ya es un dict
    if isinstance(parser, dict):
        return parser

    # preferidos
    for attr in ("train_cfg", "cfg", "config", "data"):
        if hasattr(parser, attr):
            root = getattr(parser, attr)
            if isinstance(root, dict):
                return root
    # método to_dict()
    if hasattr(parser, "to_dict"):
        try:
            maybe = parser.to_dict()
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            pass
    # como último recurso: usar __dict__ superficial
    if hasattr(parser, "__dict__") and isinstance(parser.__dict__, dict):
        return dict(parser.__dict__)
    raise ValueError("No se pudo resolver un diccionario de configuración desde parser_yaml")


def _get_section(root: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    """Busca una sección por nombres posibles, en raíz y dentro de 'normalized' si existe."""
    # directo
    for n in names:
        if n in root and isinstance(root[n], dict):
            return root[n]
    # dentro de 'normalized'
    if "normalized" in root and isinstance(root["normalized"], dict):
        norm = root["normalized"]
        for n in names:
            if n in norm and isinstance(norm[n], dict):
                return norm[n]
    # si la raíz es algo tipo {"train": {...}}, prueba un nivel
    for _, v in list(root.items()):
        if isinstance(v, dict):
            for n in names:
                if n in v and isinstance(v[n], dict):
                    return v[n]
    return {}


def _coerce_scheduler(optim_sec: Dict[str, Any], default: str) -> str:
    """Acepta tanto "scheduler: cosine" como banderas booleanas (cosine/linear/step/one_cycle)."""
    sched = str(_first_present(optim_sec, ["scheduler", "policy"], default)).lower()
    flags = {
        "cosine": bool(optim_sec.get("cosine", False)),
        "linear": bool(optim_sec.get("linear", False)),
        "step": bool(optim_sec.get("step", False)),
        "one_cycle": bool(optim_sec.get("one_cycle", False) or optim_sec.get("onecycle", False)),
    }
    for k, v in flags.items():
        if v:
            return k
    return sched


def from_parser_to_optimcfg(parser: Any, *, iters_per_epoch: int, batch_per_gpu: int, world_size: int) -> OptimConfig:
    """Construye un OptimConfig a partir de parser_yaml + train.yaml (tolerante a estructura).

    Claves soportadas (preferencia en 'normalized'):
    - optim: optimizer, lr0, lrf, weight_decay, momentum, betas, scheduler, epochs, nbs, warmup, step_size, gamma,
    - train: grad_accum, max_grad_norm, epochs, nbs
    - extras: head_lr_mult, eps
    """
    root = _resolve_train_root(parser)

    optim_sec = _get_section(root, ["optim", "optimizer", "optimization"])
    train_sec = _get_section(root, ["train", "training"])

    cfg = OptimConfig()

    # asignaciones optim básicas
    cfg.optimizer = str(_first_present(optim_sec, ["optimizer", "type"], cfg.optimizer)).lower()
    cfg.lr = float(_first_present(optim_sec, ["lr0", "lr", "learning_rate"], cfg.lr))

    lrf = _first_present(optim_sec, ["lrf", "min_lr_ratio"], None)
    if lrf is None:
        lrf = _first_present(train_sec, ["lrf", "min_lr_ratio"], cfg.lrf)
    cfg.lrf = float(lrf)
    cfg.min_lr_ratio = cfg.lrf

    cfg.weight_decay = float(_first_present(optim_sec, ["weight_decay", "wd"],
                                           _first_present(train_sec, ["weight_decay", "wd"], cfg.weight_decay)))
    cfg.momentum = float(_first_present(optim_sec, ["momentum"], cfg.momentum))

    betas = _first_present(optim_sec, ["betas"], None)
    if isinstance(betas, (list, tuple)) and len(betas) >= 2:
        cfg.betas = (float(betas[0]), float(betas[1]))

    cfg.eps = float(_first_present(optim_sec, ["eps"], cfg.eps))

    # epochs puede residir en optim o en train
    cfg.epochs = int(_first_present(optim_sec, ["epochs"], _first_present(train_sec, ["epochs"], cfg.epochs)))

    # scheduler + flags
    cfg.scheduler = _coerce_scheduler(optim_sec, cfg.scheduler)

    # warmup: admitir epochs o steps, pudiendo estar en optim o train
    warm_src = optim_sec if any(k in optim_sec for k in ("warmup", "warmup_epochs", "warmup_iters")) else train_sec
    warm = _first_present(warm_src, ["warmup", "warmup_epochs", "warmup_iters"], 0)
    if isinstance(warm, (float, int)) and warm > 0:
        if "warmup_iters" in warm_src:
            cfg.warmup_iters = int(warm)
            cfg.warmup_epochs = 0
        else:
            cfg.warmup_epochs = int(warm)
            cfg.warmup_iters = 0
    elif isinstance(warm, dict):
        cfg.warmup_epochs = int(warm.get("epochs", 0))
        cfg.warmup_iters = int(warm.get("iters", 0))

    # scheduler params adicionales
    cfg.step_size = int(_first_present(optim_sec, ["step_size"], 0)) or None
    cfg.gamma = float(_first_present(optim_sec, ["gamma"], cfg.gamma))

    cfg.one_cycle_pct_start = float(_first_present(optim_sec, ["one_cycle_pct_start", "pct_start"], cfg.one_cycle_pct_start))
    cfg.one_cycle_div_factor = float(_first_present(optim_sec, ["one_cycle_div_factor", "div_factor"], cfg.one_cycle_div_factor))
    oc_final = _first_present(optim_sec, ["one_cycle_final_div_factor", "final_div_factor"], None)
    if oc_final is not None:
        cfg.one_cycle_final_div_factor = float(oc_final)

    # nbs y lr_head_mult (pueden venir en optim o train)
    cfg.nbs = int(_first_present(optim_sec, ["nbs"], _first_present(train_sec, ["nbs", "nominal_batch_size"], cfg.nbs)))
    cfg.lr_head_mult = float(_first_present(optim_sec, ["head_lr_mult"], cfg.lr_head_mult))

    # clip grad (usualmente en train)
    cfg.clip_norm = float(_first_present(train_sec, ["max_grad_norm", "clip_norm"], cfg.clip_norm))

    # iters por época provenientes del Trainer/loader
    cfg.iters_per_epoch = int(iters_per_epoch)

    # determinar accumulate (si train.grad_accum)
    grad_accum = _first_present(train_sec, ["grad_accum", "accumulate"], None)

    _log(
        "OptimConfig<-parser: opt={optimizer}, lr0={lr:.4g}, lrf={lrf}, sched={scheduler}, epochs={epochs}, nbs={nbs}, warmup_ep={we}, warmup_it={wi}".format(
            optimizer=cfg.optimizer,
            lr=cfg.lr,
            lrf=cfg.lrf,
            scheduler=cfg.scheduler,
            epochs=cfg.epochs,
            nbs=cfg.nbs,
            we=cfg.warmup_epochs,
            wi=cfg.warmup_iters,
        ),
        cfg,
        1,
    )

    # devolver cfg más un posible accumulate forzado
    forced_acc = int(grad_accum) if isinstance(grad_accum, (int, float)) and grad_accum > 0 else None
    return cfg if forced_acc is None else (cfg, forced_acc)


def build_optim_from_parser(model: nn.Module,
                            parser: Optional[Any] = None,
                            *,
                            iters_per_epoch: int,
                            batch_per_gpu: int,
                            world_size: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int, OptimConfig]:
    """Convenience helper: obtiene OptimConfig del parser y construye opt/sched/accumulate.

    Si `parser` es None, se auto‑carga `ConfigParserYaml(project_root=…/YOLOv11)` y se usa su salida.
    Retorna (optimizer, scheduler, accumulate, cfg).
    """
    if parser is None:
        try:
            from YOLOv11.models.parser_yaml import ConfigParserYaml
        except Exception:
            from models.parser_yaml import ConfigParserYaml  # fallback
        project_root = Path(__file__).resolve().parents[1]  # …/<repo>/YOLOv11
        parser = ConfigParserYaml(project_root=str(project_root)).load()

    parsed = from_parser_to_optimcfg(parser, iters_per_epoch=iters_per_epoch,
                                     batch_per_gpu=batch_per_gpu, world_size=world_size)
    if isinstance(parsed, tuple):
        cfg, forced_acc = parsed  # type: ignore[assignment]
    else:
        cfg, forced_acc = parsed, None  # type: ignore[assignment]

    opt, sched, acc = build_optimizer_and_scheduler(
        model,
        cfg,
        batch_per_gpu=batch_per_gpu,
        world_size=world_size,
        force_accumulate=forced_acc,
    )
    return opt, sched, acc, cfg


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

    # demo sin parser (standalone)
    net = Toy()
    cfg = OptimConfig(
        optimizer="adamw", lr=0.002, weight_decay=0.0005,
        epochs=2, iters_per_epoch=5, warmup_epochs=1,
        scheduler="cosine", lrf=0.01
    )
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
