# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/engine/bn2gn_patch.py
# Descripción: Política BatchNorm2d → GroupNorm para entornos ROCm.
#              Modos: off | on | on_error. Incluye utilidades de
#              sustitución segura, copia de parámetros y envoltorio
#              de reintento en forward ante errores típicos MIOpen.
#==============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "BN2GNConfig",
    "apply_bn2gn_patch",
    "replace_bn_with_gn",
    "wrap_forward_on_error",
    "count_bn_layers",
    "count_gn_layers",
]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class BN2GNConfig:
    policy: str = "on_error"  # "off" | "on" | "on_error"
    max_groups: int = 32       # tope superior de grupos
    min_channels_per_group: int = 1  # evitar grupos con <1 canal
    verbose: int = 1           # 0: silencioso, 1: info, 2: detalle


# -------------------------------
# Utilidades
# -------------------------------


def _log(msg: str, cfg: Optional[BN2GNConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[bn2gn] {msg}")


def _best_group_count(C: int, max_groups: int = 32, min_channels_per_group: int = 1) -> int:
    """Selecciona número de grupos para GroupNorm dados C canales.

    Preferimos divisores potencias de dos hasta `max_groups` y <= C.
    Si no hay divisor en potencias de dos, buscamos el mayor divisor <= max_groups.
    """
    # 1) Preferencias potencias de 2
    cand_pow2 = [g for g in (32, 16, 8, 4, 2, 1) if g <= max_groups and C % g == 0]
    for g in cand_pow2:
        if C // g >= min_channels_per_group:
            return g
    # 2) Mayor divisor <= max_groups
    for g in range(min(max_groups, C), 0, -1):
        if C % g == 0 and C // g >= min_channels_per_group:
            return g
    # Fallback seguro
    return 1


def _make_gn_from_bn(bn: nn.BatchNorm2d, groups: int) -> nn.GroupNorm:
    """Crea una GroupNorm clonando parámetros de una BatchNorm/SyncBatchNorm.

    **Punto crítico**: la nueva capa se coloca explícitamente en el mismo
    dispositivo y con el mismo dtype que la BN original para evitar mismatches
    tipo "weight en cpu, input en cuda" cuando el modelo ya está en GPU.
    """
    C = bn.num_features

    # Inferir device y dtype desde los parámetros/buffers de BN
    if bn.affine and bn.weight is not None:
        dev = bn.weight.device
        dtype = bn.weight.dtype
    else:
        # BatchNorm siempre tiene running_mean/var como buffers
        dev = bn.running_mean.device
        dtype = bn.running_mean.dtype

    # Crear GN y moverla al mismo device/dtype que la BN original
    gn = nn.GroupNorm(num_groups=groups, num_channels=C, eps=bn.eps, affine=True)
    gn.to(device=dev, dtype=dtype)

    # Copiar parámetros si existen (gamma/beta ↦ weight/bias)
    with torch.no_grad():
        if bn.affine:
            gn.weight.copy_(bn.weight.to(device=dev, dtype=dtype))
            gn.bias.copy_(bn.bias.to(device=dev, dtype=dtype))

    return gn


# -------------------------------
# Reemplazo recursivo BN → GN
# -------------------------------


def _iter_modules_with_parent(module: nn.Module) -> Iterable[Tuple[nn.Module, nn.Module, str]]:
    for name, child in module.named_children():
        yield module, child, name
        yield from _iter_modules_with_parent(child)


def replace_bn_with_gn(model: nn.Module, cfg: BN2GNConfig) -> int:
    """Reemplaza nn.BatchNorm2d/nn.SyncBatchNorm por GroupNorm in-place.

    Retorna el número de capas sustituidas.
    """
    replaced = 0
    for parent, child, name in list(_iter_modules_with_parent(model)):
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            C = child.num_features
            g = _best_group_count(C, cfg.max_groups, cfg.min_channels_per_group)
            gn = _make_gn_from_bn(child, g)
            setattr(parent, name, gn)
            replaced += 1
            _log(f"Reemplazo: {parent.__class__.__name__}.{name}: BN({C}) → GN(groups={g})", cfg, 2)
    if cfg.verbose:
        # Contabilizar tanto BN remanentes como GN actuales para tener panorama completo
        bn_restantes = count_bn_layers(model)
        gn_total = count_gn_layers(model)
        _log(
            f"Total BN→GN reemplazadas: {replaced} | BN restantes: {bn_restantes} | GroupNorm actuales: {gn_total}",
            cfg,
            1,
        )
    return replaced


# -------------------------------
# Detección de errores típicos MIOpen/BN
# -------------------------------

_MIOPEN_BN_PATTERNS = [
    r"miopen.*BatchNorm",          # firmas genéricas
    r"MIOpen Error",               # encabezados
    r"SQLite.*no such column: mode",  # DB rota/completa parcial
    r"BatchNorm.*size mismatch",   # descriptores inconsistentes
    r"miopenStatusInternalError",  # error genérico expuesto por PyTorch/ROCm
]


def _is_miopen_bn_error(err: BaseException) -> bool:
    msg = str(err)
    for pat in _MIOPEN_BN_PATTERNS:
        if re.search(pat, msg, flags=re.IGNORECASE):
            return True
    return False


# -------------------------------
# Envoltorio on_error
# -------------------------------


def wrap_forward_on_error(model: nn.Module, cfg: BN2GNConfig) -> None:
    """Envuelve model.forward para hacer BN→GN al vuelo si ocurre un error MIOpen.

    • Primer forward: intenta normal.
    • Si falla con patrón MIOpen/BN: aplica reemplazo BN→GN y reintenta una vez.
    • Reemplaza definitivamente forward por el del modelo ya parcheado.
    """
    if cfg.policy != "on_error":
        return

    orig_forward = model.forward

    def _wrapped_forward(*args, **kwargs):  # type: ignore[override]
        try:
            return orig_forward(*args, **kwargs)
        except Exception as e:
            if _is_miopen_bn_error(e):
                _log("Error MIOpen/BN detectado. Aplicando parche BN→GN y reintentando…", cfg, 1)
                n = replace_bn_with_gn(model, cfg)
                if n == 0:
                    _log("No se encontraron capas BN para reemplazar.", cfg, 1)
                # importante: usar el forward real del modelo tras el parche
                return model.forward(*args, **kwargs)
            raise

    model.forward = _wrapped_forward  # type: ignore[assignment]
    _log("Forward envuelto para política on_error.", cfg, 2)


# -------------------------------
# Interfaz principal
# -------------------------------


def count_bn_layers(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            n += 1
    return n


def count_gn_layers(model: nn.Module) -> int:
    """Cuenta el número total de capas GroupNorm presentes en el modelo."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            n += 1
    return n


def apply_bn2gn_patch(
    model: nn.Module,
    policy: str = "on_error",
    *,
    max_groups: int = 32,
    min_channels_per_group: int = 1,
    verbose: int = 1,
) -> int:
    """Aplica la política BN→GN.

    Retorna número de reemplazos realizados (0 si off o on_error sin disparo).
    """
    cfg = BN2GNConfig(
        policy=policy,
        max_groups=max_groups,
        min_channels_per_group=min_channels_per_group,
        verbose=verbose,
    )

    if policy not in {"off", "on", "on_error"}:
        raise ValueError("policy debe ser 'off' | 'on' | 'on_error'")

    _log(f"Política BN→GN: {policy}", cfg, 1)

    if policy == "off":
        return 0

    if policy == "on":
        return replace_bn_with_gn(model, cfg)

    # on_error: no reemplaza de inmediato; envuelve forward
    wrap_forward_on_error(model, cfg)
    return 0


# -------------------------------
# Prueba local mínima (opcional)
# -------------------------------
if __name__ == "__main__":  # pragma: no cover

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.SyncBatchNorm(32),
            )

        def forward(self, x):
            return self.seq(x)

    net = Toy()
    print("BN layers antes:", count_bn_layers(net))
    print("GN layers antes:", count_gn_layers(net))
    replaced = apply_bn2gn_patch(net, policy="on", verbose=2)
    print("Reemplazadas:", replaced)
    print("BN layers después:", count_bn_layers(net))
    print("GN layers después:", count_gn_layers(net))
