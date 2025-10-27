# -*- coding: utf-8 -*-
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: conv.py
Bloques convolucionales fundamentales de YOLOv11.
Define las operaciones atómicas: convolución, normalización
y activación. Base para todos los bloques estructurales.
-------------------------------------------------------------
"""

from typing import Optional
import torch
import torch.nn as nn
from .activation import get_activation


# ============================================================
# UTILIDADES DE NORMALIZACIÓN Y PADDING
# ============================================================

def make_norm(norm_type, num_channels: int, gn_groups: int = 32):
    """
    Crea capa de normalización según tipo solicitado (robusta a tipos mixtos).
    Acepta string, int, bool o None.
    """
    # --- Normalización segura de entrada ---
    if isinstance(norm_type, (int, bool)):
        norm_type = "bn" if bool(norm_type) else "id"
    elif norm_type is None:
        norm_type = "bn"
    else:
        norm_type = str(norm_type).replace("\n", "").strip().lower()

    # --- Tipos admitidos ---
    if norm_type in ("bn", "batch", "batchnorm", "batchnorm2d"):
        return nn.BatchNorm2d(num_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    if norm_type in ("gn", "group", "groupnorm"):
        # Ajuste dinámico de grupos
        g = min(gn_groups, num_channels)
        for cand in (gn_groups, 32, 16, 8, 4, 2, 1):
            if num_channels % cand == 0:
                g = cand
                break
        return nn.GroupNorm(g, num_channels, affine=True)

    if norm_type in ("in", "instancenorm", "instance"):
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=True)

    if norm_type in ("ln", "layer", "layernorm"):
        # LayerNorm 2D simple usando GroupNorm(1, C)
        return nn.GroupNorm(1, num_channels)

    if norm_type in ("id", "none", "identity", "false", "0"):
        return nn.Identity()

    # --- Fallback por defecto ---
    print(f"⚠️ Tipo de normalización desconocido '{norm_type}', usando BatchNorm2d por defecto.")
    return nn.BatchNorm2d(num_channels)


def autopad(k: int, p: Optional[int] = None) -> int:
    """Calcula padding automático (p = k//2) para mantener tamaño espacial."""
    return k // 2 if p is None else p


# ============================================================
# BLOQUES CONVOLUCIONALES
# ============================================================

class Conv(nn.Module):
    """Bloque Conv2D + Normalización + Activación (SiLU por defecto)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 p: Optional[int] = None, groups: int = 1,
                 activation: str | None = "silu",
                 norm_type: str | int | bool | None = "bn",
                 gn_groups: int = 32):
        super().__init__()
        p = autopad(k, p)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = make_norm(norm_type, out_ch, gn_groups)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable convolution (reduce cómputo)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 activation: str | None = "silu",
                 norm_type: str | int | bool | None = "bn",
                 gn_groups: int = 32):
        super().__init__()
        self.dw = Conv(in_ch, in_ch, k=k, s=s, groups=in_ch,
                       activation=activation, norm_type=norm_type, gn_groups=gn_groups)
        self.pw = Conv(in_ch, out_ch, k=1, s=1,
                       activation=activation, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x):
        return self.pw(self.dw(x))
