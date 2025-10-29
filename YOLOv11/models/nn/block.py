# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: block.py
# Bloques compuestos de la arquitectura (Bottleneck, C3k, C3k2, SPPF, C2PSA/PSABlock) utilizados en backbone/neck/head.
#==============================================================

"""
Composite blocks for YOLOv11 (backbone/neck).
Incluye implementaciones compatibles con C3k2 y C2PSA descritas para YOLOv11.

Diseño:
- Bottleneck/C3k/C3k2 para extracción jerárquica.
- SPPF para contexto multi-escala.
- C2PSA (Partial Self-Attention en rama profunda) para modelado global en la etapa de menor resolución.

Notas de ingeniería para el dataset de correas:
- C2PSA se usa solo en el mapa de menor resolución (p.ej., 20x20 en imgsz 640) para limitar costo.
- Para clases locales y bordes finos (Tear, Wear), C3k2 mejora la agregación de camino corto y largo.
"""
from __future__ import annotations

from typing import List, Optional
import math
import torch
import torch.nn as nn

from .conv import Conv, DWConv, Concat, autopad
from .activation import get_activation

__all__ = [
    "Bottleneck",
    "C3k",
    "C3k2",
    "SPPF",
    "MHSA2D",
    "PSABlock",
    "C2PSA",
]


# -----------------------------------------------------------------------------
# 1) Bloques básicos
# -----------------------------------------------------------------------------
class Bottleneck(nn.Module):
    """
    Bloque tipo ResNet: 1x1 -> 3x3 (opcional atajo).
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5,
                 act: bool | str | nn.Module = True) -> None:
        super().__init__()
        c_ = int(c2 * e)  # canales intermedios
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """
    C3k: 3 convoluciones 1x1 con pila interna de Bottlenecks (n).
    Estructura: (1x1 -> n*Bottleneck) + (1x1 camino atajo) -> concat -> 1x1
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5,
                 act: bool | str | nn.Module = True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), 1))


class C3k2(nn.Module):
    """
    C3k2 (YOLOv11): evolución ligera de C2f que permite usar C3k como unidad interna.
    - Si `use_c3k=True` usa C3k en la rama profunda, de lo contrario usa Bottleneck(s).
    - `n` controla la profundidad.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5,
                 use_c3k: bool = True, act: bool | str | nn.Module = True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)   # rama profunda
        self.cv2 = Conv(c1, c_, 1, 1, act=act)   # rama atajo
        if use_c3k:
            self.deep = C3k(c_, c_, n=n, shortcut=shortcut, g=g, e=1.0, act=act)
        else:
            self.deep = nn.Sequential(*[Bottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.deep(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), 1))


# -----------------------------------------------------------------------------
# 2) Contexto multi-escala
# -----------------------------------------------------------------------------
class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast
    Conv(1x1) -> [MP(5), MP(5), MP(5)] -> Concat -> Conv(1x1)
    """
    def __init__(self, c1: int, c2: int, k: int = 5, act: bool | str | nn.Module = True):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# -----------------------------------------------------------------------------
# 3) Atención (MHSA 2D) y C2PSA
# -----------------------------------------------------------------------------
class MHSA2D(nn.Module):
    """
    Multi-Head Self-Attention 2D (espacio HW a secuencia).
    Implementación compacta con proyecciones 1x1 para Q, K, V.
    """
    def __init__(self, c: int, num_heads: int = 8, bias: bool = True):
        super().__init__()
        assert c % num_heads == 0, "c debe ser divisible por num_heads"
        self.c, self.h = c, num_heads
        self.q = nn.Conv2d(c, c, 1, bias=bias)
        self.k = nn.Conv2d(c, c, 1, bias=bias)
        self.v = nn.Conv2d(c, c, 1, bias=bias)
        self.proj = nn.Conv2d(c, c, 1, bias=bias)
        self.scale = (c // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, self.h, C // self.h, H * W)            # (B, h, d, N)
        k = self.k(x).view(B, self.h, C // self.h, H * W)             # (B, h, d, N)
        v = self.v(x).view(B, self.h, C // self.h, H * W)             # (B, h, d, N)
        attn = torch.softmax(torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale, dim=-1)  # (B,h,N,N)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).contiguous()   # (B,h,d,N)
        out = out.view(B, C, H, W)
        return self.proj(out)


class PSABlock(nn.Module):
    """
    Partial Self-Attention block:
    MHSA2D -> FFN (Conv1x1 -> Conv1x1 sin activación)
    """
    def __init__(self, c: int, num_heads: int = 8, act: bool | str | nn.Module = True):
        super().__init__()
        self.attn = MHSA2D(c, num_heads=num_heads)
        self.ffn1 = Conv(c, c, 1, 1, act=act)
        self.ffn2 = Conv(c, c, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn2(self.ffn1(x))
        return x


class C2PSA(nn.Module):
    """
    C2PSA (YOLOv11): divide canales en 2 ramas;
    - Rama A: camino directo (atajo).
    - Rama B: n*PSABlock (atención) en bajo costo (menor resolución).

    Al final concatena y proyecta a c2 canales.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, num_heads: int | None = None,
                 act: bool | str | nn.Module = True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1, act=act)
        self.cv2 = Conv(c2, c2, 1, 1, act=act)
        if num_heads is None:
            num_heads = max(1, min(8, c2 // 64))
        self.blocks = nn.Sequential(*[PSABlock(c2 // 2, num_heads=num_heads, act=act) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        c = x.shape[1]
        c_attn = c // 2
        x1 = x[:, :c_attn]           # rama con atención
        x2 = x[:, c_attn:]           # rama directa
        x1 = self.blocks(x1)
        y = torch.cat([x1, x2], dim=1)
        return self.cv2(y)
