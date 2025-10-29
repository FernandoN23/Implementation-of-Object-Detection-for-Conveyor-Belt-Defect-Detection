# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: backbone.py
# Backbone de YOLOv11: stem + etapas C3k2 con downsampling 2×.
# Devuelve características P3, P4 y P5, escaladas por (d, w, mc).
#==============================================================

from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn

from .nn.conv import Conv
from .nn.block import C3k2

__all__ = ["Backbone", "scale_depth", "scale_width"]


def make_divisible(x: int, divisor: int = 8) -> int:
    """Ajusta al múltiplo más cercano (por defecto /8) para eficiencia en kernels/GEMM."""
    return int(math.ceil(x / divisor) * divisor)


def scale_width(base: int, w: float, mc: int) -> int:
    """Escalado de canales: aplica width_multiple y topa por max_channels."""
    return make_divisible(min(base, mc) * w)


def scale_depth(n: int, d: float) -> int:
    """Escalado de profundidad: al menos 1 repetición."""
    return max(int(round(n * d)), 1)


class Backbone(nn.Module):
    """
    YOLOv11 Backbone (anchor-free)
    - Estructura (imagen de referencia): Conv s=2 → Conv s=2 → C3k2 → Conv s=2 → C3k2 → Conv s=2 → C3k2 → Conv s=2 → C3k2
    - Puntos de salida: P3 (80×80), P4 (40×40), P5 (20×20)
    """

    def __init__(
        self,
        d: float = 0.50,
        w: float = 0.25,
        mc: int = 1024,
        in_ch: int = 3,
        divisor: int = 8,
    ) -> None:
        super().__init__()
        self.d, self.w, self.mc, self.divisor = d, w, mc, divisor

        # Canales base (estándar YOLO): 64, 128, 256, 512, 1024
        c64 = scale_width(64, w, mc)
        c128 = scale_width(128, w, mc)
        c256 = scale_width(256, w, mc)
        c512 = scale_width(512, w, mc)
        c1024 = scale_width(1024, w, mc)

        # Stem
        self.stem0 = Conv(in_ch, c64, k=3, s=2)     # 640 -> 320
        self.stem1 = Conv(c64, c128, k=3, s=2)      # 320 -> 160

        # Etapas (ver diagrama): n=2*d; e=0.25; c3k=False en primeras etapas, True en profundas
        n2 = scale_depth(2, d)

        # 160×160
        self.c3k2_2 = C3k2(c128, c256, n=n2, c3k=False, e=0.25)

        # 80×80 (P3)
        self.down_3 = Conv(c256, c256, k=3, s=2)
        self.c3k2_4 = C3k2(c256, c256, n=n2, c3k=False, e=0.25)

        # 40×40 (P4)
        self.down_5 = Conv(c256, c512, k=3, s=2)
        self.c3k2_6 = C3k2(c512, c512, n=n2, c3k=True, e=0.25)

        # 20×20 (P5)
        self.down_7 = Conv(c512, c1024, k=3, s=2)
        self.c3k2_8 = C3k2(c1024, c1024, n=n2, c3k=True, e=0.25)

        # Exponer los canales de salida para la Neck/Head
        self.out_channels: Tuple[int, int, int] = (c256, c512, c1024)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem0(x)
        x = self.stem1(x)

        x = self.c3k2_2(x)

        x = self.down_3(x)
        p3 = self.c3k2_4(x)     # 80×80

        x = self.down_5(p3)
        p4 = self.c3k2_6(x)     # 40×40

        x = self.down_7(p4)
        p5 = self.c3k2_8(x)     # 20×20

        return [p3, p4, p5]
