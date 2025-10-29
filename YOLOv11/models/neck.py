# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: neck.py
# Neck PAN-FPN para YOLOv11 con SPPF + C2PSA (en P5).
# Top-down: P5 → up → concat(P4) → C3k2 → up → concat(P3) → C3k2
# Bottom-up: down → concat → C3k2 ... (tres niveles de salida).
#==============================================================

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

from .nn.conv import Conv, Concat
from .nn.block import C3k2, SPPF, C2PSA
from .backbone import scale_depth

__all__ = ["Neck"]


class Neck(nn.Module):
    """
    SPPF + C2PSA en P5 (baja resolución) para eficiencia, seguido por PAN (upsample/concat)
    y dos caminos bottom-up para reintroducir semántica profunda.
    """

    def __init__(
        self,
        ch: Tuple[int, int, int],   # (cP3, cP4, cP5) desde el Backbone
        d: float = 0.50,
    ) -> None:
        super().__init__()
        c3, c4, c5 = ch
        n2 = scale_depth(2, d)

        # Bloques en P5 (20×20)
        self.sppf = SPPF(c5, c5, k=5)
        self.c2psa = C2PSA(c5, c5, n=n2)

        # Operadores auxiliares
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = Concat(dim=1)

        # Top-down
        self.c3k2_13 = C3k2(c5 + c4, c4, n=n2, c3k=False, e=0.25)   # 40×40
        self.c3k2_16 = C3k2(c4 + c3, c3, n=n2, c3k=False, e=0.25)   # 80×80 (salida pequeña)

        # Bottom-up
        self.down_17 = Conv(c3, c4, k=3, s=2)                       # 80→40
        self.c3k2_19 = C3k2(c4 + c4, c4, n=n2, c3k=False, e=0.25)   # 40×40 (salida media)

        self.down_20 = Conv(c4, c5, k=3, s=2)                       # 40→20
        self.c3k2_22 = C3k2(c5 + c5, c5, n=n2, c3k=False, e=0.25)   # 20×20 (salida grande)

        # Exponer canales de salida por nivel
        self.out_channels: Tuple[int, int, int] = (c3, c4, c5)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = feats  # 80×80, 40×40, 20×20

        # P5 -> SPPF -> C2PSA
        x5 = self.sppf(p5)
        x5 = self.c2psa(x5)

        # Top-down
        u4 = self.upsample(x5)
        x4 = self.concat([u4, p4])
        x4 = self.c3k2_13(x4)

        u3 = self.upsample(x4)
        x3 = self.concat([u3, p3])
        x3 = self.c3k2_16(x3)  # salida pequeña (80×80)

        # Bottom-up
        d4 = self.down_17(x3)
        y4 = self.concat([d4, x4])
        y4 = self.c3k2_19(y4)  # salida media (40×40)

        d5 = self.down_20(y4)
        y5 = self.concat([d5, x5])
        y5 = self.c3k2_22(y5)  # salida grande (20×20)

        return [x3, y4, y5]  # [P3_out, P4_out, P5_out]
