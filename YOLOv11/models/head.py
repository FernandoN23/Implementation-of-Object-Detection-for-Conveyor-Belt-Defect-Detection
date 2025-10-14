# models/head.py
"""
Head de detección del modelo YOLOv11.
Toma los mapas multi-escala del Neck y produce predicciones anchor-free.
"""

import torch
import torch.nn as nn
from blocks import DWConv, Conv


class Detect(nn.Module):
    """
    Módulo de detección YOLOv11 (anchor-free).
    Produce una salida por nivel de escala.
    """
    def __init__(self, ch, nc=80):
        """
        ch: lista de canales [P3, P4, P5]
        nc: número de clases
        """
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # 4 coordenadas + objectness + clases
        self.stride = [8, 16, 32]

        # Por cada nivel de escala: bloque depthwise + conv + salida final
        self.m = nn.ModuleList(
            [nn.Sequential(
                DWConv(c, c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, self.no, 1)
            ) for c in ch]
        )

    def forward(self, feats):
        """
        feats: [P3, P4, P5]
        return: lista de tensores [(B, nc+5, H, W), ...]
        """
        outputs = []
        for i, f in enumerate(feats):
            outputs.append(self.m[i](f))
        return outputs
