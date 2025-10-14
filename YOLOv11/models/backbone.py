# models/backbone.py
"""
Backbone del modelo YOLOv11.
Encargado de la extracción jerárquica de características visuales.
Basado en los bloques: Conv, C3k2, SPPF y C2PSA.
"""

import torch
import torch.nn as nn
from blocks import Conv, C3k2, SPPF, C2PSA


class YOLOv11Backbone(nn.Module):
    """
    Backbone compuesto por:
      - Etapas convolucionales de downsampling (Conv stride=2)
      - Bloques C3k2 repetidos
      - Bloque SPPF para contexto global
      - Módulo C2PSA (Partial Self Attention)
    """

    def __init__(self, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        # Escalado de canales
        def c(ch): return max(16, int(ch * width_multiple))

        # ------------------------------
        # Definición de etapas (según YAML de YOLO11)
        # ------------------------------
        self.layer0 = Conv(3, c(64), 3, 2)        # P1/2
        self.layer1 = Conv(c(64), c(128), 3, 2)   # P2/4
        self.layer2 = C3k2(c(128), c(256), n=2)   # Bloque C3k2
        self.layer3 = Conv(c(256), c(256), 3, 2)  # P3/8
        self.layer4 = C3k2(c(256), c(512), n=2)
        self.layer5 = Conv(c(512), c(512), 3, 2)  # P4/16
        self.layer6 = C3k2(c(512), c(512), n=2, c3k_flag=True)
        self.layer7 = Conv(c(512), c(1024), 3, 2) # P5/32
        self.layer8 = C3k2(c(1024), c(1024), n=2, c3k_flag=True)
        self.layer9 = SPPF(c(1024), k=5)
        self.layer10 = C2PSA(c(1024))             # Atención parcial

        # Lista de canales de salida (útil para Neck)
        self.out_channels = [c(256), c(512), c(1024)]

    def forward(self, x):
        """
        Retorna los mapas de características de tres escalas:
        P3 (pequeño), P4 (mediano), P5 (grande)
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        p3 = self.layer3(x)
        p3 = self.layer4(p3)
        p4 = self.layer5(p3)
        p4 = self.layer6(p4)
        p5 = self.layer7(p4)
        p5 = self.layer8(p5)
        p5 = self.layer9(p5)
        p5 = self.layer10(p5)
        return [p3, p4, p5]
