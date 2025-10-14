# models/neck.py
"""
Neck del modelo YOLOv11.
Combina características de múltiples escalas (FPN/PAN).
Incorpora concatenaciones y upsample para mejorar la resolución espacial.
"""

import torch
import torch.nn as nn
from blocks import Conv, C3k2, Concat


class YOLOv11Neck(nn.Module):
    """
    Implementa un FPN/PAN simplificado inspirado en el YAML oficial:
      - Upsample + Concat + C3k2 (fusión ascendente)
      - Downsample + Concat + C3k2 (fusión descendente)
    """

    def __init__(self, ch: list):
        """
        ch: lista de canales de entrada [P3, P4, P5] del backbone.
        """
        super().__init__()
        c3, c4, c5 = ch

        # Fusión ascendente (de P5 -> P3)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(c5, c4, 1, 1)            # reducir canales
        self.concat1 = Concat()
        self.c3k2_1 = C3k2(c4 * 2, c4)             # mezcla P5+P4

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv(c4, c3, 1, 1)
        self.concat2 = Concat()
        self.c3k2_2 = C3k2(c3 * 2, c3)             # mezcla P4+P3

        # Fusión descendente (de P3 -> P5)
        self.down1 = Conv(c3, c3, 3, 2)            # downsample
        self.concat3 = Concat()
        self.c3k2_3 = C3k2(c3 + c4, c4)

        self.down2 = Conv(c4, c4, 3, 2)
        self.concat4 = Concat()
        self.c3k2_4 = C3k2(c4 + c5, c5, c3k_flag=True)

        # Canales de salida
        self.out_channels = [c3, c4, c5]

    def forward(self, feats):
        """
        feats: lista de feature maps [P3, P4, P5]
        return: [P3_out, P4_out, P5_out]
        """
        p3, p4, p5 = feats

        # ---------- Fusión ascendente ----------
        p5_up = self.upsample1(self.conv1(p5))
        p4_up = self.c3k2_1(self.concat1([p5_up, p4]))

        p4_up_2x = self.upsample2(self.conv2(p4_up))
        p3_out = self.c3k2_2(self.concat2([p4_up_2x, p3]))

        # ---------- Fusión descendente ----------
        p4_down = self.c3k2_3(self.concat3([self.down1(p3_out), p4_up]))
        p5_out = self.c3k2_4(self.concat4([self.down2(p4_down), p5]))

        return [p3_out, p4_down, p5_out]
