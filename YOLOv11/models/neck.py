"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título: "Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: neck.py
YOLOv11 Neck
Combina FPN (arriba-abajo) y PAN (abajo-arriba) para fusionar
características multiescala provenientes del backbone.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# FPN (Feature Pyramid Network):
#   - Reduce canales y propaga de p5→p3 mediante upsampling y concat.
#   - C3k2 refina las fusiones inter-escala.
#
# PAN (Path Aggregation Network):
#   - Reinvierte el flujo: p3→p5 usando downsampling.
#   - Genera salidas refinadas n4 y n5.
#
# Conexiones:
#   (x3, x4, x5) → backbone outputs
#   (p3, n4, n5) → salidas hacia el head.
# -------------------------------------------------------------

import torch
import torch.nn as nn
from YOLOv11.models.nn import Conv, C3k2

class YOLOv11Neck(nn.Module):
    """
    YOLOv11 Neck
    ------------
    Combina FPN + PAN con normalización configurable.
    """

    def __init__(self, base_channels=64,
                 norm_type="bn", gn_groups=32):
        super().__init__()
        self.norm_type = norm_type
        self.gn_groups = gn_groups

        # -------- FPN --------
        self.reduce_conv_p5 = Conv(base_channels * 16, base_channels * 8, k=1, s=1,
                                   norm_type=norm_type, gn_groups=gn_groups)
        self.upsample_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p4 = C3k2(base_channels * 16, base_channels * 8,
                          norm_type=norm_type, gn_groups=gn_groups)

        self.reduce_conv_p4 = Conv(base_channels * 8, base_channels * 4, k=1, s=1,
                                   norm_type=norm_type, gn_groups=gn_groups)
        self.upsample_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p3 = C3k2(base_channels * 8, base_channels * 4,
                          norm_type=norm_type, gn_groups=gn_groups)

        # -------- PAN --------
        self.down_p3 = Conv(base_channels * 4, base_channels * 4, k=3, s=2,
                            norm_type=norm_type, gn_groups=gn_groups)
        self.c3_n4 = C3k2(base_channels * 12, base_channels * 8,
                          norm_type=norm_type, gn_groups=gn_groups)

        self.down_p4 = Conv(base_channels * 8, base_channels * 8, k=3, s=2,
                            norm_type=norm_type, gn_groups=gn_groups)
        self.c3_n5 = C3k2(base_channels * 24, base_channels * 16,
                          norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x3, x4, x5):
        # ------- FPN -------
        p5 = self.reduce_conv_p5(x5)
        p5_up = self.upsample_p5(p5)
        p4 = self.c3_p4(torch.cat([p5_up, x4], dim=1))

        p4_reduced = self.reduce_conv_p4(p4)
        p4_up = self.upsample_p4(p4_reduced)
        p3 = self.c3_p3(torch.cat([p4_up, x3], dim=1))

        # ------- PAN -------
        n4 = self.c3_n4(torch.cat([self.down_p3(p3), p4], dim=1))
        n5 = self.c3_n5(torch.cat([self.down_p4(n4), x5], dim=1))

        return p3, n4, n5
