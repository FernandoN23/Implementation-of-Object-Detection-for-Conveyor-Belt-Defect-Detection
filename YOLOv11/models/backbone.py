# backbone.py
import torch
import torch.nn as nn
from .blocks import Conv, C3k2

class YOLOv11Backbone(nn.Module):
    """
    YOLOv11 Backbone
    ----------------
    Extrae características jerárquicas a múltiples escalas.
    Ahora admite normalización configurable (bn, gn, in, id).
    """

    def __init__(self, in_channels=3, base_channels=64,
                 norm_type="bn", gn_groups=32):
        super().__init__()
        self.norm_type = norm_type
        self.gn_groups = gn_groups

        # Etapa 1: reducción inicial
        self.stage1 = nn.Sequential(
            Conv(in_channels, base_channels, k=3, s=2,
                 norm_type=norm_type, gn_groups=gn_groups),
            Conv(base_channels, base_channels, k=3, s=1,
                 norm_type=norm_type, gn_groups=gn_groups)
        )

        # Etapa 2
        self.stage2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, k=3, s=2,
                 norm_type=norm_type, gn_groups=gn_groups),
            C3k2(base_channels * 2, base_channels * 2,
                 norm_type=norm_type, gn_groups=gn_groups)
        )

        # Etapa 3
        self.stage3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, k=3, s=2,
                 norm_type=norm_type, gn_groups=gn_groups),
            C3k2(base_channels * 4, base_channels * 4,
                 norm_type=norm_type, gn_groups=gn_groups)
        )

        # Etapa 4
        self.stage4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, k=3, s=2,
                 norm_type=norm_type, gn_groups=gn_groups),
            C3k2(base_channels * 8, base_channels * 8,
                 norm_type=norm_type, gn_groups=gn_groups)
        )

        # Etapa 5
        self.stage5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, k=3, s=2,
                 norm_type=norm_type, gn_groups=gn_groups),
            C3k2(base_channels * 16, base_channels * 16,
                 norm_type=norm_type, gn_groups=gn_groups)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x3, x4, x5
