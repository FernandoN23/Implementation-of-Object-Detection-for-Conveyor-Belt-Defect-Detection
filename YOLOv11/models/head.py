# head.py
import torch
import torch.nn as nn
from .blocks import Conv

class YOLOv11Head(nn.Module):
    """
    YOLOv11 Head
    ------------
    Genera predicciones multi-escala con normalización configurable.
    """

    def __init__(self, num_classes=5, base_channels=64, anchors=3,
                 norm_type="bn", gn_groups=32):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = anchors * (num_classes + 5)
        self.norm_type = norm_type
        self.gn_groups = gn_groups

        # Detectores por escala
        self.detect_p3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 4, k=3, s=1,
                 norm_type=norm_type, gn_groups=gn_groups),
            nn.Conv2d(base_channels * 4, self.out_channels, 1)
        )
        self.detect_n4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 8, k=3, s=1,
                 norm_type=norm_type, gn_groups=gn_groups),
            nn.Conv2d(base_channels * 8, self.out_channels, 1)
        )
        self.detect_n5 = nn.Sequential(
            Conv(base_channels * 16, base_channels * 16, k=3, s=1,
                 norm_type=norm_type, gn_groups=gn_groups),
            nn.Conv2d(base_channels * 16, self.out_channels, 1)
        )

    def forward(self, p3, n4, n5):
        y3 = self.detect_p3(p3)
        y4 = self.detect_n4(n4)
        y5 = self.detect_n5(n5)
        return [y3, y4, y5]
