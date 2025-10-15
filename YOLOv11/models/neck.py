import torch
import torch.nn as nn
from .blocks import Conv, C3k2, Concat, Upsample

class YOLOv11Neck(nn.Module):
    """
    YOLOv11 Neck
    ------------
    Fusiona las características del backbone en múltiples resoluciones.
    Usa un camino de fusión tipo FPN-PAN con conexiones ascendentes
    y descendentes para mantener información semántica y espacial.
    """

    def __init__(self, base_channels=64):
        super().__init__()

        # Upsampling path (de P5 a P3)
        self.up1 = Upsample(scale=2)
        self.reduce1 = Conv(base_channels * 16, base_channels * 8, k=1, s=1)
        self.c3_p4 = C3k2(base_channels * 16, base_channels * 8)

        self.up2 = Upsample(scale=2)
        self.reduce2 = Conv(base_channels * 8, base_channels * 4, k=1, s=1)
        self.c3_p3 = C3k2(base_channels * 8, base_channels * 4)

        # Down path (PAN)
        self.down1 = Conv(base_channels * 4, base_channels * 4, k=3, s=2)
        self.c3_n4 = C3k2(base_channels * 8, base_channels * 8)

        self.down2 = Conv(base_channels * 8, base_channels * 8, k=3, s=2)
        self.c3_n5 = C3k2(base_channels * 16, base_channels * 16)

    def forward(self, x3, x4, x5):
        """
        Recibe: P3, P4, P5 del backbone
        Retorna: mapas refinados (N3, N4, N5)
        """
        # Upsample path
        p5_up = self.reduce1(x5)
        p4 = self.c3_p4(torch.cat([p5_up, x4], dim=1))

        p4_up = self.reduce2(self.up2(p4))
        p3 = self.c3_p3(torch.cat([p4_up, x3], dim=1))

        # Downsample path (PAN)
        n4 = self.c3_n4(torch.cat([self.down1(p3), p4], dim=1))
        n5 = self.c3_n5(torch.cat([self.down2(n4), x5], dim=1))

        return p3, n4, n5
