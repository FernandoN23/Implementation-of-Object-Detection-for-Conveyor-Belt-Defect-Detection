import torch
import torch.nn as nn
from .blocks import Conv, C3k2

class YOLOv11Neck(nn.Module):
    """
    YOLOv11 Neck (corregido final)
    ------------------------------
    FPN + PAN bidireccional que combina características del backbone.
    Se corrigen los desajustes de dimensiones detectados en las concatenaciones:
    - Uso de `Upsample` con `mode='nearest'` y `align_corners=False`
    - Ajuste de canales intermedios para compatibilidad con el head.
    """

    def __init__(self, base_channels=64):
        super().__init__()

        # -------- FPN (de profundo a superficial) --------
        # Reducción de canales del mapa más profundo (P5)
        self.reduce_conv_p5 = Conv(base_channels * 16, base_channels * 8, k=1, s=1)
        # Upsample mantiene el tamaño espacial exacto con nearest neighbor
        self.upsample_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p4 = C3k2(base_channels * 16, base_channels * 8)  # concat(P5_up, P4)

        # Reducción de canales en P4 y upsample
        self.reduce_conv_p4 = Conv(base_channels * 8, base_channels * 4, k=1, s=1)
        self.upsample_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p3 = C3k2(base_channels * 8, base_channels * 4)  # concat(P4_up, P3)

        # -------- PAN (de superficial a profundo) --------
        self.down_p3 = Conv(base_channels * 4, base_channels * 4, k=3, s=2)
        self.c3_n4 = C3k2(base_channels * 12, base_channels * 8)  # 256 + 512 = 768

        self.down_p4 = Conv(base_channels * 8, base_channels * 8, k=3, s=2)
        self.c3_n5 = C3k2(base_channels * 24, base_channels * 16)  # 512 + 1024 = 1536


    def forward(self, x3, x4, x5):
        """
        Entradas del backbone:
            x3: [B, 256, 80, 80]  ← P3/8
            x4: [B, 512, 40, 40]  ← P4/16
            x5: [B,1024, 20, 20]  ← P5/32

        Salidas del neck:
            p3: [B, 256, 80, 80]
            n4: [B, 512, 40, 40]
            n5: [B,1024, 20, 20]
        """

        # ------- FPN: de profundo a superficial -------
        p5 = self.reduce_conv_p5(x5)                          # [B,512,20,20]
        p5_up = self.upsample_p5(p5)                          # [B,512,40,40]
        p4 = self.c3_p4(torch.cat([p5_up, x4], dim=1))        # [B,512,40,40]

        p4_reduced = self.reduce_conv_p4(p4)                  # [B,256,40,40]
        p4_up = self.upsample_p4(p4_reduced)                  # [B,256,80,80]
        p3 = self.c3_p3(torch.cat([p4_up, x3], dim=1))        # [B,256,80,80]

        # ------- PAN: de superficial a profundo -------
        n4 = self.c3_n4(torch.cat([self.down_p3(p3), p4], dim=1))  # [B,512,40,40]
        n5 = self.c3_n5(torch.cat([self.down_p4(n4), x5], dim=1))  # [B,1024,20,20]

        return p3, n4, n5
