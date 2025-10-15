import torch
import torch.nn as nn
from .blocks import Conv, C3k2

class YOLOv11Backbone(nn.Module):
    """
    YOLOv11 Backbone
    ----------------
    Extrae características jerárquicas a múltiples escalas.
    Se basa en bloques convolucionales y módulos C3k2 para un
    balance entre eficiencia y poder de representación.
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Etapa 1: reducción de resolución (stride=2)
        self.stage1 = nn.Sequential(
            Conv(in_channels, base_channels, k=3, s=2),   # 640 -> 320
            Conv(base_channels, base_channels, k=3, s=1)
        )

        # Etapa 2: segunda reducción + C3k2
        self.stage2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, k=3, s=2),  # 320 -> 160
            C3k2(base_channels * 2, base_channels * 2)
        )

        # Etapa 3: tercera reducción + C3k2
        self.stage3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, k=3, s=2),  # 160 -> 80
            C3k2(base_channels * 4, base_channels * 4)
        )

        # Etapa 4: cuarta reducción + C3k2
        self.stage4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, k=3, s=2),  # 80 -> 40
            C3k2(base_channels * 8, base_channels * 8)
        )

        # Etapa 5: salida de mayor profundidad (P5)
        self.stage5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, k=3, s=2),  # 40 -> 20
            C3k2(base_channels * 16, base_channels * 16)
        )

    def forward(self, x):
        """
        Devuelve mapas de características multi-escala.
        """
        x1 = self.stage1(x)  # P1
        x2 = self.stage2(x1) # P2
        x3 = self.stage3(x2) # P3
        x4 = self.stage4(x3) # P4
        x5 = self.stage5(x4) # P5
        return x3, x4, x5     # Se usan en el neck
