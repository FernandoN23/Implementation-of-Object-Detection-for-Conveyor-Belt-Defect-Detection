import torch
import torch.nn as nn
from .blocks import Conv

class YOLOv11Head(nn.Module):
    """
    YOLOv11 Head (corregido)
    ------------------------
    Genera predicciones multi-escala a partir de las salidas del neck.
    Asegura que las dimensiones de entrada coincidan con p3, n4, n5:
    (80x80), (40x40), (20x20)
    """

    def __init__(self, num_classes=80, base_channels=64, anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = anchors * (num_classes + 5)

        # Predictores multi-escala
        self.detect_p3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 4, k=3, s=1),
            nn.Conv2d(base_channels * 4, self.out_channels, 1)
        )
        self.detect_n4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 8, k=3, s=1),
            nn.Conv2d(base_channels * 8, self.out_channels, 1)
        )
        self.detect_n5 = nn.Sequential(
            Conv(base_channels * 16, base_channels * 16, k=3, s=1),
            nn.Conv2d(base_channels * 16, self.out_channels, 1)
        )

    def forward(self, p3, n4, n5):
        """
        Devuelve los mapas de detección por nivel:
            y3: 80x80 → objetos pequeños
            y4: 40x40 → medianos
            y5: 20x20 → grandes
        """
        y3 = self.detect_p3(p3)
        y4 = self.detect_n4(n4)
        y5 = self.detect_n5(n5)
        return [y3, y4, y5]
