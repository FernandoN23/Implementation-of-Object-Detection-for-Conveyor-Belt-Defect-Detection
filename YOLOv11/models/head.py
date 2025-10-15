import torch
import torch.nn as nn
from .blocks import Conv

class YOLOv11Head(nn.Module):
    """
    YOLOv11 Detection Head
    ----------------------
    Encargado de generar las predicciones finales (bounding boxes, clase y confianza).
    Cada nivel (P3, P4, P5) predice objetos de diferente tamaño.
    """

    def __init__(self, num_classes=80, base_channels=64, anchors=3):
        super().__init__()
        out_channels = anchors * (num_classes + 5)  # 5 = (x, y, w, h, obj)

        # Tres cabezas (multi-escala)
        self.detect3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 4, k=3, s=1),
            nn.Conv2d(base_channels * 4, out_channels, 1)
        )
        self.detect4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 8, k=3, s=1),
            nn.Conv2d(base_channels * 8, out_channels, 1)
        )
        self.detect5 = nn.Sequential(
            Conv(base_channels * 16, base_channels * 16, k=3, s=1),
            nn.Conv2d(base_channels * 16, out_channels, 1)
        )

    def forward(self, p3, n4, n5):
        """
        Devuelve los mapas de detección por nivel.
        """
        y3 = self.detect3(p3)
        y4 = self.detect4(n4)
        y5 = self.detect5(n5)
        return [y3, y4, y5]
