# head.py
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título: "Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: head.py
YOLOv11 Head
Genera las predicciones multiescala (p3, n4, n5)
para bounding boxes, clases e información de objeto.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Estructura:
#   - Tres detectores independientes (p3, n4, n5)
#   - Cada uno produce (B, anchors*(5+num_classes), H, W)
#       5 → (x, y, w, h, conf)
#   - Conv3x3 + Conv1x1 por escala
#
# Conexiones:
#   p3 ← salida de FPN
#   n4, n5 ← salidas del PAN
# Salida final:
#   lista [y3, y4, y5] → entrada al post-procesamiento (NMS)
# -------------------------------------------------------------

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
        # === Debug shapes ===
        #if not torch.jit.is_scripting():
        #    print(f"[HEAD DEBUG] y3={list(y3.shape)}, y4={list(y4.shape)}, y5={list(y5.shape)}")
        #    total_preds = sum(y.shape[2] * y.shape[3] for y in [y3, y4, y5])
        #    print(f"[HEAD DEBUG] Total cells per image: {total_preds}")
        return [y3, y4, y5]
