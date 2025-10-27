"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: neck.py
YOLOv11 Neck (versión para clasificación)
-------------------------------------------------------------

Esta versión está optimizada para la tarea de clasificación.
En lugar de FPN + PAN multiescala, realiza un refinamiento
ligero sobre el mapa de características final del backbone.

Flujo:
    x → Conv → Normalización → SiLU → SPPF → salida única
-------------------------------------------------------------
"""

import torch
import torch.nn as nn
from models.nn import Conv


class YOLOv11Neck(nn.Module):
    """
    YOLOv11 Neck (Clasificación)
    -----------------------------
    Refinamiento simple de características finales del backbone.
    Mantiene compatibilidad con GN (ROCm) y BN estándar.
    """

    def __init__(self, base_channels=64, norm_type="bn", gn_groups=32):
        super().__init__()
        self.norm_type = norm_type
        self.gn_groups = gn_groups

        # Convolución de refinamiento
        self.refine = Conv(
            base_channels * 16, base_channels * 16,
            k=3, s=1, p=1,
            norm_type=norm_type,
            gn_groups=gn_groups
        )

        # Bloque SPPF (Spatial Pyramid Pooling - Fast)
        self.sppf = SPPF(base_channels * 16, base_channels * 16)

    def forward(self, x):
        """
        Forward simplificado:
            entrada única del backbone → salida única al head.
        """
        x = self.refine(x)
        x = self.sppf(x)
        return x


# -------------------------------------------------------------
# Implementación ligera de SPPF (tomada del diseño YOLOv8/11)
# -------------------------------------------------------------
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        hidden = c_in // 2
        self.conv1 = nn.Conv2d(c_in, hidden, 1, 1)
        self.act1 = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv2 = nn.Conv2d(hidden * 4, c_out, 1, 1)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.act2(self.conv2(torch.cat([x, y1, y2, y3], 1)))
