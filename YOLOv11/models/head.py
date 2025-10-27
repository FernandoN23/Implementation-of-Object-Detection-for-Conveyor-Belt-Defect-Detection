"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: head.py
YOLOv11 Head (Clasificación)
Recibe una sola feature map [B, 1024, H, W] y produce logits [B, num_classes].
-------------------------------------------------------------
"""

import torch
import torch.nn as nn


class YOLOv11Classify(nn.Module):
    """
    Cabeza de clasificación YOLOv11.
    Flujo: Conv → Normalización → SiLU → Global Pool → Dropout → FC
    """

    def __init__(self, num_classes=5, c_in=1024, dropout=0.0, norm_type="bn", gn_groups=32):
        super().__init__()

        # Validación de dropout
        if not isinstance(dropout, (int, float)):
            print(f"⚠️ Valor inválido de dropout '{dropout}', usando 0.0")
            dropout = 0.0

        hidden_ch = c_in // 2  # 512 para base_channels=64

        self.conv = nn.Sequential(
            nn.Conv2d(c_in, hidden_ch, kernel_size=1, stride=1, padding=0, bias=False),
            self._norm_layer(hidden_ch, norm_type, gn_groups),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=float(dropout), inplace=False)
        self.fc = nn.Linear(hidden_ch, num_classes)

    def _norm_layer(self, num_features, norm_type, gn_groups):
        if norm_type == "gn":
            return nn.GroupNorm(gn_groups, num_features)
        else:
            return nn.BatchNorm2d(num_features)

    def forward(self, x):
        """
        x: tensor [B, 1024, H, W]
        salida: tensor [B, num_classes]
        """
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
