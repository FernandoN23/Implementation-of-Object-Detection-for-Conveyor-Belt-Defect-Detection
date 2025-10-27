"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título: "Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: head.py
YOLOv11 Head
Incluye los detectores multiescala y el clasificador de imágenes.
-------------------------------------------------------------
"""

import torch
import torch.nn as nn
from YOLOv11.models.nn import Conv


# ============================================================
# CLASIFICADOR YOLOv11
# ============================================================

class YOLOv11Classify(nn.Module):
    """
    YOLOv11 Classification Head
    ---------------------------
    Módulo de clasificación por imagen basado en el head oficial
    de YOLOv11, adaptado para uso general y compatible con
    normalización configurable (BN o GN).

    Convierte mapas de características (b, c_in, h, w)
    en predicciones de clases (b, n_classes).

    Atributos:
        export (bool): Modo exportación (ONNX/TFLite).
        conv (Conv): Bloque convolucional de proyección.
        pool (nn.AdaptiveAvgPool2d): Pooling global espacial.
        drop (nn.Dropout): Regularización.
        linear (nn.Linear): Capa final de clasificación.
    """

    export = False  # para compatibilidad con exportadores

    def __init__(self, c_in: int = 1024, n_classes: int = 1000,
                 hidden_ch: int = 1280, dropout: float = 0.0,
                 norm_type: str = "bn", gn_groups: int = 32):
        """
        Inicializa el clasificador YOLOv11.

        Args:
            c_in (int): Número de canales de entrada (última capa del backbone).
            n_classes (int): Número de clases de salida.
            hidden_ch (int): Dimensión intermedia del embedding.
            dropout (float): Tasa de dropout para regularización.
            norm_type (str): Tipo de normalización ('bn', 'gn', 'in', 'id').
            gn_groups (int): Número de grupos si se usa GroupNorm.
        """
        super().__init__()
        self.n_classes = n_classes

        # Bloque de proyección convolucional (1x1 por defecto)
        self.conv = Conv(c_in, hidden_ch, k=1, s=1,
                         norm_type=norm_type, gn_groups=gn_groups)

        # Pooling y capas finales
        self.pool = nn.AdaptiveAvgPool2d(1)  # reduce a (b, c, 1, 1)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.linear = nn.Linear(hidden_ch, n_classes)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | tuple:
        """
        Propagación hacia adelante del clasificador.

        Args:
            x (torch.Tensor | list[torch.Tensor]): Tensor o lista de tensores de características.

        Returns:
            torch.Tensor | tuple: Predicción (softmax) y logits si no está en modo training.
        """
        if isinstance(x, list):
            # concatenar canales de múltiples niveles (opcional)
            x = torch.cat(x, 1)

        # flujo clásico: conv → pool → dropout → linear
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        logits = self.linear(self.drop(x))

        if self.training:
            return logits  # solo logits durante entrenamiento

        probs = logits.softmax(dim=1)
        return probs if self.export else (probs, logits)
