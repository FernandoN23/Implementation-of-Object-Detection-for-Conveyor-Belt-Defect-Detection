# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: __init__.py
# Inicializador de paquete 'nn' para YOLOv11. Expone bloques, convs y activaciones
# para su uso desde 'models.nn' en backbone/neck/head y otros módulos.
#==============================================================

"""
Paquete `models.nn` — YOLOv11
-----------------------------
Este paquete reúne:
- Activaciones y fábrica (`activation.py`)
- Bloques compuestos (`block.py`)
- Convoluciones y utilitarios (`conv.py`)

Uso típico:
    from models.nn import Conv, C3k2, SPPF, get_activation

Se expone un `__all__` explícito para importaciones controladas.
"""
from __future__ import annotations

# Versionado simple del subpaquete nn (ajústese si es necesario).
__version__ = "0.1.0"

# -- Activaciones --
from .activation import (
    get_activation,
    ActivationFactory,
    Identity,
    SiLU,
    Mish,
    Hardswish,
)

# -- Convoluciones y utilidades --
from .conv import (
    autopad,
    Conv,
    DWConv,
    Conv2,
    ConvTranspose,
    DWConvTranspose2d,
    GhostConv,
    RepConv,
    Focus,
    ChannelAttention,
    SpatialAttention,
    CBAM,
    Concat,
    Index,
)

# -- Bloques compuestos --
from .block import (
    Bottleneck,
    C3k,
    C3k2,
    SPPF,
    MHSA2D,
    PSABlock,
    C2PSA,
)

# -- Alias mínimos de conveniencia --
build_activation = get_activation  # alias legible

# -- API pública --
__all__ = [
    # activation
    "get_activation", "ActivationFactory", "Identity", "SiLU", "Mish", "Hardswish",
    # conv
    "autopad", "Conv", "DWConv", "Conv2", "ConvTranspose", "DWConvTranspose2d",
    "GhostConv", "RepConv", "Focus", "ChannelAttention", "SpatialAttention", "CBAM",
    "Concat", "Index",
    # blocks
    "Bottleneck", "C3k", "C3k2", "SPPF", "MHSA2D", "PSABlock", "C2PSA",
    # convenience
    "build_activation",
]
