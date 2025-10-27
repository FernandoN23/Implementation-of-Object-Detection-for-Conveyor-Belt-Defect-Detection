"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título: "Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: __init__.py
Inicializador del submódulo 'nn' para YOLOv11.
Expone las primitivas y bloques neuronales fundamentales
de la arquitectura (activaciones, convoluciones y bloques).
-------------------------------------------------------------
"""

# ============================================================
# IMPORTS BASE
# ============================================================

from .activation import (
    SiLU,
    Hardswish,
    Mish,
    Identity,
    get_activation
)

from .conv import (
    Conv,
    DWConv,
    make_norm,
    autopad
)

from .block import (
    Bottleneck,
    C3,
    C3k,
    C3k2,
    SPPF,
    Concat,
    Upsample,
    Focus
)

# ============================================================
# INTERFAZ PÚBLICA
# ============================================================

__all__ = [
    # Activations
    "SiLU", "Hardswish", "Mish", "Identity", "get_activation",

    # Convolutions
    "Conv", "DWConv", "make_norm", "autopad",

    # Blocks
    "Bottleneck", "C3", "C3k", "C3k2",
    "SPPF", "Concat", "Upsample", "Focus",
]
