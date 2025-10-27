"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: __init__.py
Inicializador del submódulo 'models' del proyecto YOLOv11.
Expone los componentes principales del modelo:
Backbone, Neck, Head, Classify y definiciones completas.
-------------------------------------------------------------
"""

# ============================================================
# IMPORTS PRINCIPALES
# ============================================================

from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import YOLOv11Head, YOLOv11Classify
from .yolo11 import YOLOv11
from .parser_yaml import ModelParser
from . import nn


# ============================================================
# INTERFAZ PÚBLICA
# ============================================================

__all__ = [
    # Submódulo base de capas neuronales
    "nn",

    # Componentes estructurales
    "YOLOv11Backbone",
    "YOLOv11Neck",
    "YOLOv11Head",
    "YOLOv11Classify",

    # Modelo completo
    "YOLOv11",

    # Parser de configuración
    "ModelParser",
]
