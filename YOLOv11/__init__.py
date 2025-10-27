"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: __init__.py
Inicializador del paquete YOLOv11.
Permite importar directamente los componentes principales
del modelo y utilidades asociadas.
-------------------------------------------------------------
"""

# ============================================================
# IMPORTS PRINCIPALES
# ============================================================

from .models import (
    YOLOv11,
    YOLOv11Backbone,
    YOLOv11Neck,
    YOLOv11Head,
    YOLOv11Classify,
    ModelParser,
)

# ============================================================
# INFORMACIÓN DEL PAQUETE
# ============================================================

__version__ = "1.0.0"
__author__ = "Fernando N. - Universidad de Chile"
__license__ = "AGPL-3.0 License (Ultralytics-compatible)"

# ============================================================
# INTERFAZ PÚBLICA
# ============================================================

__all__ = [
    "YOLOv11",
    "YOLOv11Backbone",
    "YOLOv11Neck",
    "YOLOv11Classify",
    "ModelParser",
]
