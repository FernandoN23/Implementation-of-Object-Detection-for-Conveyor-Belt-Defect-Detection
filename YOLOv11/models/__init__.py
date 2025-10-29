# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: __init__.py
# Inicializa el paquete YOLOv11.models, reexportando clases clave
# (Backbone, Neck, DetectHead, YOLOv11) y utilidades (build_model,
# ConfigParserYaml, bloques de nn) para uso externo.
#==============================================================

from __future__ import annotations

# Ensamblado principal y utilidades de construcción
from .yolo11 import YOLOv11, VARIANTS, build_model
from .parser_yaml import ConfigParserYaml

# Componentes de arquitectura
from .backbone import Backbone, scale_depth, scale_width
from .neck import Neck
from .head import DetectHead

# Bloques y utilidades NN (subset más usados)
from .nn.activation import get_activation, ActivationFactory
from .nn.conv import (
    Conv, DWConv, ConvTranspose, DWConvTranspose2d,
    Focus, GhostConv, RepConv,
    ChannelAttention, SpatialAttention, CBAM,
    Concat, Index,
)
from .nn.block import (
    Bottleneck, C3k, C3k2, SPPF, MHSA2D, PSABlock, C2PSA,
)

__all__ = [
    # Ensamblado y parser
    "YOLOv11", "VARIANTS", "build_model", "ConfigParserYaml",
    # Arquitectura
    "Backbone", "Neck", "DetectHead", "scale_depth", "scale_width",
    # Bloques/NN
    "get_activation", "ActivationFactory",
    "Conv", "DWConv", "ConvTranspose", "DWConvTranspose2d",
    "Focus", "GhostConv", "RepConv",
    "ChannelAttention", "SpatialAttention", "CBAM",
    "Concat", "Index",
    "Bottleneck", "C3k", "C3k2", "SPPF", "MHSA2D", "PSABlock", "C2PSA",
]


def make_model_from_configs(
    project_root: str | None = None,
    parser_yaml_path: str | None = None,
    **build_kwargs,
):
    """
    Atajo para cargar YAMLs y construir el modelo:
        model, parser = make_model_from_configs(project_root=..., variant="m")

    Args:
        project_root: ruta a la raíz del proyecto (YOLOv11/). Si None, se infiere.
        parser_yaml_path: ruta explícita a configs/parser.yaml (opcional).
        **build_kwargs: argumentos que se pasan a parser.build_model(...).

    Returns:
        Tuple[YOLOv11, ConfigParserYaml]: (modelo construido, parser con configs cargadas)
    """
    parser = ConfigParserYaml(
        project_root=project_root,
        parser_yaml_path=parser_yaml_path,
    ).load()
    model = parser.build_model(**build_kwargs)
    return model, parser
