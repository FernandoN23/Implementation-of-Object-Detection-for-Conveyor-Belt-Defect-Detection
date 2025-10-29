# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: activation.py
# Definición de funciones de activación y fábrica asociada para uso en los bloques de YOLOv11 (compatibles con PyTorch 2.x).
#==============================================================

"""
Activations for YOLOv11 project.
Diseñado para integrarse con los módulos en models/nn/ y permitir configuración vía YAML o código.

Notas:
- Evita dependencias circulares manteniendo este archivo autónomo.
- `get_activation` acepta tanto cadenas abreviadas como módulos instanciados.
- Versiones recientes de PyTorch ya incluyen SiLU/Swish, GELU y Hardswish.
"""
from __future__ import annotations

from typing import Optional, Union, Dict
import torch
import torch.nn as nn

__all__ = ["get_activation", "ActivationFactory", "Identity", "SiLU", "Mish", "Hardswish"]


# -- Envoltorios explícitos (útiles para trazabilidad en gráficos/ONNX) --
class Identity(nn.Module):
    """Módulo identidad (sin operación)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SiLU(nn.SiLU):
    """Alias explícito para SiLU/Swish (x * sigmoid(x)) por claridad en exportación."""
    pass


class Mish(nn.Module):
    """Implementación estable de Mish: x * tanh(softplus(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Hardswish(nn.Module):
    """Hard-Swish: x * ReLU6(x+3)/6"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.hardsigmoid(x)


# -- Fábrica de activaciones --
_NAME_MAP: Dict[str, nn.Module] = {
    "silu": SiLU(),
    "swish": SiLU(),
    "relu": nn.ReLU(inplace=True),
    "relu6": nn.ReLU6(inplace=True),
    "lrelu": nn.LeakyReLU(0.1, inplace=True),
    "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
    "gelu": nn.GELU(),
    "mish": Mish(),
    "hswish": Hardswish(),
    "hardswish": Hardswish(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "identity": Identity(),
    "none": Identity(),
}


def get_activation(act: Union[str, bool, nn.Module, None]) -> nn.Module:
    """
    Retorna un módulo de activación a partir de:
      - nombre (str)
      - instancia (nn.Module)
      - bool (True -> SiLU, False/None -> Identity)

    Args:
        act: selector de activación.

    Returns:
        nn.Module
    """
    if isinstance(act, nn.Module):
        return act
    if act is True:
        return _NAME_MAP["silu"]
    if act in (False, None):
        return _NAME_MAP["identity"]
    key = str(act).strip().lower()
    if key not in _NAME_MAP:
        raise ValueError(f"Activación desconocida '{act}'. Opciones: {sorted(_NAME_MAP.keys())}")
    return _NAME_MAP[key]


class ActivationFactory(nn.Module):
    """
    Pequeña envoltura para construir activaciones desde YAML/código con trazabilidad.

    Ejemplo:
        act = ActivationFactory("silu")  # o True/False/nn.Module
        y = act(x)
    """
    def __init__(self, act: Union[str, bool, nn.Module, None] = "silu") -> None:
        super().__init__()
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.act.__class__.__name__})"
