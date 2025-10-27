"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título: "Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: activation.py
Funciones de activación empleadas en YOLOv11.
Contiene implementaciones estándar y personalizadas utilizadas
en los bloques convolucionales y estructurales.
-------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ACTIVACIONES ESTÁNDAR
# ============================================================

class SiLU(nn.Module):
    """Sigmoid Linear Unit (utilizada por defecto en YOLOv11)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Hard-Swish: versión eficiente de Swish."""
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, inplace=True) / 6.0


class Mish(nn.Module):
    """Mish: función suave, no monótona."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Identity(nn.Module):
    """Usada cuando no se requiere activación."""
    def forward(self, x):
        return x


# ============================================================
# FACTORÍA DE ACTIVACIONES
# ============================================================

def get_activation(name: str = "silu") -> nn.Module:
    """
    Retorna la activación solicitada por nombre.

    Args:
        name (str): Nombre de la activación. Opciones: 'silu', 'relu', 'mish', 'hardswish', 'none'

    Returns:
        nn.Module: capa de activación correspondiente
    """
    name = (name or "silu").lower()
    if name == "silu":
        return SiLU()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "mish":
        return Mish()
    elif name in {"hardswish", "hswish"}:
        return Hardswish()
    elif name in {"none", "id", "identity"}:
        return Identity()
    else:
        raise ValueError(f"Activación '{name}' no reconocida.")
