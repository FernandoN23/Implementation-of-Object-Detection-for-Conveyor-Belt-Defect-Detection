# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/engine/bn2gn_patch.py
# Descripción: Parche dinámico BatchNorm -> GroupNorm.
#              Adaptado para detectar 'FrozenBatchNorm2d' de DINO.
# ==============================================================

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class BN2GNConfig:
    policy: str = "on"
    max_groups: int = 32
    min_channels_per_group: int = 1
    verbose: int = 1


def _get_group_count(channels, max_groups=32):
    """Busca el mejor divisor para el número de grupos."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def replace_bn_with_gn(model: nn.Module, cfg: BN2GNConfig) -> int:
    """Sustituye recursivamente capas BN por GN en el modelo DINO."""
    replaced = 0
    # DINO usa FrozenBatchNorm2d en el backbone ResNet
    target_types = ("BatchNorm2d", "SyncBatchNorm", "FrozenBatchNorm2d")

    for name, m in model.named_modules():
        for sub_name, child in m.named_children():
            if any(t in str(type(child)) for t in target_types):
                channels = child.num_features if hasattr(child, 'num_features') else child.weight.shape[0]
                groups = _get_group_count(channels, cfg.max_groups)

                gn = nn.GroupNorm(groups, channels, eps=getattr(child, 'eps', 1e-5), affine=True)

                if hasattr(child, 'weight') and child.weight is not None:
                    device = child.weight.device
                    dtype = child.weight.dtype
                else:
                    device = torch.device('cpu')
                    dtype = torch.float32

                gn.to(device=device, dtype=dtype)

                with torch.no_grad():
                    if hasattr(child, 'weight') and child.weight is not None:
                        gn.weight.copy_(child.weight)
                    if hasattr(child, 'bias') and child.bias is not None:
                        gn.bias.copy_(child.bias)

                setattr(m, sub_name, gn)
                replaced += 1

                if cfg.verbose > 1:
                    print(f"[bn2gn] {type(child).__name__} -> GN({groups}) en {name}.{sub_name}")

    if cfg.verbose > 0:
        print(f"[bn2gn] Parche aplicado con éxito. Capas reemplazadas: {replaced}")
    return replaced