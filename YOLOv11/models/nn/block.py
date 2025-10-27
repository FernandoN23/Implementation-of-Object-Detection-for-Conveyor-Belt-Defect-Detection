"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título: "Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: block.py
Bloques estructurales jerárquicos de YOLOv11.
Construidos sobre conv.py, incluyen Bottleneck, C3k, C3k2,
SPPF, Concat, Upsample y Focus.
-------------------------------------------------------------
"""

import torch
import torch.nn as nn
from .conv import Conv


# ============================================================
# BLOQUES RESIDUALES Y ESTRUCTURALES
# ============================================================

class Bottleneck(nn.Module):
    """Bloque residual básico tipo CSP."""
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, g: int = 1, expansion: float = 0.5,
                 norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.conv1 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.conv2 = Conv(hidden_ch, out_ch, k=3, s=1, groups=g, norm_type=norm_type, gn_groups=gn_groups)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y


class C3(nn.Module):
    """Bloque C3 original (YOLOv5/8)."""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, expansion: float = 0.5, shortcut: bool = True,
                 norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.cv1 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.cv2 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.m = nn.Sequential(*[
            Bottleneck(hidden_ch, hidden_ch, shortcut=shortcut, expansion=1.0,
                       norm_type=norm_type, gn_groups=gn_groups) for _ in range(n)
        ])
        self.cv3 = Conv(2 * hidden_ch, out_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class C3k(nn.Module):
    """C3k Block (YOLOv11) — tres convoluciones + bottlenecks (Fig.16 paper)."""
    def __init__(self, in_ch: int, out_ch: int, expansion: float = 0.5, shortcut: bool = True,
                 norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.cv1 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.cv2 = Conv(hidden_ch, hidden_ch, k=3, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.cv3 = Conv(hidden_ch, hidden_ch, k=3, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.bottlenecks = nn.Sequential(
            Bottleneck(hidden_ch, hidden_ch, shortcut=shortcut, expansion=1.0,
                       norm_type=norm_type, gn_groups=gn_groups),
            Bottleneck(hidden_ch, hidden_ch, shortcut=shortcut, expansion=1.0,
                       norm_type=norm_type, gn_groups=gn_groups)
        )
        self.out_conv = Conv(hidden_ch, out_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.bottlenecks(x)
        return self.out_conv(x)


class C3k2(nn.Module):
    """C3k2 Block — núcleo de YOLOv11, reemplaza C2f."""
    def __init__(self, in_ch: int, out_ch: int, n: int = 2, expansion: float = 0.5, shortcut: bool = True,
                 c3k: bool = False, norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.c3k = c3k
        self.cv1 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.cv2 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.m = C3k(hidden_ch, hidden_ch, expansion=1.0, shortcut=shortcut,
                     norm_type=norm_type, gn_groups=gn_groups) if c3k else \
                 nn.Sequential(*[
                     Bottleneck(hidden_ch, hidden_ch, shortcut=shortcut, expansion=1.0,
                                norm_type=norm_type, gn_groups=gn_groups)
                     for _ in range(n)
                 ])
        self.cv3 = Conv(2 * hidden_ch, out_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = Conv(in_ch, hidden_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.cv2 = Conv(hidden_ch * 4, out_ch, k=1, s=1, norm_type=norm_type, gn_groups=gn_groups)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), dim=1))


class Concat(nn.Module):
    """Concatenación de características (FPN/PAN)."""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class Upsample(nn.Module):
    """Reescalado espacial por interpolación."""
    def __init__(self, scale: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode)


class Focus(nn.Module):
    """Reducción espacial temprana (heredado de YOLOv5)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        self.conv = Conv(in_ch * 4, out_ch, k=k, s=s, norm_type=norm_type, gn_groups=gn_groups)
    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]), dim=1))
