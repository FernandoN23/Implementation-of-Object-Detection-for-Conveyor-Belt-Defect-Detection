# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: conv.py
# Módulos convolucionales básicos y utilitarios (Conv, DWConv, GhostConv, RepConv, CBAM, etc.) para YOLOv11.
#==============================================================

"""
Convolutional modules for YOLOv11 project.

Inspirado en implementaciones públicas de Ultralytics (AGPL-3.0) pero reescrito con
énfasis en claridad, tipado y comentarios. Mantiene compatibilidad de interfaz para
ser usado por blocks como C3k2, SPPF y C2PSA.

Advertencia de licencia: si se reutiliza/redistribuye código derivado de Ultralytics,
asegúrese de cumplir con AGPL-3.0. Este archivo es una implementación propia compatible.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn

from .activation import get_activation

__all__ = (
    "autopad",
    "Conv",
    "DWConv",
    "Conv2",
    "ConvTranspose",
    "DWConvTranspose2d",
    "GhostConv",
    "RepConv",
    "Focus",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "Index",
)


def autopad(k: int | Tuple[int, int], p: Optional[int | Tuple[int, int]] = None, d: int = 1):
    """
    Calcula el padding 'same' para kernel y dilatación dados.
    """
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = tuple(d * (x - 1) + 1 for x in k)
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = tuple(x // 2 for x in k)
    return p


class Conv(nn.Module):
    """
    Conv2d + BN + Act (por defecto SiLU).
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None,
                 g: int = 1, d: int = 1, act: bool | str | nn.Module = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Variante simplificada de RepConv: 3x3 + 1x1 en paralelo (solo en entrenamiento).
    En despliegue se fusiona en un único kernel efectivo 3x3.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None,
                 g: int = 1, d: int = 1, act: bool | str | nn.Module = True):
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def fuse_convs(self) -> None:
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        delattr(self, "cv2")
        self.forward = self.forward_fuse


class DWConv(Conv):
    """Depthwise separable conv (grupos = gcd(c1, c2))."""
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, d: int = 1,
                 act: bool | str | nn.Module = True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise ConvTranspose2d."""
    def __init__(self, c1: int, c2: int, k: int = 2, s: int = 2, p1: int = 0, p2: int = 0):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """ConvTranspose2d + BN + Act (SiLU por defecto)."""
    def __init__(self, c1: int, c2: int, k: int = 2, s: int = 2, p: int = 0,
                 bn: bool = True, act: bool | str | nn.Module = True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus: recorta y concatena 4 submuestras para concentrar información (↓W,H; ↑C).
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None,
                 g: int = 1, act: bool | str | nn.Module = True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g=g, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        return self.conv(x)


class GhostConv(nn.Module):
    """GhostConv: genera canales por 'operaciones baratas' (ghost features)."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 g: int = 1, act: bool | str | nn.Module = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, g=c_, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv (entrenamiento: 3x3 + 1x1 + BN opcional; despliegue: conv única).
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1,
                 g: int = 1, d: int = 1, act: bool | str | nn.Module = True,
                 bn_identity: bool = False):
        super().__init__()
        assert k == 3 and p == 1, "RepConv: solo soporta 3x3/k=3,p=1"
        self.g, self.c1, self.c2 = g, c1, c2
        self.act = get_activation(act)
        self.bn_id = nn.BatchNorm2d(c1) if bn_identity and c1 == c2 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, d=d, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, d=d, act=False)
        self._fused: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            return self.act(self.conv(x))
        id_out = 0 if self.bn_id is None else self.bn_id(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    @staticmethod
    def _fuse_bn_tensor(branch: Conv | nn.BatchNorm2d | None, c1: int, g: int):
        import numpy as np
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            bn = branch.bn
            running_mean, running_var = bn.running_mean, bn.running_var
            gamma, beta, eps = bn.weight, bn.bias, bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            input_dim = c1 // g
            kernel_value = np.zeros((c1, input_dim, 3, 3), dtype=np.float32)
            for i in range(c1):
                kernel_value[i, i % input_dim, 1, 1] = 1.0
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if isinstance(kernel1x1, int) and kernel1x1 == 0:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def fuse_convs(self) -> None:
        if self._fused:
            return
        k3, b3 = self._fuse_bn_tensor(self.conv1, self.c1, self.g)
        k1, b1 = self._fuse_bn_tensor(self.conv2, self.c1, self.g)
        kid, bid = self._fuse_bn_tensor(self.bn_id, self.c1, self.g)
        kernel = k3 + self._pad_1x1_to_3x3_tensor(k1) + kid
        bias = b3 + b1 + bid
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for p in self.parameters():
            p.detach_()
        del self.conv1, self.conv2
        if hasattr(self, "bn_id") and self.bn_id is not None:
            del self.bn_id
        self._fused = True


class ChannelAttention(nn.Module):
    """Atención por canal (SE-like)."""
    def __init__(self, c: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Atención espacial (conv(2->1) sobre [mean,max] por canal)."""
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in {3, 7}
        padding = 3 if kernel_size == 7 else 1
        self.cv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, 1, keepdim=True)
        x_max, _ = torch.max(x, 1, keepdim=True)
        return x * self.act(self.cv(torch.cat([x_mean, x_max], 1)))


class CBAM(nn.Module):
    """CBAM: Channel + Spatial attention en serie."""
    def __init__(self, c: int, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class Concat(nn.Module):
    """Concatena tensores en la dimensión especificada (por defecto canales)."""
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, self.dim)


class Index(nn.Module):
    """Devuelve el índice i-ésimo de una lista de tensores."""
    def __init__(self, index: int = 0) -> None:
        super().__init__()
        self.index = index

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return xs[self.index]
