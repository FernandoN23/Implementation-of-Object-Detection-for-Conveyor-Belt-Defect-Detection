# models/blocks.py
"""
Bloques fundamentales del modelo YOLOv11.
Cada bloque está diseñado como un módulo reutilizable para Backbone, Neck o Head.
Incluye comentarios detallados para claridad en investigación y desarrollo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 🔹 Bloque convolucional base (Conv)
# -----------------------------------------------------------------------------
class Conv(nn.Module):
    """
    Bloque base: Conv2d -> BatchNorm -> SiLU
    Similar al usado en YOLOv5-v11.
    Parámetros:
      c1: canales de entrada
      c2: canales de salida
      k: tamaño del kernel
      s: stride
      p: padding (por defecto automático = k//2)
      g: groups (para depthwise si g=c1)
      act: usar o no activación (True/False)
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -----------------------------------------------------------------------------
# 🔹 Convolución depthwise separable (DWConv)
# -----------------------------------------------------------------------------
class DWConv(nn.Module):
    """
    Depthwise + Pointwise Conv.
    Reduce FLOPs y parámetros separando el filtrado espacial (depthwise)
    del mezclado de canales (pointwise 1x1 conv).
    """
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        # Depthwise (filtra cada canal por separado)
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)
        # Pointwise (mezcla canales)
        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


# -----------------------------------------------------------------------------
# 🔹 Bottleneck residual (usado dentro de C3k2)
# -----------------------------------------------------------------------------
class Bottleneck(nn.Module):
    """
    Bloque residual básico: dos convoluciones (1x1 -> 3x3) con conexión residual.
    """
    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        c_ = int(c2 / 2)  # canales intermedios
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2  # usar skip connection solo si canales iguales

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


# -----------------------------------------------------------------------------
# 🔹 Bloque C3k2 (base del YOLOv11 Backbone)
# -----------------------------------------------------------------------------
class C3k2(nn.Module):
    """
    C3k2 = Bloque CSP con 2 ramas y N Bottlenecks.
    Es la base de YOLOv11, reemplaza al antiguo C2f (de YOLOv8).

    Estructura:
      Entrada -> split en 2 caminos ->
        camino1: stack de Bottlenecks
        camino2: shortcut directo
      -> concat -> conv final (fusión)
    """
    def __init__(self, c1, c2, n=1, c3k_flag=False):
        super().__init__()
        c_ = int(c2 // 2)
        # Rama 1: convolución + varios bottlenecks
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_) for _ in range(max(1, n))])
        # Rama 2: proyección directa
        self.cv2 = Conv(c1, c_, 1, 1)
        # Convolución de salida que mezcla ambas ramas
        self.cv3 = Conv(2 * c_, c2, 1)
        # Flag opcional (c3k_flag): YOLOv11 lo usa para variantes grandes (no afecta estructura aquí)
        self.c3k_flag = c3k_flag

    def forward(self, x):
        # Concatenamos la salida de la rama residual con la shortcut
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# -----------------------------------------------------------------------------
# 🔹 SPPF (Spatial Pyramid Pooling - Fast)
# -----------------------------------------------------------------------------
class SPPF(nn.Module):
    """
    Realiza pooling en múltiples escalas (kernel fijo 5 por defecto)
    para capturar contexto espacial global sin aumentar coste.
    """
    def __init__(self, c1, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c1, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # Concatena resultados de diferentes profundidades de pooling
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


# -----------------------------------------------------------------------------
# 🔹 Bloque PSA (Partial Self-Attention)
# -----------------------------------------------------------------------------
class PSABlock(nn.Module):
    """
    PSA = Partial Self-Attention Block.
    Implementa atención canal-espacial simplificada, aplicada en baja resolución.
    """
    def __init__(self, c, reduction=4):
        super().__init__()
        c_hidden = c // reduction
        self.conv1 = Conv(c, c_hidden, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c_hidden, c_hidden, 1),
            nn.Sigmoid()
        )
        self.conv2 = Conv(c_hidden, c, 1, act=False)

    def forward(self, x):
        u = self.conv1(x)
        w = self.attn(u)
        out = self.conv2(u * w)
        return x + out  # residual


# -----------------------------------------------------------------------------
# 🔹 Bloque C2PSA (usa 2 o más PSABlocks en secuencia)
# -----------------------------------------------------------------------------
class C2PSA(nn.Module):
    """
    C2PSA = bloque compuesto de atención parcial (Partial Self Attention).
    Usado en la parte final del backbone (menor resolución).
    """
    def __init__(self, c, n=2):
        super().__init__()
        self.blocks = nn.Sequential(*[PSABlock(c) for _ in range(n)])

    def forward(self, x):
        return self.blocks(x)


# -----------------------------------------------------------------------------
# 🔹 Bloque Concat (utilizado en el Neck y Head)
# -----------------------------------------------------------------------------
class Concat(nn.Module):
    """
    Simplemente concatena una lista de tensores en una dimensión dada.
    Usado en el Neck (FPN/PAN) para combinar feature maps.
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x es una lista de tensores
        return torch.cat(x, self.dim)


# -----------------------------------------------------------------------------
# 🔹 Detect (Head final YOLO anchor-free)
# -----------------------------------------------------------------------------
class Detect(nn.Module):
    """
    Módulo de detección final.
    Genera para cada escala:
      [B, nc+5, H, W]
    donde:
      - 4: coordenadas (x, y, w, h)
      - 1: objectness
      - nc: clases

    in_channels: lista con canales de entrada por nivel (P3, P4, P5)
    nc: número de clases
    """
    def __init__(self, in_channels, nc=80):
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # número de salidas por celda
        self.stride = [8, 16, 32]  # escalas típicas de salida
        self.m = nn.ModuleList(
            [nn.Sequential(DWConv(c, c, 3), Conv(c, c, 3), nn.Conv2d(c, self.no, 1))
             for c in in_channels]
        )

    def forward(self, feats):
        # feats = lista de feature maps [P3, P4, P5]
        outputs = []
        for i, x in enumerate(feats):
            outputs.append(self.m[i](x))
        return outputs  # lista de tensores
