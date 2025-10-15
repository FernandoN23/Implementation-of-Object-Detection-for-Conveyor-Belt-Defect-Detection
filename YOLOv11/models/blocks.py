# blocks.py
"""
Bloques base para YOLOv11 (PyTorch)

Contiene:
- Conv: Conv2d + BatchNorm + Activation
- DWConv: Depthwise separable conv (Conv DW + PW)
- Bottleneck: bloque residual simple
- C3: Cross Stage Partial style block (variante de C3 usado en YOLO family)
- C3k2: alias de C3 con n=2 por defecto
- Concat: concatenación simple en el canal
- Upsample: wrapper sobre nn.Upsample (nearest)
- SPPF: Spatial Pyramid Pooling - Fast
- Focus: reagrupación espacial -> canales (opcional, usado en YOLOv5)
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
print("DEBUG: entrando a blocks.py")

def autopad(k: int, p: Optional[int] = None):
    # retorna padding "same" para convs con stride=1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Conv + BN + Activation (SiLU por defecto)"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, activation: Optional[Callable] = nn.SiLU()):
        super().__init__()
        p = autopad(k, p)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x if self.act is None else self.act(x)


class DWConv(nn.Module):
    """Depthwise separable convolution: DW conv then PW conv"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, activation: Optional[Callable] = nn.SiLU()):
        super().__init__()
        self.dw = Conv(in_ch, in_ch, k=k, s=s, groups=in_ch, activation=activation)
        self.pw = Conv(in_ch, out_ch, k=1, s=1, activation=activation)

    def forward(self, x):
        return self.pw(self.dw(x))


class Bottleneck(nn.Module):
    """Bottleneck residual block: Conv(in->hidden) -> Conv(hidden->out) + shortcut optional"""
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, g: int = 1, expansion: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.conv1 = Conv(in_ch, hidden_ch, k=1, s=1)
        self.conv2 = Conv(hidden_ch, out_ch, k=3, s=1, groups=g)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y


class C3(nn.Module):
    """
    C3 block (Cross Stage Partial / CSP style)
    Typical structure:
      - split input -> path1 and path2
      - path1: sequence of n Bottlenecks
      - path2: conv 1x1
      - concat -> conv 1x1
    Args:
        in_ch, out_ch: channels
        n: number of bottlenecks in the main path
        expansion: channel expansion factor inside bottlenecks
        shortcut: use residual inside bottleneck
    """
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, expansion: float = 0.5, shortcut: bool = True):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        # convs that reduce channels for both paths
        self.cv1 = Conv(in_ch, hidden_ch, k=1, s=1)  # path a (bottlenecks)
        self.cv2 = Conv(in_ch, hidden_ch, k=1, s=1)  # path b (shortcut / identity path)
        # sequence of bottlenecks
        self.m = nn.Sequential(*[Bottleneck(hidden_ch, hidden_ch, shortcut=shortcut, expansion=1.0) for _ in range(n)])
        # final conv to mix
        self.cv3 = Conv(2 * hidden_ch, out_ch, k=1, s=1)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class C3k2(C3):
    """Alias para C3 con n=2 por defecto (C3k2)"""
    def __init__(self, in_ch: int, out_ch: int, expansion: float = 0.5, shortcut: bool = True):
        super().__init__(in_ch, out_ch, n=2, expansion=expansion, shortcut=shortcut)


class Concat(nn.Module):
    """Concatenate tensors en dim=1 (canal). Útil en la parte neck / fpn"""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x expected to be a list/tuple of tensors
        return torch.cat(x, dim=self.dim)


class Upsample(nn.Module):
    """Wrapper de Upsample (nearest) - usado para FPN/Neck"""
    def __init__(self, scale: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode)


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast variant
    SPPF: maxpool k=5 repeated to aggregate receptive fields
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = Conv(in_ch, hidden_ch, k=1, s=1)
        self.cv2 = Conv(hidden_ch * 4, out_ch, k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        # concatenar original más 2 niveles pooling (y1,y2) y pasar conv final
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), dim=1))


class Focus(nn.Module):
    """
    Focus layer (como en YOLOv5): reagrupa 4 sub-cuadrantes espaciales -> canales
    Input: (B,C,H,W) -> Output: (B, 4C, H/2, W/2)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = Conv(in_ch * 4, out_ch, k=k, s=s)

    def forward(self, x):
        # x[b, c, h, w] -> y concatenando 4 sub-samples
        # top-left, top-right, bottom-left, bottom-right
        return self.conv(torch.cat((x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]), dim=1))


# Small test to validate shapes (executar directamente para debug)
if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)
    conv = Conv(64, 128)
    print("Conv out:", conv(x).shape)
    dw = DWConv(64, 128)
    print("DWConv out:", dw(x).shape)
    bott = Bottleneck(128, 128)
    y = conv(x)
    print("Bottleneck out:", bott(y).shape)
    c3 = C3(128, 256, n=3)
    print("C3 out:", c3(y).shape)
    c3k2 = C3k2(128, 256)
    print("C3k2 out:", c3k2(y).shape)
    cat = Concat()
    print("Concat out:", cat([y, y]).shape)
    up = Upsample(2)
    print("Upsample out:", up(y).shape)
    sppf = SPPF(128, 256)
    print("SPPF out:", sppf(y).shape)
    focus = Focus(3, 32)
    im = torch.randn(1, 3, 640, 640)
    print("Focus out:", focus(im).shape)
