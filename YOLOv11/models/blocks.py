# blocks.py
"""
Bloques base para YOLOv11 (PyTorch)

Contiene:
- Conv: Conv2d + Normalization + Activation
- DWConv: Depthwise separable conv (Conv DW + PW)
- Bottleneck, C3, C3k2, Concat, Upsample, SPPF, Focus
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
#print("DEBUG: entrando a blocks.py")

# ==============================
# Normalization Factory
# ==============================
def make_norm(norm_type: str, num_channels: int, gn_groups: int = 32):
    norm_type = (norm_type or "bn").lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    if norm_type == "gn":
        g = min(gn_groups, num_channels) or 1
        for cand in (gn_groups, 32, 16, 8, 4, 2, 1):
            if num_channels % cand == 0:
                g = cand
                break
        return nn.GroupNorm(g, num_channels, affine=True)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=True)
    if norm_type == "id":
        return nn.Identity()
    return nn.BatchNorm2d(num_channels)


def autopad(k: int, p: Optional[int] = None):
    if p is None:
        p = k // 2
    return p


# ==============================
# Conv Block
# ==============================
class Conv(nn.Module):
    """Conv + Normalization + Activation (SiLU por defecto)"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, activation: Optional[Callable] = nn.SiLU(),
                 norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        p = autopad(k, p)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = make_norm(norm_type, out_ch, gn_groups)
        self.act = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x if self.act is None else self.act(x)


# ==============================
# Otros bloques (sin cambios estructurales)
# ==============================
class DWConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 activation: Optional[Callable] = nn.SiLU(),
                 norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        self.dw = Conv(in_ch, in_ch, k=k, s=s, groups=in_ch,
                       activation=activation, norm_type=norm_type, gn_groups=gn_groups)
        self.pw = Conv(in_ch, out_ch, k=1, s=1,
                       activation=activation, norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x):
        return self.pw(self.dw(x))


class Bottleneck(nn.Module):
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


class C3k2(C3):
    def __init__(self, in_ch: int, out_ch: int, expansion: float = 0.5, shortcut: bool = True,
                 norm_type: str = "bn", gn_groups: int = 32):
        super().__init__(in_ch, out_ch, n=2, expansion=expansion, shortcut=shortcut,
                         norm_type=norm_type, gn_groups=gn_groups)


class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class Upsample(nn.Module):
    def __init__(self, scale: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode)


class SPPF(nn.Module):
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


class Focus(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, norm_type: str = "bn", gn_groups: int = 32):
        super().__init__()
        self.conv = Conv(in_ch * 4, out_ch, k=k, s=s, norm_type=norm_type, gn_groups=gn_groups)
    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]), dim=1))
