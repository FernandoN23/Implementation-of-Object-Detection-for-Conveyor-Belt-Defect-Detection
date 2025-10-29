# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: head.py
# Head de detección YOLOv11 (anchor-free, decoupled head).
# - Clasificación: dos DWConv + Conv1×1 (profundidad separable, ligero).
# - Regresión: dos Conv estándar + Conv1×1 → (4 * reg_max) canales (DFL-ready).
# Devuelve listas por nivel: dict(cls=[...], reg=[...]).
#==============================================================

from __future__ import annotations
from typing import List, Tuple, Dict
import torch
import torch.nn as nn

from .nn.conv import Conv, DWConv

__all__ = ["DetectHead"]


class _ClsRegBranch(nn.Module):
    """Pequeña rama configurable (cls o reg) con 2 capas y un proyector final."""

    def __init__(self, c: int, out: int, depthwise: bool, final_bias: float = 0.0, name: str = "") -> None:
        super().__init__()
        Block = DWConv if depthwise else Conv
        self.cv1 = Block(c, c, k=3, s=1)
        self.cv2 = Block(c, c, k=3, s=1)
        self.proj = nn.Conv2d(c, out, kernel_size=1, bias=True)
        if final_bias:
            nn.init.constant_(self.proj.bias, final_bias)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cv2(self.cv1(x)))


class DetectHead(nn.Module):
    """
    Head decouplada por nivel (P3, P4, P5).
    - Clasificación con depthwise separable (ligero, como en YOLOv10/11).
    - Regresión preparada para DFL (reg_max bins por coordenada).
    """

    def __init__(
        self,
        nc: int,                          # número de clases
        ch: Tuple[int, int, int],         # canales de entrada por nivel (desde Neck)
        reg_max: int = 16,                # bins DFL (4 * reg_max canales)
        use_dw_for_cls: bool = True,      # cls con DWConv
        cls_bias: float = -4.5,           # inicialización conservadora (p~1%)
    ) -> None:
        super().__init__()
        assert len(ch) == 3, "Se esperan 3 niveles (P3, P4, P5)."
        self.nc = nc
        self.reg_max = reg_max

        c3, c4, c5 = ch
        c_in = [c3, c4, c5]

        self.cls_branches = nn.ModuleList([
            _ClsRegBranch(c=c_in[i], out=nc, depthwise=use_dw_for_cls, final_bias=cls_bias, name=f"cls_p{i+3}")
            for i in range(3)
        ])
        self.reg_branches = nn.ModuleList([
            _ClsRegBranch(c=c_in[i], out=4 * reg_max, depthwise=False, final_bias=0.0, name=f"reg_p{i+3}")
            for i in range(3)
        ])

        # Conveniencia: strides típicos (8,16,32). Se pueden recalcular en el wrapper del modelo si es necesario.
        self.register_buffer("strides", torch.tensor([8., 16., 32.]))

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            feats: [P3(80×80), P4(40×40), P5(20×20)] desde la Neck
        Returns:
            dict con:
              - 'cls': [B, nc, H, W] por nivel
              - 'reg': [B, 4*reg_max, H, W] por nivel (para DFL o box decoding)
        """
        assert len(feats) == 3, "DetectHead espera 3 mapas de características."

        cls_out, reg_out = [], []
        for i, x in enumerate(feats):
            reg_out.append(self.reg_branches[i](x))
            cls_out.append(self.cls_branches[i](x))
        return {"cls": cls_out, "reg": reg_out}
