# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: yolo11.py
# Ensamblado de YOLOv11: Backbone + Neck + Head (anchor-free).
# - Parametrización por variantes (d, w, mc).
# - Head decouplada con clasificación para nc clases y regresión DFL.
# - Utilidades: actualización de strides, decodificación de cajas.
#==============================================================

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .neck import Neck
from .head import DetectHead

__all__ = ["YOLOv11", "build_model", "VARIANTS"]

# Tabla de variantes (según diagrama del proyecto)
VARIANTS = {
    "n":  {"d": 0.50, "w": 0.25, "mc": 1024},
    "s":  {"d": 0.50, "w": 0.50, "mc": 1024},
    "m":  {"d": 0.50, "w": 1.00, "mc": 512},
    "l":  {"d": 1.00, "w": 1.50, "mc": 512},
    "xl": {"d": 1.00, "w": 1.50, "mc": 512},  # según esquema adjunto (misma cota de canales)
}


class YOLOv11(nn.Module):
    """
    Ensamble YOLOv11 (anchor-free) con 3 escalas P3/P4/P5.
    Salida cruda: dict {'cls': [B,nc,H,W]*3, 'reg': [B,4*reg_max,H,W]*3}.
    Métodos auxiliares para decodificar (DFL) y concatenar predicciones.
    """

    def __init__(
        self,
        nc: int = 5,
        variant: str = "n",
        d: Optional[float] = None,
        w: Optional[float] = None,
        mc: Optional[int] = None,
        in_ch: int = 3,
        reg_max: int = 16,
        imgsz_for_strides: int = 640,
    ) -> None:
        super().__init__()
        # Resolver hiperparámetros de escalado
        if variant not in VARIANTS and (d is None or w is None or mc is None):
            raise ValueError(f"Variante '{variant}' no válida y (d,w,mc) no especificados.")
        d = VARIANTS.get(variant, {}).get("d", d)
        w = VARIANTS.get(variant, {}).get("w", w)
        mc = VARIANTS.get(variant, {}).get("mc", mc)

        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.variant = variant
        self.hparams = {"d": float(d), "w": float(w), "mc": int(mc)}
        self.in_ch = in_ch

        # Módulos
        self.backbone = Backbone(d=d, w=w, mc=mc, in_ch=in_ch)
        self.neck = Neck(ch=self.backbone.out_channels, d=d)
        self.head = DetectHead(nc=nc, ch=self.neck.out_channels, reg_max=reg_max, use_dw_for_cls=True)

        # Inicializar strides con forward "dummy"
        self.register_buffer("_strides", torch.zeros(3))
        self.update_strides(imgsz_for_strides)

    # -----------------------
    # Construcción y utilidades
    # -----------------------
    @torch.no_grad()
    def update_strides(self, imgsz: int = 640) -> None:
        """
        Propaga un tensor simulado para calcular los strides a partir de la
        reducción espacial efectiva por nivel (imgsz / H).
        """
        device = next(self.parameters()).device
        x = torch.zeros(1, self.in_ch, imgsz, imgsz, device=device)
        feats = self.neck(self.backbone(x))
        strides = []
        for f in feats:
            H = f.shape[-2]
            strides.append(imgsz / H)
        s = torch.tensor(strides, device=device, dtype=torch.float32)
        self._strides.copy_(s)
        # Mantener coherente la Head
        self.head.strides.copy_(s)

    @property
    def strides(self) -> torch.Tensor:
        return self._strides

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, x: torch.Tensor, *, decode: bool = False, concat: bool = False) -> Dict[str, List[torch.Tensor]] | Dict[str, torch.Tensor]:
        """
        Args:
            x: imagen/es [B,3,H,W]
            decode: si True, aplica decodificación DFL -> cajas absolutas en pixeles
            concat: si True (y decode=True), concatena niveles (B, N, 4 / nc)
        Returns:
            - decode=False: {'cls': list[3], 'reg': list[3]}
            - decode=True, concat=False:
                {'cls': list[3] (B,nc,H,W), 'dist': list[3](B,4,H,W), 'boxes': list[3](B,4,H,W)}
            - decode=True, concat=True:
                {'cls': (B,N,nc), 'boxes': (B,N,4), 'strides': (3,)}
        """
        feats = self.neck(self.backbone(x))
        out = self.head(feats)  # dict {'cls': [..], 'reg': [..]}

        if not decode:
            return out

        # Decodificación DFL (distancias l,t,r,b)
        dist, boxes = [], []
        for i in range(3):
            dmap = self._dfl(out["reg"][i])  # (B,4,H,W) en celdas
            bmap = self._dist2bbox(dmap, out["reg"][i].shape[-2:], stride=float(self.strides[i]))
            dist.append(dmap)
            boxes.append(bmap)

        if not concat:
            return {"cls": out["cls"], "dist": dist, "boxes": boxes}

        # Aplanar niveles
        cls_cat, box_cat = self._concat_levels(out["cls"], boxes)
        return {"cls": cls_cat, "boxes": box_cat, "strides": self.strides.detach().cpu()}

    # -----------------------
    # Decodificación (DFL y cajas)
    # -----------------------
    def _dfl(self, reg: torch.Tensor) -> torch.Tensor:
        """
        Distribution Focal Loss decoding:
        reg: [B, 4*reg_max, H, W] -> distancias esperadas (l,t,r,b) en celdas (B,4,H,W)
        """
        B, C, H, W = reg.shape
        m = self.reg_max
        reg = reg.view(B, 4, m, H, W)
        prob = F.softmax(reg, dim=2)
        bins = torch.arange(m, device=reg.device, dtype=reg.dtype).view(1, 1, m, 1, 1)
        exp = (prob * bins).sum(dim=2)  # (B,4,H,W)
        return exp

    def _dist2bbox(self, dist: torch.Tensor, hw: Tuple[int, int], stride: float) -> torch.Tensor:
        """
        Convierte distancias (l,t,r,b) en cajas absolutas [x1,y1,x2,y2] en pixeles del input.
        dist: (B,4,H,W), hw=(H,W)
        """
        B, _, H, W = dist.shape
        # Centros de celdas (en pixeles)
        y, x = torch.meshgrid(torch.arange(H, device=dist.device),
                              torch.arange(W, device=dist.device), indexing="ij")
        cx = (x + 0.5) * stride
        cy = (y + 0.5) * stride

        l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        l = l * stride
        t = t * stride
        r = r * stride
        b = b * stride

        x1 = cx[None] - l
        y1 = cy[None] - t
        x2 = cx[None] + r
        y2 = cy[None] + b
        return torch.stack([x1, y1, x2, y2], dim=1)  # (B,4,H,W)

    def _concat_levels(self, cls_maps: List[torch.Tensor], box_maps: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplana y concatena niveles espaciales.
        Returns:
            cls_cat: (B, N, nc) logits
            box_cat: (B, N, 4)  cajas en pixeles
        """
        cls_list, box_list = [], []
        for c, b in zip(cls_maps, box_maps):
            B, Nc, H, W = c.shape
            cls_list.append(c.permute(0, 2, 3, 1).reshape(B, H * W, Nc))
            box_list.append(b.permute(0, 2, 3, 1).reshape(B, H * W, 4))
        cls_cat = torch.cat(cls_list, dim=1)
        box_cat = torch.cat(box_list, dim=1)
        return cls_cat, box_cat


# -----------------------
# Fábrica
# -----------------------
def build_model(
    variant: str = "n",
    nc: int = 5,
    *,
    d: Optional[float] = None,
    w: Optional[float] = None,
    mc: Optional[int] = None,
    in_ch: int = 3,
    reg_max: int = 16,
    imgsz_for_strides: int = 640,
) -> YOLOv11:
    """
    Helper para construir el modelo desde configs o scripts de entrenamiento.
    - Si 'variant' ∈ VARIANTS, ignora d/w/mc si no se pasan.
    - Si se proveen d/w/mc, estos predominan sobre la variante.
    """
    if d is not None and w is not None and mc is not None:
        # Permitir override explícito
        return YOLOv11(nc=nc, variant=variant, d=d, w=w, mc=mc, in_ch=in_ch, reg_max=reg_max, imgsz_for_strides=imgsz_for_strides)
    return YOLOv11(nc=nc, variant=variant, in_ch=in_ch, reg_max=reg_max, imgsz_for_strides=imgsz_for_strides)


# -----------------------
# Prueba rápida
# -----------------------
if __name__ == "__main__":
    model = build_model("m", nc=5)
    x = torch.randn(1, 3, 640, 640)
    out_raw = model(x)                           # crudo
    out_dec = model(x, decode=True, concat=True) # decodificado + concatenado

    print("Strides:", model.strides.tolist())
    for i, (c, r) in enumerate(zip(out_raw["cls"], out_raw["reg"])):
        print(f"P{i+3} -> cls: {tuple(c.shape)}, reg: {tuple(r.shape)}")
    print("Concat (cls, boxes):", out_dec["cls"].shape, out_dec["boxes"].shape)
