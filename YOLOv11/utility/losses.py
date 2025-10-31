# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: losses.py
# Implementación de pérdidas para YOLOv11 (anchor-free): DFL + CIoU +
# clasificación (positivos, ponderada por IoU). Incluye asignador
# center-based con broadcasting A×N y construcción correcta de strides
# por ancla (A,) para evitar choques de forma. Self-test ampliado con
# targets no vacíos. ***FIX***: DFL ahora recibe los *logits* de reg
# (N,4,reg_max) y los *targets* se calculan a partir de GT vs centros
# (en *bins*), no desde la predicción decodificada. Además, las
# distancias DFL y las cajas para CIoU usan stride por ancla.
#==============================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilidades generales / seeds
# -----------------------------

def _set_seed(seed: int = 0):
    """Fija semilla global para reproducibilidad mínima (self-test)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _xywhn_to_xyxy_pix(xywhn: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = img_hw
    cx, cy, w, h = xywhn.unbind(-1)
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return torch.stack([x1, y1, x2, y2], -1)


# -----------------------------
# Hiperparámetros de la pérdida
# -----------------------------

@dataclass
class LossHyperparams:
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5


# -----------------------------
# DFL (Distribution Focal Loss)
# -----------------------------

class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (N, 4, reg_max) *logits*; target: (N, 4) flotante en [0, reg_max)
        n = pred.shape[0]
        if n == 0:
            return pred.sum() * 0.0
        pred = pred.view(-1, self.reg_max)           # (N*4, reg_max)
        target = target.view(-1)                      # (N*4,)
        t_left = target.floor().clamp(0, self.reg_max - 1)
        t_right = (t_left + 1).clamp(0, self.reg_max - 1)
        wl = (t_right - target).clamp(0, 1)
        wr = 1 - wl
        loss = (F.cross_entropy(pred, t_left.long(), reduction="none") * wl +
                F.cross_entropy(pred, t_right.long(), reduction="none") * wr)
        return loss.mean()


# -----------------------------
# Asignador center-based con broadcasting A×N
# -----------------------------

class CenterAssigner:
    def __init__(self, radius_eps: float = 0.2) -> None:
        """radius_eps: radio (en múltiplos de stride) para la región central.
        Se usa rad_norm = radius_eps * stride_per_anchor / img_size_pix.
        """
        self.radius_eps = radius_eps

    def __call__(self,
                 centers_norm: torch.Tensor,            # (A,2) en [0,1]
                 stride_per_anchor: torch.Tensor,       # (A,) en pixeles
                 targets_norm: torch.Tensor,            # (N,6) [img, cls, cx, cy, w, h] norm.
                 img_size_pix: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = centers_norm.device
        A = centers_norm.shape[0]
        if targets_norm.numel() == 0:
            return (torch.zeros(A, dtype=torch.bool, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, 4, device=device))

        # Coordenadas de anchors (normalizadas)
        gx, gy = centers_norm[:, 0], centers_norm[:, 1]          # (A,)
        # Radio normalizado por ancla
        rad_norm = (self.radius_eps * stride_per_anchor.to(device).float()) / float(img_size_pix)  # (A,)

        # GTs normalizados
        cls = targets_norm[:, 1].long()                          # (N,)
        cx, cy, w, h = targets_norm[:, 2], targets_norm[:, 3], targets_norm[:, 4], targets_norm[:, 5]

        # Broadcasting A×N
        gx2 = gx.unsqueeze(1)          # (A,1)
        gy2 = gy.unsqueeze(1)          # (A,1)
        rad2 = rad_norm.unsqueeze(1)   # (A,1)
        cx2, cy2 = cx.unsqueeze(0), cy.unsqueeze(0)  # (1,N)

        mask = (gx2 >= (cx2 - rad2)) & (gx2 <= (cx2 + rad2)) & \
               (gy2 >= (cy2 - rad2)) & (gy2 <= (cy2 + rad2))      # (A,N)

        pos_mask = mask.any(dim=1)  # (A,)
        if pos_mask.any():
            # Selección simple: primer GT que cae dentro para cada ancla positiva
            sel_all = mask.float().argmax(dim=1)      # (A,)
            sel = sel_all[pos_mask]                   # (A_pos,)
            matched_cls = cls[sel]
            matched_xywh = torch.stack([cx[sel], cy[sel], w[sel], h[sel]], dim=1)
            return pos_mask, matched_cls, matched_xywh
        else:
            return (pos_mask,
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, 4, device=device))


# -----------------------------
# Utilidades de anclaje/decodif.
# -----------------------------

def make_anchors_from_shapes(shapes: List[Tuple[int, int]], strides: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # shapes: [(H3,W3),(H4,W4),(H5,W5)]
    grids = []
    centers = []
    for (H, W), s in zip(shapes, strides):
        y, x = torch.meshgrid(torch.arange(H, device=strides.device), torch.arange(W, device=strides.device), indexing='ij')
        grid = torch.stack([x, y], dim=-1).view(-1, 2)
        grids.append(grid)
        centers.append((grid + 0.5) * s)
    grid_all = torch.cat(grids, dim=0).float()      # (A,2) en celdas
    centers_all = torch.cat(centers, dim=0).float() # (A,2) en pixeles
    return grid_all, centers_all


def dist2bbox(dist: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # dist: (N,4) ltrb, centers: (N,2)
    l, t, r, b = dist.unbind(dim=-1)
    cx, cy = centers.unbind(dim=-1)
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2dist_bins(gt_xyxy: torch.Tensor, centers_pix: torch.Tensor, stride_per_anchor: torch.Tensor, reg_max: int, eps: float = 1e-3) -> torch.Tensor:
    """Convierte GT (pixeles) y centros (pixeles) a distancias l,t,r,b en *bins*.
    bins = dist_pix / stride. Clampea a [0, reg_max-ε].
    Entradas:
      - gt_xyxy: (N,4) pixeles
      - centers_pix: (N,2) pixeles
      - stride_per_anchor: (N,) pixeles
    Salida: (N,4) flotante en [0, reg_max)
    """
    cx, cy = centers_pix[:, 0], centers_pix[:, 1]
    x1, y1, x2, y2 = gt_xyxy.unbind(-1)
    s = stride_per_anchor.float().clamp(min=1.0)
    l = (cx - x1) / s
    t = (cy - y1) / s
    r = (x2 - cx) / s
    b = (y2 - cy) / s
    out = torch.stack([l, t, r, b], dim=-1)
    out = out.clamp(0.0, float(reg_max - 1) - eps)
    return out


# -----------------------------
# CIoU
# -----------------------------

def _ciou_xyxy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # pred/target: (N,4) [x1,y1,x2,y2]
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    parea = pw * ph
    tarea = tw * th
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    union = parea + tarea - inter + eps
    iou = inter / union
    # Distancia entre centros
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=eps)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou


# -----------------------------
# Pérdida principal
# -----------------------------

class YOLOLoss(nn.Module):
    """Pérdida principal YOLOv11: DFL + CIoU + BCE (positivos, ponderada por IoU)."""
    def __init__(
        self,
        nc: int,
        reg_max: int = 16,
        strides: Tuple[int, int, int] = (8, 16, 32),
        hyp: Optional[LossHyperparams] = None,
        assigner: Optional[CenterAssigner] = None,
        safe_fp32: bool = True,
        cls_pos_only: bool = True,
        use_iou_weight: bool = True,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = reg_max * 4 + nc
        self.register_buffer("strides_buf", torch.tensor(list(strides), dtype=torch.float32))
        self.hyp = hyp or LossHyperparams()
        self.assigner = assigner or CenterAssigner(radius_eps=0.2)
        self.safe_fp32 = safe_fp32
        self.cls_pos_only = cls_pos_only
        self.use_iou_weight = use_iou_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl_loss = DFLoss(reg_max=reg_max)
        # IMPORTANTE: registrar como buffer para moverse con .to(device)
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))

    @staticmethod
    def _stack_levels(xs: List[torch.Tensor]) -> torch.Tensor:
        """Concatena niveles [B,C,H,W] en (B, A, C) donde A=H*W total."""
        B, C, H, W = xs[0].shape
        return torch.cat([x.view(B, C, -1).transpose(1, 2) for x in xs], dim=1)

    def _decode_dist(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """DFL: [B,A,4*reg_max] -> distancias ltrb *en bins* [B,A,4]."""
        B, A, C = pred_dist.shape
        proj = self.proj.to(pred_dist.device, dtype=pred_dist.dtype)
        pred = pred_dist.view(B, A, 4, self.reg_max).softmax(-1).matmul(proj)
        return pred  # (B,A,4) en bins (unidad: stride)

    def forward(self, preds: Dict[str, List[torch.Tensor]], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = preds["cls"][0].device

        feats_cls = preds["cls"]
        feats_reg = preds["reg"]
        assert len(feats_cls) == len(feats_reg) == int(self.strides_buf.numel())

        # Dimensiones por nivel
        feat_shapes = [tuple(x.shape[2:]) for x in feats_cls]  # (H,W) por nivel
        B = feats_cls[0].shape[0]

        # Agrupar niveles
        cls_all = self._stack_levels(feats_cls)           # (B,A,nc)
        reg_all_logits = self._stack_levels(feats_reg)    # (B,A,4*reg_max) *logits*
        A = cls_all.shape[1]

        # Anchors y centers en pixeles
        strides_t = self.strides_buf.to(device)
        shapes = feat_shapes
        grid, centers = make_anchors_from_shapes(shapes, strides_t)
        grid = grid.to(device)
        centers = centers.to(device)

        # Stride por ancla (A,)
        stride_per_anchor = torch.cat([
            torch.full((H * W,), float(s), device=device) for (H, W), s in zip(shapes, strides_t)
        ], dim=0)

        # Imagen "efectiva" en pixeles (lado)
        img_size_pix = float(centers.max().item())
        centers_norm = centers / img_size_pix

        # Decodificación DFL de distancias (en *bins*) y conversión a pixeles para CIoU
        pred_ltrb_bins = self._decode_dist(reg_all_logits)            # (B,A,4) en bins
        pred_ltrb_pix = pred_ltrb_bins * stride_per_anchor.view(1, A, 1)  # (B,A,4) pixeles
        pred_boxes  = dist2bbox(pred_ltrb_pix.view(-1, 4), centers.repeat(B, 1))  # (B*A,4)
        pred_boxes  = pred_boxes.view(B, A, 4)

        # Pérdidas acumuladas
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        for b in range(B):
            if targets.numel():
                tmask = (targets[:, 0].long() == b)
                t_b = targets[tmask]
            else:
                t_b = targets

            if t_b.numel():
                pos_mask, matched_cls, matched_xywh = self.assigner(centers_norm, stride_per_anchor, t_b, img_size_pix)
            else:
                pos_mask = torch.zeros(A, dtype=torch.bool, device=device)
                matched_cls = torch.empty(0, dtype=torch.long, device=device)
                matched_xywh = torch.empty(0, 4, device=device)

            pm = pos_mask
            if pm.any():
                # Boxes predichas y GT en pixeles
                pb = pred_boxes[b, pm]
                gt_xywh = matched_xywh
                gt_xyxy = _xywhn_to_xyxy_pix(gt_xywh, (int(img_size_pix), int(img_size_pix)))

                # CIoU (pixeles)
                ciou = _ciou_xyxy(pb, gt_xyxy)
                loss_box = loss_box + (1.0 - ciou).mean()

                # --- DFL ---
                # *pred* = logits (N_pos, 4, reg_max)
                reg_logits_pos = reg_all_logits[b, pm]  # (N_pos, 4*reg_max)
                reg_logits_pos = reg_logits_pos.view(-1, 4, self.reg_max)
                # *target* = distancias GT vs centro en *bins*
                centers_pos = centers[pm]
                stride_pos = stride_per_anchor[pm]
                target_bins = bbox2dist_bins(gt_xyxy, centers_pos, stride_pos, self.reg_max)
                loss_dfl = loss_dfl + self.dfl_loss(reg_logits_pos, target_bins)

                # Clasificación en positivos
                logits = cls_all[b, pm]
                cls_targets = torch.zeros_like(logits)
                cls_targets[torch.arange(logits.size(0), device=device), matched_cls] = 1.0
                if self.use_iou_weight:
                    w = ciou.detach().clamp(0, 1).unsqueeze(1)
                    loss_cls = loss_cls + (F.binary_cross_entropy_with_logits(logits, cls_targets, reduction='none') * w).mean()
                else:
                    loss_cls = loss_cls + F.binary_cross_entropy_with_logits(logits, cls_targets, reduction='mean')
            else:
                # Sin positivos: no contribuye (mantener tipo/dtype)
                loss_cls = loss_cls + (cls_all[b].sigmoid().mean() * 0.0)

        # Ponderaciones
        loss = self.hyp.box * loss_box + self.hyp.cls * loss_cls + self.hyp.dfl * loss_dfl
        scalars = {"box": float(loss_box.detach()), "cls": float(loss_cls.detach()), "dfl": float(loss_dfl.detach())}
        return loss, scalars


# -----------------------------
# Self-test ampliado (CPU/GPU)
# -----------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-test", type=str, default="CPU/GPU", choices=["CPU", "GPU", "CPU/GPU"], help="Ejecuta test determinista con/ sin GTs")
    args = p.parse_args()

    _set_seed(0)

    # Config de prueba
    B, nc, reg_max = 2, 5, 16
    imgsz = 320
    H3, W3 = imgsz // 8, imgsz // 8
    H4, W4 = imgsz // 16, imgsz // 16
    H5, W5 = imgsz // 32, imgsz // 32

    # Tensores de predicción falsos por nivel (logits para reg)
    preds_template = {
        "cls": [torch.randn(B, nc, H3, W3), torch.randn(B, nc, H4, W4), torch.randn(B, nc, H5, W5)],
        "reg": [torch.randn(B, reg_max * 4, H3, W3), torch.randn(B, reg_max * 4, H4, W4), torch.randn(B, reg_max * 4, H5, W5)],
    }

    # Targets no vacíos en formato [img_i, cls, cx, cy, w, h] normalizado
    targets = torch.tensor([
        [0, 4, 0.62, 0.65, 0.30, 0.25],  # imagen 0
        [1, 2, 0.40, 0.35, 0.20, 0.15],  # imagen 1
        [1, 0, 0.70, 0.70, 0.10, 0.10],  # imagen 1 (segundo GT)
    ], dtype=torch.float32)

    def run_on(device: torch.device) -> Dict[str, float]:
        preds = {k: [t.to(device) for t in v] for k, v in preds_template.items()}
        loss_fn = YOLOLoss(nc=nc, reg_max=reg_max).to(device)
        L, S = loss_fn(preds, targets.to(device))
        return {"loss": float(L), **S}

    results = {}
    if args.self_test in ("CPU", "CPU/GPU"):
        results["CPU"] = run_on(torch.device("cpu"))
    if args.self_test in ("GPU", "CPU/GPU") and torch.cuda.is_available():
        results["GPU"] = run_on(torch.device("cuda"))

    print("[Self-Test] Resultados")
    print(results)
