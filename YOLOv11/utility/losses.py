# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: losses.py
# Implementación de pérdidas para YOLOv11 (anchor-free) con fidelidad
# a Ultralytics (DFL + CIoU + Clasificación en positivos ponderada
# por calidad IoU). Incluye asignador center-based ligero con margen
# dependiente de stride (eps·stride) para estabilidad CPU/GPU y
# self-test determinista que reutiliza los mismos tensores en CPU/GPU.
#==============================================================

from __future__ import annotations

import argparse
import json
import math
import random
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
    """Fija semillas para reproducibilidad CPU/GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilidades geométricas básicas
# -----------------------------

def _xywhn_to_xyxy_pix(xywhn: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Convierte (cx,cy,w,h) normalizado [0,1] a (x1,y1,x2,y2) en pixeles."""
    H, W = hw
    if xywhn.numel() == 0:
        return xywhn.new_zeros((0, 4))
    cx, cy, w, h = xywhn.unbind(-1)
    x1 = (cx - w * 0.5) * W
    y1 = (cy - h * 0.5) * H
    x2 = (cx + w * 0.5) * W
    y2 = (cy + h * 0.5) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _ciou_xyxy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU entre dos sets alineados (N,4)-(N,4) en XYXY. Retorna (N,)."""
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    # IoU
    inter = (torch.minimum(pred[:, 2:], target[:, 2:]) - torch.maximum(pred[:, :2], target[:, :2])).clamp(min=0).prod(1)
    area_p = (pred[:, 2:] - pred[:, :2]).clamp(min=0).prod(1)
    area_t = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(1)
    union = area_p + area_t - inter + eps
    iou = inter / union
    # Distancia de centros
    pc = (pred[:, :2] + pred[:, 2:]) * 0.5
    tc = (target[:, :2] + target[:, 2:]) * 0.5
    center_dist = ((pc - tc) ** 2).sum(1)
    # Caja envolvente
    x1 = torch.minimum(pred[:, 0], target[:, 0])
    y1 = torch.minimum(pred[:, 1], target[:, 1])
    x2 = torch.maximum(pred[:, 2], target[:, 2])
    y2 = torch.maximum(pred[:, 3], target[:, 3])
    c2 = ((x2 - x1) ** 2 + (y2 - y1) ** 2) + eps
    # Relación de aspecto
    wp = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    hp = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    wt = (target[:, 2] - target[:, 0]).clamp(min=eps)
    ht = (target[:, 3] - target[:, 1]).clamp(min=eps)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(wp / hp) - torch.atan(wt / ht), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (center_dist / c2 + alpha * v)


# ---------------------------------
# Conversión distancias <-> bounding
# ---------------------------------

def dist2bbox(dist: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False) -> torch.Tensor:
    """Convierte distancias l,t,r,b a caja XYXY (o XYWH) respecto al punto ancla."""
    l, t, r, b = dist.unbind(-1)
    x1y1 = anchor_points - torch.stack((l, t), dim=-1)
    x2y2 = anchor_points + torch.stack((r, b), dim=-1)
    if xywh:
        cxcy = (x1y1 + x2y2) * 0.5
        wh = (x2y2 - x1y1).clamp(min=0)
        return torch.cat((cxcy, wh), -1)
    return torch.cat((x1y1, x2y2), -1)


def bbox2dist(anchor_points: torch.Tensor, boxes_xyxy: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Distancias l,t,r,b desde anchor_points a bordes de boxes_xyxy (clamp [0, reg_max])."""
    if boxes_xyxy.numel() == 0:
        return anchor_points.new_zeros((anchor_points.shape[0], 4))
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    px, py = anchor_points.unbind(-1)
    l = (px - x1).clamp(min=0, max=reg_max - 0.01)
    t = (py - y1).clamp(min=0, max=reg_max - 0.01)
    r = (x2 - px).clamp(min=0, max=reg_max - 0.01)
    b = (y2 - py).clamp(min=0, max=reg_max - 0.01)
    return torch.stack((l, t, r, b), dim=-1)


# -------------------
# Generación de anclas
# -------------------

def make_anchors_from_shapes(
    feat_shapes: List[Tuple[int, int]], strides: torch.Tensor, device: torch.device, offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Genera puntos ancla (x,y) por nivel (P3,P4,P5). Retorna (A,2) y (A,1)."""
    assert len(feat_shapes) == len(strides)
    all_points, all_strides = [], []
    for (h, w), s in zip(feat_shapes, strides):
        yv, xv = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        points = torch.stack((xv.flatten() + offset, yv.flatten() + offset), dim=-1) * float(s)
        all_points.append(points)
        all_strides.append(torch.full((h * w, 1), float(s), device=device))
    return torch.cat(all_points, 0), torch.cat(all_strides, 0)


# ---------------
# Pérdidas básicas
# ---------------

class DFLoss(nn.Module):
    """Distribution Focal Loss (GFLv2), devuelve loss por elemento (N,)."""
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred_dist.numel() == 0:
            return pred_dist.new_zeros((0,))
        t = target.clamp_(0, self.reg_max - 1 - 1e-3)
        tl = t.long()
        tr = (tl + 1).clamp(max=self.reg_max - 1)
        wl = tr - t
        wr = 1.0 - wl
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none")
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none")
        return loss_l * wl.view_as(loss_l) + loss_r * wr.view_as(loss_r)


@dataclass
class LossHyperparams:
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5


class CenterAssigner:
    """Asignador anchor-free ligero con margen dependiente de stride (eps·stride)."""
    def __init__(self, top_radius: float = 2.5, eps_rel: float = 1e-3) -> None:
        self.top_radius = top_radius
        self.eps_rel = eps_rel  # margen relativo al stride para comparaciones estrictas

    @torch.no_grad()
    def __call__(
        self,
        anchor_points: torch.Tensor,        # (A, 2) en pixeles
        strides_per_anchor: torch.Tensor,   # (A, 1)
        gt_xyxy: torch.Tensor,              # (M, 4) en pixeles
        gt_cls: torch.Tensor,               # (M,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = anchor_points.size(0)
        device = anchor_points.device

        if gt_xyxy.numel() == 0:
            # Sin GT → no positivos
            return (
                torch.zeros((A, 4), device=device),
                torch.zeros((A, 1), device=device),  # será relleno a nc aguas arriba
                torch.zeros((A,), dtype=torch.bool, device=device),
            )

        ap = anchor_points[:, None, :]  # (A,1,2)
        x1y1 = gt_xyxy[None, :, :2]
        x2y2 = gt_xyxy[None, :, 2:]

        # Margen proporcional al stride: evita flips CPU/GPU en puntos frontera
        eps = (self.eps_rel * strides_per_anchor).view(-1, 1, 1)  # (A,1,1)

        # inside e in_center con comparaciones estrictas y margen
        inside = ((ap > x1y1 + eps) & (ap < x2y2 - eps)).all(-1)  # (A,M)

        # Center sampling con radio dependiente del stride
        gtc = (gt_xyxy[:, :2] + gt_xyxy[:, 2:]) * 0.5  # (M,2)
        radius = (self.top_radius * strides_per_anchor).view(-1, 1, 1)  # (A,1,1)
        tl = gtc.unsqueeze(0) - radius
        br = gtc.unsqueeze(0) + radius
        in_center = ((ap > tl + eps) & (ap < br - eps)).all(-1)  # (A,M)

        cand = inside & in_center
        if not cand.any():
            cand = inside

        # Resolver colisiones por área mínima
        areas = ((gt_xyxy[:, 2:] - gt_xyxy[:, :2]).clamp(min=0).prod(-1))  # (M,)
        areas_expand = areas[None, :].repeat(A, 1)
        areas_expand[~cand] = float("inf")
        min_idx = areas_expand.argmin(dim=1)  # (A,)
        fg_mask = cand.gather(1, min_idx.unsqueeze(1)).squeeze(1)  # (A,)

        # Targets
        target_bboxes = gt_xyxy[min_idx]  # (A,4)
        target_scores = torch.zeros((A, int(gt_cls.max().item()) + 1 if gt_cls.numel() else 1), device=device)
        if fg_mask.any():
            target_scores[fg_mask, gt_cls[min_idx[fg_mask]].long()] = 1.0
        return target_bboxes, target_scores, fg_mask


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
        self.assigner = assigner or CenterAssigner(top_radius=2.5)
        self.safe_fp32 = safe_fp32
        self.cls_pos_only = cls_pos_only
        self.use_iou_weight = use_iou_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl_loss = DFLoss(reg_max=reg_max)
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))

    @staticmethod
    def _stack_levels(xs: List[torch.Tensor]) -> torch.Tensor:
        """Concatena niveles [B,C,H,W] -> [B, H*W, C]."""
        return torch.cat([x.flatten(2).transpose(1, 2).contiguous() for x in xs], dim=1)

    def _decode_dist(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """DFL: [B,A,4*reg_max] -> distancias ltrb [B,A,4]."""
        B, A, C = pred_dist.shape
        pred = pred_dist.view(B, A, 4, self.reg_max).softmax(-1).matmul(self.proj.to(pred_dist.dtype))
        return pred  # (B,A,4)

    def forward(self, preds: Dict[str, List[torch.Tensor]], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = preds["cls"][0].device

        feats_cls = preds["cls"]
        feats_reg = preds["reg"]
        assert len(feats_cls) == len(feats_reg) == int(self.strides_buf.numel())

        # Dimensiones por nivel
        feat_shapes = [tuple(x.shape[-2:]) for x in feats_cls]
        B = feats_cls[0].shape[0]
        H0, W0 = feat_shapes[0]
        imgsz_hw = (int(H0 * self.strides_buf[0].item()), int(W0 * self.strides_buf[0].item()))  # (H,W)

        # Apilado niveles -> [B, A, C] y cast seguro a FP32 (parche ROCm/autocast)
        pred_scores_all = self._stack_levels(feats_cls).float()          # (B,A,nc)
        pred_dist_all   = self._stack_levels(feats_reg).float()          # (B,A,4*reg_max)

        # Anchors
        anchor_points, stride_tensor = make_anchors_from_shapes(
            feat_shapes, self.strides_buf.to(device), device, offset=0.5
        )  # (A,2),(A,1)

        # DFL decode -> XYXY pred
        pred_ltrb   = self._decode_dist(pred_dist_all)                                   # (B,A,4)
        pred_bboxes = dist2bbox(pred_ltrb, anchor_points[None, :, :], xywh=False)        # (B,A,4) pix

        # Preparar GT por imagen (soporte de imágenes sin etiquetas)
        A = pred_scores_all.size(1)
        target_scores = torch.zeros((B, A, self.nc), device=device)
        target_bboxes = torch.zeros((B, A, 4), device=device)
        fg_mask       = torch.zeros((B, A), dtype=torch.bool, device=device)

        if targets.numel() > 0:
            img_ids     = targets[:, 0].long()
            gt_cls_all  = targets[:, 1].long()
            gt_xywhn    = targets[:, 2:]
            gt_xyxy_pix = _xywhn_to_xyxy_pix(gt_xywhn, imgsz_hw)  # (N,4)

            for b in range(B):
                m = (img_ids == b)
                gt_b = gt_xyxy_pix[m]
                gt_c = gt_cls_all[m]
                tb, ts, fm = self.assigner(anchor_points, stride_tensor, gt_b, gt_c)
                # Asegura (A, nc) aunque no aparezcan todas las clases en la imagen
                if ts.size(1) != self.nc:
                    ts = F.pad(ts, (0, self.nc - ts.size(1)))
                target_bboxes[b] = tb
                target_scores[b] = ts
                fg_mask[b]       = fm

        # Conteo de positivos (global en el batch) y denominadores seguros
        num_pos = int(fg_mask.sum().detach().item())
        pos_mask_any = fg_mask.any()
        den_all = torch.clamp(target_scores.sum(), min=1.0)  # ≈ número de positivos (suma de one-hot)

        # === Clasificación (por defecto: sólo positivos; opcional: ponderación por IoU) ===
        if self.cls_pos_only:
            if pos_mask_any:
                ts_pos   = target_scores[fg_mask]
                pred_pos = pred_scores_all[fg_mask][:, :self.nc]
                if self.use_iou_weight:
                    with torch.no_grad():
                        iou_pos = _ciou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask]).clamp(min=0.0, max=1.0)
                        qual = iou_pos.unsqueeze(-1)  # (Npos,1)
                    ts_pos = (ts_pos * qual).to(pred_pos.dtype)
                den_cls  = torch.clamp(ts_pos.sum(), min=1.0)
                cls_loss = self.bce(pred_pos, ts_pos).sum() / den_cls
            else:
                cls_loss = pred_scores_all.sum() * 0.0
        else:
            # Alternativa (no recomendada): BCE en todos los anclajes normalizada por positivos
            cls_loss = self.bce(pred_scores_all[:, :, :self.nc], target_scores).sum() / den_all

        # === DFL + Caja (sólo positivos) ===
        if pos_mask_any:
            # Objetivos DFL: distancias ltrb desde anchor al GT
            ap_expanded      = anchor_points.unsqueeze(0).expand(B, -1, 2).reshape(-1, 2)        # (B*A,2)
            target_ltrb_full = bbox2dist(ap_expanded, target_bboxes.reshape(-1, 4), self.reg_max).view(B, -1, 4)
            # DFL en positivos
            pred_dist_pos   = pred_dist_all[fg_mask].contiguous().view(-1, self.reg_max)          # (Npos*4, reg_max)
            target_ltrb_pos = target_ltrb_full[fg_mask].reshape(-1)                               # (Npos*4,)
            dfl_loss        = self.dfl_loss(pred_dist_pos, target_ltrb_pos).sum() / den_all
            # CIoU en positivos
            iou      = _ciou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask]).clamp(min=-1.0, max=1.0)
            box_loss = (1.0 - iou).sum() / den_all
        else:
            dfl_loss = pred_dist_all.sum() * 0.0
            box_loss = pred_dist_all.sum() * 0.0

        # Ponderación final
        loss = self.hyp.box * box_loss + self.hyp.cls * cls_loss + self.hyp.dfl * dfl_loss
        scalars = {
            "loss": float(loss.detach().item()),
            "loss_box": float(box_loss.detach().item()),
            "loss_cls": float(cls_loss.detach().item()),
            "loss_dfl": float(dfl_loss.detach().item()),
            "num_pos": num_pos,
        }
        return loss, scalars


# ------------------
# Prueba autónoma (CPU/GPU) determinista
# ------------------

def _synthetic_preds(B=2, nc=5, reg_max=16, device="cpu", requires_grad=False, base=None):
    """Genera predicciones sintéticas con shapes P3/P4/P5 para imgsz=640 (strides 8/16/32).
       Si base es un dict con tensores en CPU, replica exactamente esos valores en 'device'.
    """
    shapes = [(80, 80), (40, 40), (20, 20)]
    if base is None:
        feats_cls = [torch.randn(B, nc, h, w, device=device, requires_grad=requires_grad) for (h, w) in shapes]
        feats_reg = [torch.randn(B, 4 * reg_max, h, w, device=device, requires_grad=requires_grad) for (h, w) in shapes]
    else:
        feats_cls = [t.detach().to(device).requires_grad_(requires_grad) for t in base["cls"]]
        feats_reg = [t.detach().to(device).requires_grad_(requires_grad) for t in base["reg"]]
    return {"cls": feats_cls, "reg": feats_reg}


def _synthetic_targets(B=2, num_per_img=2, nc=5, device="cpu", include_empty=True, base=None):
    """Targets normalizados [img_i, cls, cx, cy, w, h]. Puede incluir imágenes sin etiquetas.
       Si base es un tensor en CPU, replica exactamente esos valores en 'device'.
    """
    if base is not None:
        return base.detach().to(device)
    all_t = []
    for b in range(B):
        k = num_per_img
        if include_empty and b == 0:
            k = 0  # fuerza una imagen sin etiquetas para validar el flujo
        for _ in range(k):
            cls = torch.randint(0, nc, (1,), device=device).float()
            w = torch.rand(1, device=device) * 0.4 + 0.1
            h = torch.rand(1, device=device) * 0.4 + 0.1
            cx = torch.rand(1, device=device) * (1 - w) + 0.5 * w
            cy = torch.rand(1, device=device) * (1 - h) + 0.5 * h
            row = torch.cat([torch.tensor([b], device=device).float(), cls, cx, cy, w, h])
            all_t.append(row)
    if len(all_t) == 0:
        return torch.zeros((0, 6), device=device)
    return torch.stack(all_t, dim=0)


def _select_gpu_device() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


def _self_test(mode: str = "CPU"):
    """Ejecuta pruebas sintéticas en CPU y/o GPU con TENSORES IDÉNTICOS.
       Valores: 'CPU' | 'GPU' | 'CPU/GPU'.
    """
    mode = mode.strip().upper()
    run_cpu = mode in ("CPU", "CPU/GPU")
    run_gpu = mode in ("GPU", "CPU/GPU")

    results = {}
    B, nc, reg_max = 2, 5, 16

    # 1) Genera SIEMPRE datos base EN CPU con semilla fija
    _set_seed(0)
    preds_base_cpu = _synthetic_preds(B=B, nc=nc, reg_max=reg_max, device="cpu", requires_grad=True, base=None)
    targets_base_cpu = _synthetic_targets(B=B, num_per_img=2, nc=nc, device="cpu", include_empty=True, base=None)

    # 2) Eval CPU
    if run_cpu:
        preds = _synthetic_preds(B=B, nc=nc, reg_max=reg_max, device="cpu", requires_grad=True, base=preds_base_cpu)
        targets = _synthetic_targets(B=B, num_per_img=2, nc=nc, device="cpu", include_empty=True, base=targets_base_cpu)
        criterion = YOLOLoss(nc=nc, reg_max=reg_max, strides=(8, 16, 32))
        loss, scalars = criterion(preds, targets)
        loss.backward()
        results["CPU"] = scalars

    # 3) Eval GPU con EXACTOS MISMOS TENSORES
    if run_gpu:
        dev = _select_gpu_device()
        if dev is None:
            results["GPU"] = {"error": "No GPU disponible (torch.cuda/mps no detectado)."}
        else:
            preds = _synthetic_preds(B=B, nc=nc, reg_max=reg_max, device=dev, requires_grad=True, base=preds_base_cpu)
            targets = _synthetic_targets(B=B, num_per_img=2, nc=nc, device=dev, include_empty=True, base=targets_base_cpu)
            criterion = YOLOLoss(nc=nc, reg_max=reg_max, strides=(8, 16, 32)).to(dev)
            loss, scalars = criterion(preds, targets)
            loss.backward()
            if dev == "cuda":
                torch.cuda.synchronize()
            results["GPU"] = scalars

    print("[Self-Test] Resultados")
    print(json.dumps(results, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 Utility Losses — Self test")
    parser.add_argument(
        "--self-test",
        nargs="?",
        const="CPU",
        default=None,
        help="Ejecuta prueba sintética. Valores: CPU, GPU, CPU/GPU (por defecto CPU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.self_test:
        _self_test(args.self_test)
    else:
        print("Uso: python losses.py --self-test [CPU | GPU | CPU/GPU]")