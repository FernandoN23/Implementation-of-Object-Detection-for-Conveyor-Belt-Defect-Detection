"""
losses.py (versión estable)
---------------------------------
Adaptada para YOLOv11.
Asegura compatibilidad entre predicciones densas [B, N, C]
y targets de dataset [B, M, C].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloLoss(nn.Module):
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, preds, targets):
        """
        Pérdida YOLOv11 simplificada con soporte multi-escala.
        Normaliza PRED y TARGET a [B, N, C_pred] y concatena en N.
        """

        # ===== Normalizar PREDICCIONES a [B, N, C_pred] =====
        if isinstance(preds, (list, tuple)):
            p_list = []
            for p in preds:
                if p.ndim == 5:  # [B, A, H, W, C]
                    B, A, H, W, C = p.shape
                    p = p.view(B, A * H * W, C)
                elif p.ndim == 4:  # [B, C, H, W]
                    B, C, H, W = p.shape
                    p = p.view(B, C, H * W).permute(0, 2, 1)
                elif p.ndim == 3:  # [B, N, C]
                    pass
                else:
                    raise ValueError(f"Formato de pred inesperado: {p.shape}")
                p_list.append(p)
            preds = torch.cat(p_list, dim=1)  # [B, N_total, C]
        elif preds.ndim == 4:  # [B, C, H, W]
            B, C, H, W = preds.shape
            preds = preds.view(B, C, H * W).permute(0, 2, 1)
        elif preds.ndim == 3:
            pass
        else:
            raise ValueError(f"Formato de pred inesperado: {preds.shape}")

        B, N_pred, C_pred = preds.shape

        # --- helper: normaliza un tensor target cualquiera a [B, N, C_pred]
        def _to_BNC(t):
            # 5D: [B, A, H, W, C?]
            if t.ndim == 5:
                B_t, A_t, H_t, W_t, C_t = t.shape
                # si el último canal no es C_pred, recortamos/expandimos
                if C_t != C_pred:
                    C_use = min(C_t, C_pred)
                    t = t[..., :C_use]
                    if C_use < C_pred:
                        pad = C_pred - C_use
                        t = torch.cat([t, torch.zeros(B_t, A_t, H_t, W_t, pad, device=t.device, dtype=t.dtype)], dim=-1)
                return t.view(B_t, A_t * H_t * W_t, C_pred)

            # 4D: puede ser [B, C, H, W] o [B, H, W, C]
            if t.ndim == 4:
                B_t, d1, d2, d3 = t.shape
                # caso canal-last: [B, H, W, C]
                if d3 == C_pred:
                    return t.view(B_t, d1 * d2, C_pred)
                # caso canal-first: [B, C, H, W]
                if d1 == C_pred:
                    return t.view(B_t, C_pred, d2 * d3).permute(0, 2, 1)
                # si ninguno coincide con C_pred, intentamos inferir y adaptar
                # hipótesis 1: d1 es H, d2 es W, d3 es C distinto
                if d3 in (4, 5, 25, 75):  # C típico
                    C_use = min(d3, C_pred)
                    t = t[..., :C_use]
                    if C_use < C_pred:
                        pad = C_pred - C_use
                        t = torch.cat([t, torch.zeros(B_t, d1, d2, pad, device=t.device, dtype=t.dtype)], dim=-1)
                    return t.view(B_t, d1 * d2, C_pred)
                # hipótesis 2: d1 es C distinto
                if d1 in (4, 5, 25, 75):
                    C_use = min(d1, C_pred)
                    t = t[:, :C_use, :, :]
                    if C_use < C_pred:
                        pad = C_pred - C_use
                        t = torch.cat([t, torch.zeros(B_t, pad, d2, d3, device=t.device, dtype=t.dtype)], dim=1)
                    return t.view(B_t, C_pred, d2 * d3).permute(0, 2, 1)
                raise ValueError(f"No se pudo ajustar target 4D a [B,N,C_pred]: {t.shape}, C_pred={C_pred}")

            # 3D: [B, N, C?] o [B, C?, N]
            if t.ndim == 3:
                B_t, a, b = t.shape
                # caso [B, N, C?]
                if b == C_pred:
                    return t
                # caso [B, C?, N]
                if a == C_pred:
                    return t.permute(0, 2, 1).contiguous()
                # si ninguna coincide, forzamos a que el último sea C_pred
                if b in (4, 5, 25, 75):
                    C_use = min(b, C_pred)
                    t = t[..., :C_use]
                    if C_use < C_pred:
                        pad = C_pred - C_use
                        t = torch.cat([t, torch.zeros(B_t, a, pad, device=t.device, dtype=t.dtype)], dim=-1)
                    return t
                if a in (4, 5, 25, 75):
                    C_use = min(a, C_pred)
                    t = t[:, :C_use, :]
                    if C_use < C_pred:
                        pad = C_pred - C_use
                        t = torch.cat([t, torch.zeros(B_t, pad, b, device=t.device, dtype=t.dtype)], dim=1)
                    return t.permute(0, 2, 1).contiguous()
                raise ValueError(f"No se pudo ajustar target 3D a [B,N,C_pred]: {t.shape}, C_pred={C_pred}")

            raise ValueError(f"Formato de target inesperado: {t.shape}")

        # ===== Normalizar TARGETS a [B, N_total, C_pred] =====
        if isinstance(targets, (list, tuple)):
            t_list = []
            for t in targets:
                t = t.to(dtype=preds.dtype, device=preds.device)
                t = _to_BNC(t)
                # validar batch y C
                if t.shape[0] != B:
                    raise ValueError(f"Batch mismatch en targets: {t.shape[0]} vs {B}")
                if t.shape[2] != C_pred:
                    raise ValueError(f"C mismatch en targets: {t.shape[2]} vs {C_pred}")
                t_list.append(t)
            targets = torch.cat(t_list, dim=1)  # [B, N_total, C_pred]
        else:
            targets = targets.to(dtype=preds.dtype, device=preds.device)
            targets = _to_BNC(targets)
            if targets.shape[0] != B or targets.shape[2] != C_pred:
                raise ValueError(f"Target único no compatible: {targets.shape}, esperado B={B}, C={C_pred}")

        # ===== Split componentes =====
        pred_box = preds[..., 0:4]
        pred_obj = preds[..., 4:5]
        pred_cls = preds[..., 5:]

        targ_box = targets[..., 0:4]
        targ_obj = targets[..., 4:5]
        targ_cls = targets[..., 5:]

        # ===== Ajuste clases si difieren =====
        if targ_cls.shape[-1] != pred_cls.shape[-1]:
            m = min(targ_cls.shape[-1], pred_cls.shape[-1])
            pred_cls = pred_cls[..., :m]
            targ_cls = targ_cls[..., :m]

        # ===== Pérdidas =====
        box_loss = F.smooth_l1_loss(pred_box, targ_box, reduction='mean')
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, targ_obj, reduction='mean')
        cls_loss = F.mse_loss(pred_cls, targ_cls, reduction='mean')

        total_loss = (
                self.lambda_box * box_loss +
                self.lambda_obj * obj_loss +
                self.lambda_cls * cls_loss
        )

        loss_items = {
            "box_loss": box_loss.item(),
            "obj_loss": obj_loss.item(),
            "cls_loss": cls_loss.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, loss_items
