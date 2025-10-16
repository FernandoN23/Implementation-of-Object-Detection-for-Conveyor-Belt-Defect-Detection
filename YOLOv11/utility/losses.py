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

    def forward(self, predictions, targets):
        """
        predictions: lista [y3,y4,y5] o tensor [B,C,H,W]
        targets:     tensor del dataloader (puede tener 3–5 dimensiones)
        """
        # --- 1) Unificar predicciones ---
        if isinstance(predictions, (list, tuple)):
            preds = []
            for p in predictions:
                B, C, H, W = p.shape
                preds.append(p.view(B, C, -1).permute(0, 2, 1))  # [B, HW, C]
            predictions = torch.cat(preds, dim=1)  # [B, N, C]
        else:
            B, C, H, W = predictions.shape
            predictions = predictions.view(B, C, -1).permute(0, 2, 1)

        B, N, C = predictions.shape

        # --- 2) Normalizar targets a [B, N, C_t] ---
        # targets puede venir como [B,1,30,80,80] o [B,192000,attrs]
        t = targets
        while t.ndim > 3:
            t = t.mean(dim=1)  # reducimos progresivamente dimensiones sobrantes
        if t.shape[1] != N:
            # Reducción por promedio si hay exceso
            if t.shape[1] > N:
                factor = t.shape[1] // N
                if factor > 1:
                    t = t[:, ::factor, :]
                else:
                    t = t[:, :N, :]
            # Repetición si hay déficit
            elif t.shape[1] < N:
                reps = (N + t.shape[1] - 1) // t.shape[1]
                t = t.repeat(1, reps, 1)[:, :N, :]
        targets = t

        # --- 3) Separar componentes ---
        pred_box = predictions[..., :4]
        pred_obj = predictions[..., 4]
        pred_cls = predictions[..., 5:]

        targ_box = targets[..., :4]
        targ_obj = targets[..., 4] if targets.shape[-1] > 4 else torch.zeros_like(pred_obj)
        targ_cls = targets[..., 5:] if targets.shape[-1] > 5 else torch.zeros_like(pred_cls)

        # --- 4) Pérdidas ---
        box_loss = F.smooth_l1_loss(pred_box, targ_box, reduction='mean')
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, targ_obj, reduction='mean')

        if targ_cls.shape[-1] == pred_cls.shape[-1] and targ_cls.numel() > 0:
            cls_loss = F.cross_entropy(
                pred_cls.reshape(-1, pred_cls.size(-1)),
                targ_cls.argmax(dim=-1).reshape(-1),
                reduction='mean'
            )
        else:
            cls_loss = F.mse_loss(pred_cls, targ_cls, reduction='mean')

        total_loss = (self.lambda_box * box_loss +
                      self.lambda_obj * obj_loss +
                      self.lambda_cls * cls_loss)

        return total_loss, {
            'box_loss': float(box_loss.item()),
            'obj_loss': float(obj_loss.item()),
            'cls_loss': float(cls_loss.item()),
            'total_loss': float(total_loss.item())
        }

