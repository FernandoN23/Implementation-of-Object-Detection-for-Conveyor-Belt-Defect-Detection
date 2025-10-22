"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: losses.py (versión funcional)
Define la pérdida YOLOv11 simplificada basada en BCE + IoU + Focal Loss.
-------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Función auxiliar: IoU entre cajas (en formato xywh)
# -------------------------------------------------------------
def bbox_iou(box1, box2, eps=1e-7):
    # Convierte xywh → xyxy
    def xywh2xyxy(x):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    b1, b2 = xywh2xyxy(box1), xywh2xyxy(box2)
    inter = (torch.min(b1[:, 2], b2[:, 2]) - torch.max(b1[:, 0], b2[:, 0])).clamp(0) * \
            (torch.min(b1[:, 3], b2[:, 3]) - torch.max(b1[:, 1], b2[:, 1])).clamp(0)
    area1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    return inter / (area1 + area2 - inter + eps)


# -------------------------------------------------------------
# Focal Loss (BCE modificado)
# -------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        p_t = target * prob + (1 - target) * (1 - prob)
        mod_factor = (1 - p_t) ** self.gamma
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_factor * mod_factor * bce_loss).mean()


# -------------------------------------------------------------
# YoloLoss funcional
# -------------------------------------------------------------
class YoloLoss(nn.Module):
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.focal_loss = FocalLoss()

    def forward(self, preds, targets):
        """
        preds: lista de 3 tensores [y3, y4, y5], cada uno [B, C, H, W]
        targets: lista de tensores [N_i,5] o tensor [N,6] con [img_idx, cls, x, y, w, h]
        """
        device = preds[0].device
        B = preds[0].shape[0]

        # Unifica predicciones de las 3 escalas
        preds_cat = []
        for p in preds:
            b, c, h, w = p.shape
            p = p.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
            preds_cat.append(p)
        preds = torch.cat(preds_cat, dim=1)  # [B, N_pred, C]
        pred_box = preds[..., :4]
        pred_obj = preds[..., 4:5]
        pred_cls = preds[..., 5:]

        # Targets vacíos → pérdida nula
        if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "box_loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0, "total_loss": 0.0
            }

        # Inicializa acumuladores
        box_loss, obj_loss, cls_loss = 0.0, 0.0, 0.0
        total_pos = 0

        # Procesa por imagen
        for b in range(B):
            t = targets[targets[:, 0] == b]
            if t.numel() == 0:
                continue
            total_pos += len(t)

            # Predicciones globales (no grid): selección simple por índice aleatorio
            pred_b = pred_box[b].sigmoid()
            pred_o = pred_obj[b]
            pred_c = pred_cls[b]

            # Matching simplificado: busca el pixel más cercano (puedes mejorar con IoU)
            for row in t:
                cls, x, y, w, h = row[1:].to(device)
                gt_box = torch.tensor([[x, y, w, h]], device=device)
                ious = bbox_iou(pred_b, gt_box.repeat(len(pred_b), 1))
                idx = ious.argmax()
                iou = ious[idx]

                # L_box = (1 - IoU)
                box_loss += (1.0 - iou)

                # L_obj = FocalLoss sobre objectness
                obj_target = torch.zeros_like(pred_o)
                obj_target[idx] = 1.0
                obj_loss += self.focal_loss(pred_o, obj_target)

                # L_cls = BCE/Focal sobre clases
                cls_target = torch.zeros_like(pred_c)
                cls_target[idx, int(cls)] = 1.0
                cls_loss += self.focal_loss(pred_c, cls_target)

        # Normalización
        total_pos = max(total_pos, 1)
        box_loss /= total_pos
        obj_loss /= total_pos
        cls_loss /= total_pos

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