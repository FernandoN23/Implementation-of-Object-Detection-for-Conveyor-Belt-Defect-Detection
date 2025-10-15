"""
losses.py
---------------------------------
Definición de pérdidas principales utilizadas por YOLOv11.
Incluye: pérdida de bounding boxes, clasificación y confianza.
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
        predictions: salida del modelo [B, anchors, grid, attributes]
        targets: etiquetas reales [B, num_targets, attributes]
        """
        box_loss = F.smooth_l1_loss(predictions[..., :4], targets[..., :4])
        obj_loss = F.binary_cross_entropy_with_logits(predictions[..., 4], targets[..., 4])
        cls_loss = F.cross_entropy(predictions[..., 5:], targets[..., 5:].argmax(dim=-1))

        total_loss = (self.lambda_box * box_loss +
                      self.lambda_obj * obj_loss +
                      self.lambda_cls * cls_loss)

        return total_loss, {
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
