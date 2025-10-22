# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# IoU (xywh en [0,1] esperado)
# ---------------------------
def bbox_iou(box1, box2, eps=1e-7):
    def xywh2xyxy(x):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    b1, b2 = xywh2xyxy(box1), xywh2xyxy(box2)
    inter_w = (torch.min(b1[:, 2], b2[:, 2]) - torch.max(b1[:, 0], b2[:, 0])).clamp(min=0)
    inter_h = (torch.min(b1[:, 3], b2[:, 3]) - torch.max(b1[:, 1], b2[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred_logits, target):
        # pred_logits: cualquier forma; target del mismo shape
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        prob = torch.sigmoid(pred_logits).detach()  # estabiliza el modulating term
        p_t = target * prob + (1 - target) * (1 - prob)
        mod = (1 - p_t).clamp(min=0, max=1) ** self.gamma
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_t * mod * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class YoloLoss(nn.Module):
    """
    Pérdida YOLOv11 simplificada (sin decodificación de grid/anchors):
    - Box: 1 - IoU en el índice "mejor emparejado" por IoU
    - Obj: Focal en positivos + minado duro de negativos (top-k por logit)
    - Cls: Focal en el índice positivo (one-hot)

    NOTA: Se espera que las GT estén normalizadas a [0,1] (xywh).
    Acepta `targets` como:
      * Tensor [N,6]: [img_idx, cls, x, y, w, h]
      * Lista len=B de tensores [N_i,5]: [cls, x, y, w, h]
    """
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5,
                 hard_neg_ratio=3, max_negatives=300):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.focal = FocalLoss(gamma=2.0, alpha=0.25)
        self.hard_neg_ratio = hard_neg_ratio
        self.max_negatives = max_negatives

    @staticmethod
    def _unify_targets(targets, batch_size, device):
        """
        Devuelve una lista de long B con tensores [N_i,5]: [cls, x, y, w, h]
        """
        if isinstance(targets, torch.Tensor):
            if targets.numel() == 0:
                return [torch.empty((0, 5), device=device) for _ in range(batch_size)]
            # targets: [N,6] -> split por img_idx
            out = [torch.empty((0, 5), device=device) for _ in range(batch_size)]
            img_idx = targets[:, 0].long()
            cls_xywh = targets[:, 1:].to(device)
            for b in range(batch_size):
                mask = img_idx == b
                out[b] = cls_xywh[mask]
            return out
        elif isinstance(targets, (list, tuple)):
            # lista por-imagen
            out = []
            for t in targets:
                if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
                    out.append(torch.empty((0, 5), device=device))
                else:
                    out.append(t.to(device))
            # padding si fuese más corta que B
            while len(out) < batch_size:
                out.append(torch.empty((0, 5), device=device))
            return out[:batch_size]
        else:
            # Cualquier otro caso -> batch vacío
            return [torch.empty((0, 5), device=device) for _ in range(batch_size)]

    def forward(self, preds, targets):
        """
        preds: lista [P3, P4, P5], cada uno [B, C, H, W] con C = 5 + num_classes
               (orden: x,y,w,h,obj, cls0..clsC-1) en logits sin activación.
        targets: ver _unify_targets
        """
        device = preds[0].device
        B = preds[0].shape[0]

        # Concatena predicciones de escalas -> [B, N_pred, C]
        feats = []
        for p in preds:
            b, c, h, w = p.shape
            feats.append(p.view(b, c, h * w).permute(0, 2, 1))
        pred_all = torch.cat(feats, dim=1)  # [B, N, C]

        pred_box = pred_all[..., :4]          # logits → usaremos sigmoid
        pred_obj = pred_all[..., 4]           # logits [B, N]
        pred_cls = pred_all[..., 5:]          # logits [B, N, num_classes]
        num_classes = pred_cls.shape[-1]

        # Unificar targets a lista por-imagen
        t_list = self._unify_targets(targets, B, device)

        total_box = pred_box.new_zeros(())
        total_obj = pred_obj.new_zeros(())
        total_cls = pred_cls.new_zeros(())

        total_pos = 0
        total_neg = 0

        for b in range(B):
            t = t_list[b]  # [N_gt,5] = [cls, x, y, w, h]
            if t.numel() == 0:
                # Solo negativos (minado duro)
                # Selecciona los top-k por logit para empujar a 0
                N = pred_obj.shape[1]
                k = min(self.max_negatives, max(1, N // max(1, self.hard_neg_ratio)))
                with torch.no_grad():
                    # mayores logits (más "confianza" errónea)
                    topk_idx = torch.topk(pred_obj[b], k=k, largest=True).indices
                neg_t = torch.zeros_like(pred_obj[b][topk_idx])
                total_obj = total_obj + self.focal(pred_obj[b][topk_idx], neg_t)
                total_neg += k
                continue

            # Preds de la imagen b
            pb = torch.sigmoid(pred_box[b])     # [N_pred, 4] en [0,1]
            po = pred_obj[b]                    # [N_pred] logits
            pc = pred_cls[b]                    # [N_pred, C] logits

            N_pred = pb.shape[0]
            pos_indices = []
            # Acumular pérdidas por cada GT
            for row in t:
                cls = int(row[0].item())
                cls = max(0, min(num_classes - 1, cls))  # clamp por seguridad
                gt_box = row[1:].unsqueeze(0)            # [1,4] en [0,1]

                # Emparejamiento simple por IoU máx
                ious = bbox_iou(pb, gt_box.repeat(N_pred, 1))
                idx = int(torch.argmax(ious).item())
                pos_indices.append(idx)

                # BOX: 1 - IoU
                iou = ious[idx].clamp(min=0, max=1)
                total_box = total_box + (1.0 - iou)

                # OBJ positivo en idx
                total_obj = total_obj + self.focal(po[idx:idx+1], po.new_ones((1,)))

                # CLS en idx (one-hot en focal)
                cls_target = pc.new_zeros((1, num_classes))
                cls_target[0, cls] = 1.0
                total_cls = total_cls + self.focal(pc[idx:idx+1, :], cls_target)

            # Minado duro de negativos (no repetir positivos)
            pos_mask = torch.zeros(N_pred, dtype=torch.bool, device=device)
            if len(pos_indices):
                pos_mask[torch.tensor(pos_indices, device=device)] = True
            neg_candidates = (~pos_mask)
            n_neg = neg_candidates.sum().item()
            if n_neg > 0:
                k = min(self.max_negatives, self.hard_neg_ratio * len(pos_indices), n_neg)
                if k > 0:
                    with torch.no_grad():
                        # entre candidatos negativos, tomar los de mayor logit
                        logits = po.clone()
                        logits[~neg_candidates] = float("-inf")
                        topk_idx = torch.topk(logits, k=k, largest=True).indices
                    neg_t = torch.zeros_like(po[topk_idx])
                    total_obj = total_obj + self.focal(po[topk_idx], neg_t)
                    total_neg += k

            total_pos += len(pos_indices)

        # Normalización estable: por nº de positivos (y algunos negativos)
        norm_pos = max(total_pos, 1)
        box_loss = total_box / norm_pos
        cls_loss = total_cls / norm_pos

        # Para objectness, normaliza por (pos + neg) para no diluir la señal
        norm_obj = max(total_pos + total_neg, 1)
        obj_loss = total_obj / norm_obj

        total_loss = self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_cls * cls_loss

        loss_items = {
            "box_loss": float(box_loss.detach().item()),
            "obj_loss": float(obj_loss.detach().item()),
            "cls_loss": float(cls_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "pos": int(total_pos),
            "neg": int(total_neg)
        }
        return total_loss, loss_items
