# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utilidades de cajas
# ---------------------------
def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def bbox_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # box1, box2: [N,4] en xyxy
    inter_x1 = torch.maximum(box1[:, 0], box2[:, 0])
    inter_y1 = torch.maximum(box1[:, 1], box2[:, 1])
    inter_x2 = torch.minimum(box1[:, 2], box2[:, 2])
    inter_y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)

def bbox_ciou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # Implementación compacta de CIoU en xyxy normalizado
    iou = bbox_iou_xyxy(box1, box2, eps)
    # centros y dimensiones
    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)
    c1x = (box1[:, 0] + box1[:, 2]) / 2
    c1y = (box1[:, 1] + box1[:, 3]) / 2
    c2x = (box2[:, 0] + box2[:, 2]) / 2
    c2y = (box2[:, 1] + box2[:, 3]) / 2
    # distancia de centros
    rho2 = (c1x - c2x) ** 2 + (c1y - c2y) ** 2
    # caja mínima envolvente
    cw = torch.maximum(box1[:, 2], box2[:, 2]) - torch.minimum(box1[:, 0], box2[:, 0])
    ch = torch.maximum(box1[:, 3], box2[:, 3]) - torch.minimum(box1[:, 1], box2[:, 1])
    c2 = cw * cw + ch * ch + eps
    # consistencia de aspecto
    v = (4 / (3.14159265 ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)

# ---------------------------
# Focal Loss estable (BCE logits)
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = target * prob + (1 - target) * (1 - prob)
        mod = (1.0 - p_t).clamp(0, 1) ** self.gamma
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_t * mod * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# ---------------------------
# YoloLoss (head legacy con obj)
# ---------------------------
class YoloLoss(nn.Module):
    """
    Pérdida para head legacy: [x,y,w,h,obj, cls0..clsC-1] (logits).
    - BOX: 1 - CIoU en el índice asignado (uno a uno sin repetición).
    - OBJ: Focal en positivos y hard-negative mining.
    - CLS: Focal one-hot en positivos.
    Requisitos: GT en [0,1] (xywh). Pred boxes tratadas con sigmoid→[0,1].
    """
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5,
                 hard_neg_ratio=3, max_negatives=300, use_ciou=True):
        super().__init__()
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)
        self.focal = FocalLoss(gamma=2.0, alpha=0.25, reduction="sum")
        self.hard_neg_ratio = int(hard_neg_ratio)
        self.max_negatives = int(max_negatives)
        self.use_ciou = bool(use_ciou)

    @staticmethod
    def _unify_targets(targets, batch_size, device):
        # Devuelve lista de B tensores [Ni,5] = [cls,x,y,w,h]
        if isinstance(targets, torch.Tensor):
            out = [torch.empty((0, 5), device=device) for _ in range(batch_size)]
            if targets.numel() == 0:
                return out
            idx = targets[:, 0].long()
            rest = targets[:, 1:].to(device)
            for b in range(batch_size):
                out[b] = rest[idx == b]
            return out
        elif isinstance(targets, (list, tuple)):
            out = []
            for t in targets:
                if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
                    out.append(torch.empty((0, 5), device=device))
                else:
                    out.append(t.to(device))
            while len(out) < batch_size:
                out.append(torch.empty((0, 5), device=device))
            return out[:batch_size]
        else:
            return [torch.empty((0, 5), device=device) for _ in range(batch_size)]

    def forward(self, preds, targets):
        """
        preds: lista [P3,P4,P5], cada P: [B, 5+nc, H, W] (logits).
        targets: ver _unify_targets.
        """
        device = preds[0].device
        B = preds[0].shape[0]

        # Concat escalas -> [B, N, C]
        feats = []
        for p in preds:
            b, c, h, w = p.shape
            feats.append(p.view(b, c, h * w).permute(0, 2, 1))
        pred_all = torch.cat(feats, dim=1)
        pred_box = pred_all[..., :4]
        pred_obj = pred_all[..., 4]
        pred_cls = pred_all[..., 5:]
        C = pred_cls.shape[-1]

        t_list = self._unify_targets(targets, B, device)

        total_box = pred_box.new_tensor(0.0)
        total_obj = pred_obj.new_tensor(0.0)
        total_cls = pred_cls.new_tensor(0.0)
        total_pos = 0
        total_neg = 0

        for b in range(B):
            gt = t_list[b]  # [Ng,5]
            pb = torch.sigmoid(pred_box[b]).clamp_(0, 1)  # [N,4] xywh norm
            po = pred_obj[b]  # [N]
            pc = pred_cls[b]  # [N,C]

            N = pb.shape[0]
            if gt.numel() == 0:
                # Solo minado duro
                k = min(self.max_negatives, max(1, N // max(1, self.hard_neg_ratio)))
                with torch.no_grad():
                    topk = torch.topk(po, k=k, largest=True).indices
                zeros = po.new_zeros((k,))
                total_obj += self.focal(po[topk], zeros)
                total_neg += k
                continue

            # pool positivo sin duplicados
            used = torch.zeros(N, dtype=torch.bool, device=device)

            # Prepara cajas en xyxy para IoU/CIoU
            pb_xyxy = _xywh2xyxy(pb)
            pos_idx = []

            for row in gt:
                cls = int(row[0].clamp(0, C - 1).item())
                g = row[1:].unsqueeze(0)  # [1,4] xywh
                g_xyxy = _xywh2xyxy(g)

                # iou contra todos los no usados
                mask = ~used
                if not mask.any():
                    break
                ious = bbox_ciou_xyxy(pb_xyxy[mask], g_xyxy.repeat(mask.sum(), 1)) if self.use_ciou \
                       else bbox_iou_xyxy(pb_xyxy[mask], g_xyxy.repeat(mask.sum(), 1))
                local = int(torch.argmax(ious).item())
                idx = torch.nonzero(mask, as_tuple=False)[local].item()

                used[idx] = True
                pos_idx.append(idx)

                # BOX
                iou_val = ious[local].clamp(0, 1)
                total_box += (1.0 - iou_val)

                # OBJ (positivo)
                total_obj += self.focal(po[idx:idx+1], po.new_ones((1,)))

                # CLS (one-hot focal)
                t = pc.new_zeros((1, C))
                t[0, cls] = 1.0
                total_cls += self.focal(pc[idx:idx+1, :], t)

            # Hard negative mining (excluir positivos)
            pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
            if pos_idx:
                pos_mask[torch.tensor(pos_idx, device=device)] = True
            neg_mask = ~pos_mask
            n_neg = int(neg_mask.sum().item())
            n_pos = len(pos_idx)

            if n_neg > 0:
                k = min(self.max_negatives, max(n_pos * self.hard_neg_ratio, 1), n_neg)
                with torch.no_grad():
                    logits = po.clone()
                    logits[~neg_mask] = float("-inf")  # garante no-positivos
                    topk = torch.topk(logits, k=k, largest=True).indices
                zeros = po.new_zeros((k,))
                total_obj += self.focal(po[topk], zeros)
                total_neg += k

            total_pos += n_pos

        # Normalizaciones
        norm_pos = max(total_pos, 1)
        norm_obj = max(total_pos + total_neg, 1)
        box_loss = total_box / norm_pos
        cls_loss = total_cls / norm_pos
        obj_loss = total_obj / norm_obj

        total = self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_cls * cls_loss
        return total, {
            "box_loss": float(box_loss.detach()),
            "obj_loss": float(obj_loss.detach()),
            "cls_loss": float(cls_loss.detach()),
            "total_loss": float(total.detach()),
            "pos": int(total_pos),
            "neg": int(total_neg),
        }
