"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: losses.py
Define la función de pérdida principal de YOLOv11.
Soporta formato multiclase y predicciones multi-escala.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Clase: YoloLoss
#   - Integra tres términos ponderados:
#       λ_box → pérdida de regresión de caja (SmoothL1)
#       λ_obj → pérdida de confianza (BCE)
#       λ_cls → pérdida de clasificación (MSE)
#
# Flujo interno:
#   1. Unifica targets de lista → tensor [N,6]
#   2. Normaliza predicciones → [B, N, C]
#   3. Genera targets simulados (dummy) para prueba funcional
#   4. Calcula pérdidas y devuelve total + desglose
#
# Uso:
#   Llamado en train.py dentro del bucle principal
#   junto a las salidas del modelo (head).
# -------------------------------------------------------------


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
        preds: puede ser lista de mapas multi-escala o tensor [B, N, C]
        targets: lista de tensores [N_i,5] o tensor [N,6] con [img_idx, cls, x, y, w, h]
        """

        # =============================================================
        # 🔹 1. Convertir targets de lista → tensor [N_total, 6]
        # =============================================================
        if isinstance(targets, list):
            merged = []
            for i, t in enumerate(targets):
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, dtype=torch.float32)
                if t.numel() == 0:
                    continue
                if t.shape[1] == 5:  # [cls, x, y, w, h]
                    img_idx = torch.full((t.size(0), 1), i, dtype=t.dtype, device=t.device)
                    t = torch.cat([img_idx, t], dim=1)  # [N,6]
                elif t.shape[1] != 6:
                    raise ValueError(f"Target inválido: {t.shape}")
                merged.append(t)
            if len(merged):
                targets = torch.cat(merged, dim=0)
            else:
                targets = torch.zeros((0, 6), dtype=torch.float32, device=preds[0].device)
        elif isinstance(targets, torch.Tensor) and targets.shape[1] == 5:
            # Un solo batch plano [N,5]
            img_idx = torch.zeros((targets.size(0), 1), dtype=targets.dtype, device=targets.device)
            targets = torch.cat([img_idx, targets], dim=1)

        # =============================================================
        # 🔹 2. Normalizar predicciones a [B, N, C]
        # =============================================================
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
            preds = torch.cat(p_list, dim=1)
        elif preds.ndim == 4:  # [B, C, H, W]
            B, C, H, W = preds.shape
            preds = preds.view(B, C, H * W).permute(0, 2, 1)
        elif preds.ndim == 3:
            pass
        else:
            raise ValueError(f"Formato de pred inesperado: {preds.shape}")

        B, N_pred, C_pred = preds.shape

        # =============================================================
        # 🔹 3. Generar targets densos simulados (dummy) del mismo tamaño
        # =============================================================
        # En esta versión simple, igualamos dimensiones por broadcast
        # para evitar errores de shape (prototipo de prueba funcional).
        if targets.numel() == 0:
            targ_box = torch.zeros((B, N_pred, 4), device=preds.device)
            targ_obj = torch.zeros((B, N_pred, 1), device=preds.device)
            targ_cls = torch.zeros((B, N_pred, max(C_pred - 5, 1)), device=preds.device)
        else:
            # Asignación simplificada (se puede mejorar con matching IoU)
            targ_box = torch.zeros((B, N_pred, 4), device=preds.device)
            targ_obj = torch.zeros((B, N_pred, 1), device=preds.device)
            targ_cls = torch.zeros((B, N_pred, max(C_pred - 5, 1)), device=preds.device)

            for row in targets:
                b = int(row[0].item())
                if b >= B:
                    continue
                cls = int(row[1].item())
                x, y, w, h = row[2:].tolist()
                # se proyecta a una posición pseudoaleatoria (prototipo)
                idx = torch.randint(0, N_pred, (1,)).item()
                targ_box[b, idx] = torch.tensor([x, y, w, h], device=preds.device)
                targ_obj[b, idx] = 1.0
                if cls < targ_cls.shape[-1]:
                    targ_cls[b, idx, cls] = 1.0

        # =============================================================
        # 🔹 4. Dividir predicciones
        # =============================================================
        pred_box = preds[..., 0:4]
        pred_obj = preds[..., 4:5]
        pred_cls = preds[..., 5:]

        # =============================================================
        # 🔹 5. Calcular pérdidas
        # =============================================================
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
