# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Validator.py
# Descripción: Motor de validación para DETR. Evalúa el modelo
#              usando la API COCO virtual generada en el loader.
#              Incluye generación de reportes completos (P/R/F1)
#              y visualización de inferencias (Bounding Boxes).
#              *Actualizado con corrección de padding para BBoxes*
# ==============================================================

import torch
import numpy as np
import cv2
import sys
import random
from pathlib import Path

# --- INTEGRACIÓN DE SUBMÓDULO DETR ---
FILE = Path(__file__).resolve()
ENGINE_ROOT = FILE.parent
DETR_ROOT = ENGINE_ROOT.parent
DETR_SUBMODULE = DETR_ROOT / "detr"

if str(DETR_SUBMODULE) not in sys.path:
    sys.path.append(str(DETR_SUBMODULE))

try:
    from datasets.coco_eval import CocoEvaluator
    from engine.bootstrap_miopen import MuteStderr
    from utility.metrics import plot_validation_report
except ImportError as e:
    print(f"[Validator] ERROR: No se pudo importar dependencias: {e}")


class Validator:
    def __init__(self, model, criterion, postprocessors, device):
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.device = device

    @torch.no_grad()
    def validate(self, loader, output_dir):
        """Validación rápida para el ciclo de entrenamiento (mAP COCO)."""
        self.model.eval()
        self.criterion.eval()

        base_ds = loader.dataset
        evaluator = CocoEvaluator(base_ds.coco, iou_types=['bbox'])

        stats = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "class_error": 0.0}

        print("[Validator] Iniciando validación rápida...", flush=True)

        with MuteStderr():
            for samples, targets in loader:
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(samples)

                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                stats["loss"] += losses.item()
                stats["loss_ce"] += loss_dict["loss_ce"].item()
                stats["loss_bbox"] += loss_dict["loss_bbox"].item()
                stats["loss_giou"] += loss_dict["loss_giou"].item()
                if "class_error" in loss_dict:
                    stats["class_error"] += loss_dict["class_error"].item()

                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = self.postprocessors['bbox'](outputs, orig_target_sizes)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                evaluator.update(res)

        num_batches = len(loader)
        final_stats = {k: v / num_batches for k, v in stats.items()}

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        if 'bbox' in evaluator.coco_eval:
            coco_eval = evaluator.coco_eval['bbox']
            coco_stats = coco_eval.stats.tolist()

            final_stats["mAP_0.5:0.95"] = coco_stats[0]
            final_stats["mAP_0.5"] = coco_stats[1]
            final_stats["recall"] = coco_stats[8]  # AR@100

            precisions = coco_eval.eval['precision'][0, :, :, 0, 2]
            valid_precisions = precisions[precisions > -1]
            p = valid_precisions.mean() if len(valid_precisions) > 0 else 0.0
            r = final_stats["recall"]

            final_stats["precision"] = float(p)
            final_stats["F1"] = float(2 * (p * r) / (p + r + 1e-16))

        return final_stats

    @torch.no_grad()
    def run_full_report(self, loader, save_dir, class_names, plot_ratio=0.20, max_images=50):
        """Genera el reporte completo de validación (Curvas, Matriz, IoU e Imágenes)."""
        self.model.eval()
        all_preds = []
        all_gts = []

        img_dir = save_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        total_images = len(loader.dataset)
        num_images_to_plot = min(int(total_images * plot_ratio), max_images)
        indices_to_plot = set(random.sample(range(total_images), num_images_to_plot))

        # Integrar CocoEvaluator para obtener mAP oficial
        base_ds = loader.dataset
        evaluator = CocoEvaluator(base_ds.coco, iou_types=['bbox'])

        print(
            f"[Validator] Recolectando predicciones para reporte completo ({num_images_to_plot} imágenes de muestra)...")

        for batch_idx, (samples, targets) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"[Validator] Evaluando Batch [{batch_idx}/{len(loader)}]", flush=True)

            with MuteStderr():
                samples = samples.to(self.device)
                targets_gpu = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(samples)

                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(self.device)
                results = self.postprocessors['bbox'](outputs, orig_target_sizes)

                # Actualizar CocoEvaluator
                res_coco = {target['image_id'].item(): output for target, output in zip(targets_gpu, results)}
                evaluator.update(res_coco)

                for i, (target, res) in enumerate(zip(targets, results)):
                    global_idx = batch_idx * loader.batch_size + i

                    gt_boxes = target['boxes'].cpu()
                    h, w = target['orig_size'].cpu()

                    gt_xyxy = gt_boxes.clone()
                    if len(gt_boxes) > 0:
                        gt_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * w
                        gt_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * h
                        gt_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * w
                        gt_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * h

                    all_gts.append({'boxes': gt_xyxy, 'labels': target['labels'].cpu()})

                    pred_boxes = res['boxes'].cpu()
                    pred_scores = res['scores'].cpu()
                    pred_labels = res['labels'].cpu()

                    all_preds.append({'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels})

                    if global_idx in indices_to_plot:
                        # Obtener dimensiones reales para reconstruir la imagen sin padding
                        orig_h, orig_w = target['orig_size'].cpu().numpy()
                        resized_h, resized_w = target['size'].cpu().numpy()

                        img_tensor = samples.tensors[i].cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean

                        # 1. Recortar el padding del batch usando el tamaño redimensionado real
                        img_tensor = img_tensor[:, :int(resized_h), :int(resized_w)]

                        # 2. Convertir a numpy y BGR
                        img_np = img_tensor.permute(1, 2, 0).numpy()
                        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                        # 3. Redimensionar a la resolución original para que calcen los bboxes
                        img_bgr = cv2.resize(img_bgr, (int(orig_w), int(orig_h)))

                        self._plot_single_overlay(
                            img_bgr, pred_boxes, pred_scores, pred_labels,
                            gt_xyxy, target['labels'].cpu(), class_names,
                            img_dir / f"val_img_{global_idx}.jpg"
                        )

        print("[Validator] Generando gráficos y métricas finales...")

        # 1. Obtener métricas de curvas (P, R, F1)
        curve_metrics = plot_validation_report(all_preds, all_gts, class_names, save_dir)

        # 2. Obtener mAP oficial de COCO
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        map_50 = 0.0
        map_50_95 = 0.0

        if 'bbox' in evaluator.coco_eval:
            coco_stats = evaluator.coco_eval['bbox'].stats.tolist()
            map_50_95 = coco_stats[0]
            map_50 = coco_stats[1]

        # 3. Calcular Fitness (Estándar YOLO: 0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95)
        fitness = (0.1 * map_50) + (0.9 * map_50_95)

        # 4. Ensamblar diccionario final
        final_metrics = {
            'F1': curve_metrics['F1'],
            'fitness': float(fitness),
            'mAP_0.5': float(map_50),
            'mAP_0.5_0.95': float(map_50_95),
            'precision': curve_metrics['precision'],
            'recall': curve_metrics['recall']
        }

        return final_metrics

    def _plot_single_overlay(self, img_bgr, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, class_names,
                             save_path):
        color_gt = (0, 255, 0)  # Verde (Real)
        color_pred = (0, 165, 255)  # Naranja (Predicción)
        VIS_CONF_THRESH = 0.25

        # 1. Dibujar las cajas
        for box, label_idx in zip(gt_boxes.numpy(), gt_labels.numpy()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_gt, 2)
            # Omitimos el texto en el GT para no saturar la imagen, la caja verde basta.

        for box, score, label_idx in zip(pred_boxes.numpy(), pred_scores.numpy(), pred_labels.numpy()):
            if score < VIS_CONF_THRESH: continue
            x1, y1, x2, y2 = map(int, box)
            label = class_names[int(label_idx)]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_pred, 2)
            text = f"{label} {score:.2f}"
            cv2.putText(img_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_pred, 2, cv2.LINE_AA)

        # 2. Lógica de Leyenda Dinámica (Buscar la esquina más despejada)
        h, w = img_bgr.shape[:2]
        lw, lh = 160, 65  # Ancho y alto de la leyenda

        # Posibles esquinas: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        corners = [
            (10, 10),
            (w - lw - 10, 10),
            (10, h - lh - 10),
            (w - lw - 10, h - lh - 10)
        ]

        # Recopilar todas las cajas válidas para calcular solapamiento
        all_boxes = list(gt_boxes.numpy()) + [b for b, s in zip(pred_boxes.numpy(), pred_scores.numpy()) if
                                              s >= VIS_CONF_THRESH]

        best_corner = corners[0]
        min_overlap = float('inf')

        for cx, cy in corners:
            overlap_area = 0
            lx1, ly1, lx2, ly2 = cx, cy, cx + lw, cy + lh  # Rectángulo de la leyenda

            for box in all_boxes:
                bx1, by1, bx2, by2 = box
                # Calcular área de intersección
                ix1 = max(lx1, bx1)
                iy1 = max(ly1, by1)
                ix2 = min(lx2, bx2)
                iy2 = min(ly2, by2)
                if ix1 < ix2 and iy1 < iy2:
                    overlap_area += (ix2 - ix1) * (iy2 - iy1)

            if overlap_area < min_overlap:
                min_overlap = overlap_area
                best_corner = (cx, cy)
            if min_overlap == 0:
                break  # Esquina perfecta sin solapamiento encontrada

        # 3. Dibujar la leyenda en la mejor esquina
        lx, ly = best_corner
        overlay = img_bgr.copy()

        # Fondo semi-transparente oscuro
        cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)

        # Cuadro de color y texto para GT (Real)
        cv2.rectangle(img_bgr, (lx + 10, ly + 15), (lx + 25, ly + 30), color_gt, -1)
        cv2.putText(img_bgr, "Real (GT)", (lx + 35, ly + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Cuadro de color y texto para Predicción
        cv2.rectangle(img_bgr, (lx + 10, ly + 40), (lx + 25, ly + 55), color_pred, -1)
        cv2.putText(img_bgr, "Prediccion", (lx + 35, ly + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.imwrite(str(save_path), img_bgr)