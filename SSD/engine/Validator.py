# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/engine/Validator.py
# Descripción: Motor de validación e inferencia.
#              Genera métricas (mAP) y gráficos estilo YOLO
#              (PR-Curve, F1-Curve, Confusion Matrix, IoU Hist).
# ==============================================================

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Importar utilidades internas de SSD
# Asumimos que SSD/ssd/utils/metrics.py está disponible
try:
    from ssd.utils.metrics import (
        ap_per_class,
        box_iou,
        ConfusionMatrix,
        fitness,
        smooth
    )
except ImportError:
    # Fallback si se ejecuta desde la raíz sin instalar como paquete
    sys.path.append(str(Path(__file__).parents[2] / "ssd" / "utils"))
    from metrics import ap_per_class, box_iou, ConfusionMatrix, fitness, smooth


class ValidatorSSD:
    """Motor de validación para modelos SSD.

    Realiza inferencia, calcula métricas estándar (P, R, mAP, F1) y genera
    reportes visuales detallados en el estilo de YOLOv5.
    """

    def __init__(
            self,
            model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            cfg: Any,
            class_names: List[str],
            save_dir: Path,
    ) -> None:
        """
        Args:
            model: Modelo SSD cargado (en modo eval).
            val_loader: DataLoader de validación.
            cfg: Configuración (objeto o namespace) con parámetros de inferencia.
            class_names: Lista de nombres de clases (sin background).
            save_dir: Ruta donde guardar gráficos y resultados.
        """
        self.model = model
        self.val_loader = val_loader
        self.cfg = cfg
        self.class_names = class_names
        self.save_dir = save_dir
        self.device = next(model.parameters()).device

        # Parámetros de inferencia desde config
        self.conf_thresh = getattr(cfg.inference, "conf_thresh", 0.001)
        self.nms_thresh = getattr(cfg.inference, "nms_thresh", 0.45)
        self.iou_thres = getattr(cfg.inference, "iou_thres", 0.5)

        # Crear directorios
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Configuración de Matplotlib para estilo YOLO
        plt.style.use(
            "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "seaborn-whitegrid")

    @torch.no_grad()
    def run(self) -> Dict[str, float]:
        """Ejecuta el ciclo completo de validación."""
        self.model.eval()

        # Contenedores de estadísticas
        # stats: lista de tuplas (tp, conf, pred_cls, target_cls)
        stats = []
        iou_vals_tp = []  # Para histograma de IoU

        # Matriz de confusión
        # SSD usa background en índice 0, pero nuestras métricas esperan 0..N-1
        # Ajustaremos los índices al procesar.
        confusion_matrix = ConfusionMatrix(nc=len(self.class_names), conf=self.conf_thresh, iou_thres=self.iou_thres)

        # Barra de progreso
        pbar = tqdm(self.val_loader, desc=f"Validando ({self.cfg.run_name})", unit="batch")

        for batch_i, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            # targets es una lista de tensores [num_objs, 5] (x1, y1, x2, y2, label)
            # label en dataset es 0-indexed (clases reales)

            # Inferencia
            # Salida SSD Test: [batch, num_classes, top_k, 5] -> (score, x1, y1, x2, y2)
            output = self.model(images)

            # Procesar batch
            for i, pred in enumerate(output):
                # pred: [num_classes, top_k, 5]
                # targets[i]: [num_objs, 5]

                # 1. Preparar Ground Truth
                gt = targets[i].to(self.device)
                # Escalar GT a dimensiones de imagen (están normalizadas 0-1)
                h, w = images[i].shape[1], images[i].shape[2]
                scale = torch.tensor([w, h, w, h], device=self.device)

                gt_boxes = gt[:, :4] * scale
                gt_labels = gt[:, 4]  # 0-indexed

                # 2. Preparar Predicciones
                # SSD devuelve clases separadas (índice 1..N, 0 es background).
                # Necesitamos aplanar a [N_detections, 6] -> (x1, y1, x2, y2, conf, cls)
                detections = []

                # Iterar sobre clases (saltando 0=background)
                # self.model.num_classes incluye background
                for cls_idx in range(1, self.model.num_classes):
                    # dets_cls: [top_k, 5] -> (score, x1, y1, x2, y2)
                    dets_cls = pred[cls_idx]

                    # Filtrar por confianza
                    mask = dets_cls[:, 0] >= self.conf_thresh
                    dets_cls = dets_cls[mask]

                    if dets_cls.size(0) == 0:
                        continue

                    # Formato: (x1, y1, x2, y2, conf, cls_idx-1)
                    # Restamos 1 al cls_idx para alinear con GT (0-indexed)
                    scores = dets_cls[:, 0].unsqueeze(1)
                    boxes = dets_cls[:, 1:] * scale  # Escalar
                    labels = torch.full((dets_cls.size(0), 1), cls_idx - 1, device=self.device)

                    detections.append(torch.cat((boxes, scores, labels), 1))

                if not detections:
                    pred_tensor = torch.zeros((0, 6), device=self.device)
                else:
                    pred_tensor = torch.cat(detections, 0)

                # 3. Actualizar Matriz de Confusión
                # ConfusionMatrix espera: preds [x1, y1, x2, y2, conf, cls], labels [cls, x1, y1, x2, y2]
                if len(gt) > 0:
                    gt_cm = torch.cat((gt_labels.unsqueeze(1), gt_boxes), 1)
                else:
                    gt_cm = torch.zeros((0, 5), device=self.device)

                confusion_matrix.process_batch(detections=pred_tensor, labels=gt_cm)

                # 4. Matching para mAP
                # stats.append((tp, conf, pred_cls, target_cls))
                stat, ious = self._process_batch_stats(pred_tensor, gt_boxes, gt_labels)
                stats.append(stat)
                if ious is not None:
                    iou_vals_tp.extend(ious)

        # Calcular métricas globales
        stats = [np.concatenate(x, 0) for x in zip(*stats)]

        if len(stats) and stats[0].any():
            tp, conf, pred_cls, target_cls = stats

            # ap_per_class calcula P, R, AP@IoU (definido por tp)
            # tp es una matriz [N_pred, N_iou_thresholds] si pasamos múltiples thresholds
            # Aquí _process_batch_stats devuelve tp para IoU=0.5:0.95

            tp_c, fp_c, p, r, f1, ap, ap_class = ap_per_class(
                tp, conf, pred_cls, target_cls, plot=False, save_dir=self.save_dir, names=self.class_names
            )

            # AP@0.5 es la columna 0 (si definimos los thresholds así)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=len(self.class_names))  # number of targets per class
        else:
            nt = torch.zeros(1)
            mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
            ap50, ap = np.zeros(len(self.class_names)), np.zeros(len(self.class_names))

        # Generar Gráficos
        self._plot_results(stats, iou_vals_tp, confusion_matrix, map50, map)

        # Imprimir resumen
        print(f"\n{'Class':<15} {'Images':<10} {'Labels':<10} {'P':<10} {'R':<10} {'mAP@.5':<10} {'mAP@.5:.95':<10}")
        print("-" * 90)
        print(
            f"{'all':<15} {len(self.val_loader.dataset):<10} {nt.sum():<10} {mp:.3f}      {mr:.3f}      {map50:.3f}      {map:.3f}")

        for i, c in enumerate(ap_class):
            print(
                f"{self.class_names[c]:<15} {'-':<10} {nt[c]:<10} {p[i]:.3f}      {r[i]:.3f}      {ap50[i]:.3f}      {ap[i]:.3f}")

        return {
            "precision": mp,
            "recall": mr,
            "mAP_0.5": map50,
            "mAP_0.5_0.95": map,
            "fitness": fitness(np.array([mp, mr, map50, map]).reshape(1, -1))[0]
        }

    def _process_batch_stats(self, detections, gt_boxes, gt_labels):
        """Calcula TP/FP para un batch a múltiples umbrales de IoU (0.5:0.95)."""
        # IoU thresholds: 0.5, 0.55, ..., 0.95 (10 pasos)
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        niou = iouv.numel()

        if len(detections) == 0:
            if len(gt_labels) > 0:
                return (np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), gt_labels.cpu().numpy()), None
            else:
                return (np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), np.array([])), None

        # detections: [N, 6] (x1, y1, x2, y2, conf, cls)
        pred_boxes = detections[:, :4]
        pred_scores = detections[:, 4]
        pred_labels = detections[:, 5]

        correct = torch.zeros(detections.shape[0], niou, dtype=torch.bool, device=self.device)
        detected = []  # Para evitar asignar múltiples preds al mismo GT
        ious_tp = []  # Guardar IoUs de los True Positives (para histograma)

        if len(gt_labels) > 0:
            # Matriz IoU [N_pred, N_gt]
            iou_matrix = box_iou(pred_boxes, gt_boxes)

            # Para cada umbral de IoU
            for i, iou_thres in enumerate(iouv):
                # Filtrar candidatos > threshold
                x = torch.where((iou_matrix >= iou_thres) & (pred_labels[:, None] == gt_labels[None, :]))

                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), iou_matrix[x[0], x[1]][:, None]), 1).cpu().numpy()
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                    if matches.shape[0] > 0:
                        correct[matches[:, 0].astype(int), i] = True

                        # Solo guardar IoUs para el primer threshold (0.5) o el que se use para el histograma
                        if i == 0:
                            ious_tp.extend(matches[:, 2].tolist())

        return (correct.cpu().numpy(), pred_scores.cpu().numpy(), pred_labels.cpu().numpy(),
                gt_labels.cpu().numpy()), ious_tp

    def _plot_results(self, stats, iou_vals, confusion_matrix, map50, map95):
        """Orquesta la generación de todos los gráficos."""

        # 1. Matriz de Confusión
        confusion_matrix.plot(save_dir=self.save_dir, names=self.class_names)

        # 2. Histograma de IoU
        self._plot_iou_distribution(iou_vals)

        # 3. Curvas PR, F1, P, R
        # Necesitamos recalcular curvas "suaves" para graficar
        tp, conf, pred_cls, target_cls = stats

        # Usamos ap_per_class para obtener las curvas interpoladas
        # Nota: ap_per_class en metrics.py ya tiene lógica de ploteo, pero
        # aquí la personalizamos para asegurar el estilo exacto.
        # Llamamos con plot=True para que genere los datos internos, pero sobreescribimos los archivos
        # o usamos los datos retornados si modificamos metrics.py.
        # Como no puedo modificar metrics.py ahora, implemento la lógica de ploteo aquí
        # usando los datos crudos.

        self._plot_curves_yolo_style(tp, conf, pred_cls, target_cls, map50)

    def _plot_iou_distribution(self, iou_vals):
        """Genera histograma de IoU estilo YOLO."""
        if not iou_vals:
            return

        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        ax.hist(iou_vals, bins=20, range=(0, 1), color='#4a86e8', edgecolor='black', alpha=0.8)
        ax.set_title(f"IoU distribution | {self.cfg.variant}", fontsize=14)
        ax.set_xlabel("IoU", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.savefig(self.save_dir / "iou_distribution.png", dpi=300)
        plt.close(fig)

    def _plot_curves_yolo_style(self, tp, conf, pred_cls, target_cls, map50):
        """Genera curvas PR, F1, P, R con estilo YOLO."""

        # Ordenar por confianza
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Clases únicas
        unique_classes = np.unique(target_cls)

        # Diccionarios para guardar curvas por clase
        p_curves, r_curves, f1_curves, pr_curves = {}, {}, {}, {}
        px = np.linspace(0, 1, 1000)  # Eje X para curvas interpoladas

        # Colores
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

        fig_pr, ax_pr = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
        fig_f1, ax_f1 = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
        fig_p, ax_p = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
        fig_r, ax_r = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)

        # Acumuladores para promedio
        mean_p = np.zeros_like(px)
        mean_r = np.zeros_like(px)
        mean_f1 = np.zeros_like(px)
        mean_pr = np.zeros_like(px)  # PR curve interpolada sobre recall

        valid_classes = 0

        for ci, c in enumerate(unique_classes):
            c = int(c)
            i = pred_cls == c
            n_l = (target_cls == c).sum()
            n_p = i.sum()

            if n_p == 0 or n_l == 0:
                continue

            valid_classes += 1

            # TP acumulado (usando IoU@0.5, columna 0)
            fpc = (1 - tp[i, 0]).cumsum(0)
            tpc = tp[i, 0].cumsum(0)

            # Recall y Precision
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc)

            # Interpolación para gráficos vs Confidence
            # Invertimos conf para que sea creciente para np.interp
            # px es confidence (0 a 1)
            p_interp = np.interp(px, np.flip(conf[i]), np.flip(precision))
            r_interp = np.interp(px, np.flip(conf[i]), np.flip(recall))
            f1_interp = 2 * p_interp * r_interp / (p_interp + r_interp + 1e-16)

            # PR Curve (Precision vs Recall)
            # Interpolamos precision sobre recall (0 a 1)
            # mrec, mpre = compute_ap logic
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            pr_interp = np.interp(px, mrec, mpre)

            # Calcular AP de esta clase
            ap_val = np.trapz(pr_interp, px)

            # Acumular
            mean_p += p_interp
            mean_r += r_interp
            mean_f1 += f1_interp
            mean_pr += pr_interp

            # Plotear líneas finas por clase
            label = f"{self.class_names[c]} {ap_val:.3f}"
            ax_pr.plot(px, pr_interp, linewidth=1, color=colors[c], label=label)
            ax_f1.plot(px, f1_interp, linewidth=1, color=colors[c], label=self.class_names[c])
            ax_p.plot(px, p_interp, linewidth=1, color=colors[c], label=self.class_names[c])
            ax_r.plot(px, r_interp, linewidth=1, color=colors[c], label=self.class_names[c])

        # Promedios
        if valid_classes > 0:
            mean_p /= valid_classes
            mean_r /= valid_classes
            mean_f1 /= valid_classes
            mean_pr /= valid_classes

        # Plotear líneas gruesas promedio
        # PR Curve
        ax_pr.plot(px, mean_pr, linewidth=4, color="blue", label=f"all classes {map50:.3f} mAP@0.5")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_pr.grid(True, linestyle='--', alpha=0.5)

        # F1 Curve
        best_f1_idx = mean_f1.argmax()
        best_f1 = mean_f1[best_f1_idx]
        best_conf = px[best_f1_idx]
        ax_f1.plot(px, mean_f1, linewidth=4, color="blue", label=f"all classes {best_f1:.2f} at {best_conf:.3f}")
        ax_f1.set_xlabel("Confidence")
        ax_f1.set_ylabel("F1")
        ax_f1.set_title("F1-Confidence Curve")
        ax_f1.set_xlim(0, 1)
        ax_f1.set_ylim(0, 1)
        ax_f1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_f1.grid(True, linestyle='--', alpha=0.5)

        # P Curve
        ax_p.plot(px, mean_p, linewidth=4, color="blue",
                  label=f"all classes {mean_p[int(0.5 * 1000)]:.2f} at 0.500")  # Aprox
        ax_p.set_xlabel("Confidence")
        ax_p.set_ylabel("Precision")
        ax_p.set_title("Precision-Confidence Curve")
        ax_p.set_xlim(0, 1)
        ax_p.set_ylim(0, 1)
        ax_p.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_p.grid(True, linestyle='--', alpha=0.5)

        # R Curve
        ax_r.plot(px, mean_r, linewidth=4, color="blue", label=f"all classes {mean_r[int(0.5 * 1000)]:.2f} at 0.500")
        ax_r.set_xlabel("Confidence")
        ax_r.set_ylabel("Recall")
        ax_r.set_title("Recall-Confidence Curve")
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_r.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_r.grid(True, linestyle='--', alpha=0.5)

        # Guardar
        fig_pr.savefig(self.save_dir / "PR_curve.png", dpi=300, bbox_inches='tight')
        fig_f1.savefig(self.save_dir / "F1_curve.png", dpi=300, bbox_inches='tight')
        fig_p.savefig(self.save_dir / "P_curve.png", dpi=300, bbox_inches='tight')
        fig_r.savefig(self.save_dir / "R_curve.png", dpi=300, bbox_inches='tight')

        plt.close(fig_pr)
        plt.close(fig_f1)
        plt.close(fig_p)
        plt.close(fig_r)