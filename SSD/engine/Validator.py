# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/engine/Validator.py
# Descripción: Motor de validación e inferencia.
#              Genera métricas (mAP) y gráficos estilo YOLO.
# ==============================================================

from __future__ import annotations

import sys
import time
import warnings
import random
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

# --------------------------------------------------------------
# Rutas base y Carga dinámica local
# --------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parents[1]  # .../SSD
_METRICS_PATH = SSD_ROOT / "ssd" / "utils" / "metrics.py"


def _load_module_from_local(path: Path, name: str):
    """Carga dinámica de un módulo Python desde un path arbitrario (local)."""
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo requerido en: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para módulo: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    # Inyección temporal de path.parent en sys.path
    module_dir = str(path.parent)
    sys.path.insert(0, module_dir)

    try:
        spec.loader.exec_module(module)
    except Exception:
        if name in sys.modules:
            del sys.modules[name]
        raise
    finally:
        if module_dir in sys.path:
            sys.path.remove(module_dir)

    return module


# Carga dinámica de métricas
try:
    _metrics_mod = _load_module_from_local(_METRICS_PATH, "ssd_metrics_validator")
    ap_per_class: Callable = _metrics_mod.ap_per_class
    box_iou: Callable = _metrics_mod.box_iou
    ConfusionMatrix: type = _metrics_mod.ConfusionMatrix
    fitness: Callable = _metrics_mod.fitness
    smooth: Callable = _metrics_mod.smooth
except Exception as e:
    raise ImportError(f"Fallo al cargar dependencias de métricas en Validator.py: {e}")


# --------------------------------------------------------------
# Funciones de Ayuda
# --------------------------------------------------------------

def _process_batch_stats(detections, gt_boxes, gt_labels, iouv):
    """Calcula TP/FP para un batch a múltiples umbrales de IoU (0.5:0.95)."""

    device = detections.device
    gt_boxes = gt_boxes.to(device)
    gt_labels = gt_labels.to(device)
    iouv = iouv.to(device)

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

    correct = torch.zeros(detections.shape[0], niou, dtype=torch.bool, device=device)
    ious_tp = []  # Guardar IoUs de los True Positives

    if len(gt_labels) > 0:
        # Matriz IoU [N_pred, N_gt]
        iou_matrix = box_iou(pred_boxes, gt_boxes)

        # Para cada umbral de IoU
        for i, iou_thres in enumerate(iouv):
            iou_thres = iou_thres.to(device)

            # Filtrar candidatos > threshold
            x = torch.where((iou_matrix >= iou_thres) & (pred_labels[:, None] == gt_labels[None, :]))

            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1).float(), iou_matrix[x[0], x[1]][:, None]), 1).cpu().numpy()

                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                if matches.shape[0] > 0:
                    idx_pred = torch.tensor(matches[:, 0].astype(int), device=device)
                    correct[idx_pred, i] = True

                    if i == 0:
                        ious_tp.extend(matches[:, 2].tolist())

    return (correct.cpu().numpy(), pred_scores.cpu().numpy(), pred_labels.cpu().numpy(),
            gt_labels.cpu().numpy()), ious_tp


# --------------------------------------------------------------
# Clase Principal
# --------------------------------------------------------------

class ValidatorSSD:
    """Motor de validación para modelos SSD."""

    def __init__(
            self,
            model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            cfg: Any,
            class_names: List[str],
            save_dir: Path,
    ) -> None:
        self.model = model
        self.val_loader = val_loader
        self.cfg = cfg
        self.class_names = class_names
        self.save_dir = save_dir
        self.device = next(model.parameters()).device

        self.conf_thresh = getattr(cfg, "conf_thresh", 0.001)
        self.nms_thresh = getattr(cfg, "nms_thresh", 0.45)
        self.iou_thres = getattr(cfg, "iou_thres", 0.5)
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "images").mkdir(parents=True, exist_ok=True)

        plt.style.use(
            "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "seaborn-whitegrid")

    @staticmethod
    def _get_detections_from_output(output: torch.Tensor, img_size: Tuple[int, int], conf_thresh: float,
                                    device: torch.device) -> torch.Tensor:
        """Convierte la salida cruda de SSD (modo test) a formato [N, 6]."""
        detections = []
        h, w = img_size
        scale = torch.tensor([w, h, w, h], device=device).float()

        for cls_idx in range(1, output.size(1)):
            dets_cls = output[0, cls_idx]
            mask = dets_cls[:, 0] >= conf_thresh
            dets_cls = dets_cls[mask]

            if dets_cls.size(0) == 0:
                continue

            scores = dets_cls[:, 0].unsqueeze(1)
            boxes = dets_cls[:, 1:] * scale
            labels = torch.full((dets_cls.size(0), 1), cls_idx - 1, device=device, dtype=torch.float32)

            detections.append(torch.cat((boxes, scores, labels), 1))

        if not detections:
            return torch.zeros((0, 6), device=device)
        else:
            return torch.cat(detections, 0)

    @staticmethod
    def calculate_metrics_only(
            model: nn.Module,
            data_loader: torch.utils.data.DataLoader,
            cfg: Any,
            class_names: List[str],
    ) -> Dict[str, float]:
        """Ejecuta la validación para obtener solo métricas numéricas."""
        model.eval()
        device = next(model.parameters()).device
        iouv = torch.linspace(0.5, 0.95, 10, device=device)

        stats = []
        conf_thresh = getattr(cfg, "conf_thresh", 0.01)

        for images, targets in data_loader:
            images = images.to(device)
            output = model(images)

            for i, pred_raw in enumerate(output):
                img_size = images[i].shape[1:]

                pred_tensor = ValidatorSSD._get_detections_from_output(
                    pred_raw.unsqueeze(0), img_size, conf_thresh, device
                )

                gt = targets[i].to(device)
                scale_tensor = torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], device=device).float()
                gt_boxes = gt[:, :4] * scale_tensor
                gt_labels = gt[:, 4]

                stat, _ = _process_batch_stats(pred_tensor, gt_boxes, gt_labels, iouv)
                stats.append(stat)

        # FIX: Eliminar check incorrecto de .any() en tupla
        if not stats:
            return {k: 0.0 for k in ["mAP_0.5", "mAP_0.5_0.95", "P", "R", "F1"]}

        # Concatenar estadísticas
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        tp, conf, pred_cls, target_cls = stats

        # FIX: Verificar si hay predicciones después de concatenar
        if tp.size == 0:
            return {k: 0.0 for k in ["mAP_0.5", "mAP_0.5_0.95", "P", "R", "F1"]}

        # Calcular P, R, F1, AP
        tp_c, fp_c, p, r, f1, ap, ap_class = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=False, names=class_names
        )

        map50 = ap[:, 0].mean()
        map95 = ap.mean()
        mp = p.mean()
        mr = r.mean()
        mf1 = f1.mean()

        return {
            "mAP_0.5": float(map50),
            "mAP_0.5_0.95": float(map95),
            "P": float(mp),
            "R": float(mr),
            "F1": float(mf1),
        }

    @torch.no_grad()
    def run_full_report(self, num_images_to_plot: int = 32) -> Dict[str, float]:
        """Ejecuta el ciclo completo de validación con gráficos."""
        self.model.eval()

        stats = []
        iou_vals_tp = []
        confusion_matrix = ConfusionMatrix(nc=len(self.class_names), conf=self.conf_thresh, iou_thres=self.iou_thres)

        images_to_visualize = self._select_images_for_visualization(num_images_to_plot)

        pbar = tqdm(self.val_loader, desc=f"Validando ({self.cfg.run_name})", unit="batch")

        for batch_i, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            output = self.model(images)

            for i, pred_raw in enumerate(output):
                img_size = images[i].shape[1:]

                pred_tensor = self._get_detections_from_output(
                    pred_raw.unsqueeze(0), img_size, self.conf_thresh, self.device
                )

                gt = targets[i].to(self.device)
                h, w = img_size
                scale = torch.tensor([w, h, w, h], device=self.device).float()
                gt_boxes = gt[:, :4] * scale
                gt_labels = gt[:, 4]

                if len(gt) > 0:
                    gt_cm = torch.cat((gt_labels.unsqueeze(1), gt_boxes), 1)
                else:
                    gt_cm = torch.zeros((0, 5), device=self.device)
                confusion_matrix.process_batch(detections=pred_tensor, labels=gt_cm)

                stat, ious = _process_batch_stats(pred_tensor, gt_boxes, gt_labels, self.iouv)
                stats.append(stat)
                if ious is not None:
                    iou_vals_tp.extend(ious)

                current_index = batch_i * self.cfg.batch_size + i
                if current_index < len(self.val_loader.dataset.samples) and self.val_loader.dataset.samples[
                    current_index].image_path.stem in images_to_visualize:
                    self._plot_single_overlay(
                        images[i],
                        pred_tensor,
                        gt_boxes,
                        gt_labels,
                        self.val_loader.dataset.samples[current_index].image_path.stem
                    )

        # FIX: Eliminar check incorrecto de .any() en tupla
        if not stats:
            print("\n[ValidatorSSD] No se encontraron estadísticas.")
            return {k: 0.0 for k in ["mAP_0.5", "mAP_0.5_0.95", "P", "R", "F1", "fitness"]}

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        tp, conf, pred_cls, target_cls = stats

        # FIX: Verificar si hay predicciones después de concatenar
        if tp.size == 0:
            print("\n[ValidatorSSD] No se encontraron detecciones válidas.")
            return {k: 0.0 for k in ["mAP_0.5", "mAP_0.5_0.95", "P", "R", "F1", "fitness"]}

        tp_c, fp_c, p, r, f1, ap, ap_class = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=False, names=self.class_names
        )

        map50, map95 = ap[:, 0].mean(), ap.mean()
        mp, mr, mf1 = p.mean(), r.mean(), f1.mean()
        ap50 = ap[:, 0]

        self._plot_results(stats, iou_vals_tp, confusion_matrix, map50, map95)

        nt = np.bincount(target_cls.astype(int), minlength=len(self.class_names))
        print(f"\n{'Class':<15} {'Labels':<10} {'P':<10} {'R':<10} {'mAP@.5':<10} {'mAP@.5:.95':<10}")
        print("-" * 90)
        print(f"{'all':<15} {nt.sum():<10} {mp:.3f}      {mr:.3f}      {map50:.3f}      {map95:.3f}")

        for i, c in enumerate(ap_class):
            # FIX: Usar ap[i].mean() para obtener el escalar mAP@0.5:0.95 de la clase
            print(
                f"{self.class_names[c]:<15} {nt[c]:<10} {p[i]:.3f}      {r[i]:.3f}      {ap50[i]:.3f}      {ap[i].mean():.3f}")

        return {
            "precision": float(mp),
            "recall": float(mr),
            "mAP_0.5": float(map50),
            "mAP_0.5_0.95": float(map95),
            "F1": float(mf1),
            "fitness": float(fitness(np.array([mp, mr, map50, map95]).reshape(1, -1))[0])
        }

    def _select_images_for_visualization(self, n: int) -> set:
        seed = getattr(self.cfg, "seed", 0)
        random.seed(seed)
        all_stems = [s.image_path.stem for s in self.val_loader.dataset.samples]
        return set(random.sample(all_stems, min(n, len(all_stems))))

    def _plot_single_overlay(self, img_tensor, pred_tensor, gt_boxes, gt_labels, stem):
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

        # FIX: Acceso seguro a la configuración de aumentación
        aug = getattr(self.cfg, "augmentation", None)
        if aug and hasattr(aug, "mean"):
            mean = aug.mean
        else:
            mean = [104, 117, 123]  # Default SSD mean

        img_np = img_np + np.array(mean)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        color_gt = (0, 255, 0)
        color_pred = (0, 165, 255)

        for box, label_idx in zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            label = self.class_names[int(label_idx)]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_gt, 2)
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_gt, 2)

        VIS_CONF_THRESH = 0.25
        for box_score_label in pred_tensor.cpu().numpy():
            x1, y1, x2, y2, score, label_idx = box_score_label
            if score < VIS_CONF_THRESH: continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = self.class_names[int(label_idx)]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_pred, 2)
            text = f"{label} {score:.2f}"
            cv2.putText(img_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_pred, 2)

        save_path = self.save_dir / "images" / f"{stem}_pred.jpg"
        cv2.imwrite(str(save_path), img_bgr)

    def _plot_results(self, stats, iou_vals, confusion_matrix, map50, map95):
        confusion_matrix.plot(save_dir=self.save_dir, names=self.class_names)
        self._plot_iou_distribution(iou_vals)
        self._plot_curves_yolo_style(stats, map50)

    def _plot_iou_distribution(self, iou_vals):
        if not iou_vals: return
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        ax.hist(iou_vals, bins=20, range=(0, 1), color='#4a86e8', edgecolor='black', alpha=0.8)
        ax.set_title(f"IoU distribution | {self.cfg.variant}", fontsize=14)
        ax.set_xlabel("IoU", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        fig.savefig(self.save_dir / "iou_distribution.png", dpi=300)
        plt.close(fig)

    def _plot_curves_yolo_style(self, stats, map50):
        tp, conf, pred_cls, target_cls = stats
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        unique_classes = np.unique(target_cls)
        px = np.linspace(0, 1, 1000)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

        fig_pr, ax_pr = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
        fig_f1, ax_f1 = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
        fig_p, ax_p = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
        fig_r, ax_r = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)

        mean_p, mean_r, mean_f1, mean_pr = np.zeros_like(px), np.zeros_like(px), np.zeros_like(px), np.zeros_like(px)
        valid_classes = 0

        for ci, c in enumerate(unique_classes):
            c = int(c)
            i = pred_cls == c
            n_l = (target_cls == c).sum()
            n_p = i.sum()
            if n_p == 0 or n_l == 0: continue

            valid_classes += 1
            fpc = (1 - tp[i, 0]).cumsum(0)
            tpc = tp[i, 0].cumsum(0)
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc)

            p_interp = np.interp(px, np.flip(conf[i]), np.flip(precision))
            r_interp = np.interp(px, np.flip(conf[i]), np.flip(recall))
            f1_interp = 2 * p_interp * r_interp / (p_interp + r_interp + 1e-16)

            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            pr_interp = np.interp(px, mrec, mpre)
            ap_val = np.trapz(pr_interp, px)

            mean_p += p_interp
            mean_r += r_interp
            mean_f1 += f1_interp
            mean_pr += pr_interp

            label = f"{self.class_names[c]} {ap_val:.3f}"
            ax_pr.plot(px, pr_interp, linewidth=1, color=colors[c], label=label)
            ax_f1.plot(px, f1_interp, linewidth=1, color=colors[c], label=self.class_names[c])
            ax_p.plot(px, p_interp, linewidth=1, color=colors[c], label=self.class_names[c])
            ax_r.plot(px, r_interp, linewidth=1, color=colors[c], label=self.class_names[c])

        if valid_classes > 0:
            mean_p /= valid_classes
            mean_r /= valid_classes
            mean_f1 /= valid_classes
            mean_pr /= valid_classes

        ax_pr.plot(px, mean_pr, linewidth=4, color="blue", label=f"all classes {map50:.3f} mAP@0.5")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_pr.grid(True, linestyle='--', alpha=0.5)

        best_f1_idx = mean_f1.argmax()
        ax_f1.plot(px, mean_f1, linewidth=4, color="blue",
                   label=f"all classes {mean_f1[best_f1_idx]:.2f} at {px[best_f1_idx]:.3f}")
        ax_f1.set_xlabel("Confidence")
        ax_f1.set_ylabel("F1")
        ax_f1.set_title("F1-Confidence Curve")
        ax_f1.set_xlim(0, 1)
        ax_f1.set_ylim(0, 1)
        ax_f1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_f1.grid(True, linestyle='--', alpha=0.5)

        ax_p.plot(px, mean_p, linewidth=4, color="blue", label=f"all classes {mean_p[500]:.2f} at 0.500")
        ax_p.set_xlabel("Confidence")
        ax_p.set_ylabel("Precision")
        ax_p.set_title("Precision-Confidence Curve")
        ax_p.set_xlim(0, 1)
        ax_p.set_ylim(0, 1)
        ax_p.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_p.grid(True, linestyle='--', alpha=0.5)

        ax_r.plot(px, mean_r, linewidth=4, color="blue", label=f"all classes {mean_r[500]:.2f} at 0.500")
        ax_r.set_xlabel("Confidence")
        ax_r.set_ylabel("Recall")
        ax_r.set_title("Recall-Confidence Curve")
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_r.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax_r.grid(True, linestyle='--', alpha=0.5)

        fig_pr.savefig(self.save_dir / "PR_curve.png", dpi=300, bbox_inches='tight')
        fig_f1.savefig(self.save_dir / "F1_curve.png", dpi=300, bbox_inches='tight')
        fig_p.savefig(self.save_dir / "P_curve.png", dpi=300, bbox_inches='tight')
        fig_r.savefig(self.save_dir / "R_curve.png", dpi=300, bbox_inches='tight')

        plt.close(fig_pr)
        plt.close(fig_f1)
        plt.close(fig_p)
        plt.close(fig_r)