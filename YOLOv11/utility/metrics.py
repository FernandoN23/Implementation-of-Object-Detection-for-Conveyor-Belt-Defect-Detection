"""
YOLOv11 - Robust Metrics Module
------------------------------------------------------------
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"

Autor: Fernando N.
------------------------------------------------------------
Este módulo unifica la compatibilidad con métricas oficiales
de Ultralytics (mAP50-95, F1, Precision, Recall) y tu sistema
local de guardado y visualización en TensorBoard.
------------------------------------------------------------
"""

import os
import time
import math
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from ultralytics.utils.metrics import ap_per_class, box_iou  # métricas oficiales
from utility.visualization import TensorboardVisualizer


# ============================================================
#   FUNCIONES AUXILIARES
# ============================================================
def bbox_iou_np_1v1(box1, box2, eps=1e-6):
    """
    IoU escalar entre dos cajas [x1, y1, x2, y2] usando NumPy.
    Robusto a cajas vacías o coordenadas fuera de rango.
    """
    box1, box2 = np.array(box1, float), np.array(box2, float)
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
    area1 = max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0)
    area2 = max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0)
    return inter / (area1 + area2 - inter + eps)


def _to_numpy(x):
    """Convierte tensores o listas anidadas a NumPy recursivamente."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return [_to_numpy(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return np.array(x)


# ============================================================
#   EVALUACIÓN GLOBAL Y POR CLASE
# ============================================================
def calculate_metrics(preds, targets, class_names=None, iou_threshold=0.5, beta=1.0):
    """
    Evalúa precisión, recall, F-beta e IoU promedio por clase y global.

    Args:
        preds: lista [batch, detecciones[N, x1,y1,x2,y2,conf,cls...]]
        targets: lista [batch, etiquetas[M, x1,y1,x2,y2,cls]]
        class_names: nombres de clases (list[str])
        iou_threshold: umbral mínimo IoU para TP
        beta: ponderación de F-beta

    Returns:
        global_metrics (dict), per_class_metrics (dict)
    """
    preds, targets = _to_numpy(preds), _to_numpy(targets)
    classes = class_names or []
    n_cls = len(classes) if classes else 5
    per_class = {classes[i] if i < len(classes) else f"class_{i}": {"tp": 0, "fp": 0, "fn": 0, "ious": []}
                 for i in range(n_cls)}
    tp = fp = fn = 0
    all_ious = []

    def normalize_scale(a, b):
        """Alinea escalas [0,1] vs [0,640]"""
        if a.size == 0 or b.size == 0:
            return a, b
        max_a, max_b = np.max(a[..., :4], initial=0), np.max(b[..., :4], initial=0)
        if max_a <= 1.5 and max_b > 1.5:
            a[..., :4] *= max_b
        elif max_b <= 1.5 and max_a > 1.5:
            b[..., :4] *= max_a
        return a, b

    for pb, gb in zip(preds, targets):
        if pb is None or gb is None or len(gb) == 0:
            continue
        pb, gb = normalize_scale(pb.copy(), gb.copy())
        matched = set()

        for p in pb:
            cls = int(np.argmax(p[5:])) if p.shape[-1] > 6 else 0
            cname = classes[cls] if cls < len(classes) else f"class_{cls}"
            ious = [bbox_iou_np_1v1(p[:4], g[:4]) for g in gb]
            best = np.argmax(ious)
            iou_val = ious[best]
            all_ious.append(iou_val)
            per_class[cname]["ious"].append(iou_val)
            gcls = int(gb[best, 5]) if gb.shape[-1] > 5 else 0
            if iou_val >= iou_threshold and best not in matched and cls == gcls:
                tp += 1
                per_class[cname]["tp"] += 1
                matched.add(best)
            else:
                fp += 1
                per_class[cname]["fp"] += 1
        fn += max(0, len(gb) - len(matched))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
    global_metrics = {
        "Precision": precision,
        "Recall": recall,
        "F_beta": f_beta,
        "IoU_mean": float(np.mean(all_ious) if all_ious else 0.0)
    }

    per_class_metrics = {}
    for cls, d in per_class.items():
        ctp, cfp, cfn = d["tp"], d["fp"], d["fn"]
        p = ctp / (ctp + cfp + 1e-9)
        r = ctp / (ctp + cfn + 1e-9)
        f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-9)
        iou = float(np.mean(d["ious"])) if d["ious"] else 0.0
        per_class_metrics[cls] = {"Precision": p, "Recall": r, "F_beta": f, "IoU": iou}
    return global_metrics, per_class_metrics


# ============================================================
#   AP/MAP OFICIAL (basado en Ultralytics)
# ============================================================
def compute_map(preds, targets, class_names, save_dir=None):
    """
    Calcula mAP@0.5 y mAP@0.5:0.95 usando el método oficial de Ultralytics.
    """
    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []
    for pb, gb in zip(preds, targets):
        if pb is None or gb is None or len(pb) == 0 or len(gb) == 0:
            continue
        boxes_p = torch.tensor(pb[:, :4])
        conf = torch.tensor(pb[:, 4])
        cls_p = torch.tensor(np.argmax(pb[:, 5:], axis=1)) if pb.shape[-1] > 6 else torch.zeros(len(pb))
        boxes_g = torch.tensor(gb[:, :4])
        cls_g = torch.tensor(gb[:, 5] if gb.shape[-1] > 5 else np.zeros(len(gb)))

        iou_mat = box_iou(boxes_g, boxes_p)
        matches = (iou_mat > 0.5).float().sum(0)
        tp = matches.unsqueeze(1)
        tp_list.append(tp.numpy())
        conf_list.append(conf.numpy())
        pred_cls_list.append(cls_p.numpy())
        target_cls_list.append(cls_g.numpy())

    if len(tp_list) == 0:
        return {"mAP50": 0.0, "mAP50-95": 0.0}

    tp = np.concatenate(tp_list, 0)
    conf = np.concatenate(conf_list, 0)
    pred_cls = np.concatenate(pred_cls_list, 0)
    target_cls = np.concatenate(target_cls_list, 0)

    _, _, p, r, f1, ap, _, _, _, _, _, _ = ap_per_class(tp, conf, pred_cls, target_cls, names={i: n for i, n in enumerate(class_names)})
    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean())
    }


# ============================================================
#   VISUALIZACIÓN Y GUARDADO
# ============================================================
def create_metrics_folder(model_variant="n", phase="valid"):
    base = Path(__file__).resolve().parents[1] / "metrics" / model_variant / phase
    base.mkdir(parents=True, exist_ok=True)
    idx = len([d for d in os.listdir(base) if d.startswith("test_")]) + 1
    path = base / f"test_{idx:04d}"
    path.mkdir(exist_ok=True)
    return str(path)


def save_metrics_plots(global_metrics, per_class_metrics, save_dir, model_variant="n"):
    plt.figure(figsize=(8, 5))
    plt.bar(global_metrics.keys(), global_metrics.values(), color="steelblue")
    plt.title(f"Global Metrics - YOLOv11-{model_variant.upper()}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"global_{model_variant}.png"))
    plt.close()

    for cls, vals in per_class_metrics.items():
        plt.figure(figsize=(6, 4))
        plt.bar(vals.keys(), vals.values(), color="indianred")
        plt.title(f"{cls} - {model_variant.upper()}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{cls}_{model_variant}.png"))
        plt.close()


def save_metrics_summary(global_metrics, per_class_metrics, save_dir, model_variant="n", phase="valid"):
    path = os.path.join(save_dir, f"metrics_summary_{model_variant}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"📅 {datetime.now()}\n🔧 Modelo: YOLOv11-{model_variant.upper()} | Fase: {phase}\n\n")
        f.write("=== MÉTRICAS GLOBALES ===\n")
        for k, v in global_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== MÉTRICAS POR CLASE ===\n")
        for c, m in per_class_metrics.items():
            f.write(f"\n[{c}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"📄 Resumen guardado en {path}")


def measure_fps(model, sample_input, device="cpu", runs=20):
    """Mide FPS promedio en inferencia."""
    model.eval()
    sample_input = sample_input.to(device)
    torch.cuda.synchronize() if device != "cpu" and torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(sample_input)
    torch.cuda.synchronize() if device != "cpu" and torch.cuda.is_available() else None
    return runs / (time.time() - t0)


# ============================================================
#   INTERFAZ PRINCIPAL
# ============================================================
def evaluate_model(preds, targets, save_results=True, model_variant="n", phase="valid"):
    """
    Evalúa un conjunto de predicciones/targets y guarda resultados + TensorBoard.
    """
    try:
        cfg = OmegaConf.load("YOLOv11/configs/yolo11.yaml")
        names = cfg.get("names", [])
    except Exception:
        names = []

    # 1️⃣ Métricas globales y por clase (precisión/recall)
    global_metrics, per_class_metrics = calculate_metrics(preds, targets, names)

    # 2️⃣ Métricas oficiales (mAP)
    try:
        map_results = compute_map(preds, targets, names)
        global_metrics.update(map_results)
    except Exception as e:
        print(f"⚠️ Error al calcular mAP: {e}")

    # 3️⃣ Guardado
    if save_results:
        save_dir = create_metrics_folder(model_variant, phase)
        save_metrics_plots(global_metrics, per_class_metrics, save_dir, model_variant)
        save_metrics_summary(global_metrics, per_class_metrics, save_dir, model_variant, phase)
        print(f"✅ Resultados guardados en {save_dir} (YOLOv11-{model_variant.upper()})")
    else:
        save_dir = None

    # 4️⃣ TensorBoard
    try:
        tb = TensorboardVisualizer(log_dir="YOLOv11/runs", model_variant=f"{model_variant}/{phase}")
        tb.log_metrics(global_metrics, 0, phase)
        for c, v in per_class_metrics.items():
            tb.log_metrics(v, 0, phase, class_name=c)
        if save_results:
            tb.log_images_folder(save_dir, 0, phase)
        tb.close()
    except Exception as e:
        print(f"⚠️ TensorBoard error: {e}")

    return global_metrics, per_class_metrics
