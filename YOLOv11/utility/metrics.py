"""
metrics.py
---------------------------------
Cálculo y registro visual de métricas para YOLOv11.

Métricas incluidas:
- Precision
- Recall
- Average Precision (AP)
- mean Average Precision (mAP)
- F-beta score
- Intersection over Union (IoU)
- FPS (opcional)

Cada test genera una carpeta:
    YOLOv11/metrics/test_0001/
donde se guardan gráficos .png y un resumen en .txt
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time


# ---------------------------
#     MÉTRICAS BÁSICAS
# ---------------------------
def bbox_iou(box1, box2, eps=1e-6):
    """Calcula IoU entre dos cajas [x1, y1, x2, y2]."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = box1_area + box2_area - inter_area

    return inter_area / (union + eps)


def calculate_metrics(preds, targets, iou_threshold=0.5, beta=1.0):
    """
    preds y targets son listas de cajas: [[x1, y1, x2, y2, conf, cls], ...]
    """
    tp, fp, fn, ious = 0, 0, 0, []

    for pred, gt in zip(preds, targets):
        iou = bbox_iou(pred[:4], gt[:4])
        ious.append(iou)
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    fn = max(0, len(targets) - tp)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)
    ap = precision * recall  # simplificación conceptual
    iou_mean = np.mean(ious) if ious else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "AP": ap,
        "mAP": ap,  # por ahora igual al AP global (puede promediarse por clases)
        "F_beta": f_beta,
        "IoU": iou_mean
    }


# ---------------------------
#     VISUALIZACIÓN Y LOG
# ---------------------------
def create_metrics_folder(base_dir="metrics"):
    """Crea carpeta incremental de prueba (test_0001, test_0002, ...)."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("test_")]
    new_idx = len(existing) + 1
    folder_name = f"test_{new_idx:04d}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_metrics_plots(metrics_dict, save_dir):
    """Genera gráficos y guarda .png con las métricas calculadas."""
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Gráfico de barras
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.title("YOLOv11 Evaluation Metrics")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_overview.png"))
    plt.close()

    # IoU específico
    plt.figure()
    plt.bar(["IoU"], [metrics_dict["IoU"]])
    plt.title("Mean IoU")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iou.png"))
    plt.close()


def save_metrics_summary(metrics_dict, save_dir):
    """Guarda resumen numérico en .txt."""
    summary_path = os.path.join(save_dir, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"📅 Test generado: {datetime.now()}\n\n")
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"📄 Resumen guardado en {summary_path}")


# ---------------------------
#     FPS OPCIONAL
# ---------------------------
def measure_fps(model, sample_input, device="cpu", runs=20):
    """Evalúa FPS promedio del modelo."""
    model.eval()
    sample_input = sample_input.to(device)
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(sample_input)
    total = time.time() - start
    return runs / total


# ---------------------------
#     PIPELINE PRINCIPAL
# ---------------------------
def evaluate_model(preds, targets, save_results=True):
    """Evalúa métricas y opcionalmente guarda resultados visuales."""
    metrics = calculate_metrics(preds, targets)

    if save_results:
        save_dir = create_metrics_folder(base_dir="metrics")
        save_metrics_plots(metrics, save_dir)
        save_metrics_summary(metrics, save_dir)
        print(f"✅ Resultados guardados en {save_dir}")

    return metrics


# ---------------------------
#   EJEMPLO DE USO LOCAL
# ---------------------------
if __name__ == "__main__":
    # Ejemplo de prueba local (dummy data)
    preds = [
        [0.1, 0.1, 0.4, 0.4, 0.9, 0],
        [0.5, 0.5, 0.8, 0.8, 0.8, 1]
    ]
    targets = [
        [0.12, 0.1, 0.38, 0.4, 1.0, 0],
        [0.52, 0.5, 0.78, 0.79, 1.0, 1]
    ]
    metrics = evaluate_model(preds, targets)
    print(metrics)
