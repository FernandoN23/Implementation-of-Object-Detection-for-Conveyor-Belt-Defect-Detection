"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: metrics.py
Cálculo, almacenamiento y visualización de métricas de
evaluación del modelo YOLOv11.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Métricas calculadas:
#   • Precision, Recall, AP, mAP
#   • F_beta, IoU medio
#   • FPS opcional (rendimiento)
#
# Estructura de guardado:
#   YOLOv11/metrics/{variant}/test_XXXX/
#
# Funciones principales:
#   - calculate_metrics(): evalúa IoU y métricas básicas
#   - save_metrics_plots(): guarda gráficos .png
#   - save_metrics_summary(): exporta resumen .txt
#   - evaluate_model(): pipeline completo de evaluación
#
# Conexión:
#   Invocado tras validación o test final, genera registros
#   visuales por variante del modelo (N, S, M, L, X).
# -------------------------------------------------------------


import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Evita Tkinter, usa backend no interactivo
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path


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
    """Calcula Precision, Recall, F_beta, AP, mAP e IoU promedio."""
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
    ap = precision * recall
    iou_mean = np.mean(ious) if ious else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "AP": ap,
        "mAP": ap,
        "F_beta": f_beta,
        "IoU": iou_mean
    }


# ---------------------------
#     VISUALIZACIÓN Y LOG
# ---------------------------
def create_metrics_folder(model_variant="n"):
    """
    Crea estructura de carpetas:
    YOLOv11/metrics/{variant}/test_000X/
    """
    base_dir = Path(__file__).resolve().parents[1] / "metrics" / model_variant
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = [d for d in os.listdir(base_dir) if d.startswith("test_")]
    new_idx = len(existing) + 1
    folder_name = f"test_{new_idx:04d}"
    path = base_dir / folder_name
    path.mkdir(exist_ok=True)

    return str(path)


def save_metrics_plots(metrics_dict, save_dir, model_variant="n"):
    """Genera gráficos y guarda .png con las métricas calculadas."""
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color="cornflowerblue", alpha=0.9)
    plt.title(f"YOLOv11-{model_variant.upper()} Evaluation Metrics")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics_overview_{model_variant}.png"))
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.bar(["IoU"], [metrics_dict["IoU"]], color="lightseagreen")
    plt.title(f"Mean IoU - YOLOv11-{model_variant.upper()}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"iou_{model_variant}.png"))
    plt.close()


def save_metrics_summary(metrics_dict, save_dir, model_variant="n"):
    """Guarda resumen numérico y modelo en .txt."""
    summary_path = os.path.join(save_dir, f"metrics_summary_{model_variant}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"📅 Test generado: {datetime.now()}\n")
        f.write(f"🔧 Modelo evaluado: YOLOv11-{model_variant.upper()}\n\n")
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
def evaluate_model(preds, targets, save_results=True, model_variant="n"):
    """Evalúa métricas y guarda resultados visuales en carpeta del modelo."""
    metrics = calculate_metrics(preds, targets)

    if save_results:
        save_dir = create_metrics_folder(model_variant)
        save_metrics_plots(metrics, save_dir, model_variant)
        save_metrics_summary(metrics, save_dir, model_variant)
        print(f"✅ Resultados guardados en {save_dir} para modelo YOLOv11-{model_variant.upper()}")

    return metrics


# ---------------------------
#   EJEMPLO DE USO LOCAL
# ---------------------------
if __name__ == "__main__":
    preds = [
        [0.1, 0.1, 0.4, 0.4, 0.9, 0],
        [0.5, 0.5, 0.8, 0.8, 0.8, 1]
    ]
    targets = [
        [0.12, 0.1, 0.38, 0.4, 1.0, 0],
        [0.52, 0.5, 0.78, 0.79, 1.0, 1]
    ]
    metrics = evaluate_model(preds, targets, model_variant="l")
    print(metrics)
