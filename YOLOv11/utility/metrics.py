import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path


# ============================================================
#                FUNCIONES BÁSICAS DE MÉTRICAS
# ============================================================
def bbox_iou(box1, box2, eps=1e-6):
    """Calcula IoU entre dos cajas [x1, y1, x2, y2]."""
    inter_x1 = max(float(box1[0]), float(box2[0]))
    inter_y1 = max(float(box1[1]), float(box2[1]))
    inter_x2 = min(float(box1[2]), float(box2[2]))
    inter_y2 = min(float(box1[3]), float(box2[3]))

    inter_area = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
    box1_area = max(float(box1[2]) - float(box1[0]), 0.0) * max(float(box1[3]) - float(box1[1]), 0.0)
    box2_area = max(float(box2[2]) - float(box2[0]), 0.0) * max(float(box2[3]) - float(box2[1]), 0.0)
    union = box1_area + box2_area - inter_area
    return inter_area / (union + eps)


def _to_numpy(x):
    """Convierte a numpy de forma segura (soporta tensores en CUDA/CPU, listas o np arrays)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x
    return np.array(x, dtype=np.float32)


def calculate_metrics(preds, targets, iou_threshold=0.5, beta=1.0):
    """Calcula Precision, Recall, AP, mAP, F_beta e IoU promedio."""
    tp, fp, fn, ious = 0, 0, 0, []

    preds = [ _to_numpy(p) for p in preds ]
    targets = [ _to_numpy(t) for t in targets ]

    for pred_boxes, gt_boxes in zip(preds, targets):
        if pred_boxes is None or gt_boxes is None:
            continue

        if pred_boxes.size == 0 or gt_boxes.size == 0:
            # Si no hay predicciones o GT en la muestra actual, contar FNs en bloque
            fn += int(gt_boxes.shape[0])
            continue

        # Normalizar a [x1,y1,x2,y2,score,cls] si vienen con extras; limitar a 6 cols mín.
        pred_boxes = pred_boxes.reshape(-1, min(6, pred_boxes.shape[-1]))
        gt_boxes   = gt_boxes.reshape(-1, min(6, gt_boxes.shape[-1]))

        matched_gt = set()
        for pb in pred_boxes:
            ious_local = [(bbox_iou(pb[:4], gb[:4]), i) for i, gb in enumerate(gt_boxes)]
            if not ious_local:
                fp += 1
                continue

            best_iou, best_idx = max(ious_local, key=lambda x: x[0])
            ious.append(best_iou)

            if best_iou >= iou_threshold and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += max(0, len(gt_boxes) - len(matched_gt))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)
    ap = precision * recall
    iou_mean = float(np.mean(ious)) if ious else 0.0

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "AP": float(ap),
        "mAP": float(ap),
        "F_beta": float(f_beta),
        "IoU": iou_mean
    }


# ============================================================
#          VISUALIZACIÓN Y ALMACENAMIENTO DE RESULTADOS
# ============================================================
def create_metrics_folder(model_variant="n", phase="valid"):
    base_dir = Path(__file__).resolve().parents[1] / "metrics" / model_variant / phase
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("test_")]
    folder_name = f"test_{len(existing) + 1:04d}"
    path = base_dir / folder_name
    path.mkdir(exist_ok=True)
    return str(path)


def save_metrics_plots(metrics_dict, save_dir, model_variant="n"):
    names, values = list(metrics_dict.keys()), list(metrics_dict.values())
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, alpha=0.9)
    plt.title(f"YOLOv11-{model_variant.upper()} Evaluation Metrics")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics_overview_{model_variant}.png"))
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.bar(["IoU"], [metrics_dict.get("IoU", 0.0)])
    plt.title(f"Mean IoU - YOLOv11-{model_variant.upper()}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"iou_{model_variant}.png"))
    plt.close()


def save_metrics_summary(metrics_dict, save_dir, model_variant="n"):
    summary_path = os.path.join(save_dir, f"metrics_summary_{model_variant}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"📅 Test generado: {datetime.now()}\n")
        f.write(f"🔧 Modelo evaluado: YOLOv11-{model_variant.upper()}\n\n")
        for k, v in metrics_dict.items():
            f.write(f"{k}: {float(v):.4f}\n")
    print(f"📄 Resumen guardado en {summary_path}")


def measure_fps(model, sample_input, device="cpu", runs=20):
    model.eval()
    sample_input = sample_input.to(device)
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(sample_input)
    total = time.time() - start
    return runs / total


def evaluate_model(preds, targets, save_results=True, model_variant="n", phase="valid"):
    metrics = calculate_metrics(preds, targets)
    if save_results:
        save_dir = create_metrics_folder(model_variant, phase=phase)
        save_metrics_plots(metrics, save_dir, model_variant)
        save_metrics_summary(metrics, save_dir, model_variant)
        print(f"✅ Resultados guardados en {save_dir} para modelo YOLOv11-{model_variant.upper()}")
    return metrics
