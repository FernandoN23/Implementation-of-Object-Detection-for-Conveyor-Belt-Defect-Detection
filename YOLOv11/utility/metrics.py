import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path
from omegaconf import OmegaConf

# ============================================================
#   🔧 Asegurar acceso al paquete raíz YOLOv11
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utility.visualization import TensorboardVisualizer


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
    """Convierte a numpy de forma segura."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        return [_to_numpy(i) for i in x]
    elif isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    else:
        return x


# ============================================================
#        MÉTRICAS GENERALES Y POR CLASE (MULTICLASE)
# ============================================================
def calculate_metrics(preds, targets, class_names=None, iou_threshold=0.5, beta=1.0):
    """Calcula métricas globales y por clase."""
    preds = [_to_numpy(p) for p in preds]
    targets = [_to_numpy(t) for t in targets]
    classes = class_names or []

    # Inicializar estructura por clase
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for c in classes}

    tp, fp, fn, ious = 0, 0, 0, []

    # Aplanar listas anidadas
    flat_preds, flat_targets = [], []
    for p in preds:
        flat_preds.extend(p if isinstance(p, (list, tuple)) else [p])
    for t in targets:
        flat_targets.extend(t if isinstance(t, (list, tuple)) else [t])
    preds, targets = flat_preds, flat_targets

    for pred_boxes, gt_boxes in zip(preds, targets):
        if pred_boxes is None or gt_boxes is None:
            continue
        if isinstance(pred_boxes, (float, int)) or isinstance(gt_boxes, (float, int)):
            continue
        if not hasattr(pred_boxes, "shape") or not hasattr(gt_boxes, "shape"):
            continue
        if pred_boxes.size == 0 or gt_boxes.size == 0:
            fn += int(getattr(gt_boxes, "shape", [0])[0])
            continue

        # Asegurar formato [x1,y1,x2,y2,score,cls]
        try:
            pred_boxes = pred_boxes.reshape(-1, min(6, pred_boxes.shape[-1]))
            gt_boxes = gt_boxes.reshape(-1, min(6, gt_boxes.shape[-1]))
        except Exception:
            continue

        matched_gt = set()
        for pb in pred_boxes:
            cls = int(pb[5]) if pb.shape[-1] > 5 else 0
            class_name = classes[cls] if cls < len(classes) else f"class_{cls}"
            ious_local = [(bbox_iou(pb[:4], gb[:4]), i, int(gb[5]) if gb.shape[-1] > 5 else 0)
                          for i, gb in enumerate(gt_boxes)]
            if not ious_local:
                fp += 1
                per_class[class_name]["fp"] += 1
                continue

            best_iou, best_idx, gt_cls = max(ious_local, key=lambda x: x[0])
            ious.append(best_iou)
            per_class[class_name]["ious"].append(best_iou)

            if best_iou >= iou_threshold and best_idx not in matched_gt and cls == gt_cls:
                tp += 1
                per_class[class_name]["tp"] += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
                per_class[class_name]["fp"] += 1

        fn += max(0, len(gt_boxes) - len(matched_gt))
        for gb in gt_boxes:
            cls = int(gb[5]) if gb.shape[-1] > 5 else 0
            class_name = classes[cls] if cls < len(classes) else f"class_{cls}"
            if cls not in [int(pb[5]) for pb in pred_boxes]:
                per_class[class_name]["fn"] += 1

    # ==== MÉTRICAS GLOBALES ====
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)
    ap = precision * recall
    iou_mean = float(np.mean(ious)) if ious else 0.0

    global_metrics = {
        "Precision": float(precision),
        "Recall": float(recall),
        "AP": float(ap),
        "mAP": float(ap),
        "F_beta": float(f_beta),
        "IoU": iou_mean
    }

    # ==== MÉTRICAS POR CLASE ====
    per_class_metrics = {}
    for cls, data in per_class.items():
        c_tp, c_fp, c_fn = data["tp"], data["fp"], data["fn"]
        c_ious = data["ious"]
        c_prec = c_tp / (c_tp + c_fp + 1e-6)
        c_rec = c_tp / (c_tp + c_fn + 1e-6)
        c_f = (1 + beta**2) * (c_prec * c_rec) / (beta**2 * c_prec + c_rec + 1e-6)
        c_ap = c_prec * c_rec
        c_iou = float(np.mean(c_ious)) if c_ious else 0.0

        per_class_metrics[cls] = {
            "Precision": c_prec,
            "Recall": c_rec,
            "AP": c_ap,
            "F_beta": c_f,
            "IoU": c_iou
        }

    return global_metrics, per_class_metrics


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


def save_metrics_plots(global_metrics, per_class_metrics, save_dir, model_variant="n"):
    # === Gráfico global ===
    names, values = list(global_metrics.keys()), list(global_metrics.values())
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, alpha=0.9)
    plt.title(f"YOLOv11-{model_variant.upper()} Global Metrics")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"global_metrics_{model_variant}.png"))
    plt.close()

    # === Gráfico por clase ===
    for cls, cls_metrics in per_class_metrics.items():
        plt.figure(figsize=(6, 4))
        plt.bar(cls_metrics.keys(), cls_metrics.values(), alpha=0.85)
        plt.title(f"{cls} - Metrics ({model_variant.upper()})")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{cls}_metrics_{model_variant}.png"))
        plt.close()


def save_metrics_summary(global_metrics, per_class_metrics, save_dir, model_variant="n", phase="valid"):
    summary_path = os.path.join(save_dir, f"metrics_summary_{model_variant}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"📅 {datetime.now()}\n")
        f.write(f"🔧 Modelo: YOLOv11-{model_variant.upper()} | Fase: {phase}\n\n")

        f.write("=== MÉTRICAS GLOBALES ===\n")
        for k, v in global_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n=== MÉTRICAS POR CLASE ===\n")
        for cls, cls_metrics in per_class_metrics.items():
            f.write(f"\n[{cls}]\n")
            for k, v in cls_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")

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


# ============================================================
#            EVALUACIÓN Y REGISTRO EN TENSORBOARD
# ============================================================
def evaluate_model(preds, targets, save_results=True, model_variant="n", phase="valid"):
    # Leer nombres de clases desde configs/yolo11.yaml
    try:
        cfg = OmegaConf.load("YOLOv11/configs/yolo11.yaml")
        class_names = cfg.get("names", [])
    except Exception:
        class_names = []

    global_metrics, per_class_metrics = calculate_metrics(preds, targets, class_names)

    if save_results:
        save_dir = create_metrics_folder(model_variant, phase=phase)
        save_metrics_plots(global_metrics, per_class_metrics, save_dir, model_variant)
        save_metrics_summary(global_metrics, per_class_metrics, save_dir, model_variant, phase)
        print(f"✅ Resultados guardados en {save_dir} para modelo YOLOv11-{model_variant.upper()}")

    # === Integración con TensorBoard ===
    try:
        tb = TensorboardVisualizer(model_variant=model_variant)
        tb.log_metrics(global_metrics, step=0, phase=phase)

        for class_name, cls_metrics in per_class_metrics.items():
            tb.log_metrics(cls_metrics, step=0, phase=phase, class_name=class_name)

        tb.log_images_folder(save_dir, step=0, phase=phase)
        tb.flush()
        tb.close()

    except Exception as e:
        print(f"⚠️ Error al registrar métricas en TensorBoard: {e}")

    return global_metrics
