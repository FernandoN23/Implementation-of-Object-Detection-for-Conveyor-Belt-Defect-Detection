import os, sys, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path
from omegaconf import OmegaConf
from utility.visualization import TensorboardVisualizer

# ============================================================
#   FUNCIONES BÁSICAS
# ============================================================
def bbox_iou(box1, box2, eps=1e-6):
    """Calcula IoU entre dos cajas [x1,y1,x2,y2] (torch o np)."""
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
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)): return [_to_numpy(i) for i in x]
    if isinstance(x, dict): return {k: _to_numpy(v) for k, v in x.items()}
    return np.array(x)

# ============================================================
#   MÉTRICAS PRINCIPALES
# ============================================================
def calculate_metrics(preds, targets, class_names=None, iou_threshold=0.5, beta=1.0):
    preds, targets = _to_numpy(preds), _to_numpy(targets)
    classes = class_names or []
    per_class = {f"class_{i}": {"tp": 0, "fp": 0, "fn": 0, "ious": []} for i in range(len(classes) or 5)}
    tp = fp = fn = 0
    all_ious = []

    # --- Normalizar espacio de coordenadas ---
    def normalize_scale(a, b):
        """Si uno está en [0,1] y otro en [0,640], iguala escala."""
        max_a, max_b = np.max(a[..., :4], initial=0), np.max(b[..., :4], initial=0)
        if max_a <= 1.5 and max_b > 1.5: a[..., :4] *= max_b
        elif max_b <= 1.5 and max_a > 1.5: b[..., :4] *= max_a
        return a, b

    for pb, gb in zip(preds, targets):
        if pb is None or gb is None or len(pb) == 0 or len(gb) == 0:
            fn += len(gb)
            continue
        pb, gb = pb.copy(), gb.copy()
        pb, gb = normalize_scale(pb, gb)
        matched = set()

        for p in pb:
            cls = int(np.argmax(p[5:])) if p.shape[-1] > 6 else 0
            cname = classes[cls] if cls < len(classes) else f"class_{cls}"
            ious = [bbox_iou(p[:4], g[:4]) for g in gb]
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
    ap = precision * recall
    global_metrics = {
        "Precision": precision, "Recall": recall,
        "AP": ap, "mAP": ap, "F_beta": f_beta,
        "IoU": float(np.mean(all_ious) if all_ious else 0.0)
    }

    per_class_metrics = {}
    for cls, d in per_class.items():
        ctp, cfp, cfn = d["tp"], d["fp"], d["fn"]
        p = ctp / (ctp + cfp + 1e-9)
        r = ctp / (ctp + cfn + 1e-9)
        f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-9)
        ap = p * r
        iou = float(np.mean(d["ious"])) if d["ious"] else 0.0
        per_class_metrics[cls] = {"Precision": p, "Recall": r, "AP": ap, "F_beta": f, "IoU": iou}
    return global_metrics, per_class_metrics

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
    plt.figure(figsize=(8,5))
    plt.bar(global_metrics.keys(), global_metrics.values(), color="skyblue")
    plt.title(f"Global Metrics - YOLOv11-{model_variant.upper()}")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"global_{model_variant}.png"))
    plt.close()
    for cls, vals in per_class_metrics.items():
        plt.figure(figsize=(6,4))
        plt.bar(vals.keys(), vals.values(), color="lightcoral")
        plt.title(f"{cls} - {model_variant.upper()}")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{cls}_{model_variant}.png"))
        plt.close()

def save_metrics_summary(global_metrics, per_class_metrics, save_dir, model_variant="n", phase="valid"):
    path = os.path.join(save_dir, f"metrics_summary_{model_variant}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"📅 {datetime.now()}\n🔧 Modelo: YOLOv11-{model_variant.upper()} | Fase: {phase}\n\n")
        f.write("=== MÉTRICAS GLOBALES ===\n")
        for k,v in global_metrics.items(): f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== MÉTRICAS POR CLASE ===\n")
        for c, m in per_class_metrics.items():
            f.write(f"\n[{c}]\n")
            for k,v in m.items(): f.write(f"  {k}: {v:.4f}\n")
    print(f"📄 Resumen guardado en {path}")

def measure_fps(model, sample_input, device="cpu", runs=20):
    model.eval()
    sample_input = sample_input.to(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs): _ = model(sample_input)
    return runs / (time.time() - t0)

# ============================================================
#   INTERFAZ PRINCIPAL
# ============================================================
def evaluate_model(preds, targets, save_results=True, model_variant="n", phase="valid"):
    try:
        cfg = OmegaConf.load("YOLOv11/configs/yolo11.yaml")
        names = cfg.get("names", [])
    except Exception: names = []
    g, pc = calculate_metrics(preds, targets, names)
    if save_results:
        d = create_metrics_folder(model_variant, phase)
        save_metrics_plots(g, pc, d, model_variant)
        save_metrics_summary(g, pc, d, model_variant, phase)
        print(f"✅ Resultados guardados en {d} (YOLOv11-{model_variant.upper()})")
    try:
        tb = TensorboardVisualizer(log_dir="YOLOv11/runs", model_variant=f"{model_variant}/{phase}")
        tb.log_metrics(g, 0, phase)
        for c,v in pc.items(): tb.log_metrics(v, 0, phase, class_name=c)
        if save_results: tb.log_images_folder(d, 0, phase)
        tb.close()
    except Exception as e:
        print(f"⚠️ TensorBoard error: {e}")
    return g, pc
