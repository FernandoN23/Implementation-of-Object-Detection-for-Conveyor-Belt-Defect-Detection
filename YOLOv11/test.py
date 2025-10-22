"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: test.py
=============================================================

Evaluación visual del modelo YOLOv11 en el conjunto de test.
Muestra predicciones y bboxes reales en una sola ventana,
con etiquetas visibles, leyenda lateral y métricas por imagen.
=============================================================
"""

import os, cv2, yaml, torch, numpy as np
from collections import defaultdict
from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.weights import load_checkpoint


# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
DATASET_DIR = "./Dataset"
IMG_EXT = (".jpg", ".jpeg", ".png")

def load_classes():
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names") or data.get("classes")

def load_split(split="test"):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    imgs = []
    for e in IMG_EXT:
        imgs.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(e)])
    imgs.sort()
    return imgs, lbl_dir

def load_gt_boxes(label_file):
    boxes = []
    if not os.path.exists(label_file):
        return boxes
    with open(label_file, "r") as f:
        for line in f:
            c, x, y, w, h = map(float, line.strip().split()[:5])
            boxes.append((c, x, y, w, h))
    return boxes


# =============================================================
# MÉTRICAS LOCALES
# =============================================================
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def compute_image_metrics(preds, gts, iou_thr=0.5):
    if len(preds) == 0 and len(gts) == 0:
        return 1.0, 1.0, 1.0
    if len(preds) == 0 or len(gts) == 0:
        return 0.0, 0.0, 0.0

    ious, tp, fp = [], 0, 0
    matched_gt = set()

    for p in preds:
        pbox = p[:4]
        best_iou, best_idx = 0, -1
        for i, g in enumerate(gts):
            iou = compute_iou(pbox, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thr and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    mean_iou = np.mean(ious) if ious else 0
    return precision, recall, mean_iou


# =============================================================
# VISUALIZACIÓN Y LEYENDA
# =============================================================
def put_label(img, x1, y1, text, color):
    """Dibuja fondo y texto de etiqueta evitando límites."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.6, 1
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)

    text_y1 = max(0, y1 - th - bl - 3)
    text_y2 = text_y1 + th + bl + 3
    text_x1 = max(0, x1)
    text_x2 = text_x1 + tw + 6

    cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1)
    cv2.putText(img, text, (text_x1 + 3, text_y2 - 4), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return img


def draw_combined(img, gt_boxes, pred_boxes, class_names, colors, idx, total, metrics):
    h, w = img.shape[:2]
    img_vis = img.copy()
    class_counts_gt, class_counts_pred = defaultdict(int), defaultdict(int)

    # --- Dibujar ground truth (verde) ---
    for c, x, y, bw, bh in gt_boxes:
        x1, y1 = int((x - bw/2)*w), int((y - bh/2)*h)
        x2, y2 = int((x + bw/2)*w), int((y + bh/2)*h)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (60,220,60), 2)
        img_vis = put_label(img_vis, x1, y1, class_names[int(c)], (60,220,60))
        class_counts_gt[int(c)] += 1

    # --- Dibujar predicciones (color clase) ---
    for p in pred_boxes:
        x1, y1, x2, y2, conf, cls = map(int, p[:6])
        color = colors[cls % len(colors)]
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]} {conf:.2f}"
        img_vis = put_label(img_vis, x1, y1, label, color)
        class_counts_pred[cls] += 1

    # --- Crear leyenda lateral ---
    legend_w = 260
    legend = np.zeros((h, legend_w, 3), dtype=np.uint8)
    legend[:] = (40, 50, 80)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y, step = 30, 23

    lines = [
        f"Imagen: {idx+1}/{total}",
        f"Etiquetas tot: {sum(class_counts_gt.values())}",
        f"Pred tot: {sum(class_counts_pred.values())}",
        "",
        f"Precisión: {metrics[0]*100:.1f}%",
        f"Recall: {metrics[1]*100:.1f}%",
        f"IoU medio: {metrics[2]*100:.1f}%",
        "",
        "Comandos:",
        "'a' -> anterior",
        "'d' -> siguiente",
        "ESC -> salir",
        "",
        "Conteo por clase:"
    ]

    for line in lines:
        color = (255,255,0) if "IoU" in line or "Precisión" in line else (255,255,255)
        cv2.putText(legend, line, (10, y), font, 0.55, color, 1, cv2.LINE_AA)
        y += step

    for i, cls in enumerate(class_names):
        et, pred = class_counts_gt.get(i, 0), class_counts_pred.get(i, 0)
        color = colors[i % len(colors)]
        txt = f"{cls}: ET={et}, P={pred}"
        cv2.circle(legend, (15, y - 8), 5, color, -1)
        cv2.putText(legend, txt, (35, y), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += step

    combined = np.hstack((img_vis, legend))
    return combined


# =============================================================
# MAIN
# =============================================================
def main():
    base_path = "YOLOv11/weights"
    variants = [v for v in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, v))]
    print("📂 Variantes disponibles:")
    for v in variants: print(" •", v)
    variant = input("👉 Variante a testear: ").strip().lower() or "n"

    ckpt_dir = os.path.join(base_path, variant, "train")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"📦 Usando checkpoint: {ckpt_path}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    cfg = parser.parse_model_config()
    model = YOLOv11(cfg_path=model_cfg_path, num_classes=cfg.get("nc", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, path=ckpt_path, device=device)
    model.to(device).eval()

    class_names = load_classes()
    colors = [tuple(np.random.randint(0,255,3).tolist()) for _ in range(len(class_names))]
    imgs, lbl_dir = load_split("test")
    idx = 0

    print("\n🧩 Controles: [a]=anterior | [d]=siguiente | [ESC]=salir")

    while True:
        if idx < 0: idx = 0
        if idx >= len(imgs): idx = len(imgs) - 1

        img_path = imgs[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        gt_boxes = load_gt_boxes(lbl_path)

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = torch.tensor(img_in).permute(2,0,1).unsqueeze(0).float()/255.0
        img_in = img_in.to(device)

        with torch.no_grad():
            preds = model(img_in)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        pred_boxes = []
        for p in np.atleast_2d(preds):
            p = np.array(p).flatten()
            if len(p) >= 6:
                pred_boxes.append(p[:6])

        # Calcular métricas por imagen
        gts_xyxy = []
        for c, x, y, bw, bh in gt_boxes:
            x1, y1 = (x - bw/2)*w, (y - bh/2)*h
            x2, y2 = (x + bw/2)*w, (y + bh/2)*h
            gts_xyxy.append([x1, y1, x2, y2])
        preds_xyxy = [p[:4] for p in pred_boxes]
        metrics = compute_image_metrics(preds_xyxy, gts_xyxy)

        combined = draw_combined(img, gt_boxes, pred_boxes, class_names, colors, idx, len(imgs), metrics)
        combined = cv2.resize(combined, (1280, 720))
        cv2.imshow("YOLOv11 - Test Viewer", combined)

        k = cv2.waitKey(0) & 0xFF
        if k == 27: break
        elif k == ord('d'): idx += 1
        elif k == ord('a'): idx -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
