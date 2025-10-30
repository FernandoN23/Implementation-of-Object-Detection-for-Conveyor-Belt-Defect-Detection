# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: test_metrics.py
# Prueba funcional de métricas para YOLOv11. Toma una imagen del
# dataset (aleatoria o por índice), lee su etiqueta YOLO (cx,cy,w,h),
# genera una detección sintética con leve desplazamiento/escala y
# valida el cálculo de Precision/Recall, mAP@50/50-95, matriz de
# confusión y curvas PR. Guarda resultados en YOLOv11/metrics/...
#==============================================================
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# --- Inyección robusta del project root en sys.path ---
FILE = Path(__file__).resolve()
PROJ = FILE.parents[1]  # YOLOv11/
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
UTIL = FILE.parent  # YOLOv11/utility
if str(UTIL) not in sys.path:
    sys.path.insert(0, str(UTIL))

from metrics import DetMetricsYOLOv11

SUPPORTED_IM_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _images_dir_from_split(ds: dict, split: str) -> Path:
    if split in ds and ds[split]:
        return Path(ds[split])
    base = Path(ds.get("path", "."))
    return base / split / "images"


def _labels_dir_for_images_dir(images_dir: Path) -> Path:
    """Devuelve la carpeta 'labels' hermana de 'images' sin usar secuencias de escape problemáticas."""
    # Caso típico: el nombre del directorio es exactamente 'images'
    if images_dir.name.lower() == "images":
        return images_dir.with_name("labels")
    # Fallback robusto: reemplazo de segmento en la cadena (Windows/Unix)
    p = str(images_dir)
    p = p.replace("/images", "/labels")
    p = p.replace("\\\\images", "\\\\labels")  # dos backslashes en la cadena
    p = p.replace("\\images", "\\labels")      # caso normal en Windows (un backslash)
    return Path(p)


def _list_images(images_dir: Path) -> List[Path]:
    ims: List[Path] = []
    for ext in SUPPORTED_IM_EXTS:
        ims.extend(images_dir.rglob(f"*{ext}"))
        ims.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(ims)


def _label_path_for_image(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")


def _load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            c = float(parts[0])
            x, y, w, h = map(float, parts[1:5])
            rows.append([c, x, y, w, h])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _xywhn_to_xyxy_pix(xywhn: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    cx, cy, w, h = xywhn.T
    x1 = (cx - w * 0.5) * W
    y1 = (cy - h * 0.5) * H
    x2 = (cx + w * 0.5) * W
    y2 = (cy + h * 0.5) * H
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom) if denom > 0 else 0.0


def _clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def _build_pred_from_gt(gt_row: np.ndarray,
                        hw: Tuple[int, int],
                        jitter: float = 0.03,
                        scale_jitter: float = 0.00,
                        conf: float = 0.9,
                        wrong_class: bool = False,
                        num_classes: int | None = None) -> np.ndarray:
    c, cx, cy, w, h = gt_row.astype(np.float32)
    cx_p = cx + np.random.randn() * jitter
    cy_p = cy + np.random.randn() * jitter
    w_p = w * (1.0 + np.random.randn() * scale_jitter)
    h_p = h * (1.0 + np.random.randn() * scale_jitter)
    cx_p, cy_p, w_p, h_p = _clamp01(np.array([cx_p, cy_p, w_p, h_p], dtype=np.float32))

    xyxy = _xywhn_to_xyxy_pix(np.array([[cx_p, cy_p, w_p, h_p]], dtype=np.float32), hw)[0]
    cls = int(c)
    if wrong_class:
        cls = (cls + 1) if (num_classes is None) else ((cls + 1) % max(1, int(num_classes)))
    return np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, float(cls)], dtype=np.float32)


def _overlay_debug(image_bgr: np.ndarray,
                   gt_xyxy: Optional[np.ndarray],
                   pred_xyxy: Optional[np.ndarray],
                   out_path: Path) -> None:
    im = image_bgr.copy()
    if gt_xyxy is not None:
        x1, y1, x2, y2 = map(int, gt_xyxy)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, "GT", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if pred_xyxy is not None:
        x1, y1, x2, y2 = map(int, pred_xyxy)
        cv2.rectangle(im, (x1, y1), (x2, y2), (36, 36, 255), 2)
        cv2.putText(im, "PRED", (x1, min(im.shape[0] - 1, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 36, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)


def main() -> None:
    p = argparse.ArgumentParser("Prueba de métricas YOLOv11 (smoke test)")
    p.add_argument("--variant", default="m", type=str, help="Variante del modelo (n/s/m/l/xl) para rutas de logging")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split a muestrear")
    p.add_argument("--index", default="random", help="Índice de imagen o `random`")
    p.add_argument("--jitter", type=float, default=0.03, help="Desplazamiento gaussiano (px normalizado) del bbox respecto al GT")
    p.add_argument("--scale_jitter", type=float, default=0.00, help="Jitter de escala relativo en w/h")
    p.add_argument("--conf", type=float, default=0.90, help="Confianza de la predicción sintética")
    p.add_argument("--wrong_class", action="store_true", help="Usa clase incorrecta para forzar FP")
    p.add_argument("--extra_fp", type=int, default=0, help="Cantidad de falsos positivos aleatorios a inyectar")
    p.add_argument("--allow_negative", action="store_true", help="Permite probar con imagen sin etiquetas (negativa)")
    p.add_argument("--seed", type=int, default=123, help="Semilla para reproducibilidad del jitter")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- Configs/Dataset ---
    ds_yaml = PROJ / "configs" / "dataset.yaml"
    assert ds_yaml.exists(), f"No se encontró {ds_yaml}"
    ds = _read_yaml(ds_yaml)

    names = ds.get("names")
    if isinstance(names, dict):
        class_names: Dict[int, str] = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        class_names = {i: str(n) for i, n in enumerate(names)}
    else:
        raise RuntimeError("'names' no válido en dataset.yaml")

    images_dir = _images_dir_from_split(ds, args.split)
    labels_dir = _labels_dir_for_images_dir(images_dir)
    assert images_dir.exists(), f"No existe el directorio de imágenes: {images_dir}"

    imgs = _list_images(images_dir)
    assert len(imgs) > 0, f"No se encontraron imágenes en {images_dir}"

    # --- Selección de imagen ---
    if args.index == "random":
        idx = random.randrange(len(imgs))
    else:
        try:
            idx = int(args.index)
        except Exception:
            raise SystemExit("--index debe ser entero o 'random'")
        assert 0 <= idx < len(imgs), f"Índice fuera de rango (0..{len(imgs)-1})"

    img_path = imgs[idx]
    lbl_path = _label_path_for_image(img_path, labels_dir)

    # --- Carga imagen y etiqueta ---
    im = cv2.imread(str(img_path))
    assert im is not None, f"No se pudo leer la imagen: {img_path}"
    H, W = im.shape[:2]

    y = _load_yolo_labels(lbl_path)  # (N,5): [cls,cx,cy,w,h]
    print(f"[Info] Imagen: {img_path}")
    print(f"[Info] Label:  {lbl_path}  (N={len(y)})")

    run_name = f"metrics_smoketest_{_now_tag()}"

    # --- Construcción de GT y pred ---
    targets_list: List[torch.Tensor] = []
    preds_list: List[torch.Tensor] = []
    sizes: List[Tuple[int, int]] = []

    if len(y) == 0:
        if not args.allow_negative:
            raise SystemExit("La imagen seleccionada no posee etiquetas. Use --allow_negative o elija otra.")
        gt = torch.zeros((0, 6), dtype=torch.float32)
        preds = []
        for _ in range(args.extra_fp):
            cx, cy, w, h = [random.random() for _ in range(4)]
            xyxy = _xywhn_to_xyxy_pix(np.array([[cx, cy, w * 0.2, h * 0.2]], dtype=np.float32), (H, W))[0]
            c = float(random.randrange(len(class_names)))
            preds.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], args.conf, c])
        pred_t = torch.tensor(preds, dtype=torch.float32) if preds else torch.zeros((0, 6), dtype=torch.float32)
    else:
        areas = y[:, 3] * y[:, 4]
        gt_row = y[int(np.argmax(areas))]  # [cls,cx,cy,w,h]
        gt = torch.tensor([[0.0, gt_row[0], gt_row[1], gt_row[2], gt_row[3], gt_row[4]]], dtype=torch.float32)
        pred_np = _build_pred_from_gt(gt_row, (H, W), jitter=args.jitter, scale_jitter=args.scale_jitter,
                                      conf=args.conf, wrong_class=args.wrong_class, num_classes=len(class_names))
        preds = [pred_np]
        for _ in range(args.extra_fp):
            cx, cy, w, h = [random.random() for _ in range(4)]
            xyxy = _xywhn_to_xyxy_pix(np.array([[cx, cy, w * 0.2, h * 0.2]], dtype=np.float32), (H, W))[0]
            c = float(random.randrange(len(class_names)))
            preds.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], max(0.1, args.conf - 0.2), c])
        pred_t = torch.tensor(np.array(preds, dtype=np.float32), dtype=torch.float32)

        gt_xyxy = _xywhn_to_xyxy_pix(np.array([gt_row[1:]], dtype=np.float32), (H, W))[0]
        out_overlay = PROJ / "metrics" / "test_metrics" / args.variant / run_name / "overlay.png"
        _overlay_debug(im, gt_xyxy, pred_np[:4], out_overlay)
        # Debug IoU puntual (visual)
        iou_dbg = _iou_xyxy(gt_xyxy, pred_np[:4])
        print(f"[Debug] IoU(GT,PRED) = {iou_dbg:.3f}")
        print(f"[Info] Overlay guardado en: {out_overlay}")

    targets_list.append(gt)
    preds_list.append(pred_t)
    sizes.append((H, W))

    # --- Métricas ---
    save_dir = PROJ / "metrics" / "test_metrics" / args.variant / run_name
    metricor = DetMetricsYOLOv11(class_names=class_names, save_dir=save_dir)
    metricor.add_batch(preds_list, targets_list, sizes, labels_is_xywhn=True, iou_match_for_cm=0.50)
    summary, _ = metricor.finalize()

    # --- Logging CSV ---
    csv_path = save_dir / "test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","split","metric","value"])
        for k, v in summary.to_dict().items():
            w.writerow([1, "test", k, float(v)])

    print("\n=== Resumen de métricas (JSON) ===")
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"\nFiguras/MC/PR guardadas en: {save_dir}")


if __name__ == "__main__":
    main()
