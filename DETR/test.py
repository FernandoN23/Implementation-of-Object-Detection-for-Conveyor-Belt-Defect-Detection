# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/test.py
# Descripción: Punto de entrada principal para pruebas de inferencia
#              sobre modelos DETR entrenados (detección en imágenes
#              de la partición Dataset/test con comparación
#              predicción vs. etiqueta real).
# ==============================================================

"""Punto de entrada CLI para testear modelos DETR ya entrenados.

Visor interactivo orientado a inspección fina de desempeño:
- Carga un modelo DETR (por defecto, variante r50).
- Recorre exclusivamente la partición Dataset/test.
- Muestra, para cada imagen, los bounding boxes **reales** (labels)
  y las **predicciones** del modelo con colores diferenciados.
- Calcula métricas locales por imagen (Precision, Recall, IoU por
  clase y promedios) usando un umbral de IoU configurable para el
  matching.
- Permite ocultar/mostrar las predicciones manteniendo siempre
  visibles las cajas reales.

La implementación se apoya en OpenCV para la interfaz gráfica y en la
implementación local de DETR para la carga/inferencia del modelo.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Rutas base de proyecto
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parent  # .../DETR
PROJECT_ROOT = DETR_ROOT.parent  # raíz del repositorio
CONFIGS_ROOT = DETR_ROOT / "configs"  # DETR/configs

# Submódulo oficial de DETR
DETR_SUBMODULE = DETR_ROOT / "detr"
if DETR_SUBMODULE.is_dir() and str(DETR_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(DETR_SUBMODULE))

# Dataset principal
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"

# Mitigación opcional MIOpen y sistema de warnings del proyecto
try:
    from engine.bootstrap_miopen import MIOpenConfig, bootstrap, MuteStderr
except Exception as e:
    print(f"[test.py] ERROR: No se pudo importar bootstrap_miopen: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """Representa un bounding box en coordenadas absolutas (imagen)."""
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


@dataclass
class ClassStats:
    """Acumuladores por clase para una imagen."""
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    matches: int = 0


@dataclass
class ModelContext:
    """Contenedor de contexto para inferencia DETR."""
    model: Any
    postprocessors: Any
    device: Any


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def _resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_class_names() -> List[str]:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"[test.py] No se encontró el archivo de clases: {DATA_YAML}")

    data = load_yaml(DATA_YAML)
    names = data.get("names") or data.get("classes")

    if isinstance(names, dict):
        names = list(names.values())

    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(f"[test.py] El archivo {DATA_YAML} debe definir 'names' o 'classes'.")

    return [str(n) for n in names]


def load_test_images(image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Tuple[List[Path], Path]:
    images_dir = DATASET_ROOT / "test" / "images"
    labels_dir = DATASET_ROOT / "test" / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"[test.py] No se encontró el directorio de imágenes de test: {images_dir}")

    image_paths: List[Path] = []
    for ext in image_exts:
        image_paths.extend(sorted(images_dir.glob(f"*{ext}")))

    if not image_paths:
        raise ValueError(f"[test.py] No se encontraron imágenes en {images_dir}")

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"[test.py] No se encontró el directorio de labels de test: {labels_dir}")

    return image_paths, labels_dir


def build_color_palettes(num_classes: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    gt_color = (0, 0, 160)  # Rojo oscuro para GT
    pred_color = (0, 160, 0)  # Verde oscuro para Predicciones

    colors_gt = [gt_color for _ in range(num_classes)]
    colors_pred = [pred_color for _ in range(num_classes)]

    return colors_gt, colors_pred


# ---------------------------------------------------------------------------
# Carga de labels y modelo
# ---------------------------------------------------------------------------

def load_gt_boxes(label_file: Path, img_shape: Tuple[int, int, int]) -> List[Box]:
    h, w = img_shape[:2]
    if not label_file.is_file():
        return []

    boxes: List[Box] = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
            cls_id_int = int(cls_id)

            x_c *= w
            y_c *= h
            bw *= w
            bh *= h

            x1 = x_c - bw / 2.0
            y1 = y_c - bh / 2.0
            x2 = x_c + bw / 2.0
            y2 = y_c + bh / 2.0

            boxes.append(Box(cls_id=cls_id_int, x1=x1, y1=y1, x2=x2, y2=y2, conf=1.0))

    return boxes


def load_model(weights: Path, variant: str, device_str: str, num_classes: int) -> ModelContext:
    import torch
    import torch.nn as nn
    from models import build_model
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig

    if not weights.is_file():
        raise FileNotFoundError(f"[test.py] No se encontró el archivo de pesos: {weights}")

    device = torch.device(device_str if device_str else "cuda" if torch.cuda.is_available() else "cpu")

    # Cargar configuración de la variante
    variants_cfg = load_yaml(CONFIGS_ROOT / "model_variants.yaml")
    if variant not in variants_cfg['variants']:
        raise ValueError(f"[test.py] Variante '{variant}' no encontrada en model_variants.yaml")

    v_params = variants_cfg['variants'][variant]

    # Construir Namespace Dummy
    base_args = {
        'lr_backbone': 0, 'masks': False, 'frozen_weights': None,
        'aux_loss': False, 'set_cost_class': 1.0, 'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0, 'bbox_loss_coef': 5.0, 'giou_loss_coef': 2.0,
        'eos_coef': 0.1, 'dataset_file': 'coco', 'device': str(device)
    }
    base_args.update(v_params)
    model_args = argparse.Namespace(**base_args)

    print(f"[test.py] Construyendo arquitectura DETR ({variant})...")
    model, criterion, postprocessors = build_model(model_args)

    # Ajustar cabezal de clasificación
    hidden_dim = model.transformer.d_model
    model.class_embed = nn.Linear(hidden_dim, num_classes + 1)

    # Aplicar parche BN2GN
    replace_bn_with_gn(model, BN2GNConfig(policy='on', verbose=0))

    print(f"[test.py] Cargando pesos desde {weights}...")
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    model.to(device)
    model.eval()

    return ModelContext(model=model, postprocessors=postprocessors, device=device)


# ---------------------------------------------------------------------------
# Inferencia por imagen (pipeline DETR)
# ---------------------------------------------------------------------------

def infer_image(ctx: ModelContext, img_bgr: np.ndarray, conf_thres: float) -> List[Box]:
    import torch
    import torchvision.transforms.functional as F

    # 1) BGR a RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 2) Convertir a Tensor y Normalizar (ImageNet stats)
    tensor = F.to_tensor(img_rgb)
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(ctx.device)

    # 3) Forward Pass
    with torch.no_grad():
        with MuteStderr():
            outputs = ctx.model(tensor)

    # 4) Postprocesamiento (Convertir cxcywh a xyxy absoluto)
    orig_target_sizes = torch.tensor([[h, w]], device=ctx.device)
    results = ctx.postprocessors['bbox'](outputs, orig_target_sizes)[0]

    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()

    boxes_out: List[Box] = []

    # 5) Filtrar por confianza
    for score, label, box in zip(scores, labels, boxes):
        if score >= conf_thres:
            x1, y1, x2, y2 = box
            boxes_out.append(
                Box(
                    cls_id=int(label),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=float(score),
                )
            )

    return boxes_out


# ---------------------------------------------------------------------------
# Métricas P/R/IoU por clase (por imagen)
# ---------------------------------------------------------------------------

def box_iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def evaluate_image(
        gt_boxes: List[Box],
        pred_boxes: List[Box],
        num_classes: int,
        iou_match: float,
) -> Tuple[Dict[int, ClassStats], Dict[str, float]]:
    stats: Dict[int, ClassStats] = {c: ClassStats() for c in range(num_classes)}

    for box in gt_boxes:
        if 0 <= box.cls_id < num_classes:
            stats[box.cls_id].n_gt += 1

    for box in pred_boxes:
        if 0 <= box.cls_id < num_classes:
            stats[box.cls_id].n_pred += 1

    for cls in range(num_classes):
        gt_c = [b for b in gt_boxes if b.cls_id == cls]
        pred_c = sorted([b for b in pred_boxes if b.cls_id == cls], key=lambda b: b.conf, reverse=True)

        used_gt = [False] * len(gt_c)

        for pred in pred_c:
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gt_c):
                if used_gt[i]: continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_match and best_idx >= 0:
                used_gt[best_idx] = True
                stats[cls].tp += 1
                stats[cls].matches += 1
                stats[cls].iou_sum += best_iou
            else:
                stats[cls].fp += 1

        fn_count = sum(1 for u in used_gt if not u)
        stats[cls].fn += fn_count

    global_metrics: Dict[str, float] = {"P_macro": 0.0, "R_macro": 0.0, "IoU_macro": 0.0}
    p_list, r_list, iou_list = [], [], []

    for cls in range(num_classes):
        s = stats[cls]
        denom_p = s.tp + s.fp
        denom_r = s.tp + s.fn

        p = float(s.tp / denom_p) if denom_p > 0 else 0.0
        r = float(s.tp / denom_r) if denom_r > 0 else 0.0
        iou_mean = float(s.iou_sum / s.matches) if s.matches > 0 else 0.0

        s.precision = p  # type: ignore
        s.recall = r  # type: ignore
        s.iou_mean = iou_mean  # type: ignore

        if s.n_gt > 0 or s.n_pred > 0:
            p_list.append(p)
            r_list.append(r)
            iou_list.append(iou_mean)

    if p_list: global_metrics["P_macro"] = float(np.mean(p_list))
    if r_list: global_metrics["R_macro"] = float(np.mean(r_list))
    if iou_list: global_metrics["IoU_macro"] = float(np.mean(iou_list))

    return stats, global_metrics


# ---------------------------------------------------------------------------
# Dibujo de boxes y leyenda
# ---------------------------------------------------------------------------

def draw_boxes(
        img: np.ndarray,
        boxes: List[Box],
        class_names: List[str],
        colors: List[Tuple[int, int, int]],
        thickness: int = 2,
        draw_conf: bool = False,
) -> np.ndarray:
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        if not (0 <= box.cls_id < len(colors)):
            continue
        color = colors[box.cls_id]

        x1 = int(max(0, min(w - 1, box.x1)))
        y1 = int(max(0, min(h - 1, box.y1)))
        x2 = int(max(0, min(w - 1, box.x2)))
        y2 = int(max(0, min(h - 1, box.y2)))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = class_names[box.cls_id]
        if draw_conf:
            label = f"{label} {box.conf:.2f}"

        font_scale = 0.5
        t = 1
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, t)

        tx1 = x1
        ty1 = y1 - th - bl - 3
        tx2 = x1 + tw + 4
        ty2 = y1

        if ty1 < 0:
            ty1 = y2 + 3
            ty2 = y2 + th + bl + 3

        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(img, label, (tx1 + 2, ty2 - 4), font, font_scale, (255, 255, 255), t, cv2.LINE_AA)

    return img


def draw_legend(
        split: str,
        idx: int,
        num_images: int,
        model_name: str,
        class_names: List[str],
        stats: Dict[int, ClassStats],
        global_metrics: Dict[str, float],
        show_pred: bool,
        colors_gt: List[Tuple[int, int, int]],
        colors_pred: List[Tuple[int, int, int]],
        height: int,
) -> np.ndarray:
    legend_width = 460
    legend_height = max(height, 600)
    canvas = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    canvas[:] = (50, 60, 90)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    t = 1
    y = 30
    lh = 22

    def put(line: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        nonlocal y
        cv2.putText(canvas, line, (10, y), font, fs, color, t, cv2.LINE_AA)
        y += lh

    put(f"Split: {split}", (255, 255, 0))
    put(f"Imagen: {idx + 1}/{num_images}", (255, 255, 0))
    put(f"Modelo: {os.path.basename(model_name)}", (200, 255, 255))
    put("", (255, 255, 255))

    put("Metricas por imagen (macro):", (0, 255, 255))
    put(f"P:   {global_metrics.get('P_macro', 0.0):.3f}")
    put(f"R:   {global_metrics.get('R_macro', 0.0):.3f}")
    put(f"IoU: {global_metrics.get('IoU_macro', 0.0):.3f}")
    put("", (255, 255, 255))

    put("Leyenda bboxes:", (0, 255, 255))
    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_gt[0] if colors_gt else (0, 0, 160), -1)
    cv2.putText(canvas, "GT (etiqueta real)", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA)
    y += lh

    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_pred[0] if colors_pred else (0, 160, 0), -1)
    cv2.putText(canvas, "Prediccion modelo", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA)
    y += lh

    put("", (255, 255, 255))
    put("Comandos:", (0, 255, 255))
    put("<- / 'a': imagen anterior")
    put("-> / 'd': imagen siguiente")
    put("'h': mostrar/ocultar pred.")
    put("ESC: salir")
    put("", (255, 255, 255))
    put(f"Predicciones visibles: {'Si' if show_pred else 'No'}")
    put("", (255, 255, 255))

    put("Metricas por clase:", (0, 255, 255))
    fs_cls = 0.45
    lh_cls = 18

    def put_cls(line: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        nonlocal y
        cv2.putText(canvas, line, (10, y), font, fs_cls, color, t, cv2.LINE_AA)
        y += lh_cls

    for cls_id, s in stats.items():
        if s.n_gt == 0 and s.n_pred == 0:
            continue
        if y + 2 * lh_cls > legend_height - 10:
            put_cls("...", (200, 200, 200))
            break

        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        precision = getattr(s, "precision", 0.0)
        recall = getattr(s, "recall", 0.0)
        iou_mean = getattr(s, "iou_mean", 0.0)

        put_cls(f"[{cls_id}] {name}", (255, 255, 0))
        put_cls(
            f"GT:{s.n_gt} Pred:{s.n_pred} TP:{s.tp} FP:{s.fp} FN:{s.fn} "
            f"P:{precision:.2f} R:{recall:.2f} IoU:{iou_mean:.2f}",
            (220, 220, 220),
        )

    return canvas


# ---------------------------------------------------------------------------
# Bucle principal de visualización
# ---------------------------------------------------------------------------

def run_viewer(args: argparse.Namespace) -> None:
    class_names = load_class_names()
    num_classes = len(class_names)
    colors_gt, colors_pred = build_color_palettes(num_classes)

    weights_path = _resolve_path(args.weights, PROJECT_ROOT)
    ctx = load_model(weights_path, args.variant, args.device, num_classes)

    image_paths, labels_dir = load_test_images()

    idx = 0
    num_images = len(image_paths)
    show_pred = True
    split = "test"

    window_name = "DETR Test Viewer"

    while True:
        if idx < 0: idx = 0
        if idx >= num_images: idx = num_images - 1

        img_path = image_paths[idx]
        filename = img_path.stem
        label_file = labels_dir / f"{filename}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[test.py] WARN: No se pudo leer la imagen: {img_path}")
            idx += 1
            if idx >= num_images: break
            continue

        gt_boxes = load_gt_boxes(label_file, img.shape)
        pred_boxes = infer_image(
            ctx=ctx,
            img_bgr=img,
            conf_thres=args.conf_thres
        )

        stats, global_metrics = evaluate_image(
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            num_classes=num_classes,
            iou_match=args.iou_match,
        )

        img_vis = img.copy()
        img_vis = draw_boxes(img_vis, gt_boxes, class_names, colors_gt, thickness=2, draw_conf=False)
        if show_pred:
            img_vis = draw_boxes(img_vis, pred_boxes, class_names, colors_pred, thickness=1, draw_conf=True)

        legend = draw_legend(
            split=split, idx=idx, num_images=num_images,
            model_name=str(weights_path), class_names=class_names,
            stats=stats, global_metrics=global_metrics,
            show_pred=show_pred, colors_gt=colors_gt, colors_pred=colors_pred,
            height=img_vis.shape[0],
        )

        if legend.shape[0] != img_vis.shape[0]:
            legend = cv2.resize(legend, (legend.shape[1], img_vis.shape[0]))

        combined = np.hstack((img_vis, legend))
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord("d"), 83):  # siguiente
            idx = min(num_images - 1, idx + 1)
        elif key in (ord("a"), 81):  # anterior
            idx = max(0, idx - 1)
        elif key in (ord("h"), ord("p")):
            show_pred = not show_pred

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="DETR.test",
        description=(
            "Visor interactivo para testear un modelo DETR entrenado sobre la "
            "partición Dataset/test, mostrando boxes reales vs. predichos y "
            "métricas locales por imagen."
        ),
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Ruta al archivo de pesos .pt del modelo DETR a testear.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="r50",
        choices=["r50", "r50_dc5", "r101", "r101_dc5"],
        help="Variante de la arquitectura DETR (ej. r50, r101).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo para inferencia: 'cuda' o 'cpu'.",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Umbral de confianza mínimo para visualizar predicciones.",
    )

    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="IoU mínimo para considerar una predicción como TP frente a un GT.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    print(f"[test.py] Iniciando visor interactivo DETR...")
    args = parse_args(argv)

    # Bootstrap MIOpen
    cfg = MIOpenConfig(
        find_mode="FAST",
        user_db_path=None,
        disable_cache=True,
        log_level=0,
        verbose=1,
    )
    bootstrap(cfg)

    # Filtros de warnings
    try:
        from engine.warnings import install_global_warning_filters
        install_global_warning_filters()
    except Exception:
        pass

    run_viewer(args)


if __name__ == "__main__":
    main()