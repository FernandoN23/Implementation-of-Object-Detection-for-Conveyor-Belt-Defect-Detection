# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/test.py
# Descripción: Punto de entrada principal para pruebas de inferencia
#              sobre modelos YOLOv5 entrenados (detección en imágenes
#              de la partición Dataset/test con comparación
#              predicción vs. etiqueta real).
#==============================================================

"""Punto de entrada CLI para testear modelos YOLOv5 ya entrenados.

Visor interactivo orientado a inspección fina de desempeño:
- Carga un modelo YOLOv5 (por defecto, el checkpoint final entrenado
  para fallas en correas transportadoras).
- Recorre exclusivamente la partición Dataset/test.
- Muestra, para cada imagen, los bounding boxes **reales** (labels)
  y las **predicciones** del modelo con colores diferenciados.
- Calcula métricas locales por imagen (Precision, Recall, IoU por
  clase y promedios) usando un umbral de IoU configurable para el
  matching.
- Permite ocultar/mostrar las predicciones manteniendo siempre
  visibles las cajas reales.

La implementación se apoya en OpenCV para la interfaz gráfica y en la
implementación local de YOLOv5 (carpeta YOLO/yolov5) para la
carga/inferencia del modelo, reutilizando el mismo pipeline básico de
preprocesamiento, forward y NMS que en los scripts oficiales
(detect/val).

Nota importante de diseño
-------------------------
Para respetar la política de `engine.bootstrap_miopen` (que requiere
configurar MIOpen **antes** de importar PyTorch), los imports de
PyTorch y de los módulos de YOLOv5 se realizan de forma perezosa dentro
de las funciones de carga e inferencia. Adicionalmente, se aplica el
parche BN→GN definido en `engine.bn2gn_patch` con política `on_error`
sobre el modelo cargado, de forma análoga a la utilizada en
entrenamiento y validación.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Mitigación opcional MIOpen y sistema de warnings del proyecto
try:  # pragma: no cover - entorno mínimo sin estos módulos
    from engine.bootstrap_miopen import MIOpenConfig, bootstrap  # type: ignore
except Exception:  # pragma: no cover
    MIOpenConfig = None  # type: ignore
    bootstrap = None  # type: ignore

# ---------------------------------------------------------------------------
# Rutas base de proyecto
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parent                 # .../YOLO
PROJECT_ROOT = YOLO_ROOT.parent         # raíz del repositorio
CONFIGS_ROOT = YOLO_ROOT / "configs"    # YOLO/configs

# Raíz local del repositorio YOLOv5 clonado en el proyecto
YOLOV5_ROOT = YOLO_ROOT / "yolov5"
if YOLOV5_ROOT.is_dir() and str(YOLOV5_ROOT) not in sys.path:
    # Permite importar `models`, `utils`, etc. como en los scripts oficiales
    sys.path.insert(0, str(YOLOV5_ROOT))

# Dataset principal (mismo criterio que view_dataset.py)
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"

# Ruta por defecto al modelo entrenado indicada por el usuario
DEFAULT_WEIGHTS = PROJECT_ROOT / "YOLO" / "weights" / "detect" / "s" / "train" / "s_belt_defects_yolov5s_final_best.pt"


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
    """Contenedor de contexto para inferencia YOLOv5."""

    model: Any
    device: Any
    stride: int
    imgsz: int


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def _resolve_path(path: str | Path, base: Path) -> Path:
    """Resuelve una ruta relativa contra `base`, dejando rutas absolutas intactas."""

    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_class_names() -> List[str]:
    """Carga los nombres de clases desde Dataset/data.yaml.

    Se admite tanto la clave `names` como `classes`.
    """

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {DATA_YAML}")

    with DATA_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names") or data.get("classes")
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(
            f"El archivo {DATA_YAML} debe definir una lista 'names' o 'classes' con las etiquetas."
        )

    return [str(n) for n in names]


def load_test_images(image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Tuple[List[Path], Path]:
    """Carga la lista de imágenes y la ruta base de labels de Dataset/test.

    Estructura esperada:
      Dataset/
        test/
          images/
          labels/
    """

    images_dir = DATASET_ROOT / "test" / "images"
    labels_dir = DATASET_ROOT / "test" / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes de test: {images_dir}")

    image_paths: List[Path] = []
    for ext in image_exts:
        image_paths.extend(sorted(images_dir.glob(f"*{ext}")))

    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en {images_dir}")

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No se encontró el directorio de labels de test: {labels_dir}")

    return image_paths, labels_dir


def build_color_palettes(num_classes: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Genera paletas de colores separadas para GT y predicciones.

    Para asegurar coherencia visual con la leyenda, se usan colores
    fijos por tipo de caja (GT vs predicción), independientemente de
    la clase. Los nombres de clase distinguen el tipo de defecto.

    Retorna (colors_gt, colors_pred), cada una como lista de BGR.
    """

    # Rojo oscuro para GT (etiquetas reales, alta visibilidad)
    gt_color = (0, 0, 160)
    # Verde oscuro para predicciones del modelo (mejor contraste para texto blanco)
    pred_color = (0, 160, 0)

    colors_gt = [gt_color for _ in range(num_classes)]
    colors_pred = [pred_color for _ in range(num_classes)]

    return colors_gt, colors_pred


# ---------------------------------------------------------------------------
# Carga de labels y modelo
# ---------------------------------------------------------------------------


def load_gt_boxes(label_file: Path, img_shape: Tuple[int, int, int]) -> List[Box]:
    """Carga bounding boxes reales (GT) desde un archivo de labels YOLO.

    Formato esperado por línea: cls xc yc w h (normalizados).
    """

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


def load_model(weights: Path, device_str: str, imgsz: int) -> ModelContext:
    """Carga el modelo YOLOv5 local usando los módulos del repositorio.

    - Usa `select_device` y `attempt_load` como en detect/val.
    - Ajusta `imgsz` al múltiplo de stride adecuado mediante `check_img_size`.
    - Aplica el parche BN→GN (`engine.bn2gn_patch`) con política `on_error`
      si el módulo está disponible.
    """

    try:
        import torch  # noqa: F401  # se usa para tipos y contexto
        from models.experimental import attempt_load  # type: ignore
        from utils.general import check_img_size  # type: ignore
        from utils.torch_utils import select_device  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No se pudieron importar los módulos de YOLOv5 (`models`, `utils`). "
            "Verifique que el repositorio YOLO/yolov5 está presente y accesible."
        ) from e

    if not weights.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights}")

    device = select_device(device_str or "")
    model = attempt_load(str(weights), device=device)
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    imgsz_adj = check_img_size(imgsz, s=stride)

    # Aplicar parche BN→GN si el módulo está disponible
    try:  # pragma: no cover - en entornos mínimos puede no existir
        from engine.bn2gn_patch import apply_bn2gn_patch  # type: ignore
    except Exception:
        apply_bn2gn_patch = None  # type: ignore

    if apply_bn2gn_patch is not None:
        try:
            apply_bn2gn_patch(
                model=model,
                policy="on_error",          # parche bajo demanda ante errores MIOpen/BN
                max_groups=32,
                min_channels_per_group=1,
                verbose=1,
            )
        except Exception as e:  # pragma: no cover
            print(f"[test.bn2gn] Advertencia: fallo al aplicar parche BN→GN: {e}")

    model.eval()

    return ModelContext(model=model, device=device, stride=stride, imgsz=imgsz_adj)


# ---------------------------------------------------------------------------
# Inferencia por imagen (pipeline YOLOv5)
# ---------------------------------------------------------------------------


def infer_image(
    ctx: ModelContext,
    img_bgr: np.ndarray,
    conf_thres: float,
    iou_nms: float,
    max_det: int,
) -> List[Box]:
    """Ejecuta inferencia YOLOv5 sobre una imagen BGR y retorna boxes predichos.

    Pipeline alineado con detect/val:
    - letterbox → tensor → forward → non_max_suppression → scale_boxes.
    """

    try:
        import torch
        from utils.augmentations import letterbox  # type: ignore
        from utils.general import non_max_suppression, scale_boxes  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No se pudieron importar las utilidades de YOLOv5 (`utils.augmentations`, `utils.general`)."
        ) from e

    img0 = img_bgr.copy()

    # 1) Letterbox al tamaño esperado y múltiplo de stride
    img = letterbox(img0, ctx.imgsz, stride=ctx.stride, auto=True)[0]  # BGR, HWC

    # 2) BGR→RGB, HWC→CHW, contiguous
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # 3) A tensor
    im = torch.from_numpy(img).to(ctx.device)
    im = im.float() / 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)

    # 4) Forward
    with torch.no_grad():
        pred = ctx.model(im)[0]

    # 5) NMS
    preds = non_max_suppression(
        pred,
        conf_thres,
        iou_nms,
        classes=None,
        agnostic=False,
        max_det=max_det,
    )

    boxes_out: List[Box] = []

    if not preds or preds[0] is None or len(preds[0]) == 0:
        return boxes_out

    det = preds[0]

    # 6) Reescalar de resolución letterbox → resolución original
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()

    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = xyxy
        boxes_out.append(
            Box(
                cls_id=int(cls),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf),
            )
        )

    return boxes_out


# ---------------------------------------------------------------------------
# Métricas P/R/IoU por clase (por imagen)
# ---------------------------------------------------------------------------


def box_iou(a: Box, b: Box) -> float:
    """Calcula IoU entre dos boxes (formato xyxy)."""

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
    """Evalúa una imagen calculando TP/FP/FN y métricas P/R/IoU por clase.

    Estrategia por clase (independiente):
      - Predicciones ordenadas por confianza (descendente).
      - Matching greedy con IoU >= iou_match.
      - Cada GT participa en a lo más un match.
    """

    stats: Dict[int, ClassStats] = {c: ClassStats() for c in range(num_classes)}

    # Contadores base
    for box in gt_boxes:
        if 0 <= box.cls_id < num_classes:
            stats[box.cls_id].n_gt += 1

    for box in pred_boxes:
        if 0 <= box.cls_id < num_classes:
            stats[box.cls_id].n_pred += 1

    # Procesamiento por clase
    for cls in range(num_classes):
        gt_c = [b for b in gt_boxes if b.cls_id == cls]
        pred_c = sorted(
            [b for b in pred_boxes if b.cls_id == cls],
            key=lambda b: b.conf,
            reverse=True,
        )

        used_gt = [False] * len(gt_c)

        for pred in pred_c:
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gt_c):
                if used_gt[i]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_match and best_idx >= 0:
                # Verdadero positivo
                used_gt[best_idx] = True
                stats[cls].tp += 1
                stats[cls].matches += 1
                stats[cls].iou_sum += best_iou
            else:
                # Falso positivo
                stats[cls].fp += 1

        # Cualquier GT no emparejado es un falso negativo
        fn_count = sum(1 for u in used_gt if not u)
        stats[cls].fn += fn_count

    # Cálculo de métricas derivadas (por clase y globales)
    global_metrics: Dict[str, float] = {
        "P_macro": 0.0,
        "R_macro": 0.0,
        "IoU_macro": 0.0,
    }

    p_list: List[float] = []
    r_list: List[float] = []
    iou_list: List[float] = []

    for cls in range(num_classes):
        s = stats[cls]
        denom_p = s.tp + s.fp
        denom_r = s.tp + s.fn

        p = float(s.tp / denom_p) if denom_p > 0 else 0.0
        r = float(s.tp / denom_r) if denom_r > 0 else 0.0
        iou_mean = float(s.iou_sum / s.matches) if s.matches > 0 else 0.0

        # Guardamos estos valores aprovechando el dataclass
        s.precision = p  # type: ignore[attr-defined]
        s.recall = r  # type: ignore[attr-defined]
        s.iou_mean = iou_mean  # type: ignore[attr-defined]

        if s.n_gt > 0 or s.n_pred > 0:
            p_list.append(p)
            r_list.append(r)
            iou_list.append(iou_mean)

    if p_list:
        global_metrics["P_macro"] = float(np.mean(p_list))
    if r_list:
        global_metrics["R_macro"] = float(np.mean(r_list))
    if iou_list:
        global_metrics["IoU_macro"] = float(np.mean(iou_list))

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
    """Dibuja una lista de boxes sobre la imagen en BGR."""

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

        # Intentar poner arriba del bbox
        tx1 = x1
        ty1 = y1 - th - bl - 3
        tx2 = x1 + tw + 4
        ty2 = y1

        if ty1 < 0:  # si se sale, mover abajo
            ty1 = y2 + 3
            ty2 = y2 + th + bl + 3

        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            img,
            label,
            (tx1 + 2, ty2 - 4),
            font,
            font_scale,
            (255, 255, 255),
            t,
            cv2.LINE_AA,
        )

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
    """Dibuja panel lateral con información de imagen, métricas y ayuda de teclas."""

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

    # Cabecera
    put(f"Split: {split}", (255, 255, 0))
    put(f"Imagen: {idx + 1}/{num_images}", (255, 255, 0))
    put(f"Modelo: {os.path.basename(model_name)}", (200, 255, 255))
    put("", (255, 255, 255))

    put("Metricas por imagen (macro):", (0, 255, 255))
    put(f"P:   {global_metrics.get('P_macro', 0.0):.3f}")
    put(f"R:   {global_metrics.get('R_macro', 0.0):.3f}")
    put(f"IoU: {global_metrics.get('IoU_macro', 0.0):.3f}")
    put("", (255, 255, 255))

    # Leyenda de colores
    put("Leyenda bboxes:", (0, 255, 255))

    # GT
    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_gt[0] if colors_gt else (0, 0, 160), -1)
    cv2.putText(canvas, "GT (etiqueta real)", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA)
    y += lh

    # Pred
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

    # Métricas por clase (si hay espacio)
    put("Metricas por clase:", (0, 255, 255))

    # Usar una tipografía ligeramente más pequeña y líneas compactas
    fs_cls = 0.45
    lh_cls = 18

    def put_cls(line: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        nonlocal y
        cv2.putText(canvas, line, (10, y), font, fs_cls, color, t, cv2.LINE_AA)
        y += lh_cls

    for cls_id, s in stats.items():
        if s.n_gt == 0 and s.n_pred == 0:
            continue
        # Reservar como mínimo dos líneas por clase
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
    """Ejecuta el visor interactivo sobre Dataset/test."""

    class_names = load_class_names()
    num_classes = len(class_names)
    colors_gt, colors_pred = build_color_palettes(num_classes)

    weights_path = _resolve_path(args.weights, PROJECT_ROOT)
    ctx = load_model(weights_path, args.device or "", args.imgsz)

    image_paths, labels_dir = load_test_images()

    idx = 0
    num_images = len(image_paths)
    show_pred = True
    split = "test"

    window_name = "YOLOv5 Test Viewer"

    while True:
        if idx < 0:
            idx = 0
        if idx >= num_images:
            idx = num_images - 1

        img_path = image_paths[idx]
        filename = img_path.stem
        label_file = labels_dir / f"{filename}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer la imagen: {img_path}")
            idx += 1
            if idx >= num_images:
                break
            continue

        gt_boxes = load_gt_boxes(label_file, img.shape)
        pred_boxes = infer_image(
            ctx=ctx,
            img_bgr=img,
            conf_thres=args.conf_thres,
            iou_nms=args.iou_nms,
            max_det=args.max_det,
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
            img_vis = draw_boxes(
                img_vis,
                pred_boxes,
                class_names,
                colors_pred,
                thickness=1,
                draw_conf=True,
            )

        legend = draw_legend(
            split=split,
            idx=idx,
            num_images=num_images,
            model_name=str(weights_path),
            class_names=class_names,
            stats=stats,
            global_metrics=global_metrics,
            show_pred=show_pred,
            colors_gt=colors_gt,
            colors_pred=colors_pred,
            height=img_vis.shape[0],
        )

        if legend.shape[0] != img_vis.shape[0]:
            legend = cv2.resize(legend, (legend.shape[1], img_vis.shape[0]))

        combined = np.hstack((img_vis, legend))
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord("d"), 83):  # siguiente (tecla 'd' o flecha derecha)
            idx = min(num_images - 1, idx + 1)
        elif key in (ord("a"), 81):  # anterior (tecla 'a' o flecha izquierda)
            idx = max(0, idx - 1)
        elif key in (ord("h"), ord("p")):
            show_pred = not show_pred

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comando para el test YOLOv5."""

    parser = argparse.ArgumentParser(
        prog="YOLOv5.test",
        description=(
            "Visor interactivo para testear un modelo YOLOv5 entrenado sobre la "
            "partición Dataset/test, mostrando boxes reales vs. predichos y "
            "métricas locales por imagen."
        ),
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=(
            "Ruta al archivo de pesos .pt del modelo YOLOv5 a testear. "
            "Por defecto, el checkpoint final entrenado para fallas en correas."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo para inferencia: '', '0', '0,1', 'cpu', etc.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen para inferencia (lado mayor).",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Umbral de confianza mínimo para visualizar predicciones.",
    )

    parser.add_argument(
        "--iou-nms",
        type=float,
        default=0.6,
        help="IoU para NMS interno del modelo (no confundir con IoU de matching métrico).",
    )

    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="IoU mínimo para considerar una predicción como TP frente a un GT (métricas P/R/IoU).",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Máximo de detecciones por imagen.",
    )

    parser.add_argument(
        "--no-bootstrap-miopen",
        action="store_true",
        help="Desactiva el bootstrap MIOpen previo a la carga de Torch/YOLOv5, si aplica.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Punto de entrada del script de test YOLOv5."""

    args = parse_args(argv)

    # Bootstrap MIOpen (si el módulo está disponible) antes de cualquier
    # inicialización pesada de Torch/YOLOv5.
    if not args.no_bootstrap_miopen and MIOpenConfig is not None and bootstrap is not None:
        cfg = MIOpenConfig(
            find_mode="FAST",
            user_db_path=None,
            disable_cache=True,
            log_level=0,
            extra_env={},
            strict_before_torch=True,
            verbose=1,
        )
        bootstrap(cfg)

    # Instalación opcional de filtros de warnings del proyecto (import perezoso tras bootstrap)
    try:
        from engine.warnings import install_global_warning_filters  # type: ignore
    except Exception:
        install_global_warning_filters = None  # type: ignore

    if install_global_warning_filters is not None:
        install_global_warning_filters(force=False)

    run_viewer(args)


if __name__ == "__main__":  # pragma: no cover
    main()
