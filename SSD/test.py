# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/test.py
# Descripción: Punto de entrada principal para pruebas de inferencia
#              sobre modelos SSD entrenados (detección en imágenes
#              de la partición Dataset/test con comparación
#              predicción vs. etiqueta real).
# ==============================================================

from __future__ import annotations

import argparse
import os
import sys
import types
import yaml
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Rutas base de proyecto
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parent  # .../SSD
PROJECT_ROOT = SSD_ROOT.parent  # raíz del repositorio
CONFIGS_ROOT = SSD_ROOT / "configs"  # SSD/configs

# Rutas de módulos clave
SSD_MODEL_PATH = SSD_ROOT / "ssd" / "ssd.py"
VALIDATOR_PATH = SSD_ROOT / "engine" / "Validator.py"

# Dataset principal (mismo criterio que train/valid.py)
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATA_YAML = CONFIGS_ROOT / "dataset.yaml"  # Usamos la config del proyecto


# ---------------------------------------------------------------------------
# Utilidad de carga dinámica de módulos (copiada de train/valid.py)
# ---------------------------------------------------------------------------

def _load_module_from(path: Path, name: str):
    """Carga dinámica de un módulo Python desde un path arbitrario."""
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo requerido en: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para módulo: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    module_dir = str(path.parent)
    sys.path.insert(0, module_dir)

    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except Exception:
        if name in sys.modules:
            del sys.modules[name]
        raise
    finally:
        if module_dir in sys.path:
            sys.path.remove(module_dir)

    return module


# ---------------------------------------------------------------------------
# Mocking Legacy (Igual que en train/valid.py)
# ---------------------------------------------------------------------------

def _mock_legacy_coco_dependency():
    """Neutraliza dependencia de COCO para evitar errores de importación."""

    class DummyDataset:
        def __init__(self, *args, **kwargs): pass

    class DummyTransform:
        def __init__(self, *args, **kwargs): pass

    mock_coco = types.ModuleType("data.coco")
    mock_coco.COCODetection = DummyDataset
    mock_coco.COCOAnnotationTransform = DummyTransform
    mock_coco.COCO_CLASSES = []
    mock_coco.COCO_ROOT = ""
    mock_coco.get_label_map = lambda x: {}
    sys.modules["data.coco"] = mock_coco
    sys.modules["ssd.data.coco"] = mock_coco


# ---------------------------------------------------------------------------
# Estructuras de datos (Adaptadas de YOLO/test.py)
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
    """Contenedor de contexto para inferencia SSD."""
    model: nn.Module
    device: torch.device
    img_dim: int
    validator_cls: Any  # Referencia a ValidatorSSD


# ---------------------------------------------------------------------------
# Utilidades generales (Adaptadas de YOLO/test.py)
# ---------------------------------------------------------------------------

def _resolve_path(path: str | Path, base: Path) -> Path:
    """Resuelve una ruta relativa contra `base`, dejando rutas absolutas intactas."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_test_images(image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Tuple[List[Path], Path]:
    """Carga la lista de imágenes y la ruta base de labels de Dataset/test."""
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


def load_gt_boxes(label_file: Path, img_shape: Tuple[int, int, int]) -> List[Box]:
    """Carga bounding boxes reales (GT) desde un archivo de labels YOLO (normalizados)."""
    h, w = img_shape[:2]
    if not label_file.is_file():
        return []
    boxes: List[Box] = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
            cls_id_int = int(cls_id)
            x_c *= w;
            y_c *= h;
            bw *= w;
            bh *= h
            x1 = x_c - bw / 2.0;
            y1 = y_c - bh / 2.0
            x2 = x_c + bw / 2.0;
            y2 = y_c + bh / 2.0
            boxes.append(Box(cls_id=cls_id_int, x1=x1, y1=y1, x2=x2, y2=y2, conf=1.0))
    return boxes


def build_color_palettes(num_classes: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Genera paletas de colores separadas para GT y predicciones (BGR)."""
    gt_color = (0, 0, 160)  # Rojo oscuro para GT
    pred_color = (0, 160, 0)  # Verde oscuro para Predicciones
    colors_gt = [gt_color for _ in range(num_classes)]
    colors_pred = [pred_color for _ in range(num_classes)]
    return colors_gt, colors_pred


# ---------------------------------------------------------------------------
# Métricas P/R/IoU por clase (por imagen) - Copiadas de YOLO/test.py
# ---------------------------------------------------------------------------

def box_iou(a: Box, b: Box) -> float:
    """Calcula IoU entre dos boxes (formato xyxy)."""
    inter_x1 = max(a.x1, b.x1);
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2);
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1);
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0.0 else 0.0


def evaluate_image(
        gt_boxes: List[Box],
        pred_boxes: List[Box],
        num_classes: int,
        iou_match: float,
) -> Tuple[Dict[int, ClassStats], Dict[str, float]]:
    """Evalúa una imagen calculando TP/FP/FN y métricas P/R/IoU por clase."""
    stats: Dict[int, ClassStats] = {c: ClassStats() for c in range(num_classes)}
    for box in gt_boxes:
        if 0 <= box.cls_id < num_classes: stats[box.cls_id].n_gt += 1
    for box in pred_boxes:
        if 0 <= box.cls_id < num_classes: stats[box.cls_id].n_pred += 1

    for cls in range(num_classes):
        gt_c = [b for b in gt_boxes if b.cls_id == cls]
        pred_c = sorted([b for b in pred_boxes if b.cls_id == cls], key=lambda b: b.conf, reverse=True)
        used_gt = [False] * len(gt_c)

        for pred in pred_c:
            best_iou = 0.0;
            best_idx = -1
            for i, gt in enumerate(gt_c):
                if used_gt[i]: continue
                iou = box_iou(pred, gt)
                if iou > best_iou: best_iou = iou; best_idx = i

            if best_iou >= iou_match and best_idx >= 0:
                used_gt[best_idx] = True
                stats[cls].tp += 1
                stats[cls].matches += 1
                stats[cls].iou_sum += best_iou
            else:
                stats[cls].fp += 1

        stats[cls].fn += sum(1 for u in used_gt if not u)

    global_metrics: Dict[str, float] = {"P_macro": 0.0, "R_macro": 0.0, "IoU_macro": 0.0}
    p_list: List[float] = [];
    r_list: List[float] = [];
    iou_list: List[float] = []

    for cls in range(num_classes):
        s = stats[cls]
        denom_p = s.tp + s.fp;
        denom_r = s.tp + s.fn
        p = float(s.tp / denom_p) if denom_p > 0 else 0.0
        r = float(s.tp / denom_r) if denom_r > 0 else 0.0
        iou_mean = float(s.iou_sum / s.matches) if s.matches > 0 else 0.0

        setattr(s, "precision", p)
        setattr(s, "recall", r)
        setattr(s, "iou_mean", iou_mean)

        if s.n_gt > 0 or s.n_pred > 0:
            p_list.append(p);
            r_list.append(r);
            iou_list.append(iou_mean)

    if p_list: global_metrics["P_macro"] = float(np.mean(p_list))
    if r_list: global_metrics["R_macro"] = float(np.mean(r_list))
    if iou_list: global_metrics["IoU_macro"] = float(np.mean(iou_list))

    return stats, global_metrics


# ---------------------------------------------------------------------------
# Dibujo de boxes y leyenda - Copiadas de YOLO/test.py
# ---------------------------------------------------------------------------

def draw_boxes(
        img: np.ndarray, boxes: List[Box], class_names: List[str], colors: List[Tuple[int, int, int]],
        thickness: int = 2, draw_conf: bool = False,
) -> np.ndarray:
    """Dibuja una lista de boxes sobre la imagen en BGR."""
    h, w = img.shape[:2];
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        if not (0 <= box.cls_id < len(colors)): continue
        color = colors[box.cls_id]
        x1 = int(max(0, min(w - 1, box.x1)));
        y1 = int(max(0, min(h - 1, box.y1)))
        x2 = int(max(0, min(w - 1, box.x2)));
        y2 = int(max(0, min(h - 1, box.y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = class_names[box.cls_id]
        if draw_conf: label = f"{label} {box.conf:.2f}"
        font_scale = 0.5;
        t = 1
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, t)
        tx1 = x1;
        ty1 = y1 - th - bl - 3;
        tx2 = x1 + tw + 4;
        ty2 = y1
        if ty1 < 0: ty1 = y2 + 3; ty2 = y2 + th + bl + 3
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(img, label, (tx1 + 2, ty2 - 4), font, font_scale, (255, 255, 255), t, cv2.LINE_AA)
    return img


def draw_legend(
        split: str, idx: int, num_images: int, model_name: str, class_names: List[str],
        stats: Dict[int, ClassStats], global_metrics: Dict[str, float], show_pred: bool,
        colors_gt: List[Tuple[int, int, int]], colors_pred: List[Tuple[int, int, int]], height: int,
) -> np.ndarray:
    """Dibuja panel lateral con información de imagen, métricas y ayuda de teclas."""
    legend_width = 460;
    legend_height = max(height, 600)
    canvas = np.zeros((legend_height, legend_width, 3), dtype=np.uint8);
    canvas[:] = (50, 60, 90)
    font = cv2.FONT_HERSHEY_SIMPLEX;
    fs = 0.55;
    t = 1;
    y = 30;
    lh = 22

    def put(line: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        nonlocal y;
        cv2.putText(canvas, line, (10, y), font, fs, color, t, cv2.LINE_AA);
        y += lh

    put(f"Split: {split}", (255, 255, 0));
    put(f"Imagen: {idx + 1}/{num_images}", (255, 255, 0))
    put(f"Modelo: {os.path.basename(model_name)}", (200, 255, 255));
    put("")
    put("Metricas por imagen (macro):", (0, 255, 255))
    put(f"P:   {global_metrics.get('P_macro', 0.0):.3f}");
    put(f"R:   {global_metrics.get('R_macro', 0.0):.3f}")
    put(f"IoU: {global_metrics.get('IoU_macro', 0.0):.3f}");
    put("")
    put("Leyenda bboxes:", (0, 255, 255))
    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_gt[0] if colors_gt else (0, 0, 160), -1)
    cv2.putText(canvas, "GT (etiqueta real)", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA);
    y += lh
    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_pred[0] if colors_pred else (0, 160, 0), -1)
    cv2.putText(canvas, "Prediccion modelo", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA);
    y += lh
    put("");
    put("Comandos:", (0, 255, 255))
    put("<- / 'a': imagen anterior");
    put("-> / 'd': imagen siguiente")
    put("'h': mostrar/ocultar pred.");
    put("ESC: salir");
    put("")
    put(f"Predicciones visibles: {'Si' if show_pred else 'No'}");
    put("")
    put("Metricas por clase:", (0, 255, 255));
    fs_cls = 0.45;
    lh_cls = 18

    def put_cls(line: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        nonlocal y;
        cv2.putText(canvas, line, (10, y), font, fs_cls, color, t, cv2.LINE_AA);
        y += lh_cls

    for cls_id, s in stats.items():
        if s.n_gt == 0 and s.n_pred == 0: continue
        if y + 2 * lh_cls > legend_height - 10: put_cls("...", (200, 200, 200)); break
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        precision = getattr(s, "precision", 0.0);
        recall = getattr(s, "recall", 0.0)
        iou_mean = getattr(s, "iou_mean", 0.0)
        put_cls(f"[{cls_id}] {name}", (255, 255, 0))
        put_cls(
            f"GT:{s.n_gt} Pred:{s.n_pred} TP:{s.tp} FP:{s.fp} FN:{s.fn} "
            f"P:{precision:.2f} R:{recall:.2f} IoU:{iou_mean:.2f}", (220, 220, 220),
        )
    return canvas


# ---------------------------------------------------------------------------
# Lógica de Inferencia Específica de SSD
# ---------------------------------------------------------------------------

def _get_ssd_mean() -> np.ndarray:
    """Media de normalización por defecto para SSD (BGR)."""
    return np.array([104, 117, 123], dtype=np.float32)


def _preprocess_image_ssd(img_bgr: np.ndarray, img_dim: int) -> torch.Tensor:
    """Preprocesa la imagen BGR para la entrada del modelo SSD."""
    # 1. Redimensionar
    img_resized = cv2.resize(img_bgr, (img_dim, img_dim)).astype(np.float32)

    # 2. Restar media (BGR)
    img_norm = img_resized - _get_ssd_mean()

    # 3. HWC -> CHW
    img_chw = img_norm.transpose((2, 0, 1))

    # 4. A tensor y unsqueeze (batch size 1)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor


def infer_image_ssd(
        ctx: ModelContext,
        img_bgr: np.ndarray,
        conf_thres: float,
) -> List[Box]:
    """Ejecuta inferencia SSD y retorna boxes predichos en coordenadas originales."""

    # 1. Preprocesamiento
    img_tensor = _preprocess_image_ssd(img_bgr, ctx.img_dim).to(ctx.device)

    # 2. Forward
    with torch.no_grad():
        output = ctx.model(img_tensor)

    # 3. Post-procesamiento (usando la utilidad de ValidatorSSD)
    # La salida de SSD en modo 'test' es un tensor [1, num_classes, top_k, 5]
    # El tamaño de la imagen de entrada al modelo es (img_dim, img_dim)
    img_size_model = (ctx.img_dim, ctx.img_dim)

    # Usamos el método estático del Validator para obtener el tensor [N, 6]
    pred_tensor_norm = ctx.validator_cls._get_detections_from_output(
        output=output,
        img_size=img_size_model,
        conf_thresh=conf_thres,
        device=ctx.device
    )

    if pred_tensor_norm.size(0) == 0:
        return []

    # 4. Reescalar a coordenadas de imagen original
    h_orig, w_orig = img_bgr.shape[:2]
    scale_x = w_orig / ctx.img_dim
    scale_y = h_orig / ctx.img_dim

    # pred_tensor_norm es [x1, y1, x2, y2, conf, cls] en coordenadas normalizadas a img_dim
    boxes_scaled = pred_tensor_norm.clone()
    boxes_scaled[:, 0] *= scale_x
    boxes_scaled[:, 2] *= scale_x
    boxes_scaled[:, 1] *= scale_y
    boxes_scaled[:, 3] *= scale_y

    boxes_out: List[Box] = []
    for *xyxy, conf, cls in boxes_scaled.cpu().numpy():
        boxes_out.append(
            Box(
                cls_id=int(cls),
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                conf=float(conf),
            )
        )

    return boxes_out


def load_model_ssd(weights: Path, device_str: str, img_dim: int, num_classes: int) -> ModelContext:
    """Carga el modelo SSD y sus pesos."""

    if not weights.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights}")

    # Carga dinámica de módulos
    ssd_mod = _load_module_from(SSD_MODEL_PATH, "ssd_model")
    build_ssd = ssd_mod.build_ssd
    val_mod = _load_module_from(VALIDATOR_PATH, "ssd_validator")
    ValidatorSSD = val_mod.ValidatorSSD

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[SSD/test] Construyendo SSD{img_dim} (clases={num_classes})...")
    # Importante: construir en fase 'test' para que la capa Detect esté activa
    model = build_ssd("test", img_dim, num_classes)

    # Cargar estado
    state_dict = torch.load(weights, map_location=device, weights_only=False)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Limpiar prefijo 'module.' si existe
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return ModelContext(model=model, device=device, img_dim=img_dim, validator_cls=ValidatorSSD)


# ---------------------------------------------------------------------------
# Bucle principal de visualización
# ---------------------------------------------------------------------------

def run_viewer(args: argparse.Namespace) -> None:
    """Ejecuta el visor interactivo sobre Dataset/test."""

    # 1. Cargar nombres de clases
    try:
        # Importación estándar para que los workers de DataLoader puedan resolver el módulo
        if str(SSD_ROOT) not in sys.path: sys.path.append(str(SSD_ROOT))
        from utility import data_loader as _data_loader
        ds_cfg_dict = _data_loader.load_dataset_config(DATA_YAML)
        class_names = list(ds_cfg_dict["names"].values())
        num_classes = len(class_names)
    except Exception as e:
        raise RuntimeError(f"Fallo al cargar la configuración del dataset: {e}")

    colors_gt, colors_pred = build_color_palettes(num_classes)

    # 2. Cargar modelo
    weights_path = _resolve_path(args.weights, PROJECT_ROOT)
    ctx = load_model_ssd(weights_path, args.device or "", args.img_dim, num_classes + 1)  # +1 por background

    # 3. Cargar imágenes de test
    image_paths, labels_dir = load_test_images()

    idx = 0
    num_images = len(image_paths)
    show_pred = True
    split = "test"

    window_name = "SSD Test Viewer"

    while True:
        if idx < 0: idx = 0
        if idx >= num_images: idx = num_images - 1

        img_path = image_paths[idx]
        filename = img_path.stem
        label_file = labels_dir / f"{filename}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer la imagen: {img_path}")
            idx += 1
            if idx >= num_images: break
            continue

        gt_boxes = load_gt_boxes(label_file, img.shape)
        pred_boxes = infer_image_ssd(
            ctx=ctx,
            img_bgr=img,
            conf_thres=args.conf_thres,
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
                img_vis, pred_boxes, class_names, colors_pred, thickness=1, draw_conf=True,
            )

        legend = draw_legend(
            split=split, idx=idx, num_images=num_images, model_name=str(weights_path),
            class_names=class_names, stats=stats, global_metrics=global_metrics,
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
    """Define y parsea los argumentos de línea de comando para el test SSD."""

    parser = argparse.ArgumentParser(
        prog="SSD.test",
        description=(
            "Visor interactivo para testear un modelo SSD entrenado sobre la "
            "partición Dataset/test, mostrando boxes reales vs. predichos y "
            "métricas locales por imagen."
        ),
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=str(SSD_ROOT / "weights" / "detect" / "ssd300" / "train" / "best.pth"),
        help="Ruta al archivo de pesos .pth del modelo SSD a testear.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo para inferencia: '', '0', '0,1', 'cpu', etc.",
    )

    parser.add_argument(
        "--img-dim",
        type=int,
        default=300,
        help="Dimensión de imagen para inferencia (lado mayor).",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Umbral de confianza mínimo para visualizar predicciones.",
    )

    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="IoU mínimo para considerar una predicción como TP frente a un GT (métricas P/R/IoU).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Punto de entrada del script de test SSD."""

    args = parse_args(argv)
    _mock_legacy_coco_dependency()

    # Nota: Se omite el bootstrap MIOpen aquí, ya que el Trainer.py lo maneja
    # y no es estrictamente necesario para la inferencia si el modelo ya está cargado.

    run_viewer(args)


if __name__ == "__main__":
    main()