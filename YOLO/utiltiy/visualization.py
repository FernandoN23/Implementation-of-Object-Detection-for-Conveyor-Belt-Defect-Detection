# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: visualization.py
# Utilidades de visualización con TensorBoard para YOLOv11.
# - Registro de pérdidas por época (series temporales)
# - Registro de métricas agregadas (IoU, mAP, P/R, TP/FP/FN, etc.)
# - Registro de imágenes de referencia (train/val) como grids
# - Overlay de GT ("real") y Pred ("pred") sobre imágenes pivote
# - Registro de overlay.png generado por test_metrics.py
# - Demo que lee métricas de test_metrics y crea una pérdida constante (0.1)
#==============================================================
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# --- Dependencias (opcional) ---
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception as e:  # pragma: no cover
    SummaryWriter = None  # type: ignore
    _TB_IMPORT_ERROR = e
else:
    _TB_IMPORT_ERROR = None

# Flag para evitar spam del banner de TensorBoard en cada época
_TB_BANNER_SHOWN: bool = False

# Carga/transformaciones de imágenes
try:
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import torchvision.transforms.functional as TF
    from torchvision.utils import make_grid
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    TF = None  # type: ignore
    make_grid = None  # type: ignore
    _VISION_IMPORT_ERROR = e
else:
    _VISION_IMPORT_ERROR = None


# =============================
# Descubrimiento de la raíz del proyecto
# =============================
FILE = Path(__file__).resolve()
PROJ = FILE.parents[1]  # .../YOLOv11/

# Ruta base del dataset (por defecto según memoria del proyecto)
DEFAULT_DATASET_BASE = Path(
    r"C:/Users/memorista/Desktop/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection/Dataset"
)


def _ensure_tensorboard_available() -> None:
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard no disponible. Instale con 'pip install tensorboard'\n"
            f"Detalle de importación: {_TB_IMPORT_ERROR}"
        )


def _ensure_vision_available() -> None:
    if any(x is None for x in (torch, Image, TF, make_grid, ImageDraw)):
        raise RuntimeError(
            "Visión no disponible (torch/torchvision/PIL).\n"
            "Instale PyTorch + torchvision y Pillow.\n"
            f"Detalle de importación: {_VISION_IMPORT_ERROR}"
        )


@dataclass
class TBConfig:
    variant: str
    phase: str  # 'train' | 'val' | 'test'
    run_name: str
    runs_root: Path = PROJ / "runs"

    @property
    def logdir(self) -> Path:
        return self.runs_root / self.variant / self.phase / self.run_name


class TBVisualization:
    """Wrapper minimalista sobre SummaryWriter con nombres consistentes."""

    def __init__(self, cfg: TBConfig, flush_secs: int = 10) -> None:
        _ensure_tensorboard_available()
        self.cfg = cfg
        self.cfg.logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.cfg.logdir), flush_secs=flush_secs)

        # Banner informativo: sólo se imprime una vez por proceso para evitar
        # ruido en consola cuando se loguea por época (val_int, sesiones, etc.).
        global _TB_BANNER_SHOWN
        if not _TB_BANNER_SHOWN:
            print(
                f"[TB] Activo en: {self.cfg.logdir}"
                f"    Inicie con: tensorboard --logdir {self.cfg.runs_root}"
            )
            _TB_BANNER_SHOWN = True

    # ----------- API de scalars -----------
    def log_train_loss_epoch(self, scalars: Dict[str, float], epoch: int) -> None:
        """Registra pérdidas por época (p.ej., 'loss/total', 'loss/box', 'loss/cls', 'loss/dfl')."""
        for k, v in scalars.items():
            self.writer.add_scalar(k, float(v), epoch)

    def log_metrics_epoch(self, scalars: Dict[str, float], epoch: int, phase: Optional[str] = None) -> None:
        """Registra métricas agregadas (precision/recall/mAP/IoU y stats)."""
        for k, v in scalars.items():
            tag = k if (phase is None) else f"{phase}/{k}"
            self.writer.add_scalar(tag, float(v), epoch)

    # ----------- API de imágenes -----------
    def log_image(self, tag: str, img_tensor, epoch: int) -> None:
        """Registra una imagen CHW en [0,1]."""
        self.writer.add_image(tag, img_tensor, epoch)

    def log_image_grid(self, tag: str, img_tensors: Sequence, epoch: int, nrow: int = 4) -> None:
        _ensure_vision_available()
        grid = make_grid(torch.stack(img_tensors, dim=0), nrow=nrow, padding=2)
        self.writer.add_image(tag, grid, epoch)

    # ----------- Opcional: grafo del modelo -----------
    def add_graph(self, model, imgsz: int = 640, device: Optional[str] = None, dtype=None) -> None:
        try:
            model.eval()
            im = torch.zeros((1, 3, imgsz, imgsz), device=device or (next(model.parameters()).device), dtype=dtype)
            self.writer.add_graph(torch.jit.trace(model, im, strict=False), [])
            print("[TB] Grafo del modelo añadido.")
        except Exception as e:  # pragma: no cover
            print(f"[TB] Aviso: no fue posible trazar el grafo ({e}).")

    def close(self) -> None:
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass


# =============================
# Curvas de pérdida (Training Loss) en TensorBoard
# =============================

def log_train_loss_curve_to_tb(
    variant: str,
    run_name: str,
    curve: Dict[str, Any],
    *,
    phase: str = "train",
    scalar_tag: str = "loss/train",
    log_image: bool = False,
    image_tag: str = "images/train_loss_curve",
    image_epoch: int = 0,
) -> Path:
    """Registra en TensorBoard la curva de pérdida de entrenamiento.

    Parámetros
    ----------
    variant:
        Variante del modelo (n/s/m/l/xl), coherente con la estructura de
        ``runs/<variant>/<phase>/<run_name>``.
    run_name:
        Nombre del run (habitualmente, el mismo ``cfg.name`` del Trainer).
    curve:
        Diccionario devuelto por ``metrics.build_train_loss_curve`` con las
        claves ``"epochs"`` (lista de int), ``"losses"`` (lista de float) y
        opcionalmente ``"path"`` (ruta del PNG generado).
    phase:
        Fase lógica para TensorBoard (por defecto ``"train"``).
    scalar_tag:
        Nombre base del scalar donde se registrará la pérdida. Por defecto
        ``"loss/train"`` (aparecerá como tal en TB).
    log_image:
        Si es ``True`` y ``curve["path"]`` apunta a un PNG existente, también
        se registrará la imagen en TB bajo ``image_tag``.
    image_tag:
        Tag de la imagen en TB (por defecto ``"images/train_loss_curve"``).
    image_epoch:
        Paso/época con el que se asociará la imagen en TB.
    """

    _ensure_tensorboard_available()
    epochs = list(curve.get("epochs") or [])
    losses = list(curve.get("losses") or [])

    tb = TBVisualization(TBConfig(variant=variant, phase=phase, run_name=run_name))

    # Serie temporal de scalars (un punto por época)
    if epochs and losses and len(epochs) == len(losses):
        for ep, loss in zip(epochs, losses):
            try:
                tb.log_train_loss_epoch({scalar_tag: float(loss)}, int(ep))
            except Exception:
                # No romper la sesión por un dato atípico
                continue

    # Imagen opcional de la curva (PNG generado por metrics.build_train_loss_curve)
    if log_image:
        try:
            path_val = curve.get("path")
            if path_val is not None:
                p = Path(path_val)
                if p.exists():
                    img_t = _load_image_as_tensor(p, size=(640, 640))
                    tb.log_image(image_tag, img_t, int(image_epoch))
        except Exception:
            # El fallo en la imagen no debe interrumpir el resto de logging
            pass

    tb.close()
    return tb.cfg.logdir


# Helper específico para el slot canónico de entrenamiento: train/final

def log_train_loss_curve_to_tb_final(
    variant: str,
    curve: Dict[str, Any],
    *,
    scalar_tag: str = "loss/train",
    log_image: bool = True,
    image_tag: str = "images/train_loss_curve",
    image_epoch: int = 0,
) -> Path:
    """Convenience wrapper para registrar la curva de pérdida en train/final.

    Este helper fuerza el uso del slot estable ``runs/<variant>/train/final``
    para la fase de entrenamiento, de modo que no se creen subcarpetas con
    timestamps bajo ``train/``. Está pensado para ser usado por el Trainer al
    final del entrenamiento, reescribiendo siempre sobre el mismo run lógico
    (limpiado previamente por ExperimentLogger si corresponde).
    """

    return log_train_loss_curve_to_tb(
        variant=variant,
        run_name="final",
        curve=curve,
        phase="train",
        scalar_tag=scalar_tag,
        log_image=log_image,
        image_tag=image_tag,
        image_epoch=image_epoch,
    )


# =============================
# Utilidades de imágenes (pivotes, etiquetas y overlay)
# =============================

# Pivotes entregados por el usuario (NO considerar 'healthy')
TRAIN_PIVOT_IMAGES = [
    "0864.jpg",  # Hole
    "0451.jpg",  # Impact Damage
    "0297.jpg",  # Puncture
    "0128.jpg",  # Tear
    "0393.jpg",  # Wear
]

VALID_PIVOT_IMAGES = [
    "0039.jpg",  # Hole
    "0065.jpg",  # Impact Damage
    "0085.jpg",  # Puncture
    "0129.jpg",  # Tear
    "0113.jpg",  # Wear
]


def _resolve_dataset_base(dataset_base: Optional[Path]) -> Path:
    return Path(dataset_base) if dataset_base is not None else DEFAULT_DATASET_BASE


def _img_paths_from_pivots(dataset_base: Path, split: str, names: Sequence[str]) -> List[Path]:
    base = dataset_base / split / "images"
    paths = [base / n for n in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("No se encontraron las imágenes pivote:\n" + "\n".join(missing))
    return paths


def _labels_path_from_image(img_path: Path) -> Path:
    # Sustituye /images -> /labels y .jpg|.png -> .txt
    lbl_dir = img_path.parent.parent / "labels"
    stem = img_path.stem
    return lbl_dir / f"{stem}.txt"


def _read_yolo_labels(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        boxes.append((c, cx, cy, w, h))
    return boxes


def _xywhn_to_xyxy_pix(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _draw_label(draw: ImageDraw.ImageDraw, x1: int, y1: int, text: str, color: Tuple[int, int, int]) -> None:
    """Dibuja una pequeña caja de texto sólida con el label encima del bbox."""
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # Pillow siempre tiene una fuente por defecto, pero por si acaso.

    pad = 2
    if hasattr(draw, "textbbox"):
        # type: ignore[attr-defined]
        bbox = draw.textbbox((x1 + pad, y1 - 100), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    else:
        # Aproximación conservadora
        tw = 6 * len(text)
        th = 12

    rect = [x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1]
    draw.rectangle(rect, fill=color)
    draw.text((x1 + pad, rect[1] + pad), text, fill=(0, 0, 0), font=font)


def _draw_boxes(
    img: Image.Image,
    boxes_xywhn: List[Tuple[int, float, float, float, float]],
    color: Tuple[int, int, int],
    label_text: str,
    W: int,
    H: int,
    thickness: int = 3,
) -> None:
    draw = ImageDraw.Draw(img)
    for (_cls_id, cx, cy, w, h) in boxes_xywhn:
        x1, y1, x2, y2 = _xywhn_to_xyxy_pix(cx, cy, w, h, W, H)
        # Rectángulo
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        _draw_label(draw, x1, y1, label_text, color)


def _load_image_as_tensor(path: Path, size: Tuple[int, int] = (640, 640)):
    _ensure_vision_available()
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    t = TF.to_tensor(img)  # [0,1] CHW
    return t


def _load_image_as_pil(path: Path, size: Tuple[int, int] = (640, 640)) -> Image.Image:
    _ensure_vision_available()
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img


def make_ref_grid(dataset_base: Optional[Path], split: str, size: Tuple[int, int] = (640, 640)) -> List:
    """Devuelve tensores CHW [0,1] sin dibujo de cajas."""
    db = _resolve_dataset_base(dataset_base)
    names = TRAIN_PIVOT_IMAGES if split == "train" else VALID_PIVOT_IMAGES
    paths = _img_paths_from_pivots(db, split, names)
    tensors = [_load_image_as_tensor(p, size=size) for p in paths]
    return tensors


def make_ref_grid_with_gt(dataset_base: Optional[Path], split: str, size: Tuple[int, int] = (640, 640)) -> List:
    """Devuelve tensores con **GT dibujado** sobre las imágenes pivote."""
    db = _resolve_dataset_base(dataset_base)
    names = TRAIN_PIVOT_IMAGES if split == "train" else VALID_PIVOT_IMAGES
    paths = _img_paths_from_pivots(db, split, names)
    out: List = []
    for p in paths:
        img = _load_image_as_pil(p, size=size)
        W, H = img.size
        gt = _read_yolo_labels(_labels_path_from_image(p))
        _draw_boxes(img, gt, color=(0, 220, 130), label_text="real", W=W, H=H)
        out.append(TF.to_tensor(img))
    return out


def make_ref_grid_with_gt_and_pred(
    dataset_base: Optional[Path],
    split: str,
    preds_by_file: Optional[Dict[str, List[Dict]]] = None,
    size: Tuple[int, int] = (640, 640),
    conf_thr: float = 0.25,
    topk: int = 5,
) -> List:
    """Devuelve tensores con **GT** y **Pred** dibujados.

    preds_by_file: dict opcional con clave = nombre de archivo (p.ej. '0864.jpg') y valor = lista de dicts:
      {"bbox_xywh": [cx,cy,w,h] en [0,1], "conf": float, "cls": int}
    """
    db = _resolve_dataset_base(dataset_base)
    names = TRAIN_PIVOT_IMAGES if split == "train" else VALID_PIVOT_IMAGES
    paths = _img_paths_from_pivots(db, split, names)
    out: List = []
    for p in paths:
        img = _load_image_as_pil(p, size=size)
        W, H = img.size
        # GT
        gt = _read_yolo_labels(_labels_path_from_image(p))
        _draw_boxes(img, gt, color=(0, 220, 130), label_text="real", W=W, H=H)
        # Preds (si existen)
        if preds_by_file and p.name in preds_by_file:
            preds = [d for d in preds_by_file[p.name] if float(d.get("conf", 0.0)) >= conf_thr]
            preds = sorted(preds, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[: topk]
            preds_xywh = [(int(d.get("cls", -1)), *map(float, d.get("bbox_xywh", [0, 0, 0, 0]))) for d in preds]
            _draw_boxes(img, preds_xywh, color=(255, 80, 90), label_text="pred", W=W, H=H)
        out.append(TF.to_tensor(img))
    return out


def save_reference_overlays_png(
    out_path: Path,
    split: str,
    dataset_base: Optional[Path] = None,
    preds_by_file: Optional[Dict[str, List[Dict]]] = None,
    conf_thr: float = 0.25,
    topk: int = 5,
    nrow: int = 3,
    size: Tuple[int, int] = (640, 640),
) -> Path:
    """Genera y guarda un PNG con overlays GT+Pred sobre imágenes pivote.

    Parámetros
    ----------
    out_path:
        Ruta de salida del PNG. Se crearán las carpetas padre si no existen.
    split:
        Split del dataset ("train" o "val"). Determina el set de pivotes a usar.
    dataset_base:
        Ruta base del dataset. Si es None, se usa DEFAULT_DATASET_BASE.
    preds_by_file:
        Diccionario opcional con clave = nombre de archivo (p.ej. '0864.jpg') y valor = lista
        de dicts con el formato {"bbox_xywh": [cx,cy,w,h] en [0,1], "conf": float, "cls": int}.
        Si es None, se dibuja sólo GT.
    conf_thr:
        Umbral mínimo de confianza para considerar una predicción.
    topk:
        Máximo de predicciones a dibujar por imagen.
    nrow:
        Número de imágenes por fila en el grid.
    size:
        Tamaño de redimensionamiento (ancho, alto) para cada imagen pivote.
    """
    _ensure_vision_available()

    # Construir tensores CHW con GT+Pred para las imágenes pivote
    imgs = make_ref_grid_with_gt_and_pred(
        dataset_base=dataset_base,
        split=split,
        preds_by_file=preds_by_file,
        size=size,
        conf_thr=conf_thr,
        topk=topk,
    )

    if not imgs:
        raise RuntimeError("save_reference_overlays_png: no se generaron imágenes de referencia.")

    # Grid tipo torchvision.utils.make_grid
    grid = make_grid(torch.stack(imgs, dim=0), nrow=int(nrow), padding=2)
    img = TF.to_pil_image(grid)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def log_reference_images_to_tb(
    variant: str,
    split: str,
    run_name: Optional[str] = None,
    dataset_base: Optional[Path] = None,
    epoch: int = 0,
    nrow: int = 3,
    with_gt: bool = False,
) -> Path:
    """Crea un run en TB y registra un grid de imágenes de referencia del dataset."""
    _ensure_tensorboard_available()
    _ensure_vision_available()
    run_name = run_name or f"TB_{split}_refs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb = TBVisualization(TBConfig(variant=variant, phase=split, run_name=run_name))
    if with_gt:
        imgs = make_ref_grid_with_gt(dataset_base, split)
        tag = f"images/{split}_reference_grid_with_gt"
    else:
        imgs = make_ref_grid(dataset_base, split)
        tag = f"images/{split}_reference_grid"
    tb.log_image_grid(tag=tag, img_tensors=imgs, epoch=epoch, nrow=nrow)
    tb.close()
    return tb.cfg.logdir


def log_reference_overlays_to_tb(
    variant: str,
    split: str,
    run_name: Optional[str] = None,
    dataset_base: Optional[Path] = None,
    epoch: int = 0,
    nrow: int = 3,
    pred_json: Optional[Path] = None,
    conf_thr: float = 0.25,
    topk: int = 5,
) -> Path:
    """Registra grid con GT y Pred simultáneamente. Si pred_json es None, dibuja solo GT."""
    _ensure_tensorboard_available()
    _ensure_vision_available()
    run_name = run_name or f"TB_{split}_overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb = TBVisualization(TBConfig(variant=variant, phase=split, run_name=run_name))

    preds_by_file = None
    if pred_json is not None:
        preds_by_file = json.loads(Path(pred_json).read_text(encoding="utf-8"))

    imgs = make_ref_grid_with_gt_and_pred(
        dataset_base,
        split,
        preds_by_file=preds_by_file,
        conf_thr=conf_thr,
        topk=topk,
    )
    tb.log_image_grid(tag=f"images/{split}_overlays_gt_pred", img_tensors=imgs, epoch=epoch, nrow=nrow)
    tb.close()
    return tb.cfg.logdir


# =============================
# Herramientas para la demo/overlay de test
# =============================
def _latest_test_metrics_run(variant: str) -> Path:
    base = PROJ / "metrics" / "test_metrics" / variant
    if not base.exists():
        raise FileNotFoundError(
            f"No se encontró {base}. Ejecute primero test_metrics.py para generar un run de prueba."
        )
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"Carpeta {base} no contiene ejecuciones. Ejecute test_metrics.py.")
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest


def _read_test_csv(run_dir: Path) -> Dict[str, float]:
    csv_path = run_dir / "test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de métricas: {csv_path}")
    out: Dict[str, float] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["metric"].strip()
            try:
                v = float(row["value"])  # puede ser entero; lo forzamos a float
            except Exception:
                continue
            out[m] = v
    return out


def demo_from_test_metrics(variant: str = "m", epochs: int = 20, loss_value: float = 0.1) -> Path:
    """Construye un run de TB desde la última ejecución de test_metrics.

    - Copia los scalars del 'test.csv' (precision/recall/mAP/IoU y stats)
    - Genera una serie temporal artificial de pérdidas ('loss/total' = 0.1)
    """
    _ensure_tensorboard_available()
    src_run = _latest_test_metrics_run(variant)
    scalars = _read_test_csv(src_run)

    demo_name = f"TB_demo__{src_run.name}"
    tb = TBVisualization(TBConfig(variant=variant, phase="test", run_name=demo_name))

    # 1) Métricas del test como scalars (epoch=1)
    tb.log_metrics_epoch(scalars, epoch=1, phase=None)

    # 2) Serie temporal de pérdidas (recta constante en 0.1)
    for ep in range(1, int(epochs) + 1):
        tb.log_train_loss_epoch({"loss/total": float(loss_value)}, epoch=ep)

    tb.close()
    print("[TB] Demo completada.")
    return tb.cfg.logdir


def log_test_overlay_image(variant: str, tag: str = "images/overlay", epoch: int = 0) -> Path:
    """Registra en TB la imagen overlay.png generada por test_metrics.py del último run."""
    _ensure_tensorboard_available()
    _ensure_vision_available()
    src_run = _latest_test_metrics_run(variant)
    overlay = src_run / "overlay.png"
    if not overlay.exists():
        raise FileNotFoundError(f"No se encontró overlay.png en {src_run}")
    img_t = _load_image_as_tensor(overlay, size=(640, 640))
    run_name = f"TB_overlay__{src_run.name}"
    tb = TBVisualization(TBConfig(variant=variant, phase="test", run_name=run_name))
    tb.log_image(tag, img_t, epoch)
    tb.close()
    return tb.cfg.logdir



# =============================
# Sesiones por época (GT en 0, GT+Pred en >0)
# =============================
class TBRefOverlaySession:
    """
    Administra una sesión persistente de logging de imágenes pivote por época.
    - Época 0: dibuja únicamente GT real.
    - Épocas >0: dibuja GT + Pred (si se proveen).
    """
    def __init__(self, variant: str, split: str, run_name: str,
                 dataset_base: Optional[Path] = None, nrow: int = 3, size: Tuple[int, int] = (640, 640)) -> None:
        _ensure_tensorboard_available()
        _ensure_vision_available()
        self.tb = TBVisualization(TBConfig(variant=variant, phase=split, run_name=run_name))
        self.variant, self.split, self.dataset_base, self.nrow, self.size = variant, split, dataset_base, nrow, size

    def log_epoch(self, epoch: int, preds_by_file: Optional[Dict[str, List[Dict]]] = None,
                  conf_thr: float = 0.25, topk: int = 5) -> None:
        if epoch == 0:
            imgs = make_ref_grid_with_gt(self.dataset_base, self.split, size=self.size)
            tag = f"images/{self.split}_reference_grid_with_gt"
        else:
            imgs = make_ref_grid_with_gt_and_pred(self.dataset_base, self.split, preds_by_file=preds_by_file,
                                                  size=self.size, conf_thr=conf_thr, topk=topk)
            tag = f"images/{self.split}_reference_overlays_gt_pred"
        self.tb.log_image_grid(tag=tag, img_tensors=imgs, epoch=epoch, nrow=self.nrow)

    def close(self) -> None:
        self.tb.close()


def log_ref_session_epoch(variant: str, split: str, run_name: str,
                          dataset_base: Optional[Path] = None, epoch: int = 0,
                          pred_json: Optional[Path] = None, conf_thr: float = 0.25, topk: int = 5,
                          nrow: int = 3, size: Tuple[int, int] = (640, 640)) -> Path:
    """
    Versión sin estado: reabre el mismo run y registra el grid correspondiente a 'epoch'.
    Si pred_json se entrega y existe, se dibujan preds; de lo contrario solo GT.
    """
    _ensure_tensorboard_available()
    _ensure_vision_available()
    tb = TBVisualization(TBConfig(variant=variant, phase=split, run_name=run_name))

    preds_by_file = None
    if pred_json is not None:
        pjson = Path(pred_json)
        if pjson.exists():
            try:
                import json
                preds_by_file = json.loads(pjson.read_text(encoding="utf-8"))
            except Exception:
                preds_by_file = None

    if epoch == 0:
        imgs = make_ref_grid_with_gt(dataset_base, split, size=size)
        tag = f"images/{split}_reference_grid_with_gt"
    else:
        imgs = make_ref_grid_with_gt_and_pred(dataset_base, split, preds_by_file=preds_by_file,
                                              size=size, conf_thr=float(conf_thr), topk=int(topk))
        tag = f"images/{split}_reference_overlays_gt_pred"

    tb.log_image_grid(tag=tag, img_tensors=imgs, epoch=int(epoch), nrow=int(nrow))
    tb.close()
    return tb.cfg.logdir

# =============================
# CLI
# =============================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualización TensorBoard — YOLOv11")
    sub = p.add_subparsers(dest="cmd")

    # Subcomando: demo (métricas + pérdida sintética)
    d = sub.add_parser("demo", help="Crear un run de TB leyendo metrics/test_metrics/.../test.csv")
    d.add_argument("--variant", default="m", choices=["n", "s", "m", "l", "xl"], help="Variante")
    d.add_argument("--epochs", type=int, default=20, help="Épocas sintéticas para la curva de pérdida")
    d.add_argument("--loss", type=float, default=0.1, help="Valor constante de pérdida para la demo")

    # Subcomando: quick (sanity check)
    q = sub.add_parser("quick", help="Registrar un escalar de ejemplo en TB (sanity check)")
    q.add_argument("--variant", default="m", choices=["n", "s", "m", "l", "xl"], help="Variante")
    q.add_argument("--phase", default="train", choices=["train", "val", "test"], help="Fase")
    q.add_argument("--run", default=None, help="Nombre de run (por defecto TB_quick_<fecha>)")

    # Subcomando: refs (grids de imágenes pivote, con o sin GT)
    r = sub.add_parser("refs", help="Registrar grids de imágenes pivote del dataset en TB")
    r.add_argument("--variant", default="m", choices=["n", "s", "m", "l", "xl"], help="Variante")
    r.add_argument("--split", default="train", choices=["train", "val"], help="Split del dataset")
    r.add_argument("--dataset", default=str(DEFAULT_DATASET_BASE), help="Ruta base del dataset")
    r.add_argument("--nrow", type=int, default=3, help="N° de imágenes por fila en el grid")
    r.add_argument("--with-gt", action="store_true", help="Dibujar GT (real) sobre las imágenes pivote")

    # Subcomando: overlays (GT + Pred)
    o2 = sub.add_parser("overlays", help="Registrar grid con GT+Pred para imágenes pivote")
    o2.add_argument("--variant", default="m", choices=["n", "s", "m", "l", "xl"], help="Variante")
    o2.add_argument("--split", default="val", choices=["train", "val"], help="Split del dataset")
    o2.add_argument("--dataset", default=str(DEFAULT_DATASET_BASE), help="Ruta base del dataset")
    o2.add_argument("--nrow", type=int, default=3, help="N° de imágenes por fila en el grid")
    o2.add_argument("--pred-json", default=None, help="Ruta a JSON con predicciones por archivo (formato simple)")
    o2.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza para dibujar preds")
    o2.add_argument("--topk", type=int, default=5, help="Máximo de preds por imagen")
    o2.add_argument("--run", default=None, help="(Opcional) Nombre de run; si se entrega, se reutiliza.")

    # Subcomando: overlay (registrar overlay.png del último test_metrics)
    o = sub.add_parser("overlay", help="Registrar overlay.png del último run de test_metrics en TB")
    o.add_argument("--variant", default="m", choices=["n", "s", "m", "l", "xl"], help="Variante")

    # Subcomando: epoch (sesión persistente por época)
    e = sub.add_parser("epoch", help="(Entrenamiento) Registrar grid de referencia por época (GT en 0; GT+Pred en >0)")
    e.add_argument("--variant", default="m", choices=["n", "s", "m", "l", "xl"], help="Variante")
    e.add_argument("--split", default="train", choices=["train", "val"], help="Split del dataset")
    e.add_argument("--run", required=True, help="Nombre estable del run para acumular épocas (ej.: yolo11_m_train_20251030)")
    e.add_argument("--dataset", default=str(DEFAULT_DATASET_BASE), help="Ruta base del dataset")
    e.add_argument("--epoch", type=int, default=0, help="Época a registrar")
    e.add_argument("--pred-json", default=None, help="Ruta a JSON con predicciones (solo épocas >0)")
    e.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza para dibujar preds")
    e.add_argument("--topk", type=int, default=5, help="Máximo de preds por imagen")
    e.add_argument("--nrow", type=int, default=3, help="N° de imágenes por fila en el grid")

    return p.parse_args()



def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cmd_quick(variant: str, phase: str, run: Optional[str]) -> Path:
    _ensure_tensorboard_available()
    run_name = run or f"TB_quick_{_now_tag()}"
    tb = TBVisualization(TBConfig(variant=variant, phase=phase, run_name=run_name))
    tb.log_train_loss_epoch({"loss/total": 0.1, "loss/box": 0.05, "loss/cls": 0.03, "loss/dfl": 0.02}, epoch=1)
    tb.log_metrics_epoch(
        {"metrics/mAP50": 0.65, "metrics/mAP50-95": 0.42, "metrics/precision": 0.70, "metrics/recall": 0.60},
        epoch=1,
        phase=None,
    )
    tb.close()
    return tb.cfg.logdir



if __name__ == "__main__":
    args = _parse_args()
    if args.cmd == "demo":
        out = demo_from_test_metrics(variant=args.variant, epochs=args.epochs, loss_value=args.loss)
        print(f"[OK] Revise TensorBoard en: {out}")
    elif args.cmd == "quick":
        out = _cmd_quick(args.variant, args.phase, args.run)
        print(f"[OK] Revise TensorBoard en: {out}")
    elif args.cmd == "refs":
        out = log_reference_images_to_tb(
            variant=args.variant,
            split=args.split,
            dataset_base=Path(args.dataset),
            epoch=0,
            nrow=args.nrow,
            with_gt=args.with_gt,
        )
        print(f"[OK] Grid de referencias registrado en: {out}")
    elif args.cmd == "overlays":
        run_name = args.run or None
        out = log_reference_overlays_to_tb(
            variant=args.variant,
            split=args.split,
            dataset_base=Path(args.dataset),
            epoch=0,
            nrow=args.nrow,
            pred_json=Path(args.pred_json) if args.pred_json else None,
            conf_thr=args.conf,
            topk=args.topk,
        ) if run_name is None else log_ref_session_epoch(
            variant=args.variant,
            split=args.split,
            run_name=run_name,
            dataset_base=Path(args.dataset),
            epoch=0,
            pred_json=Path(args.pred_json) if args.pred_json else None,
            conf_thr=args.conf,
            topk=args.topk,
            nrow=args.nrow,
        )
        print(f"[OK] Grid con GT+Pred registrado en: {out}")
    elif args.cmd == "overlay":
        out = log_test_overlay_image(variant=args.variant, tag="images/overlay", epoch=0)
        print(f"[OK] overlay.png registrado en: {out}")
    elif args.cmd == "epoch":
        out = log_ref_session_epoch(
            variant=args.variant,
            split=args.split,
            run_name=args.run,
            dataset_base=Path(args.dataset),
            epoch=int(args.epoch),
            pred_json=Path(args.pred_json) if args.pred_json else None,
            conf_thr=float(args.conf),
            topk=int(args.topk),
            nrow=int(args.nrow),
        )
        print(f"[OK] Época {args.epoch} registrada en: {out}")
    else:
        print(
            "Uso:"
            "  python YOLOv11/utility/visualization.py demo --variant m --epochs 20 --loss 0.1"
            "  python YOLOv11/utility/visualization.py quick --variant m --phase train"
            "  python YOLOv11/utility/visualization.py refs --variant m --split train --with-gt"
            "  python YOLOv11/utility/visualization.py overlays --variant m --split val --pred-json preds.json"
            "  python YOLOv11/utility/visualization.py overlay --variant m"
            "  python YOLOv11/utility/visualization.py epoch --variant m --split train --run yolo11_m_train_20251030 --epoch 0"
        )
