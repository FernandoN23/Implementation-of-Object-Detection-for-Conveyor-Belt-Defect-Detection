# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/utility/metrics.py
# Descripción: Utilidades para estandarizar métricas de
#              entrenamiento y validación YOLO.
#==============================================================

"""Módulo de utilidades para métricas del proyecto YOLO.

Responsabilidades principales
-----------------------------
- Leer y consolidar resultados de entrenamiento/validación
  (results.csv, curvas, etc.) desde YOLO/metrics.
- Estandarizar el formato de salida de métricas en una carpeta
  `final_metrics` por variante/experimento.
- (Opcional) Calcular una distribución de IoUs sobre el split
  de validación a partir de los pesos finales.

Este módulo está pensado para ejecutarse como script:

    $ python YOLO/utility/metrics.py --variant s --train-run belt_defects_yolov5s_final \
        --val-run s_belt_defects_yolov5s_val --weights YOLO/weights/detect/s/train/....pt \
        --compute-iou

"""

from __future__ import annotations

import argparse
import traceback
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import yaml

# Raíces de proyecto (alineado con otros módulos de YOLO/)
FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parents[1]          # .../YOLO
PROJECT_ROOT = YOLO_ROOT.parent      # raíz del proyecto
METRICS_ROOT = YOLO_ROOT / "metrics"
RUNS_ROOT = YOLO_ROOT / "runs"
CONFIGS_ROOT = YOLO_ROOT / "configs"
YOLOV5_ROOT = YOLO_ROOT / "yolov5"

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

# Imports de YOLOv5 para cómputo opcional de IoU
_YOLOV5_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - sólo disponible cuando YOLOv5 está presente
    from models.common import DetectMultiBackend  # type: ignore
    from utils.dataloaders import create_dataloader  # type: ignore
    from utils.general import (  # type: ignore
        LOGGER,
        check_dataset,
        check_file,
        colorstr,
        increment_path,
        non_max_suppression,
    )
    from utils.metrics import box_iou  # type: ignore  # noqa: F401 (importado por compatibilidad, no se usa directamente)
    from utils.torch_utils import select_device  # type: ignore
except Exception as e:  # pragma: no cover
    # En contexto puramente offline (por ejemplo, análisis estático), se
    # puede importar este módulo sin YOLOv5, pero las funciones que
    # dependen de estos imports no serán utilizables.
    _YOLOV5_IMPORT_ERROR = e  # type: ignore
    print(
        f"[metrics] Advertencia: error al importar módulos YOLOv5 desde {YOLOV5_ROOT}: {e}",
        file=sys.stderr,
    )
    traceback.print_exc()
    LOGGER = None  # type: ignore
    # En contexto puramente offline (por ejemplo, análisis estático), se
    # puede importar este módulo sin YOLOv5, pero las funciones que
    # dependen de estos imports no serán utilizables.
    LOGGER = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuración de alto nivel
# ---------------------------------------------------------------------------


@dataclass
class MetricsConfig:
    """Configuración para consolidación de métricas de un experimento.

    Atributos clave
    ---------------
    task_model:
        Tipo de modelo. Por ahora sólo se soporta "detect".
    variant:
        Variante de escalado (n, s, m, l, x).
    train_run:
        Nombre lógico del experimento de entrenamiento (carpeta en
        YOLO/metrics/detect/<variant>/train/<train_run> y en
        YOLO/runs/detect/<variant>/train/<train_run>).
    val_run:
        Nombre lógico del experimento de validación (carpeta en
        YOLO/metrics/detect/<variant>/val/<val_run>).
    experiment_id:
        Identificador final para la carpeta de salida
        YOLO/metrics/detect/<variant>/final_metrics/<experiment_id>.
        Si es None, se usa train_run.
    dataset_cfg:
        Ruta al dataset.yaml utilizado para entrenar/validar.
    weights:
        Ruta a los pesos finales (.pt). Sólo obligatorio si se
        requiere calcular la distribución de IoU.
    imgsz:
        Tamaño de imagen para cómputo de IoU.
    batch_size:
        Tamaño de batch para el dataloader de validación en IoU.
    device:
        Dispositivo para inferencia ("", "0", "cpu", etc.).
    compute_iou:
        Si True, se calcula la distribución de IoUs.
    """

    task_model: str = "detect"
    variant: str = "s"
    train_run: str = ""
    val_run: str = ""
    experiment_id: Optional[str] = None

    dataset_cfg: Path = CONFIGS_ROOT / "dataset.yaml"
    weights: Optional[str] = None

    imgsz: int = 640
    batch_size: int = 4
    device: str = ""
    compute_iou: bool = False

    def final_experiment_id(self) -> str:
        if self.experiment_id:
            return self.experiment_id
        return self.train_run or f"{self.task_model}_{self.variant}_experiment"

    # Directorios derivados
    @property
    def train_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / self.variant / "train" / self.train_run

    @property
    def val_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / self.variant / "val" / self.val_run

    @property
    def final_metrics_dir(self) -> Path:
        return (
            METRICS_ROOT
            / self.task_model
            / self.variant
            / "final_metrics"
            / self.final_experiment_id()
        )

    @property
    def train_runs_root(self) -> Path:
        return RUNS_ROOT / self.task_model / self.variant / "train"


# ---------------------------------------------------------------------------
# Utilidades de archivos y verificación de precondiciones
# ---------------------------------------------------------------------------


def list_runs(root: Path) -> List[str]:
    """Devuelve una lista de nombres de subdirectorios en `root`."""
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def interactive_select(prompt: str, options: List[str]) -> str:
    """Permite seleccionar interactivamente un elemento de `options`."""
    if not options:
        raise RuntimeError(f"No hay opciones disponibles para {prompt}")
    print(prompt)
    for i, name in enumerate(options):
        print(f"  [{i}] {name}")
    while True:
        idx = input("Seleccione índice: ").strip()
        if not idx.isdigit():
            print("Ingrese un índice numérico válido.")
            continue
        i = int(idx)
        if 0 <= i < len(options):
            return options[i]
        print("Índice fuera de rango.")


def assert_train_and_val_exist(cfg: MetricsConfig) -> None:
    """Verifica que existan métricas de train y val para la configuración dada."""
    train_dir = cfg.train_metrics_dir
    val_dir = cfg.val_metrics_dir

    missing = []
    if not (train_dir / "results.csv").is_file():
        missing.append(f"results.csv no encontrado en {train_dir}")
    if not val_dir.exists():
        missing.append(f"Directorio de validación no encontrado: {val_dir}")

    if missing:
        msg = (
            "No se encontraron métricas completas de entrenamiento/validación "
            "para la configuración seleccionada:\n- "
            + "\n- ".join(missing)
        )
        raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Lectura de results.csv y metadatos
# ---------------------------------------------------------------------------


def load_results_csv(cfg: MetricsConfig) -> pd.DataFrame:
    """Carga `results.csv` del experimento de entrenamiento."""
    path = cfg.train_metrics_dir / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró results.csv en {path}")
    return pd.read_csv(path)


def load_hyp(cfg: MetricsConfig) -> Dict[str, Any]:
    """Intenta cargar hyp.yaml desde la carpeta de runs de entrenamiento.

    Si no se encuentra, devuelve un dict vacío.
    """
    hyp_path = cfg.train_runs_root / cfg.train_run / "hyp.yaml"
    if not hyp_path.is_file():
        return {}
    with open(hyp_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_stats(dataset_cfg: Path) -> Dict[str, Any]:
    """Carga información básica del dataset (número de clases, nombres y #imágenes)."""
    with open(dataset_cfg, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stats: Dict[str, Any] = {}
    stats["config"] = str(dataset_cfg)

    # Número de clases y nombres
    if "nc" in data:
        stats["num_classes"] = int(data["nc"])
    if "names" in data:
        # Puede ser dict o lista en algunos casos
        if isinstance(data["names"], dict):
            stats["class_names"] = [v for _, v in sorted(data["names"].items())]
        else:
            stats["class_names"] = list(data["names"])

    # Conteo básico de imágenes (opcional)
    def _count_images(path_key: str) -> Optional[int]:
        if path_key not in data:
            return None
        p = Path(data[path_key])
        if not p.exists():
            return None
        return sum(
            1
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for _ in p.rglob(ext)
        )

    stats["num_train_images"] = _count_images("train")
    stats["num_val_images"] = _count_images("val")

    return stats


# ---------------------------------------------------------------------------
# Cálculo de pérdidas promedio y curvas mAP
# ---------------------------------------------------------------------------


def _resolve_loss_columns(df: pd.DataFrame, split: str) -> Dict[str, str]:
    """Resuelve las columnas de pérdida (box/obj/cls) para un split dado.

    Devuelve un diccionario {"box_loss": col_name, ...} con las columnas
    encontradas. Si no encuentra ninguna, lanza un KeyError con las
    columnas disponibles en `results.csv`.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split debe ser 'train' o 'val', no {split!r}")

    base_names = ["box_loss", "obj_loss", "cls_loss"]
    col_map: Dict[str, str] = {}

    for base in base_names:
        # Candidatos en orden de prioridad
        candidates = [
            f"{split}/{base}",          # p.ej. train/box_loss
            f"{split}_{base}",          # p.ej. train_box_loss (por si acaso)
            base,                        # p.ej. box_loss
        ]
        col_found: Optional[str] = None
        for cand in candidates:
            if cand in df.columns:
                col_found = cand
                break
        # Búsqueda más laxa: columna que contenga base y split
        if col_found is None:
            for col in df.columns:
                if base in col and split in col:
                    col_found = col
                    break
        if col_found is not None:
            col_map[base] = col_found

    if not col_map:
        raise KeyError(
            "No se encontraron columnas de pérdida para "
            f"split='{split}' en results.csv. Columnas disponibles: "
            f"{list(df.columns)}"
        )

    return col_map


def compute_mean_loss(df: pd.DataFrame, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula pérdida promedio (box+obj+cls)/3 para train o val.

    La función delega la resolución de columnas a `_resolve_loss_columns`
    para ser robusta frente a pequeñas variaciones en los nombres de
    columnas de `results.csv`.

    Parameters
    ----------
    df:
        DataFrame cargado desde results.csv.
    split:
        "train" o "val".

    Returns
    -------
    epochs:
        Vector de épocas (tomadas desde la columna `epoch` si existe,
        de lo contrario se usa el índice del DataFrame).
    mean_loss:
        Vector de pérdidas promedio sobre las columnas disponibles.
    """
    col_map = _resolve_loss_columns(df, split)
    selected_cols = list(col_map.values())

    losses = df[selected_cols].to_numpy(dtype=float)
    mean_loss = losses.mean(axis=1)

    # Eje x: usar la columna `epoch` si está presente, si no el índice
    if "epoch" in df.columns:
        epochs = df["epoch"].to_numpy(dtype=int)
    else:
        epochs = df.index.to_numpy(dtype=int)

    return epochs, mean_loss


def extract_map_curves(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae curvas de mAP@0.5 y mAP@0.5:0.95 por época.

    La función es tolerante a pequeñas variaciones en el nombre de las
    columnas (espacios, prefijos/sufijos adicionales, etc.).
    """

    def _find_col(target: str) -> Optional[str]:
        # 1) Coincidencia exacta
        if target in df.columns:
            return target
        # 2) Coincidencia ignorando espacios
        for col in df.columns:
            if col.replace(" ", "") == target.replace(" ", ""):
                return col
        # 3) Cualquier columna que contenga el patrón objetivo
        for col in df.columns:
            if target in col:
                return col
        return None

    map50_col = _find_col("metrics/mAP_0.5")
    map5095_col = _find_col("metrics/mAP_0.5:0.95")

    missing: List[str] = []
    if map50_col is None:
        missing.append("metrics/mAP_0.5")
    if map5095_col is None:
        missing.append("metrics/mAP_0.5:0.95")

    if missing:
        raise KeyError(
            "Faltan columnas de mAP en results.csv: "
            + ", ".join(missing)
            + f". Columnas disponibles: {list(df.columns)}"
        )

    # Eje x: épocas
    if "epoch" in df.columns:
        epochs = df["epoch"].to_numpy(dtype=int)
    else:
        epochs = df.index.to_numpy(dtype=int)

    map50 = df[map50_col].to_numpy(dtype=float)
    map5095 = df[map5095_col].to_numpy(dtype=float)
    return epochs, map50, map5095


# ---------------------------------------------------------------------------
# Plotting estandarizado
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_title(prefix: str, cfg: MetricsConfig, hyp: Dict[str, Any], ds: Dict[str, Any]) -> str:
    epochs = hyp.get("epochs")
    batch = hyp.get("batch_size") or hyp.get("batch")
    num_classes = ds.get("num_classes")
    num_train = ds.get("num_train_images")
    num_val = ds.get("num_val_images")

    parts = [prefix, f"YOLOv5-{cfg.variant}"]
    meta = []
    if epochs is not None:
        meta.append(f"epochs={epochs}")
    if batch is not None:
        meta.append(f"batch={batch}")
    if num_classes is not None:
        meta.append(f"nc={num_classes}")
    if num_train is not None:
        meta.append(f"N_train={num_train}")
    if num_val is not None:
        meta.append(f"N_val={num_val}")

    if meta:
        parts.append(" | " + ", ".join(str(m) for m in meta))
    return " ".join(parts)


def plot_loss_curves(
    cfg: MetricsConfig,
    hyp: Dict[str, Any],
    ds_stats: Dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Genera curvas de pérdida promedio train/val y las guarda como PNG.

    `out_dir` debe corresponder a la carpeta `losses/` del experimento,
    de forma que en ese nivel quede la curva promedio.
    """
    _ensure_dir(out_dir)

    epochs_train, mean_loss_train = compute_mean_loss(df, "train")
    try:
        epochs_val, mean_loss_val = compute_mean_loss(df, "val")
        has_val = True
    except KeyError:
        epochs_val, mean_loss_val, has_val = None, None, False

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_train, mean_loss_train, label="Train mean loss")
    if has_val and epochs_val is not None and mean_loss_val is not None:
        plt.plot(epochs_val, mean_loss_val, label="Val mean loss")

    plt.xlabel("Epoch")
    plt.ylabel("Mean loss (box, obj, cls)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.title(_build_title("Loss curves |", cfg, hyp, ds_stats))
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def plot_loss_components_for_split(
    cfg: MetricsConfig,
    hyp: Dict[str, Any],
    ds_stats: Dict[str, Any],
    df: pd.DataFrame,
    split: str,
    out_dir: Path,
) -> None:
    """Genera curvas individuales de box/obj/cls loss para un split.

    Los gráficos se guardan en una subcarpeta dedicada, por ejemplo:
    `losses/train/loss_components.png` o `losses/val/loss_components.png`.
    """
    _ensure_dir(out_dir)
    col_map = _resolve_loss_columns(df, split)

    # Eje x
    if "epoch" in df.columns:
        epochs = df["epoch"].to_numpy(dtype=int)
    else:
        epochs = df.index.to_numpy(dtype=int)

    plt.figure(figsize=(8, 5))
    for base_name, col_name in col_map.items():
        values = df[col_name].to_numpy(dtype=float)
        # Etiqueta legible: "box", "obj", "cls"
        label_core = base_name.replace("_loss", "")
        label = f"{split} {label_core}"
        plt.plot(epochs, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    title_prefix = f"{split.capitalize()} loss components |"
    plt.title(_build_title(title_prefix, cfg, hyp, ds_stats))
    plt.tight_layout()
    plt.savefig(out_dir / "loss_components.png", dpi=200)
    plt.close()


def plot_map_curves(
    cfg: MetricsConfig,
    hyp: Dict[str, Any],
    ds_stats: Dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Genera curvas de mAP@0.5 y mAP@0.5:0.95 por época."""
    _ensure_dir(out_dir)
    epochs, map50, map5095 = extract_map_curves(df)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, map50, label="mAP@0.5")
    plt.plot(epochs, map5095, label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.title(_build_title("mAP curves |", cfg, hyp, ds_stats))
    plt.tight_layout()
    plt.savefig(out_dir / "map_curves.png", dpi=200)
    plt.close()

def plot_map_curves(
    cfg: MetricsConfig,
    hyp: Dict[str, Any],
    ds_stats: Dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Genera curvas de mAP@0.5 y mAP@0.5:0.95 por época."""
    _ensure_dir(out_dir)
    epochs, map50, map5095 = extract_map_curves(df)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, map50, label="mAP@0.5")
    plt.plot(epochs, map5095, label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.title(_build_title("mAP curves |", cfg, hyp, ds_stats))
    plt.tight_layout()
    plt.savefig(out_dir / "map_curves.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Empaquetado de curvas P/R/PR/F1 existentes
# ---------------------------------------------------------------------------


def copy_pr_curves(cfg: MetricsConfig, out_dir: Path) -> None:
    """Copia P/R/PR/F1 curves de train y val a la carpeta final."""
    _ensure_dir(out_dir)

    def _copy_if_exists(src_dir: Path, prefix: str) -> None:
        mapping = {
            "P_curve.png": f"{prefix}_P_curve.png",
            "R_curve.png": f"{prefix}_R_curve.png",
            "PR_curve.png": f"{prefix}_PR_curve.png",
            "F1_curve.png": f"{prefix}_F1_curve.png",
            "confusion_matrix.png": f"{prefix}_confusion_matrix.png",
        }
        for src_name, dst_name in mapping.items():
            src = src_dir / src_name
            if src.is_file():
                dst = out_dir / dst_name
                dst.write_bytes(src.read_bytes())

    _copy_if_exists(cfg.train_metrics_dir, "train")
    _copy_if_exists(cfg.val_metrics_dir, "val")


# ---------------------------------------------------------------------------
# Distribución de IoUs sobre dataset de validación
# ---------------------------------------------------------------------------


def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Calcula IoU entre dos conjuntos de cajas (xyxy).

    box1: (N, 4), box2: (M, 4) -> IoU (N, M).
    """
    # Expand dims para broadcasting
    b1 = box1[:, None, :]  # (N, 1, 4)
    b2 = box2[None, :, :]  # (1, M, 4)

    # Coordenadas de intersección
    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    # Áreas individuales
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Unión
    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / np.clip(union, a_min=1e-9, a_max=None)
    return iou


# ------- Helpers locales para compatibilidad con distintas versiones de YOLOv5 -------


def clip_coords(boxes: torch.Tensor, img_shape: Tuple[int, int]) -> None:
    """Restringe coordenadas xyxy al rango de la imagen.

    Parámetros
    ----------
    boxes:
        Tensor (N, 4) en formato xyxy.
    img_shape:
        (alto, ancho) de la imagen objetivo.
    """
    h, w = img_shape
    boxes[:, 0].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 3].clamp_(0, h)


def scale_coords(
    img1_shape: Tuple[int, int],
    coords: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[Any, Any]] = None,
) -> torch.Tensor:
    """Reescala coords (xyxy) de img1_shape a img0_shape.

    Esta implementación replica el comportamiento clásico de YOLOv5
    pero se define localmente para evitar depender de la presencia de
    `scale_coords` en `utils.general`, cuya firma puede variar entre
    versiones.

    Parámetros
    ----------
    img1_shape:
        Forma de la imagen de entrada al modelo (alto, ancho).
    coords:
        Tensor (N, 4) con cajas en coordenadas de `img1_shape`.
    img0_shape:
        Forma de la imagen original antes de letterboxing (alto, ancho).
    ratio_pad:
        Información de escalado/padding devuelta por el dataloader
        (típicamente `shapes[i][1]`). Puede ser:
        - (gain, (padw, padh)), o
        - ((gain_w, gain_h), (padw, padh)).
    """
    if ratio_pad is None:
        # Calcular gain y pad asumiendo letterbox estándar
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
        pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        rp0, rp1 = ratio_pad
        # rp0 puede ser escalar o tupla (gw, gh)
        if isinstance(rp0, (tuple, list)):
            gain = float(rp0[0])
        else:
            gain = float(rp0)
        pad_w, pad_h = float(rp1[0]), float(rp1[1])

    # Quitar padding
    coords[:, [0, 2]] -= pad_w
    coords[:, [1, 3]] -= pad_h
    # Desescalar
    coords[:, :4] /= gain
    # Ajustar a la imagen original
    clip_coords(coords, img0_shape)
    return coords


def compute_iou_distribution(
    cfg: MetricsConfig,
    conf_thres: float = 0.25,
    iou_thres: float = 0.6,
    max_det: int = 300,
) -> np.ndarray:
    """Calcula una distribución de IoUs en el split de validación.

    Para cada *ground truth* se toma la mejor predicción de la misma
    clase (después de NMS) y se almacena su IoU. Si no hay predicciones
    de esa clase para un GT dado, se registra IoU = 0.0.

    Notas
    -----
    - Los *targets* entregados por el dataloader están en formato
      (image_idx, class, x, y, w, h) normalizado (xywh). Aquí se
      convierten explícitamente a xyxy en píxeles antes de calcular
      IoU.
    - Las predicciones se reescalan a las dimensiones originales de la
      imagen mediante `scale_coords` local.
    """
    if LOGGER is None:
        raise RuntimeError(
            "YOLOv5 no está disponible; no se puede calcular la distribución de IoUs. "
            "Revisa el error de importación mostrado al cargar YOLO/utility/metrics.py."
        )
    if cfg.weights is None:
        raise ValueError("Se requiere cfg.weights para calcular la distribución de IoUs.")

    device = select_device(cfg.device)
    data = check_dataset(check_file(str(cfg.dataset_cfg)))
    val_path = data["val"]

    # Modelo
    model = DetectMultiBackend(
        cfg.weights,
        device=device,
        dnn=False,
        data=str(cfg.dataset_cfg),
        fp16=False,
    )
    stride, names = model.stride, model.names  # noqa: F841
    imgsz = cfg.imgsz
    bs = cfg.batch_size

    # Dataloader de validación (similar a val.py, simplificado)
    dataloader = create_dataloader(
        val_path,
        imgsz,
        bs,
        stride,
        single_cls=False,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.5,
        rect=False,
        rank=-1,
        workers=0,
        image_weights=False,
        prefix=colorstr("val: "),
    )[0]

    ious_all: List[float] = []

    model.model.eval()
    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        im = im.to(device, non_blocking=True)
        im = im.float() / 255.0

        with torch.no_grad():
            preds = model(im)

        preds = non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=max_det,
        )

        nb, _, h, w = im.shape  # batch size, channels, height, width
        for si in range(nb):
            pred = preds[si]
            gt = targets[targets[:, 0] == si]

            # Si no hay GT en esta imagen, no hay nada que acumular
            if len(gt) == 0:
                continue

            # Si no hay predicciones, cada GT se contabiliza con IoU=0.0
            if pred is None or len(pred) == 0:
                ious_all.extend([0.0] * len(gt))
                continue

            # Escalar predicciones a tamaño de la imagen original
            pred_boxes = pred[:, :4]
            pred_boxes = scale_coords(
                im[si].shape[1:],  # (h, w) del tensor de entrada
                pred_boxes,
                shapes[si][0],     # forma original (h0, w0)
                shapes[si][1],     # ratio_pad
            ).cpu().numpy()
            pred_cls = pred[:, 5].cpu().numpy().astype(int)

            # Ground truth: (image_index, class, x, y, w, h) normalizado
            gt_boxes_xywh = gt[:, 2:6].clone()
            h0, w0 = shapes[si][0]
            gain = np.array([w0, h0, w0, h0], dtype=np.float32)
            gt_boxes_xywh = (gt_boxes_xywh * gt_boxes_xywh.new_tensor(gain)).cpu().numpy()
            gt_cls = gt[:, 1].cpu().numpy().astype(int)

            # Conversión explícita de xywh -> xyxy en píxeles
            x_c = gt_boxes_xywh[:, 0]
            y_c = gt_boxes_xywh[:, 1]
            w_gt = gt_boxes_xywh[:, 2]
            h_gt = gt_boxes_xywh[:, 3]
            x1 = x_c - w_gt / 2.0
            y1 = y_c - h_gt / 2.0
            x2 = x_c + w_gt / 2.0
            y2 = y_c + h_gt / 2.0
            gt_boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # Para cada GT, tomamos el mejor IoU con alguna predicción
            # de la misma clase. Si no hay predicciones de esa clase,
            # se registra IoU=0.
            for g_box, g_cls in zip(gt_boxes_xyxy, gt_cls):
                mask = pred_cls == g_cls
                if not mask.any():
                    ious_all.append(0.0)
                    continue
                ious_mat = _bbox_iou(pred_boxes[mask], g_box.reshape(1, 4))  # (K, 1)
                best_iou = float(ious_mat.max())
                ious_all.append(best_iou)

    return np.asarray(ious_all, dtype=np.float32)


def save_iou_distribution(
    ious: np.ndarray,
    cfg: MetricsConfig,
    hyp: Dict[str, Any],
    ds_stats: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Guarda histograma de IoUs y el array en disco.

    El título del gráfico se construye con el mismo formato estándar
    que el resto de las curvas (loss, mAP, etc.), de forma que todas
    las figuras queden homogéneas para el informe.
    """
    _ensure_dir(out_dir)
    np.save(out_dir / "iou_distribution.npy", ious)

    plt.figure(figsize=(8, 5))
    plt.hist(ious, bins=20, range=(0.0, 1.0), alpha=0.8, edgecolor="black")
    plt.xlabel("IoU")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title(_build_title("IoU distribution |", cfg, hyp, ds_stats))
    plt.tight_layout()
    plt.savefig(out_dir / "iou_distribution.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Resumen en JSON
# ---------------------------------------------------------------------------


def build_summary(
    cfg: MetricsConfig,
    hyp: Dict[str, Any],
    ds_stats: Dict[str, Any],
    df: pd.DataFrame,
    ious: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Construye un diccionario con resumen de métricas."""

    summary: Dict[str, Any] = {
        "task_model": cfg.task_model,
        "variant": cfg.variant,
        "experiment_id": cfg.final_experiment_id(),
        "train_run": cfg.train_run,
        "val_run": cfg.val_run,
        "dataset": ds_stats,
        "hyperparams": {},
        "metrics": {},
    }

    # Hiperparámetros relevantes
    epochs = hyp.get("epochs")
    batch = hyp.get("batch_size") or hyp.get("batch")
    if epochs is not None:
        summary["hyperparams"]["epochs"] = int(epochs)
    if batch is not None:
        summary["hyperparams"]["batch_size"] = int(batch)
    summary["hyperparams"]["img_size"] = cfg.imgsz

    # Métricas globales (tomamos última fila como proxy del estado final)
    if "metrics/mAP_0.5" in df.columns:
        summary["metrics"]["final_map50"] = float(df["metrics/mAP_0.5"].iloc[-1])
    if "metrics/mAP_0.5:0.95" in df.columns:
        summary["metrics"]["final_map50_95"] = float(df["metrics/mAP_0.5:0.95"].iloc[-1])
    if "metrics/precision" in df.columns:
        summary["metrics"]["final_precision"] = float(df["metrics/precision"].iloc[-1])
    if "metrics/recall" in df.columns:
        summary["metrics"]["final_recall"] = float(df["metrics/recall"].iloc[-1])

    # Pérdidas finales (train/val)
    for split in ("train", "val"):
        try:
            _, mean_loss = compute_mean_loss(df, split)
        except KeyError:
            continue
        summary.setdefault("metrics", {})
        summary["metrics"][f"{split}_final_mean_loss"] = float(mean_loss[-1])

    # IoU distribution
    if ious is not None and ious.size > 0:
        summary["iou_distribution"] = {
            "mean": float(np.mean(ious)),
            "median": float(np.median(ious)),
            "p75": float(np.percentile(ious, 75)),
            "p90": float(np.percentile(ious, 90)),
            "num_samples": int(ious.size),
        }

    return summary


def save_summary_json(summary: Dict[str, Any], out_dir: Path) -> None:
    _ensure_dir(out_dir)
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidación y estandarización de métricas YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-model", default="detect", help="Tipo de modelo (detect/classify).")
    parser.add_argument("--variant", default="s", help="Variante de escala (n, s, m, l, x).")
    parser.add_argument("--train-run", default="", help="Nombre del experimento de entrenamiento.")
    parser.add_argument("--val-run", default="", help="Nombre del experimento de validación.")
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Identificador final para carpeta final_metrics (por defecto, train_run).",
    )
    parser.add_argument(
        "--dataset-cfg",
        default=str(CONFIGS_ROOT / "dataset.yaml"),
        help="Ruta a dataset.yaml.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Ruta a pesos finales (.pt) para cómputo opcional de IoU.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen para IoU.")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamaño de batch para IoU.")
    parser.add_argument("--device", default="", help="Dispositivo (\"\", \"0\", \"cpu\", etc.).")
    parser.add_argument(
        "--compute-iou",
        action="store_true",
        help="Calcular distribución de IoUs en validación.",
    )
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> MetricsConfig:
    cfg = MetricsConfig(
        task_model=args.task_model,
        variant=args.variant,
        train_run=args.train_run,
        val_run=args.val_run,
        experiment_id=args.experiment_id or None,
        dataset_cfg=Path(args.dataset_cfg),
        weights=args.weights or None,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        compute_iou=bool(args.compute_iou),
    )

    # Selección interactiva si faltan runs
    if not cfg.train_run:
        options = list_runs(METRICS_ROOT / cfg.task_model / cfg.variant / "train")
        cfg.train_run = interactive_select("Seleccione experimento de entrenamiento:", options)
    if not cfg.val_run:
        options = list_runs(METRICS_ROOT / cfg.task_model / cfg.variant / "val")
        cfg.val_run = interactive_select("Seleccione experimento de validación:", options)

    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = build_config_from_args(args)

    # Verificar que existan métricas de train y val
    assert_train_and_val_exist(cfg)

    # Cargar datos base
    df = load_results_csv(cfg)
    hyp = load_hyp(cfg)
    ds_stats = load_dataset_stats(cfg.dataset_cfg)

    out_dir = cfg.final_metrics_dir
    _ensure_dir(out_dir)

    # Subcarpeta para pérdidas
    losses_root = out_dir / "losses"
    _ensure_dir(losses_root)

    # Curvas de pérdida promedio (nivel `losses/`)
    plot_loss_curves(cfg, hyp, ds_stats, df, losses_root)

    # Curvas de componentes de pérdida por split en subcarpetas
    try:
        plot_loss_components_for_split(cfg, hyp, ds_stats, df, "train", losses_root / "train")
    except KeyError:
        pass
    try:
        plot_loss_components_for_split(cfg, hyp, ds_stats, df, "val", losses_root / "val")
    except KeyError:
        pass

    # Curvas de mAP se mantienen en el nivel raíz del experimento
    plot_map_curves(cfg, hyp, ds_stats, df, out_dir)

    # Copiar curvas P/R/PR/F1 y matriz de confusión
    copy_pr_curves(cfg, out_dir)

    # Distribución de IoU (opcional)
    ious: Optional[np.ndarray] = None
    if cfg.compute_iou:
        ious = compute_iou_distribution(cfg)
        if ious.size:
            save_iou_distribution(ious, cfg, hyp, ds_stats, out_dir)

    # Resumen JSON
    summary = build_summary(cfg, hyp, ds_stats, df, ious)
    save_summary_json(summary, out_dir)

    print(f"Métricas consolidadas en: {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
