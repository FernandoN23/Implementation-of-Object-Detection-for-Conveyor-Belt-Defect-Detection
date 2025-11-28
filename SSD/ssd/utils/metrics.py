# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/ssd/utils/metrics.py
# Descripción: Utilidades de métricas para SSD.
#              Cálculo de P, R, F1, mAP@[0.5,0.5:0.95], curvas PR y
#              matrices de confusión a partir de predicciones de
#              detección y etiquetas reales.
# ==============================================================

from __future__ import annotations

import math
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Decoradores internos (equivalentes simplificados a YOLO/utils)
# ---------------------------------------------------------------------------


def TryExcept(msg: str) -> Callable:
    """Decorador simple para capturar excepciones y mostrar un mensaje."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:  # pragma: no cover - uso defensivo
                print(f"{msg}: {e}")
        return wrapper

    return decorator


def threaded(func: Callable) -> Callable:
    """Ejecuta la función en un hilo separado (modo daemon)."""

    def wrapper(*args: Any, **kwargs: Any) -> threading.Thread:
        t = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t

    return wrapper


# ---------------------------------------------------------------------------
# Métricas principales
# ---------------------------------------------------------------------------


def fitness(x: np.ndarray) -> np.ndarray:
    """Calcula una medida de fitness a partir de [P, R, mAP@0.5, mAP@0.5:0.95].

    Usa una combinación lineal ponderada:
        fitness = 0*P + 0*R + 0.1*mAP50 + 0.9*mAP5095
    """
    w = [0.0, 0.0, 0.1, 0.9]
    return (x[:, :4] * w).sum(1)


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Suavizado con filtro de caja sobre el vector `y` con fracción `f`."""
    if len(y) == 0:
        return y
    nf = round(len(y) * f * 2) // 2 + 1  # número impar de elementos
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    save_dir: str | Path = ".",
    names: Dict[int, str] | Tuple[str, ...] | Iterable[str] = (),
    eps: float = 1e-16,
    prefix: str = "",
):
    """Calcula AP, precisión, recall y F1 por clase a partir de TP/FP por predicción.

    Parámetros
    ----------
    tp:
        True positives acumulados por predicción (array Nx1 o NxT).
    conf:
        Confianza de cada predicción (array N).
    pred_cls:
        Clase predicha para cada bounding box (array N).
    target_cls:
        Clase real para cada bounding box (array N).
    plot:
        Si es True, guarda PR/F1/P/R curves en `save_dir`.
    save_dir:
        Carpeta donde guardar las figuras.
    names:
        Diccionario o lista de nombres de clases.
    eps:
        Pequeña constante numérica para evitar divisiones por cero.
    prefix:
        Prefijo opcional para los nombres de archivo de salida.

    Retorna
    -------
    tp_c, fp_c, p, r, f1, ap, unique_classes
    """
    save_dir = Path(save_dir)

    # Ordenar por confianza descendente
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Clases únicas presentes en las etiquetas
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    # Curvas de precisión-recall y AP por clase
    px = np.linspace(0, 1, 1000)
    py = []  # curvas PR para ploteo
    ap = np.zeros((nc, tp.shape[1]))
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]      # nº labels de esta clase
        n_p = i.sum()     # nº predicciones de esta clase
        if n_p == 0 or n_l == 0:
            continue

        # FP/TP acumulados
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall y precisión
        recall = tpc / (n_l + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # AP a partir de la curva P-R
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))

    # F1 score
    f1 = 2 * p * r / (p + r + eps)

    # Normalizar nombres a dict {idx: name} sólo en clases presentes
    if isinstance(names, dict):
        names_list = [v for k, v in names.items() if k in unique_classes]
    else:
        names_list = list(names)
        if names_list and len(names_list) == int(unique_classes.max()) + 1:
            # Filtrar por clases presentes
            names_list = [names_list[int(i)] for i in unique_classes]
    names_dict = dict(enumerate(names_list))

    if plot:
        if py:
            plot_pr_curve(px, py, ap, save_dir / f"{prefix}PR_curve.png", names_dict)
        plot_mc_curve(px, f1, save_dir / f"{prefix}F1_curve.png", names_dict, ylabel="F1")
        plot_mc_curve(px, p, save_dir / f"{prefix}P_curve.png", names_dict, ylabel="Precision")
        plot_mc_curve(px, r, save_dir / f"{prefix}R_curve.png", names_dict, ylabel="Recall")

    # Índice de máximo F1 medio (para un umbral de confianza global)
    i_f1 = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i_f1], r[:, i_f1], f1[:, i_f1]

    tp_c = (r * nt).round()
    fp_c = (tp_c / (p + eps) - tp_c).round()

    return tp_c, fp_c, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall: np.ndarray, precision: np.ndarray):
    """Calcula promedio de precisión (AP) dado un par (recall, precision)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Envolvente de precisión
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrar área bajo la curva
    method = "interp"  # 'interp' (COCO) o 'continuous'
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mpre, mrec


# ---------------------------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------------------------


class ConfusionMatrix:
    """Matriz de confusión para evaluación de clasificación en detección."""

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45) -> None:
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.float32)
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections: torch.Tensor, labels: torch.Tensor) -> None:
        """Actualiza la matriz de confusión con un batch.

        detections: [N,6] → (x1,y1,x2,y2,conf,cls)
        labels    : [M,5] → (cls,x1,y1,x2,y2)
        """
        if detections is None or len(detections) == 0:
            gt_classes = labels[:, 0].int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        if len(detections) == 0:
            gt_classes = labels[:, 0].int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        gt_classes = labels[:, 0].int()
        det_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                1,
            ).cpu().numpy()

            # Resolver empates: cada predicción y cada GT se asocia a lo sumo una vez
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)

        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[det_classes[m1[j]], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1

        if n:
            for i, dc in enumerate(det_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1

    def tp_fp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve TP y FP por clase (excluyendo background)."""
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return tp[:-1], fp[:-1]

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    def plot(self, normalize: bool = True, save_dir: str | Path = "", names: Dict[int, str] | Tuple[str, ...] = ()):
        """Genera la figura de la matriz de confusión."""
        import seaborn as sn

        save_dir = Path(save_dir)
        array = self.matrix.copy()
        if normalize:
            col_sums = array.sum(0, keepdims=True) + 1e-9
            array = array / col_sums

        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc = self.nc
        nn = len(names)
        sn.set(font_scale=1.0 if nc < 50 else 0.8)

        labels_flag = (0 < nn < 99) and (nn == nc)
        ticklabels = ([*names, "background"]) if labels_flag else "auto"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(save_dir / "confusion_matrix.png", dpi=250)
        plt.close(fig)

    def print(self) -> None:
        """Imprime la matriz en formato texto."""
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


# ---------------------------------------------------------------------------
# Funciones geométricas (IoU, etc.)
# ---------------------------------------------------------------------------


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calcula IoU / GIoU / DIoU / CIoU entre dos conjuntos de cajas.

    box1: [N,4] o [1,4]
    box2: [M,4]
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    if xywh:
        w1, h1 = w1_, h1_
        w2, h2 = w2_, h2_

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2

        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area

    return iou


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU pairwise entre todos los elementos de box1 y box2 (en formato xyxy)."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Intersection over area de box2 (útil, por ejemplo, para ignorar regiones)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    inter_w = np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)
    inter_h = np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    inter_area = np.clip(inter_w, a_min=0, a_max=None) * np.clip(inter_h, a_min=0, a_max=None)

    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter_area / box2_area


def wh_iou(wh1: torch.Tensor, wh2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU entre vectores ancho-alto (w,h) de anchors."""
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2).prod(2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)


# ---------------------------------------------------------------------------
# Plots de curvas PR / metric-confidence
# ---------------------------------------------------------------------------


@threaded
def plot_pr_curve(
    px: np.ndarray,
    py: Iterable[np.ndarray],
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: Dict[int, str] | Tuple[str, ...] = (),
):
    """Genera la curva P–R promedio y por clase (si hay pocas clases)."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(list(py), axis=1)

    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="grey")

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: Dict[int, str] | Tuple[str, ...] = (),
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
):
    """Curva métrica–confianza (F1, P o R) para todas las clases."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")

    y_mean = smooth(py.mean(0), 0.05)
    ax.plot(px, y_mean, linewidth=3, color="blue", label=f"all classes {y_mean.max():.2f} at {px[y_mean.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


__all__ = [
    "fitness",
    "smooth",
    "ap_per_class",
    "compute_ap",
    "ConfusionMatrix",
    "bbox_iou",
    "box_iou",
    "bbox_ioa",
    "wh_iou",
    "plot_pr_curve",
    "plot_mc_curve",
]
