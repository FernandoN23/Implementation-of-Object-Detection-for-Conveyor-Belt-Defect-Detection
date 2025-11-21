# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: metrics.py
# Módulo de métricas para YOLOv11 (detección). Calcula y registra
# P, R, F1, mAP@0.50, mAP@0.50:0.95, IoU y matrices de confusión
# por variante (n/s/m/l/xl) y fase (train/val/test). Soporta
# imágenes negativas (sin etiquetas) y se integra con el logger
# del proyecto para guardar curvas y CSV por época.
# Ahora soporta "slots" de ejecución (tests/<run_name> y final/) para
# separar métricas de PRUEBAS y métricas FINALES, en espejo a logger.py y
# weights.py.
#==============================================================
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

TensorOrArray = Union[torch.Tensor, np.ndarray, None]

# ==============================================================
# Helpers de slot y proyecto
# ==============================================================

def _find_project_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()


def _resolve_slot_dir(root: Path, variant: str, phase: str, *, is_test: bool,
                      run_name: Optional[str], reset_final: bool = False,
                      base: str = "metrics") -> Path:
    """Devuelve la carpeta base (root/<base>/<variant>/<phase>/<slot>/).
    - Si is_test=True → slot = tests/<run_name> (run_name obligatorio; si None, usa timestamp).
    - Si is_test=False → slot = final (si reset_final y existe, se borra).
    """
    variant = str(variant).lower()
    phase = str(phase).lower()

    if is_test:
        rn = run_name or "test"
        slot = Path("tests") / rn
        out = root / base / variant / phase / slot
    else:
        slot = Path("final")
        out = root / base / variant / phase / slot
        if reset_final and out.exists():
            import shutil
            shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==============================================================
# Utilidades geométricas
# ==============================================================

def _box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def _as_tensor(x: TensorOrArray, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(device=device) if device is not None else t
    raise TypeError(f"Tipo no soportado: {type(x)}")


# ==============================================================
# Promedios corrientes para pérdidas (train/val)
# ==============================================================

class RunningMean:
    def __init__(self):
        self._sum: Dict[str, float] = {}
        self._n: int = 0

    def update(self, scalars: Dict[str, float], inc: int = 1) -> None:
        for k, v in scalars.items():
            self._sum[k] = self._sum.get(k, 0.0) + float(v) * inc
        self._n += inc

    def mean(self) -> Dict[str, float]:
        if self._n == 0:
            return {k: 0.0 for k in self._sum}
        return {k: s / self._n for k, s in self._sum.items()}


# ==============================================================
# Cálculo de AP y curvas PR (COCO-like)
# ==============================================================

@dataclass
class PRCurves:
    px: np.ndarray
    p_curve: np.ndarray
    r_curve: np.ndarray
    f1_curve: np.ndarray
    ap: np.ndarray

    def summary(self) -> Dict[str, float]:
        map50 = self.ap[:, 0].mean() if self.ap.size else 0.0
        map5095 = self.ap.mean() if self.ap.size else 0.0
        i = _smooth(self.f1_curve.mean(0), f=0.1).argmax() if self.f1_curve.size else 0
        p = float(self.p_curve[:, i].mean()) if self.p_curve.size else 0.0
        r = float(self.r_curve[:, i].mean()) if self.r_curve.size else 0.0
        return {"precision": p, "recall": r, "mAP50": float(map50), "mAP50_95": float(map5095)}


def _smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    if y.size == 0:
        return y
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def _ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray,
                  iouv: np.ndarray, names: Dict[int, str],
                  save_dir: Optional[Path] = None, prefix: str = "") -> Tuple[PRCurves, np.ndarray]:
    if tp.size == 0:
        T = 1000
        return PRCurves(px=np.linspace(0, 1, T), p_curve=np.zeros((0, T)), r_curve=np.zeros((0, T)),
                         f1_curve=np.zeros((0, T)), ap=np.zeros((0, len(iouv)))), np.array([], dtype=int)

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc, T = unique_classes.shape[0], 1000

    x = np.linspace(0, 1, T)
    p_curve = np.zeros((nc, T))
    r_curve = np.zeros((nc, T))
    ap = np.zeros((nc, iouv.size))

    for ci, c in enumerate(unique_classes):
        idx = pred_cls == c
        n_l = int(nt[ci])
        n_p = int(idx.sum())
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[idx]).cumsum(0)
        tpc = tp[idx].cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        r_curve[ci] = np.interp(-x, -conf[idx], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-x, -conf[idx], precision[:, 0], left=1)
        for j in range(iouv.size):
            ap[ci, j] = _compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    curves = PRCurves(px=x, p_curve=p_curve, r_curve=r_curve, f1_curve=f1_curve, ap=ap)

    if save_dir is not None:
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            _plot_pr_curve(x, p_curve, ap, save_dir / f"{prefix}PR_curve.png", names)
            _plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1")
            _plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision")
            _plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall")
        except Exception as e:
            warnings.warn(f"Fallo al guardar curvas PR/MC: {e}")

    return curves, unique_classes.astype(int)


def _plot_pr_curve(px: np.ndarray, p_curve: np.ndarray, ap: np.ndarray, path: Path,
                   names: Dict[int, str]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21 and p_curve.shape[0] == len(names):
        for i, y in enumerate(p_curve):
            label = f"{names.get(i, str(i))} {ap[i,0]:.3f}"
            ax.plot(px, y, linewidth=1, label=label)
    else:
        ax.plot(px, p_curve.T, linewidth=1)
    ax.plot(px, p_curve.mean(0), linewidth=3, label=f"all classes {ap[:,0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left"); ax.set_title("Precision-Recall Curve")
    fig.savefig(path, dpi=250); plt.close(fig)


def _plot_mc_curve(px: np.ndarray, py: np.ndarray, path: Path, names: Dict[int, str],
                   xlabel: str = "Confidence", ylabel: str = "Metric") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21 and py.shape[0] == len(names):
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names.get(i, str(i))}")
    else:
        ax.plot(px, py.T, linewidth=1)
    y = _smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left"); ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(path, dpi=250); plt.close(fig)


# ==============================================================
# Matriz de confusión (detección)
# ==============================================================

class ConfusionMatrix:
    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.50):
        self.nc = int(nc)
        self.conf = 0.25 if conf in (None, 0.001) else float(conf)
        self.iou_thres = float(iou_thres)
        self.mat = np.zeros((self.nc + 1, self.nc + 1), dtype=np.float64)

    def update(self, detections: Optional[torch.Tensor], gt_boxes: torch.Tensor, gt_cls: torch.Tensor) -> None:
        if gt_cls.numel() == 0:
            if detections is not None:
                det = detections[detections[:, 4] > self.conf]
                for dc in det[:, 5].int().tolist():
                    self.mat[int(dc), self.nc] += 1
            return

        if detections is None or detections.numel() == 0:
            for gc in gt_cls.int().tolist():
                self.mat[self.nc, int(gc)] += 1
            return

        det = detections[detections[:, 4] > self.conf]
        if det.numel() == 0:
            for gc in gt_cls.int().tolist():
                self.mat[self.nc, int(gc)] += 1
            return

        iou = _box_iou_xyxy(gt_boxes, det[:, :4])
        x = torch.where(iou > self.iou_thres)
        if x[0].numel():
            matches = torch.stack((x[0], x[1], iou[x[0], x[1]]), 1).cpu().numpy()
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        gt_classes = gt_cls.int().cpu().numpy()
        det_classes = det[:, 5].int().cpu().numpy()
        m0 = matches[:, 0].astype(int) if matches.size else np.array([], dtype=int)
        m1 = matches[:, 1].astype(int) if matches.size else np.array([], dtype=int)

        for i, gc in enumerate(gt_classes):
            j = (m0 == i)
            if m0.size and j.sum() == 1:
                dc = int(det_classes[m1[j]][0]) if det_classes[m1[j]].ndim > 0 else int(det_classes[m1[j]])
                self.mat[dc, int(gc)] += 1
            else:
                self.mat[self.nc, int(gc)] += 1

        for i, dc in enumerate(det_classes):
            if not (m1.size and (m1 == i).any()):
                self.mat[int(dc), self.nc] += 1

    def plot(self, path: Path, names: Dict[int, str]) -> None:
        if sns is None:
            warnings.warn("seaborn no disponible; no se graficará la matriz de confusión")
            return
        array = self.mat / (self.mat.sum(0, keepdims=True) + 1e-9)
        array[array < 0.005] = np.nan
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        ticklabels = list(names.values()) + ["background"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.heatmap(array, ax=ax, annot=self.nc < 30, annot_kws={"size": 8}, cmap="Blues",
                        fmt=".2f", square=True, vmin=0.0, xticklabels=ticklabels, yticklabels=ticklabels)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.set_title("Confusion Matrix (Normalized)")
        fig.savefig(path, dpi=250); plt.close(fig)


# ==============================================================
# Núcleo de métricas de detección
# ==============================================================

@dataclass
class DetMetricsSummary:
    precision: float
    recall: float
    map50: float
    map50_95: float
    tp: int
    fp: int
    fn: int
    iou_mean: float = 0.0
    iou_median: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        return {
            "metrics/precision": d["precision"],
            "metrics/recall": d["recall"],
            "metrics/mAP50": d["map50"],
            "metrics/mAP50-95": d["map50_95"],
            "metrics/IoU_mean": d.get("iou_mean", 0.0),
            "metrics/IoU_median": d.get("iou_median", 0.0),
            "stats/tp": float(d["tp"]),
            "stats/fp": float(d["fp"]),
            "stats/fn": float(d["fn"]),
        }


class DetMetricsYOLOv11:
    def __init__(self,
                 class_names: Optional[Dict[int, str]] = None,
                 save_dir: Optional[Path] = None,
                 iou_thresholds: Iterable[float] = np.linspace(0.5, 0.95, 10),
                 nc: Optional[int] = None,
                 # --- slots opcionales (si no se pasa save_dir) ---
                 project_root: Optional[Path] = None,
                 variant: Optional[str] = None,
                 phase: Optional[str] = None,
                 is_test: bool = False,
                 run_name: Optional[str] = None,
                 reset_final: bool = False,
                 base_for_save: str = "metrics") -> None:
        """
        Si 'save_dir' es None y se proveen project_root+variant+phase, se construye automáticamente
        la carpeta de guardado por slot: <project_root>/<base_for_save>/<variant>/<phase>/<slot>/
        donde slot = tests/<run_name> (is_test=True) o final (is_test=False).
        """
        if class_names is None:
            if nc is None:
                raise ValueError("DetMetricsYOLOv11: debes proveer 'class_names' o 'nc'.")
            self.names: Dict[int, str] = {i: str(i) for i in range(int(nc))}
        else:
            self.names = class_names

        if save_dir is None and (project_root is not None and variant is not None and phase is not None):
            root = Path(project_root)
            slot_dir = _resolve_slot_dir(root, str(variant), str(phase), is_test=is_test,
                                         run_name=run_name, reset_final=reset_final, base=base_for_save)
            self.save_dir = slot_dir
        else:
            self.save_dir = Path(save_dir) if save_dir is not None else None

        self.iouv = np.array(list(iou_thresholds), dtype=np.float64)
        self.stats_tp: List[np.ndarray] = []
        self.stats_conf: List[np.ndarray] = []
        self.stats_pred_cls: List[np.ndarray] = []
        self.stats_target_cls: List[np.ndarray] = []
        self.cm = ConfusionMatrix(nc=len(self.names), iou_thres=0.50)
        self.matched_ious: List[float] = []

    @staticmethod
    def _xywhn_to_xyxy_pix(xywhn: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        H, W = hw
        cx, cy, w, h = xywhn.unbind(-1)
        x1 = (cx - w * 0.5) * W
        y1 = (cy - h * 0.5) * H
        x2 = (cx + w * 0.5) * W
        y2 = (cy + h * 0.5) * H
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def add_batch(self,
                  preds: List[TensorOrArray],
                  targets: List[TensorOrArray],
                  img_hw: Union[List[Tuple[int, int]], Tuple[int, int]],
                  labels_is_xywhn: bool = True,
                  conf_min_for_cm: float = 0.25,
                  iou_match_for_cm: float = 0.50) -> None:
        if isinstance(img_hw, tuple):  # admitir (H,W) para un solo elemento
            img_hw = [img_hw]
        device = torch.device("cpu")
        if preds and isinstance(preds[0], torch.Tensor):
            device = preds[0].device
        preds_t = [_as_tensor(p, device=device) for p in preds]
        targs_t = [_as_tensor(t, device=device) for t in targets]

        for p_i, t_i, hw in zip(preds_t, targs_t, img_hw):
            if t_i is None or t_i.numel() == 0:
                tcls = torch.zeros((0,), device=device)
                tbox = torch.zeros((0, 4), device=device)
            else:
                if labels_is_xywhn:
                    if t_i.size(-1) == 6:
                        tcls = t_i[:, 1]
                        xywhn = t_i[:, 2:]
                    elif t_i.size(-1) == 5:
                        tcls = t_i[:, 0]
                        xywhn = t_i[:, 1:]
                    else:
                        raise ValueError("Formato de labels no reconocido para XYWHN")
                    tbox = self._xywhn_to_xyxy_pix(xywhn, hw)
                else:
                    if t_i.size(-1) == 5:
                        tcls = t_i[:, 0]
                        tbox = t_i[:, 1:5]
                    else:
                        raise ValueError("Formato de labels no reconocido para XYXY")

            self.cm.conf = conf_min_for_cm
            self.cm.iou_thres = iou_match_for_cm
            self.cm.update(p_i if (p_i is not None and p_i.numel()) else None, tbox, tcls)

            if p_i is None or p_i.numel() == 0:
                self.stats_target_cls.append(tcls.cpu().numpy())
                continue

            p_i = p_i[p_i[:, 4].argsort(descending=True)]
            correct = np.zeros((p_i.shape[0], self.iouv.size), dtype=bool)

            if tbox.numel():
                ious = _box_iou_xyxy(tbox, p_i[:, :4])
                x = torch.where(ious > self.iouv.min() - 1e-9)
                if x[0].numel():
                    matches = torch.stack((x[0], x[1], ious[x[0], x[1]]), 1).cpu().numpy()
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    m_pred = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    m_gt = m_pred[np.unique(m_pred[:, 0], return_index=True)[1]]
                    if m_gt.size:
                        gt_idx = m_gt[:, 0].astype(int)
                        det_idx = m_gt[:, 1].astype(int)
                        iou_vals = m_gt[:, 2]
                        gt_cls = tcls.cpu().numpy().astype(int)
                        det_cls = p_i[:, 5].cpu().numpy().astype(int)
                        for g, d, iou_val in zip(gt_idx, det_idx, iou_vals):
                            if det_cls[d] != gt_cls[g]:
                                continue
                            correct[d, iou_val >= self.iouv] = True
                            self.matched_ious.append(float(iou_val))

            self.stats_tp.append(correct)
            self.stats_conf.append(p_i[:, 4].cpu().numpy())
            self.stats_pred_cls.append(p_i[:, 5].cpu().numpy().astype(int))
            self.stats_target_cls.append(tcls.cpu().numpy())

    def finalize(self) -> Tuple['DetMetricsSummary', PRCurves]:
        if len(self.stats_tp) == 0:
            empty_curves = PRCurves(px=np.linspace(0, 1, 1000), p_curve=np.zeros((0, 1000)),
                                     r_curve=np.zeros((0, 1000)), f1_curve=np.zeros((0, 1000)), ap=np.zeros((0, 10)))
            return DetMetricsSummary(0.0, 0.0, 0.0, 0.0, 0, 0, 0), empty_curves

        tp = np.concatenate(self.stats_tp, 0)
        conf = np.concatenate(self.stats_conf, 0)
        pred_cls = np.concatenate(self.stats_pred_cls, 0)
        target_cls = np.concatenate(self.stats_target_cls, 0)

        save_dir = self.save_dir / "pr_curves" if self.save_dir is not None else None
        curves, _ = _ap_per_class(tp, conf, pred_cls, target_cls, self.iouv, self.names,
                                  save_dir=save_dir, prefix="")

        summary = curves.summary()
        iou_mean = float(np.mean(self.matched_ious)) if len(self.matched_ious) else 0.0
        iou_median = float(np.median(self.matched_ious)) if len(self.matched_ious) else 0.0

        tp_sum = int(np.diag(self.cm.mat)[:-1].sum())
        fp_sum = int(self.cm.mat[:-1, -1].sum())
        fn_sum = int(self.cm.mat[-1, :-1].sum())

        det_summary = DetMetricsSummary(
            precision=float(summary["precision"]),
            recall=float(summary["recall"]),
            map50=float(summary["mAP50"]),
            map50_95=float(summary["mAP50_95"]),
            tp=tp_sum,
            fp=fp_sum,
            fn=fn_sum,
            iou_mean=iou_mean,
            iou_median=iou_median,
        )

        if self.save_dir is not None:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                self.cm.plot(self.save_dir / "confusion_matrix.png", self.names)
                if self.matched_ious:
                    fig, ax = plt.subplots(1, 1, figsize=(7, 5), tight_layout=True)
                    ax.hist(self.matched_ious, bins=20, range=(0, 1))
                    ax.set_title("IoU distribution (TPs)")
                    ax.set_xlabel("IoU"); ax.set_ylabel("count")
                    fig.savefig(self.save_dir / "iou_hist.png", dpi=200); plt.close(fig)
            except Exception as e:
                warnings.warn(f"No se pudo guardar la matriz/figuras de métricas: {e}")

        return det_summary, curves


# ==============================================================
# Escritura de métricas por época (CSV largo) — por slot
# ==============================================================

class MetricsWriter:
    def __init__(self, root: Path, variant: str, phase: str, run_name: str,
                 *, is_test: bool = False, reset_final: bool = False):
        self.root = Path(root)
        self.variant = str(variant).lower()
        self.phase = str(phase).lower()
        self.run_name = run_name
        # Guardar CSV en logs/ por slot (tests/<run_name> o final)
        self.logs_dir = _resolve_slot_dir(self.root, self.variant, self.phase,
                                          is_test=is_test, run_name=run_name,
                                          reset_final=reset_final, base="logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.logs_dir / ("train.csv" if self.phase == "train" else f"{self.phase}.csv")
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "split", "metric", "value"])  # formato largo

    def write_epoch(self, epoch: int, scalars: Dict[str, float]) -> None:
        rows = [(epoch, self.phase, k, float(v)) for k, v in sorted(scalars.items())]
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)


class TrainEpochTracker:
    def __init__(self):
        self.rm = RunningMean()

    def update(self, loss_scalars: Dict[str, float], batch_size: int = 1) -> None:
        self.rm.update(loss_scalars, inc=batch_size)

    def finalize(self) -> Dict[str, float]:
        m = self.rm.mean()
        return {
            "loss/total": m.get("loss", 0.0),
            "loss/box": m.get("loss_box", 0.0),
            "loss/cls": m.get("loss_cls", 0.0),
            "loss/dfl": m.get("loss_dfl", 0.0),
            "stats/num_pos": m.get("num_pos", 0.0),
        }


def summarize_validation(det_summary: DetMetricsSummary, loss_means: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    d = det_summary.to_dict()
    if loss_means is not None:
        d.update({
            "loss/total": loss_means.get("loss", 0.0),
            "loss/box": loss_means.get("loss_box", 0.0),
            "loss/cls": loss_means.get("loss_cls", 0.0),
            "loss/dfl": loss_means.get("loss_dfl", 0.0),
            "stats/num_pos": loss_means.get("num_pos", 0.0),
        })
    return d


# ==============================================================
# Curva de pérdida vs época (train)
# ==============================================================

def build_train_loss_curve(
    epochs: Iterable[int],
    losses: Iterable[float],
    output_path: Optional[Path] = None,
    *,
    variant: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Epochs",
    ylabel: str = "Loss",
    dpi: int = 200,
) -> Dict[str, Optional[Union[List[float], List[int], Path]]]:
    """Construye y opcionalmente guarda la curva de pérdida de entrenamiento.

    Parámetros
    ----------
    epochs:
        Secuencia de índices de época (0-based o 1-based, se usan tal cual en
        los datos devueltos). Para el gráfico, se desplazan a 1..N sólo con
        fines de visualización.
    losses:
        Pérdida promedio por época, en el mismo orden que ``epochs``.
    output_path:
        Ruta del archivo PNG a guardar. Si es ``None``, no se guarda imagen
        y la función sólo devuelve los datos normalizados.
    variant:
        Identificador de variante (por ejemplo "n", "s", "m", "l", "x"). Si se
        proporciona y ``title`` es None, se usará para construir un título
        técnico del tipo "Train Loss Variant: YOLOv11-s — per-image averaged".
    title:
        Título del gráfico. Si es None, se construye uno genérico o específico
        según ``variant``.

    Retorna
    -------
    dict
        Diccionario con claves:
        - ``"epochs"``: lista de épocas (int) tal como se recibieron.
        - ``"losses"``: lista de pérdidas (float).
        - ``"path"``: ruta del PNG generado (Path) o ``None`` si no se guardó.

    Nota
    ----
    Esta función es deliberadamente agnóstica del origen de los datos. El
    llamado típico será desde el loop de entrenamiento (Trainer) pasando las
    listas agregadas de pérdida por época, pero también puede usarse a partir
    de CSV externos si se preprocesan previamente.
    """

    # Normalización básica
    ep_list = [int(e) for e in epochs]
    loss_list = [float(l) for l in losses]

    if not ep_list or not loss_list or len(ep_list) != len(loss_list):
        # Datos insuficientes o inconsistentes: devolvemos solo el contenedor vacío
        return {"epochs": ep_list, "losses": loss_list, "path": None}

    png_path: Optional[Path] = None

    # Normalizar variante y preparar título final
    v_norm = (variant or "").strip()
    v_norm = v_norm.lower() if v_norm else ""
    if title is None:
        if v_norm:
            final_title = f"Train Loss Variant: YOLOv11-{v_norm} — per-image averaged"
        else:
            final_title = "Training Loss vs Epoch — per-image averaged"
    else:
        final_title = title

    # Épocas para el gráfico: desplazar a 1..N sólo para visualización
    plot_epochs = [int(e) + 1 for e in ep_list]

    if output_path is not None:
        png_path = Path(output_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
            ax.plot(plot_epochs, loss_list, label="Training Loss")
            ax.set_title(final_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Forzar ticks enteros en el eje X para que las épocas sean
            # interpretables (1, 2, 3, ..., N) sin subdivisiones decimales.
            try:
                xmin, xmax = min(plot_epochs), max(plot_epochs)
                ax.set_xticks(list(range(xmin, xmax + 1)))
            except Exception:
                # Si algo falla (por ejemplo, epochs no enteros), dejamos el
                # comportamiento por defecto de Matplotlib.
                pass

            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(png_path, dpi=dpi)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"No se pudo guardar la curva de pérdida: {e}")
            png_path = None

    return {"epochs": ep_list, "losses": loss_list, "path": png_path}


if __name__ == "__main__":
    print("metrics.py módulo — slots (tests/final) listos. Integrar con train.py y logger.py")
