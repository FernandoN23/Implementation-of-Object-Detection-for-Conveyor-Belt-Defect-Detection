# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/overlays.py
# Descripción: Generación de superposiciones (pred vs GT) periódicas
#              durante el entrenamiento para control visual y QA.
#              Selecciona muestras pivote, corre inferencia y
#              guarda imágenes anotadas en runs/<exp>/overlays/.
#==============================================================

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Opcional: usar visualizador del proyecto si está disponible
try:
    from utility.visualization import draw_detections, draw_targets  # type: ignore
except Exception:
    draw_detections = None  # type: ignore
    draw_targets = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

__all__ = ["OverlayConfig", "OverlaysManager", "save_overlays"]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class OverlayConfig:
    save_dir: str
    interval: int = 10          # ejecutar cada k épocas
    num_samples: int = 8        # imágenes pivote por ejecución
    conf_thres: float = 0.25
    iou_thres: float = 0.6
    max_det: int = 300
    names: Optional[List[str]] = None
    device: str = "auto"
    seed: int = 0
    verbose: int = 1


# -------------------------------
# Utilidades
# -------------------------------

def _log(msg: str, cfg: Optional[OverlayConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[overlays] {msg}")


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
            return torch.device("hip")
        return torch.device("cpu")
    return torch.device(spec)


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    # t: [C,H,W] en [0,1] o [0,255]; devuelve uint8 BGR para OpenCV
    t = t.detach().float().cpu()
    if t.ndim == 3 and t.size(0) in (1, 3):
        if t.size(0) == 1:
            t = t.repeat(3, 1, 1)
        img = (t.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).numpy()
    elif t.ndim == 3 and t.size(-1) in (1, 3):
        img = t.byte().numpy()
    else:
        # fallback: intentar reshape
        arr = t.reshape(-1).numpy()
        side = int(np.sqrt(arr.size // 3) * 3)
        img = arr[: side * side].reshape(side, side, 3).astype(np.uint8)
    if cv2 is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _xywhn_to_xyxy(x: np.ndarray, w: int, h: int) -> np.ndarray:
    xy = x.copy()
    xy[:, 0] = (x[:, 0] - x[:, 2] / 2) * w
    xy[:, 1] = (x[:, 1] - x[:, 3] / 2) * h
    xy[:, 2] = (x[:, 0] + x[:, 2] / 2) * w
    xy[:, 3] = (x[:, 1] + x[:, 3] / 2) * h
    return xy


def _draw_boxes_cv2(img: np.ndarray,
                    boxes: np.ndarray,
                    labels: Sequence[str],
                    color: Tuple[int, int, int]) -> np.ndarray:
    if cv2 is None:
        return img
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2, conf, cls = b
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        txt = f"{labels[int(cls)] if 0 <= int(cls) < len(labels) else int(cls)} {conf:.2f}"
        cv2.putText(out, txt, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# -------------------------------
# Núcleo de overlays
# -------------------------------

class OverlaysManager:
    """Genera overlays pred vs GT a intervalos regulares de épocas."""

    def __init__(self, cfg: OverlayConfig) -> None:
        self.cfg = cfg
        self.device = _select_device(cfg.device)
        self.save_root = Path(cfg.save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.seed)

    @torch.inference_mode()
    def _predict(self, model: nn.Module, imgs: torch.Tensor) -> List[torch.Tensor]:
        model.eval()
        dev = imgs.device
        if hasattr(model, "predict"):
            out = model.predict(imgs)
        else:
            try:
                out = model(imgs)
            except Exception:
                out = model({"img": imgs})
        # Normalizar a lista
        if isinstance(out, torch.Tensor):
            preds = [o for o in out]
        else:
            preds = list(out)
        # Limitar por max_det si se puede
        results: List[torch.Tensor] = []
        for p in preds:
            if p.ndim == 2 and p.size(-1) >= 6:
                det = p
            else:
                # asumir cxcywh + conf + cls
                cxcywh = p[:, :4]
                x1y1 = cxcywh[:, :2] - cxcywh[:, 2:] / 2
                x2y2 = cxcywh[:, :2] + cxcywh[:, 2:] / 2
                det = torch.cat([x1y1, x2y2, p[:, 4:6]], 1)
            if self.cfg.max_det > 0 and det.numel():
                det = det[: self.cfg.max_det]
            results.append(det.to(dev))
        return results

    def _pick_indices(self, n: int, k: int) -> List[int]:
        if k >= n:
            return list(range(n))
        return sorted(random.sample(range(n), k))

    def _save_one(self,
                  img_t: torch.Tensor,
                  det: torch.Tensor,
                  tgt: Optional[torch.Tensor],
                  names: List[str],
                  out_path: Path) -> None:
        img = _to_numpy_image(img_t)
        H, W = img.shape[:2]

        # Dibujar GT
        if tgt is not None and tgt.numel():
            if tgt.size(-1) == 5:  # [cls, cx, cy, w, h] normalizado
                t = tgt.detach().float().cpu().numpy()
                cls = t[:, 0:1]
                xyxy = _xywhn_to_xyxy(t[:, 1:], W, H)
                gt = np.concatenate([xyxy, np.ones((xyxy.shape[0], 1)), cls], 1)
            else:
                gt = tgt.detach().float().cpu().numpy()
            if draw_targets is not None:
                img = draw_targets(img, gt, names)  # type: ignore
            else:
                img = _draw_boxes_cv2(img, gt, names, (0, 200, 255))  # naranja GT

        # Dibujar pred
        if det is not None and det.numel():
            pr = det.detach().float().cpu().numpy()
            if draw_detections is not None:
                img = draw_detections(img, pr, names)  # type: ignore
            else:
                img = _draw_boxes_cv2(img, pr, names, (50, 220, 50))  # verde pred

        # Guardar
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            cv2.imwrite(str(out_path), img)
        else:
            # fallback numpy -> .npy
            np.save(str(out_path.with_suffix(".npy")), img)

    @torch.inference_mode()
    def run(self,
            epoch: int,
            model: nn.Module,
            loader: Iterable[Dict[str, Any]],
            *,
            names: Optional[List[str]] = None) -> Optional[str]:
        """Genera overlays si epoch coincide con el intervalo.

        Retorna el path de la carpeta generada o None si se omite por intervalo.
        """
        if self.cfg.interval <= 0:
            return None
        if epoch % self.cfg.interval != 0:
            return None

        names = names or self.cfg.names or []

        out_dir = self.save_root / f"epoch_{epoch:03d}"
        _log(f"Generando overlays en {out_dir}", self.cfg, 1)

        # Tomar un batch y seleccionar n muestras
        try:
            batch = next(iter(loader))
        except StopIteration:
            return None

        imgs = batch["img"] if isinstance(batch, dict) else batch[0]
        tgts = batch.get("targets", []) if isinstance(batch, dict) else batch[1]
        imgs = imgs.to(_select_device(self.cfg.device), non_blocking=True).float()

        preds = self._predict(model, imgs)

        bs = imgs.size(0)
        idxs = self._pick_indices(bs, self.cfg.num_samples)

        for j, i in enumerate(idxs):
            img_t = imgs[i]
            det = preds[i]
            tgt = tgts[i] if isinstance(tgts, list) else tgts
            out_path = out_dir / f"overlay_{j:02d}.jpg"
            self._save_one(img_t, det, tgt, names, out_path)

        return str(out_dir)


# -------------------------------
# Función de conveniencia
# -------------------------------

def save_overlays(epoch: int,
                  model: nn.Module,
                  loader: Iterable[Dict[str, Any]],
                  save_dir: str,
                  *,
                  interval: int = 10,
                  num_samples: int = 8,
                  names: Optional[List[str]] = None,
                  device: str = "auto") -> Optional[str]:
    cfg = OverlayConfig(save_dir=save_dir, interval=interval, num_samples=num_samples, names=names, device=device)
    mgr = OverlaysManager(cfg)
    return mgr.run(epoch, model, loader, names=names)
