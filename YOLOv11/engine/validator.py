# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/validator.py
# Descripción: Bucle de validación/evaluación para detección. Ejecuta
#              inferencia, NMS, emparejamiento pred–GT y calcula
#              métricas (P/R, mAP@0.5, mAP@[.5:.95], F1, AR, fitness).
#              Se integra con utility/metrics.py para AP por clase.
#==============================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torchvision.ops import nms as tv_nms  # preferir si está
except Exception:  # pragma: no cover
    tv_nms = None  # type: ignore

# Integra utilidades del proyecto (AP por clase, IoU, etc.)
try:
    from utility.metrics import ap_per_class  # type: ignore
except Exception:  # fallback mínimo si el módulo no está disponible
    ap_per_class = None  # type: ignore

__all__ = ["ValConfig", "Validator", "validate"]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class ValConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.6
    max_det: int = 300
    agnostic_nms: bool = False
    save_json: bool = False
    save_dir: Optional[str] = None
    names: Optional[List[str]] = None  # nombres de clases
    nc: Optional[int] = None           # número de clases
    device: str = "auto"
    imgsz: int = 640
    plots: bool = False
    verbose: int = 1

    # Para cómputo de métricas
    map_iou_lo: float = 0.5
    map_iou_hi: float = 0.95
    map_iou_step: float = 0.05


# -------------------------------
# Utilidades
# -------------------------------

def _log(msg: str, cfg: Optional[ValConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[validator] {msg}")


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
            return torch.device("hip")
        return torch.device("cpu")
    return torch.device(spec)


def _xywhn_to_xyxy(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
    # x: [N, 4] en formato xywh normalizado (0-1)
    xy = x.clone()
    xy[:, 0] = (x[:, 0] - x[:, 2] / 2) * w
    xy[:, 1] = (x[:, 1] - x[:, 3] / 2) * h
    xy[:, 2] = (x[:, 0] + x[:, 2] / 2) * w
    xy[:, 3] = (x[:, 1] + x[:, 3] / 2) * h
    return xy


def _box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # box1: [N,4], box2: [M,4]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def _nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    # Implementación sencilla por si no existe torchvision.ops.nms
    keep: List[int] = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.numel() == 1:
            break
        ious = _box_iou_xyxy(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if tv_nms is not None:
        return tv_nms(boxes, scores, iou_thres)
    return _nms_pytorch(boxes, scores, iou_thres)


# -------------------------------
# Validator
# -------------------------------

class Validator:
    """Validador estilo Ultralytics: acumula TP/FP y calcula AP/mAP."""

    def __init__(self, cfg: Optional[ValConfig] = None) -> None:
        self.cfg = cfg or ValConfig()
        self.device = _select_device(self.cfg.device)
        self.stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.seen: int = 0
        self.save_dir: Optional[Path] = Path(self.cfg.save_dir) if self.cfg.save_dir else None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    # ---------- Modelo/inferencia ----------
    @torch.inference_mode()
    def _model_predict(self, model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
        """Devuelve detecciones por imagen en formato [x1,y1,x2,y2,conf,cls]."""
        model.eval()
        dev = images.device
        # Intentos progresivos para compatibilidad
        # 1) Método predict habitual
        if hasattr(model, "predict"):
            out = model.predict(images)
        else:
            # 2) Intentar forward directo
            try:
                out = model(images)
            except Exception:
                # 3) Intentar contrato tipo batch dict
                out = model({"img": images})
        # Normalizar a lista por imagen
        if isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
            preds = out
        elif isinstance(out, torch.Tensor):
            preds = [o for o in out]
        else:
            raise RuntimeError("Salida de predicción no reconocida por Validator")
        # Aplicar NMS + truncado por imagen
        results: List[torch.Tensor] = []
        for p in preds:
            if p.ndim == 2 and p.size(-1) >= 6:
                boxes_xyxy = p[:, :4]
                scores = p[:, 4]
                classes = p[:, 5].to(boxes_xyxy.dtype)
            else:
                # asumir formato [cx,cy,w,h,conf,cls]
                cxcywh = p[:, :4]
                x1y1 = cxcywh[:, :2] - cxcywh[:, 2:] / 2
                x2y2 = cxcywh[:, :2] + cxcywh[:, 2:] / 2
                boxes_xyxy = torch.cat([x1y1, x2y2], 1)
                scores = p[:, 4]
                classes = p[:, 5].to(boxes_xyxy.dtype)
            keep = _nms(boxes_xyxy, scores, self.cfg.iou_thres)
            det = torch.cat([boxes_xyxy[keep], scores[keep, None], classes[keep, None]], 1)
            if det.numel() and self.cfg.max_det > 0:
                det = det[: self.cfg.max_det]
            results.append(det.to(dev))
        return results

    # ---------- Acumulación de métricas ----------
    def _process_batch(self, detections: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Empareja predicciones con etiquetas por IoU y devuelve (tp, conf, pred_cls, target_cls).
        - detections: [N,6] (x1,y1,x2,y2,conf,cls)
        - labels: [M,5] (cls, x1,y1,x2,y2)
        """
        if detections.numel() == 0:
            return (torch.zeros(0), torch.zeros(0), torch.zeros(0), labels[:, 0])

        iou = _box_iou_xyxy(detections[:, :4], labels[:, 1:])  # [N,M]
        # Para cada etiqueta, escoger la pred con IoU máximo por clase coincidente
        correct = torch.zeros((detections.size(0),), dtype=torch.bool, device=detections.device)
        detected = []
        tcls = labels[:, 0]
        for j, lab in enumerate(labels):
            ti = int(lab[0].item())
            # filtrar por clase
            same_cls = (detections[:, 5].round().long() == ti)
            iou_j = iou[:, j] * same_cls.float()
            iou_max, imax = iou_j.max(0)
            if iou_max >= self.cfg.iou_thres and imax.item() not in detected:
                correct[imax] = True
                detected.append(int(imax))
        return correct.cpu(), detections[:, 4].cpu(), detections[:, 5].cpu(), tcls.cpu()

    # ---------- Loop principal ----------
    @torch.inference_mode()
    def validate(self,
                 model: nn.Module,
                 loader: Iterable[Dict[str, Any]],
                 *,
                 names: Optional[List[str]] = None) -> Dict[str, Any]:
        dev = self.device
        names = names or self.cfg.names or []

        self.stats.clear()
        self.seen = 0

        for batch in loader:
            # Estructura flexible del batch (compatibilidad con utility/data_loader)
            if isinstance(batch, dict) and "img" in batch:
                imgs = batch["img"]
                targets_list = batch.get("targets", [])
                paths = batch.get("paths", None)
            else:
                # fallback: suponer (imgs, targets)
                imgs, targets_list = batch
                paths = None

            imgs = imgs.to(dev, non_blocking=True).float()

            # Inferencia + NMS
            dets = self._model_predict(model, imgs)

            # Preparar GT por imagen
            bs, _, H, W = imgs.shape
            for i in range(bs):
                self.seen += 1
                targets_i = targets_list[i] if isinstance(targets_list, list) else targets_list
                if targets_i is None or len(targets_i) == 0:
                    labels = torch.zeros((0, 5), device=imgs.device)
                else:
                    # targets_i formato esperado: [cls, cx, cy, w, h] normalizado
                    if targets_i.size(-1) == 5:
                        xyxy = _xywhn_to_xyxy(targets_i[:, 1:], W, H)
                        cls = targets_i[:, 0:1].to(xyxy.dtype)
                        labels = torch.cat([cls, xyxy], 1)
                    else:
                        # si ya viene en xyxy absol.
                        labels = targets_i
                det = dets[i]
                tp, conf, pred_cls, tcls = self._process_batch(det, labels)
                self.stats.append((tp, conf, pred_cls, tcls))

        # Agregar métricas
        metrics = self._compute_metrics(names)

        # Guardado opcional JSON resumido
        if self.save_dir and self.cfg.save_json:
            out = {"metrics": metrics, "config": asdict(self.cfg)}
            p = Path(self.save_dir) / "val_metrics.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            _log(f"Métricas guardadas en {p}", self.cfg, 1)

        return metrics

    # ---------- Cálculo de métricas agregadas ----------
    def _compute_metrics(self, names: List[str]) -> Dict[str, Any]:
        if len(self.stats) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map5095": 0.0,
                "f1": 0.0,
                "seen": self.seen,
                "fitness": 0.0,
            }
        tp, conf, pred_cls, tcls = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]

        if ap_per_class is None:
            # Fallback: estimaciones mínimas (sin AP real)
            # precision/recall aproximadas
            eps = 1e-9
            precision = float((tp > 0).sum()) / float(len(tp) + eps)
            recall = precision  # sin info de GT por clase no podemos estimar recall real
            map50 = precision
            map5095 = precision
            f1 = 2 * precision * recall / (precision + recall + eps)
        else:
            p, r, ap, f1, ap_class = ap_per_class(
                tp,
                conf,
                pred_cls,
                tcls,
                iouv=torch.arange(self.cfg.map_iou_lo, self.cfg.map_iou_hi + 1e-9, self.cfg.map_iou_step),
            )
            precision = float(p.mean()) if len(p) else 0.0
            recall = float(r.mean()) if len(r) else 0.0
            ap = ap if hasattr(ap, "mean") else ap
            map50 = float(ap[:, 0].mean()) if hasattr(ap, "__getitem__") else 0.0
            map5095 = float(ap.mean()) if hasattr(ap, "mean") else 0.0
            f1 = float(f1.mean()) if hasattr(f1, "mean") else 0.0

        # Fitness estilo Ultralytics (pondera mAP50-95)
        fitness = 0.1 * map50 + 0.9 * map5095

        metrics = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "map50": round(map50, 6),
            "map5095": round(map5095, 6),
            "f1": round(f1, 6),
            "seen": int(self.seen),
            "fitness": round(fitness, 6),
        }
        _log(
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} mAP50={metrics['map50']:.4f} mAP50-95={metrics['map5095']:.4f} F1={metrics['f1']:.4f}",
            self.cfg,
            1,
        )
        return metrics


# -------------------------------
# Función de conveniencia
# -------------------------------

def validate(model: nn.Module,
             loader: Iterable[Dict[str, Any]],
             names: Optional[List[str]] = None,
             *,
             save_dir: Optional[str] = None,
             conf_thres: float = 0.25,
             iou_thres: float = 0.6,
             max_det: int = 300,
             agnostic_nms: bool = False,
             device: str = "auto",
             plots: bool = False,
             save_json: bool = False) -> Dict[str, Any]:
    cfg = ValConfig(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        device=device,
        plots=plots,
        save_json=save_json,
        save_dir=save_dir,
        names=names,
    )
    v = Validator(cfg)
    return v.validate(model, loader, names=names)
