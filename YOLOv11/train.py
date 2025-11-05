
# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: train.py
# Script principal de entrenamiento para YOLOv11 con registro completo
# en TensorBoard (scalars + overlays). Incluye warm‑up, AMP/EMA opcional,
# BN‑eval fallback (mitigación ROCm/MIOpen), validación periódica y
# guardado de checkpoints mediante ExperimentLogger.
#==============================================================

from __future__ import annotations

import os
import sys
import math
import time
import json
import types
import shutil
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Resolver raíz del proyecto (asumir que este archivo vive en YOLOv11/) ---
THIS = Path(__file__).resolve().parent
if not (THIS / "configs").exists():
    # permitir ejecutar desde la raíz del repo
    if (THIS / "YOLOv11" / "configs").exists():
        os.chdir(THIS / "YOLOv11")
        THIS = Path.cwd()

# --- Imports del proyecto ---
sys.path.insert(0, str(THIS))
from models.parser_yaml import ConfigParserYaml   # type: ignore
from utility.losses import YOLOLoss, LossHyperparams  # type: ignore
from utility.metrics import DetMetricsYOLOv11        # type: ignore
from utility.data_loader import build_yolo_dataloader # type: ignore
from utility.logger import ExperimentLogger           # type: ignore

# Visualización / TensorBoard
try:
    from utility.visualization import TBRefOverlaySession, log_ref_session_epoch  # type: ignore
except Exception:  # noqa: E722
    TBRefOverlaySession, log_ref_session_epoch = None, None

# ========================= Utilidades generales ========================= #

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

# ======== Heurísticas para adaptar salidas del modelo a (scores, boxes) ======== #
def adapt_outputs_to_scores_boxes(out: Any, nc: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Adapta la salida del modelo a (scores [B,N,C], boxes [B,N,4 xyxy]).
    Se soportan varios formatos comunes.
    """
    if out is None:
        return None, None
    # dict con claves habituales
    if isinstance(out, dict):
        scores = out.get("scores", out.get("cls", None))
        boxes = out.get("boxes", out.get("xyxy", out.get("reg", None)))
        if scores is not None and boxes is not None:
            if scores.ndim == 4:  # (B,C,H,W) -> (B,N,C)
                B, C, H, W = scores.shape
                scores = scores.flatten(2).permute(0, 2, 1).contiguous()
            if boxes.ndim == 4:   # (B,4*R,H,W) or (B,4,H,W) -> approx xyxy via max bins (DFL ya decodificado en 'boxes' suele ser xyxy)
                B, Cb, H, W = boxes.shape
                if Cb == 4:
                    boxes = boxes.flatten(2).permute(0, 2, 1).contiguous()
                else:
                    # Si viene DFL sin decodificar, aquí no sabemos mapear; devolvemos None
                    pass
            return scores, boxes
    # tupla/lista (scores, boxes)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        scores, boxes = out[0], out[1]
        # normalizar dimensiones si vienen por niveles
        if isinstance(scores, (list, tuple)):
            scores = torch.cat([s.flatten(2).permute(0, 2, 1) for s in scores], dim=1)
        if isinstance(boxes, (list, tuple)):
            boxes = torch.cat([b.flatten(2).permute(0, 2, 1) for b in boxes], dim=1)
        return scores, boxes
    # tensor simple -> no soportado aquí
    return None, None

def nms_xyxy(scores: torch.Tensor, boxes: torch.Tensor, conf_thr: float = 0.25,
             iou_thr: float = 0.7, max_det: int = 300) -> List[torch.Tensor]:
    """
    scores: (N,C)   boxes: (N,4) en xyxy (pix o norm homogéneo)
    Devuelve lista con [x1,y1,x2,y2,conf,cls] ordenados por conf.
    """
    from torchvision.ops import nms
    C = scores.shape[1]
    conf, cls = scores.max(dim=1)
    keep = conf >= conf_thr
    conf = conf[keep]
    cls = cls[keep]
    boxes = boxes[keep]

    # NMS por clase
    out = []
    for c in range(C):
        mask = (cls == c)
        if mask.sum() == 0:
            continue
        b = boxes[mask]
        s = conf[mask]
        idx = nms(b, s, iou_thr)
        idx = idx[:max_det]
        if idx.numel():
            merged = torch.cat([b[idx], s[idx].unsqueeze(1), torch.full((idx.numel(),1), float(c))], dim=1)
            out.append(merged)
    if not out:
        return [torch.zeros((0,6), dtype=boxes.dtype)]
    return [torch.cat(out, dim=0)]

def scores_boxes_to_dets(scores: torch.Tensor, boxes: torch.Tensor,
                         conf_thr: float = 0.25, iou_thr: float = 0.7, max_det: int = 300) -> List[torch.Tensor]:
    """Batched NMS. scores [B,N,C], boxes [B,N,4] -> list de [M_i,6]."""
    B = scores.shape[0]
    outs: List[torch.Tensor] = []
    for i in range(B):
        outs.append(nms_xyxy(scores[i], boxes[i], conf_thr=conf_thr, iou_thr=iou_thr, max_det=max_det)[0])
    return outs

# ========================= Entrenador ========================= #

@dataclass
class TrainArgs:
    variant: str = "n"
    epochs: int = 1
    batch: int = 2
    imgsz: int = 640
    lr0: float = 1.5e-3
    lrf: float = 0.2
    weight_decay: float = 0.01
    warmup_steps: int = 3
    amp: bool = True
    ema: bool = True
    bn_eval_fallback: bool = False
    val_interval: int = 1
    overlay_every: int = 0
    pr_curves_every: int = 0
    cm_every: int = 0
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False
    verbosity: str = "v1"
    hud: str = "one"
    conf_thr: float = 0.25
    iou_thr: float = 0.70

class Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        seed_everything(args.seed, args.deterministic)
        self.device = select_device() if args.device == "auto" else torch.device(args.device)
        self.imgsz = int(args.imgsz)
        self.epochs = int(args.epochs)
        self.batch = int(args.batch)
        self.overlay_every = int(args.overlay_every or 0)
        self.pr_curves_every = int(args.pr_curves_every or 0)
        self.cm_every = int(args.cm_every or 0)
        # Configs y modelo
        self.cfg = ConfigParserYaml(project_root=str(THIS)).load()
        self.variant_safe = str(args.variant)
        model, meta = self.cfg.build_model(variant=self.variant_safe)
        self.model = model.to(self.device)
        self.nc = int(meta.get("nc", 5))
        # Optimización
        self.amp_enabled = bool(args.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, self.epochs), eta_min=args.lr0*args.lrf)
        # EMA opcional
        self.ema = None
        if args.ema:
            try:
                from copy import deepcopy
                ema_model = deepcopy(self.model).to(self.device).eval()
                for p in ema_model.parameters(): p.requires_grad_(False)
                self.ema = types.SimpleNamespace(ema=ema_model, decay=0.9999)
            except Exception:
                self.ema = None
        # Pérdidas y métricas
        self.criterion = YOLOLoss(reg_max=int(meta.get("reg_max", 16)))
        self.metrics = DetMetricsYOLOv11(num_classes=self.nc)
        # Logger
        self.logger = ExperimentLogger(variant=self.variant_safe, phase="train")
        self.run_name = self.logger.run_name

    # ----------------- Predicciones de pivote para overlays ----------------- #
    def _dump_pivot_preds(self, epoch: int) -> Optional[str]:
        """
        Ejecuta inferencia sobre imágenes pivote del split TRAIN y guarda un JSON compatible
        con log_ref_session_epoch para dibujar GT+Pred en TB.
        """
        try:
            from PIL import Image
            import torchvision.transforms.functional as TF
            from utility.visualization import TRAIN_PIVOT_IMAGES, DEFAULT_DATASET_BASE  # type: ignore
            base = Path(DEFAULT_DATASET_BASE)
            paths = [(base / "train" / "images" / n) for n in TRAIN_PIVOT_IMAGES]
            paths = [p for p in paths if p.exists()]
            if not paths:
                return None
            tensors, names = [], []
            for p in paths:
                try:
                    im = Image.open(p).convert("RGB").resize((self.imgsz, self.imgsz))
                    tensors.append(TF.to_tensor(im))
                    names.append(p.name)
                except Exception:
                    continue
            if not tensors:
                return None
            batch = torch.stack(tensors, 0).to(self.device).float()
            mdl = self.ema.ema if self.ema is not None else self.model
            mdl.eval()
            with torch.no_grad():
                try:
                    out = mdl(batch, decode=True, concat=True)
                except TypeError:
                    out = mdl(batch)
            scores, boxes = adapt_outputs_to_scores_boxes(out, self.nc)
            if scores is None or boxes is None:
                return None
            dets = scores_boxes_to_dets(scores, boxes,
                                        conf_thr=float(self.args.conf_thr),
                                        iou_thr=float(self.args.iou_thr),
                                        max_det=50)
            W = float(self.imgsz); H = float(self.imgsz)
            result: Dict[str, Any] = {}
            for i, name in enumerate(names):
                preds: List[Dict[str, Any]] = []
                arr = dets[i].detach().cpu().tolist() if i < len(dets) else []
                for d in arr:
                    if len(d) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = d
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    cx = x1 + 0.5 * w
                    cy = y1 + 0.5 * h
                    preds.append({"bbox_xywh": [cx/W, cy/H, w/W, h/H], "conf": float(conf), "cls": int(cls)})
                result[name] = preds
            out_dir = THIS / "logs" / self.variant_safe / "train" / self.run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            jpath = out_dir / f"pivot_preds_epoch_{epoch:03d}.json"
            jpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(jpath)
        except Exception as e:
            print(f"[vis] dump preds omitido: {e}")
            return None

    # ----------------- Warm‑up y entrenamiento ----------------- #
    def warmup_if_needed(self) -> None:
        steps = max(0, int(self.args.warmup_steps))
        if steps <= 0:
            return
        self.model.train()
        B = 1
        x = torch.randn(B, 3, self.imgsz, self.imgsz, device=self.device)
        for i in range(steps):
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                out = self.model(x)
                loss = (sum(p.sum() for p in self.model.parameters())*0.0) + (out[0].mean() if isinstance(out, (list,tuple)) else 0.0)
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer); self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize() if self.device.type == "cuda" else None

    def train(self) -> None:
        # Dataloaders
        train_loader = build_yolo_dataloader(split="train", imgsz=self.imgsz, batch=self.batch, shuffle=True)
        val_loader = build_yolo_dataloader(split="val", imgsz=self.imgsz, batch=max(1, self.batch//2), shuffle=False)

        # Warm‑up
        self.warmup_if_needed()

        # Overlay época 0 (solo GT) — split 'train'
        if (TBRefOverlaySession is not None) and (self.overlay_every > 0):
            try:
                log_ref_session_epoch(
                    variant=self.variant_safe, split="train", run_name=self.run_name,
                    dataset_base=None, epoch=0, pred_json=None,
                    conf_thr=float(self.args.conf_thr), topk=5, nrow=3, size=(self.imgsz, self.imgsz)
                )
            except Exception as e:
                print(f"[vis] overlay(época 0) omitido: {e}")

        # Bucle de épocas
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
            n_batches = 0

            for imgs, targets, _meta in train_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True) if targets is not None else None
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    preds = self.model(imgs)
                    loss, parts = self.criterion(preds, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer); self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # EMA
                if self.ema is not None:
                    with torch.no_grad():
                        for e_p, p in zip(self.ema.ema.parameters(), self.model.parameters()):
                            e_p.mul_(self.ema.decay).add_(p, alpha=1.0 - self.ema.decay)
                # acumular
                running["loss"] += float(loss.detach().cpu())
                running["box"] += float(parts.get("box", 0.0))
                running["cls"] += float(parts.get("cls", 0.0))
                running["dfl"] += float(parts.get("dfl", 0.0))
                n_batches += 1

            # Promedios de entrenamiento
            loss_epoch = running["loss"] / max(1, n_batches)
            box_epoch  = running["box"]  / max(1, n_batches)
            cls_epoch  = running["cls"]  / max(1, n_batches)
            dfl_epoch  = running["dfl"]  / max(1, n_batches)

            # Validación (cada val_interval)
            val_metrics: Dict[str, Optional[float]] = {"precision": None, "recall": None, "mAP50": None, "mAP50-95": None}
            if (self.args.val_interval > 0) and (epoch % self.args.val_interval == 0):
                self.model.eval()
                m = DetMetricsYOLOv11(num_classes=self.nc)
                with torch.no_grad():
                    for imgs, targets, meta in val_loader:
                        imgs = imgs.to(self.device)
                        out = self.model(imgs, decode=True, concat=True)
                        scores, boxes = adapt_outputs_to_scores_boxes(out, self.nc)
                        if scores is None or boxes is None:
                            continue
                        dets = scores_boxes_to_dets(scores, boxes,
                                                    conf_thr=float(self.args.conf_thr),
                                                    iou_thr=float(self.args.iou_thr),
                                                    max_det=300)
                        # Convertir a lista para la métrica: [x1,y1,x2,y2,conf,cls]
                        preds_b = [d.cpu().numpy() for d in dets]
                        # meta["labels"] esperado como XYWH normalizado por imagen; aquí el DataLoader ya lo entrega
                        gts = targets.cpu().numpy() if targets is not None else None
                        m.add_batch(preds_b, gts, imgsz=self.imgsz)
                res = m.finalize(save_dir=None)
                val_metrics.update({
                    "precision": float(res.metrics["precision"]),
                    "recall": float(res.metrics["recall"]),
                    "mAP50": float(res.metrics["mAP@0.50"]),
                    "mAP50-95": float(res.metrics["mAP@0.50-0.95"]),
                })

            # === Logging a TensorBoard (train + valid) ===
            train_metrics = {
                "loss": float(loss_epoch),
                "loss_box": float(box_epoch),
                "loss_cls": float(cls_epoch),
                "loss_dfl": float(dfl_epoch),
            }
            try:
                self.logger.log_epoch(epoch, train_metrics, split="train")
                if getattr(self.logger, "tb", None) is not None:
                    try: self.logger.tb.flush()
                    except Exception: pass
            except Exception as e:
                print(f"[logger] fallo al registrar train scalars: {e}")

            valid_metrics = {k: float(v) for k, v in val_metrics.items() if v is not None}
            if valid_metrics:
                try:
                    self.logger.log_epoch(epoch, valid_metrics, split="valid")
                    if getattr(self.logger, "tb", None) is not None:
                        try: self.logger.tb.flush()
                        except Exception: pass
                except Exception as e:
                    print(f"[logger] fallo al registrar valid scalars: {e}")

            # Overlays con predicciones (train) según frecuencia
            if (TBRefOverlaySession is not None) and (self.overlay_every > 0) and (epoch % self.overlay_every == 0):
                try:
                    pj = self._dump_pivot_preds(epoch=epoch)
                    log_ref_session_epoch(
                        variant=self.variant_safe, split="train", run_name=self.run_name,
                        dataset_base=None, epoch=epoch, pred_json=pj,
                        conf_thr=float(self.args.conf_thr), topk=10, nrow=3, size=(self.imgsz, self.imgsz)
                    )
                except Exception as e:
                    print(f"[vis] overlay(pred) omitido: {e}")

            # Scheduler step por época
            try: self.scheduler.step()
            except Exception: pass

        # Cierre
        try: self.logger.close()
        except Exception: pass

# =============================== CLI =============================== #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("YOLOv11 — Entrenamiento")
    p.add_argument("--variant", type=str, default="n")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=1.5e-3)
    p.add_argument("--lrf", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--bn-eval-fallback", action="store_true")
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--overlay-every", type=int, default=0)
    p.add_argument("--pr-curves-every", type=int, default=0)
    p.add_argument("--cm-every", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--verbosity", type=str, default="v1")
    p.add_argument("--hud", type=str, default="one")
    p.add_argument("--conf-thr", type=float, default=0.25)
    p.add_argument("--iou-thr", type=float, default=0.70)
    return p

def main() -> None:
    args_ns = build_argparser().parse_args()
    args = TrainArgs(
        variant=args_ns.variant,
        epochs=args_ns.epochs,
        batch=args_ns.batch,
        imgsz=args_ns.imgsz,
        lr0=args_ns.lr0,
        lrf=args_ns.lrf,
        weight_decay=args_ns.weight_decay,
        warmup_steps=args_ns.warmup_steps,
        amp=bool(args_ns.amp),
        ema=bool(args_ns.ema),
        bn_eval_fallback=bool(args_ns.bn_eval_fallback),
        val_interval=args_ns.val_interval,
        overlay_every=args_ns.overlay_every,
        pr_curves_every=args_ns.pr_curves_every,
        cm_every=args_ns.cm_every,
        device=args_ns.device,
        seed=args_ns.seed,
        deterministic=bool(args_ns.deterministic),
        verbosity=args_ns.verbosity,
        hud=args_ns.hud,
        conf_thr=args_ns.conf_thr,
        iou_thr=args_ns.iou_thr,
    )
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
