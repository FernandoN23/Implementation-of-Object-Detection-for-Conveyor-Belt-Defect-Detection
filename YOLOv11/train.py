# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: train.py
# Entrenamiento YOLOv11 con mitigación MIOpen encapsulada en MIOpenMitigator
#==============================================================

from __future__ import annotations

import os
import sys
import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Resolver raíz del proyecto (asumir que este archivo vive en YOLOv11/)
THIS = Path(__file__).resolve().parent
if not (THIS / "configs").exists():
    if (THIS / "YOLOv11" / "configs").exists():
        os.chdir(THIS / "YOLOv11")
        THIS = Path.cwd()

# Imports del proyecto
sys.path.insert(0, str(THIS))
from models.parser_yaml import ConfigParserYaml   # type: ignore
from utility.losses import YOLOLoss               # type: ignore
from utility.metrics import DetMetricsYOLOv11     # type: ignore
from utility.data_loader import build_yolo_dataloader  # type: ignore
from utility.logger import ExperimentLogger             # type: ignore

# Visualización / TensorBoard (opcional)
try:
    from utility.visualization import TBRefOverlaySession, log_ref_session_epoch  # type: ignore
except Exception:
    TBRefOverlaySession, log_ref_session_epoch = None, None


# ============================== Utilidades básicas ============================== #

class Environment:
    @staticmethod
    def select_device(device: str = "auto") -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
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


# ============================== MIOpen Mitigator ============================== #

class MIOpenMitigator:
    # Encapsula mitigaciones frente a errores de MIOpen/SQLite en Windows (ROCm).
    def __init__(self, model: nn.Module, *, enable_bn_eval_fallback: bool = False,
                 swap_bn_to_gn: bool = False, gn_groups: int = 32, verbose: bool = True):
        self.model = model
        self.enable_bn_eval_fallback = bool(enable_bn_eval_fallback)
        self.swap_bn_to_gn = bool(swap_bn_to_gn)
        self.gn_groups = int(gn_groups)
        self.verbose = bool(verbose)

        self.applied_bn_eval: bool = False
        self.gn_replacements: int = 0

    @staticmethod
    def _set_bn_eval(module: nn.Module) -> int:
        # Coloca todas las capas BatchNorm en eval(). Devuelve cuántas afectadas.
        count = 0
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                count += 1
        return count

    @staticmethod
    def _replace_bn_with_gn(module: nn.Module, groups: int = 32, eps: float = 1e-5, affine: bool = True) -> int:
        # Reemplaza recursivamente BatchNorm2d por GroupNorm. Devuelve conteo de reemplazos.
        count = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                C = child.num_features
                g = max(1, min(groups, C))
                setattr(module, name, nn.GroupNorm(num_groups=g, num_channels=C, eps=eps, affine=affine))
                count += 1
            else:
                count += MIOpenMitigator._replace_bn_with_gn(child, groups=groups, eps=eps, affine=affine)
        return count

    def apply_startup(self) -> None:
        # Ejecuta medidas proactivas de mitigación antes de warm-up.
        if self.swap_bn_to_gn:
            n = self._replace_bn_with_gn(self.model, groups=self.gn_groups)
            self.gn_replacements = n
            if self.verbose:
                print(f"[MIOpenMitigator] BN->GN aplicado: {n} reemplazos.")
        if self.enable_bn_eval_fallback:
            affected = self._set_bn_eval(self.model)
            self.applied_bn_eval = True
            if self.verbose:
                print(f"[MIOpenMitigator] BN.eval() proactivo activado, capas afectadas: {affected}.")

    def on_exception(self, exc: Exception) -> bool:
        # Si ocurre excepción (típicamente en warm-up o backward), activar BN.eval() una vez.
        if (not self.applied_bn_eval) and self.enable_bn_eval_fallback:
            affected = self._set_bn_eval(self.model)
            self.applied_bn_eval = True
            if self.verbose:
                print(f"[MIOpenMitigator] BN.eval() activado por excepción ({type(exc).__name__}). Capas afectadas: {affected}.")
            return True
        return False

    def metrics(self) -> Dict[str, float]:
        return {
            "bn_eval": float(self.applied_bn_eval),
            "gn_replacements": float(self.gn_replacements),
        }


# ============================== Post-proceso detección ============================== #

class DetectionPostprocessor:
    @staticmethod
    def adapt_outputs_to_scores_boxes(out: Any, nc: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if out is None:
            return None, None
        if isinstance(out, dict):
            scores = out.get("scores", out.get("cls", None))
            boxes = out.get("boxes", out.get("xyxy", out.get("reg", None)))
            if scores is not None and boxes is not None:
                if isinstance(scores, (list, tuple)):
                    scores = torch.cat([s.flatten(2).permute(0, 2, 1) for s in scores], dim=1)
                elif isinstance(scores, torch.Tensor) and scores.ndim == 4:
                    scores = scores.flatten(2).permute(0, 2, 1).contiguous()
                if isinstance(boxes, (list, tuple)):
                    boxes = torch.cat([b.flatten(2).permute(0, 2, 1) for b in boxes], dim=1)
                elif isinstance(boxes, torch.Tensor) and boxes.ndim == 4:
                    B, Cb, H, W = boxes.shape
                    if Cb == 4:
                        boxes = boxes.flatten(2).permute(0, 2, 1).contiguous()
                    else:
                        return None, None
                return scores, boxes
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            scores, boxes = out[0], out[1]
            if isinstance(scores, (list, tuple)):
                scores = torch.cat([s.flatten(2).permute(0, 2, 1) for s in scores], dim=1)
            if isinstance(boxes, (list, tuple)):
                boxes = torch.cat([b.flatten(2).permute(0, 2, 1) for b in boxes], dim=1)
            return scores, boxes
        return None, None

    @staticmethod
    def _nms_xyxy(scores_1nC: torch.Tensor, boxes_1n4: torch.Tensor,
                  conf_thr: float = 0.25, iou_thr: float = 0.7, max_det: int = 300) -> torch.Tensor:
        from torchvision.ops import nms
        C = scores_1nC.shape[1]
        conf, cls = scores_1nC.max(dim=1)
        keep = conf >= conf_thr
        conf, cls, boxes = conf[keep], cls[keep], boxes_1n4[keep]
        out = []
        for c in range(C):
            m = (cls == c)
            if m.sum() == 0:
                continue
            idx = nms(boxes[m], conf[m], iou_thr)[:max_det]
            if idx.numel():
                merged = torch.cat([boxes[m][idx],
                                    conf[m][idx].unsqueeze(1),
                                    torch.full((idx.numel(), 1), float(c), device=boxes.device)], dim=1)
                out.append(merged)
        if not out:
            return boxes_1n4.new_zeros((0, 6))
        return torch.cat(out, dim=0)

    @classmethod
    def scores_boxes_to_dets(cls, scores: torch.Tensor, boxes: torch.Tensor,
                             conf_thr: float = 0.25, iou_thr: float = 0.7, max_det: int = 300) -> List[torch.Tensor]:
        B = scores.shape[0]
        outs: List[torch.Tensor] = []
        for i in range(B):
            outs.append(cls._nms_xyxy(scores[i], boxes[i], conf_thr, iou_thr, max_det))
        return outs


# ============================== EMA y Optimizador ============================== #

class EMAHelper:
    def __init__(self, model: nn.Module, enabled: bool = True, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.enabled = enabled
        self.decay = decay
        self.ema: Optional[nn.Module] = None
        if enabled:
            from copy import deepcopy
            self.ema = deepcopy(model).to(device if device is not None else next(model.parameters()).device).eval()
            for p in self.ema.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.ema is None:
            return
        for e_p, p in zip(self.ema.parameters(), model.parameters()):
            e_p.mul_(self.decay).add_(p, alpha=1.0 - self.decay)


class OptimizerFactory:
    @staticmethod
    def build(model: nn.Module, lr0: float, weight_decay: float, epochs: int, lrf: float) -> Tuple[optim.Optimizer, CosineAnnealingLR]:
        opt = optim.AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)
        sch = CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr0 * lrf)
        return opt, sch


# ============================== Overlay Manager ============================== #

class OverlayManager:
    def __init__(self, variant: str, run_name: str, imgsz: int, conf_thr: float, iou_thr: float,
                 device: torch.device, model: nn.Module, ema: Optional[EMAHelper]) -> None:
        self.variant = variant
        self.run_name = run_name
        self.imgsz = int(imgsz)
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.device = device
        self.model = model
        self.ema = ema

    def _dump_pivot_preds(self, epoch: int) -> Optional[str]:
        if TBRefOverlaySession is None or log_ref_session_epoch is None:
            return None
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
                im = Image.open(p).convert("RGB").resize((self.imgsz, self.imgsz))
                tensors.append(TF.to_tensor(im))
                names.append(p.name)
            if not tensors:
                return None

            batch = torch.stack(tensors, 0).to(self.device).float()
            mdl = self.ema.ema if (self.ema is not None and self.ema.ema is not None) else self.model
            mdl.eval()
            with torch.no_grad():
                try:
                    out = mdl(batch, decode=True, concat=True)
                except TypeError:
                    out = mdl(batch)
            scores, boxes = DetectionPostprocessor.adapt_outputs_to_scores_boxes(out, nc=999)
            if scores is None or boxes is None:
                return None
            dets = DetectionPostprocessor.scores_boxes_to_dets(scores, boxes,
                                                               conf_thr=self.conf_thr, iou_thr=self.iou_thr, max_det=50)
            W = float(self.imgsz); H = float(self.imgsz)
            result: Dict[str, Any] = {}
            for i, name in enumerate(names):
                arr = dets[i].detach().cpu().tolist() if i < len(dets) else []
                preds: List[Dict[str, Any]] = []
                for d in arr:
                    if len(d) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = d
                    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                    cx = x1 + 0.5 * w; cy = y1 + 0.5 * h
                    preds.append({"bbox_xywh": [cx/W, cy/H, w/W, h/H], "conf": float(conf), "cls": int(cls)})
                result[name] = preds
            out_dir = THIS / "logs" / self.variant / "train" / self.run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            jpath = out_dir / f"pivot_preds_epoch_{epoch:03d}.json"
            jpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(jpath)
        except Exception as e:
            print(f"[vis] dump preds omitido: {e}")
            return None

    def overlay_epoch0_gt(self, conf_thr: float) -> None:
        if TBRefOverlaySession is None or log_ref_session_epoch is None:
            return
        try:
            log_ref_session_epoch(
                variant=self.variant, split="train", run_name=self.run_name,
                dataset_base=None, epoch=0, pred_json=None,
                conf_thr=float(conf_thr), topk=5, nrow=3, size=(self.imgsz, self.imgsz)
            )
        except Exception as e:
            print(f"[vis] overlay(época 0) omitido: {e}")

    def overlay_epoch_preds(self, epoch: int) -> None:
        if TBRefOverlaySession is None or log_ref_session_epoch is None:
            return
        try:
            pj = self._dump_pivot_preds(epoch=epoch)
            log_ref_session_epoch(
                variant=self.variant, split="train", run_name=self.run_name,
                dataset_base=None, epoch=epoch, pred_json=pj,
                conf_thr=self.conf_thr, topk=10, nrow=3, size=(self.imgsz, self.imgsz)
            )
        except Exception as e:
            print(f"[vis] overlay(pred) omitido: {e}")


# ============================== Validador ============================== #

class Validator:
    def __init__(self, nc: int, imgsz: int, conf_thr: float, iou_thr: float, device: torch.device):
        self.nc = int(nc)
        self.imgsz = int(imgsz)
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.device = device

    def run(self, model: nn.Module, val_loader) -> Dict[str, float]:
        model.eval()
        m = DetMetricsYOLOv11(nc=self.nc)
        with torch.no_grad():
            for imgs, targets, meta in val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                try:
                    out = model(imgs, decode=True, concat=True)
                except TypeError:
                    out = model(imgs)
                scores, boxes = DetectionPostprocessor.adapt_outputs_to_scores_boxes(out, self.nc)
                if scores is None or boxes is None:
                    continue
                dets = DetectionPostprocessor.scores_boxes_to_dets(scores, boxes,
                                                                   conf_thr=self.conf_thr, iou_thr=self.iou_thr, max_det=300)
                preds_b = [d.cpu().numpy() for d in dets]
                gts = targets.cpu().numpy() if targets is not None else None
                m.add_batch(preds_b, gts, img_hw=(self.imgsz, self.imgsz), labels_is_xywhn=True)
        summary, _ = m.finalize()
        return {
            "precision": float(getattr(summary, "precision", 0.0)),
            "recall": float(getattr(summary, "recall", 0.0)),
            "mAP50": float(getattr(summary, "map50", 0.0)),
            "mAP50-95": float(getattr(summary, "map50_95", 0.0)),
        }


# ============================== Entrenamiento ============================== #

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
    swap_bn_to_gn: bool = False
    val_interval: int = 1
    overlay_every: int = 0
    pr_curves_every: int = 0
    cm_every: int = 0
    save_period: int = 10
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False
    verbosity: str = "v1"
    hud: str = "one"
    conf_thr: float = 0.25
    iou_thr: float = 0.70
    grad_accum: int = 1


class Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args

        Environment.seed_everything(args.seed, args.deterministic)
        self.device = Environment.select_device(args.device)
        self.imgsz = int(args.imgsz)

        self.cfg = ConfigParserYaml(project_root=str(THIS)).load()
        self.variant = str(args.variant)
        model = self.cfg.build_model(variant=self.variant)
        meta = {}
        if hasattr(self.cfg, "model_meta") and isinstance(self.cfg.model_meta, dict):
            meta = self.cfg.model_meta
        elif hasattr(model, "meta") and isinstance(getattr(model, "meta"), dict):
            meta = getattr(model, "meta")
        self.model = model.to(self.device)
        self.nc = int(meta.get("nc", getattr(model, "nc", 5)))
        self.reg_max = int(meta.get("reg_max", getattr(model, "reg_max", 16)))

        # Inicializar mitigador y aplicar medidas proactivas
        self.mio = MIOpenMitigator(self.model,
                                   enable_bn_eval_fallback=bool(args.bn_eval_fallback),
                                   swap_bn_to_gn=bool(args.swap_bn_to_gn),
                                   gn_groups=32,
                                   verbose=True)
        self.mio.apply_startup()

        self.optimizer, self.scheduler = OptimizerFactory.build(
            self.model, lr0=args.lr0, weight_decay=args.weight_decay, epochs=args.epochs, lrf=args.lrf
        )

        self.use_amp = bool(args.amp)
        self.autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        try:
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.device.type == 'cuda' else None

        self.ema = EMAHelper(self.model, enabled=bool(args.ema), decay=0.9999, device=self.device)

        self.criterion = YOLOLoss(nc=self.nc, reg_max=self.reg_max, strides=(8,16,32))
        self.criterion = self.criterion.to(self.device)
        self.metrics = DetMetricsYOLOv11(nc=self.nc)

        self.logger = ExperimentLogger(project_root=THIS, variant=self.variant, phase="train")
        self.run_name = getattr(self.logger, "run_name", "yolo11_run")

        self.weights_dir = THIS / "weights" / self.variant / "train" / self.run_name
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0

        self.overlay_mgr = OverlayManager(
            variant=self.variant, run_name=self.run_name, imgsz=self.imgsz,
            conf_thr=args.conf_thr, iou_thr=args.iou_thr,
            device=self.device, model=self.model, ema=self.ema
        )

    def warmup_if_needed(self) -> None:
        steps = max(0, int(self.args.warmup_steps))
        if steps <= 0:
            return
        self.model.train()
        x = torch.randn(1, 3, self.imgsz, self.imgsz, device=self.device)
        for _ in range(steps):
            try:
                with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
                    out = self.model(x)
                    loss = 0.0
                    if isinstance(out, torch.Tensor):
                        loss = out.float().mean()
                    elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
                        loss = out[0].float().mean()
                    else:
                        loss = sum(p.float().abs().mean() for p in self.model.parameters()) * 0.0
                if self.scaler is not None:
                    self.scaler.scale(loss).backward(retain_graph=True)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            except RuntimeError as e:
                # Activar BN.eval() una vez si procede y reintentar
                if self.mio.on_exception(e):
                    continue
                raise
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _train_step(self, imgs: torch.Tensor, targets: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        try:
            with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
                preds = self.model(imgs)
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict.get("loss", None)
                if loss is None:
                    raise RuntimeError("YOLOLoss no entregó 'loss' en el dict.")
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            parts = {
                "box": float(loss_dict.get("loss_box", 0.0)),
                "cls": float(loss_dict.get("loss_cls", 0.0)),
                "dfl": float(loss_dict.get("loss_dfl", 0.0)),
            }
            return loss.detach(), parts
        except RuntimeError as e:
            if self.mio.on_exception(e):
                with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
                    preds = self.model(imgs)
                    loss_dict = self.criterion(preds, targets)
                    loss = loss_dict.get("loss")
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                parts = {
                    "box": float(loss_dict.get("loss_box", 0.0)),
                    "cls": float(loss_dict.get("loss_cls", 0.0)),
                    "dfl": float(loss_dict.get("loss_dfl", 0.0)),
                }
                return loss.detach(), parts
            raise

    def _save_ckpt(self, epoch: int, tag: str) -> None:
        state = {
            "epoch": epoch,
            "model": (self.ema.ema.state_dict() if (self.ema is not None and self.ema.ema is not None) else self.model.state_dict()),
            "optimizer": self.optimizer.state_dict(),
            "scaler": (self.scaler.state_dict() if self.scaler is not None else None),
            "variant": self.variant,
            "imgsz": self.imgsz,
        }
        path = self.weights_dir / f"VAR_train_Epoch_{epoch:03d}_{tag}.pt"
        torch.save(state, path)

    def train(self) -> None:
        train_loader = build_yolo_dataloader(split="train", imgsz=self.imgsz, batch=int(self.args.batch), shuffle=True)
        val_loader = build_yolo_dataloader(split="val", imgsz=self.imgsz, batch=max(1, int(self.args.batch)//2), shuffle=False)

        self.warmup_if_needed()

        if (TBRefOverlaySession is not None) and (self.args.overlay_every and self.args.overlay_every > 0):
            self.overlay_mgr.overlay_epoch0_gt(conf_thr=self.args.conf_thr)

        global_step = 0
        for epoch in range(1, int(self.args.epochs) + 1):
            self.model.train()
            running_loss = 0.0
            running_parts = {"box": 0.0, "cls": 0.0, "dfl": 0.0}
            n_batches = 0

            self.optimizer.zero_grad(set_to_none=True)
            for bi, (imgs, targets, _meta) in enumerate(train_loader, start=1):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True) if targets is not None else None
                loss_t, parts = self._train_step(imgs, targets)

                if bi % max(1, int(self.args.grad_accum)) == 0:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss_t.detach().cpu())
                for k in running_parts:
                    running_parts[k] += float(parts.get(k, 0.0))
                n_batches += 1
                global_step += 1

                self.ema.update(self.model)

            loss_epoch = running_loss / max(1, n_batches)
            box_epoch  = running_parts["box"] / max(1, n_batches)
            cls_epoch  = running_parts["cls"] / max(1, n_batches)
            dfl_epoch  = running_parts["dfl"] / max(1, n_batches)

            valid_metrics: Dict[str, float] = {}
            if (self.args.val_interval > 0) and (epoch % self.args.val_interval == 0):
                validator = Validator(self.nc, self.imgsz, self.args.conf_thr, self.args.iou_thr, self.device)
                mdl_eval = self.ema.ema if (self.ema is not None and self.ema.ema is not None) else self.model
                valid_metrics = validator.run(mdl_eval, val_loader)

            try:
                # Métricas del mitigador
                mio_stats = self.mio.metrics()
                self.logger.log_epoch(epoch, {
                    "loss": float(loss_epoch),
                    "loss_box": float(box_epoch),
                    "loss_cls": float(cls_epoch),
                    "loss_dfl": float(dfl_epoch),
                    **mio_stats,
                }, split="train")
                if valid_metrics:
                    self.logger.log_epoch(epoch, {k: float(v) for k, v in valid_metrics.items()}, split="valid")
                if getattr(self.logger, "tb", None) is not None:
                    try:
                        self.logger.tb.flush()
                    except Exception:
                        pass
            except Exception as e:
                print(f="[logger] fallo al registrar métricas: {e}")

            if self.args.save_period and (epoch % int(self.args.save_period) == 0):
                self._save_ckpt(epoch, tag="period")
            if valid_metrics:
                cur = float(valid_metrics.get("mAP50-95", -1.0))
                if cur > self.best_metric:
                    self.best_metric = cur
                    self._save_ckpt(epoch, tag="best")
            self._save_ckpt(epoch, tag="last")

            if (TBRefOverlaySession is not None) and (self.args.overlay_every and self.args.overlay_every > 0) and (epoch % self.args.overlay_every == 0):
                self.overlay_mgr.overlay_epoch_preds(epoch)

            try:
                self.scheduler.step()
            except Exception:
                pass

        try:
            self.logger.close()
        except Exception:
            pass


# ================================ CLI ================================ #

def _as_dict(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    except Exception:
        return {}

def _train_cfg_dict(cfg) -> Dict[str, Any]:
    if hasattr(cfg, "train"):
        t = cfg.train
        if isinstance(t, dict):
            return t.get("config", t)
        if hasattr(t, "config") and isinstance(t.config, dict):
            return t.config
    return {}

def _runtime_dict(cfg) -> Dict[str, Any]:
    rt = getattr(cfg, "runtime", None)
    return _as_dict(rt)

def _save_dict(cfg) -> Dict[str, Any]:
    sv = getattr(cfg, "save", None)
    return _as_dict(sv)

def _req(d: Dict[str, Any], keys: Iterable[str], cast: Optional[Callable[[Any], Any]] = None, *, ctx: str = "train.yaml") -> Any:
    if isinstance(keys, str):
        keys = (keys,)
    for k in keys:
        if k in d:
            v = d[k]
            return cast(v) if cast else v
    ks = " | ".join(keys)
    raise KeyError(f"[{ctx}] Parámetro requerido ausente: {ks}")

def _to_int(v: Any) -> int:
    return int(round(float(v)))

def build_argparser() -> argparse.ArgumentParser:
    cfg = ConfigParserYaml(project_root=str(THIS)).load()
    tr = _train_cfg_dict(cfg)
    rt = _runtime_dict(cfg)
    sv = _save_dict(cfg)

    d_imgsz         = _req(tr, ("imgsz",), _to_int)
    d_epochs        = _req(tr, ("epochs",), _to_int)
    d_batch         = _req(tr, ("batch",), _to_int)
    d_lr0           = _req(tr, ("lr0",), float)
    d_lrf           = _req(tr, ("lrf",), float)
    d_wd            = _req(tr, ("weight_decay",), float)
    d_warmup_steps  = _req(tr, ("warmup_steps","warmup_epochs"), _to_int)
    d_amp           = _req(tr, ("amp",), bool)
    d_ema           = _req(tr, ("ema",), bool)
    d_grad_accum    = _req(tr, ("grad_accum","grad_accumulation"), _to_int)
    d_val_interval  = _req(tr, ("val_interval",), _to_int)
    d_overlay_every = _req(tr, ("overlay_every",), _to_int)
    d_pr_every      = _req(tr, ("pr_curves_every",), _to_int)
    d_cm_every      = _req(tr, ("cm_every",), _to_int)
    d_conf          = _req(tr, ("conf_thr","conf_thres"), float)
    d_iou           = _req(tr, ("iou_thr","iou_thres"), float)
    d_verbosity     = _req(tr, ("verbosity",), str)
    d_hud           = _req(tr, ("hud",), str)

    d_save_period   = int(sv["save_period"])
    d_device        = str(rt["device"])
    d_seed          = int(rt["seed"])
    d_det           = bool(rt["deterministic"])

    p = argparse.ArgumentParser(
        "YOLOv11 — Entrenamiento con MIOpenMitigator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", type=str,
                   default=getattr(cfg, "default_variant", getattr(cfg, "default_variant_name", "n")),
                   help="Variante del modelo (n/s/m/l/xl).")
    p.add_argument("--epochs", type=int, default=d_epochs, help="Número de épocas.")
    p.add_argument("--batch", type=int, default=d_batch, help="Tamaño de batch.")
    p.add_argument("--imgsz", type=int, default=d_imgsz, help="Tamaño de imagen.")
    p.add_argument("--lr0", type=float, default=d_lr0, help="LR inicial AdamW.")
    p.add_argument("--lrf", type=float, default=d_lrf, help="LR mínimo relativo en Cosine.")
    p.add_argument("--weight-decay", type=float, default=d_wd, help="Weight decay AdamW.")
    p.add_argument("--warmup-steps", type=int, default=d_warmup_steps, help="Pasos de warm-up.")

    p.add_argument("--amp", action=("store_false" if d_amp else "store_true"),
                   default=d_amp, help=f"Habilita AMP (YAML={d_amp}).")
    p.add_argument("--ema", action=("store_false" if d_ema else "store_true"),
                   default=d_ema, help=f"Habilita EMA (YAML={d_ema}).")
    p.add_argument("--bn-eval-fallback", action="store_true",
                   help="Pone BatchNorm en eval() proactivamente y si ocurren errores en warm-up/train (mitigación MIOpen).")
    p.add_argument("--swap-bn-to-gn", action="store_true",
                   help="Reemplaza BatchNorm2d por GroupNorm (32 grupos) para evitar MIOpen por completo.")

    p.add_argument("--val-interval", type=int, default=d_val_interval, help="Validación cada N épocas.")
    p.add_argument("--overlay-every", type=int, default=d_overlay_every, help="Overlays TB cada N épocas.")
    p.add_argument("--pr-curves-every", type=int, default=d_pr_every, help="Curvas P-R cada N épocas.")
    p.add_argument("--cm-every", type=int, default=d_cm_every, help="Matriz de confusión cada N épocas.")
    p.add_argument("--save-period", type=int, default=d_save_period, help="Checkpoint periódico.")
    p.add_argument("--grad-accum", type=int, default=d_grad_accum, help="Acumulación de gradiente.")
    p.add_argument("--device", type=str, default=d_device, help="Dispositivo.")
    p.add_argument("--seed", type=int, default=d_seed, help="Semilla.")
    p.add_argument("--deterministic", action=("store_false" if d_det else "store_true"),
                   default=d_det, help=f"CUDA determinista (YAML={d_det}).")
    p.add_argument("--verbosity", type=str, default=d_verbosity, choices=["v0", "v1", "v2"], help="Nivel de verbosidad del HUD/log.")
    p.add_argument("--hud", type=str, default=d_hud, choices=["off", "one"], help="Modo del HUD.")
    p.add_argument("--conf-thr", type=float, default=d_conf, help="Confianza mínima NMS.")
    p.add_argument("--iou-thr", type=float, default=d_iou, help="IoU para NMS.")
    return p

def _resolve_bool_flag(ns_value: bool) -> bool:
    return bool(ns_value)

def main() -> None:
    parser = build_argparser()
    args_ns = parser.parse_args()

    amp_final = _resolve_bool_flag(args_ns.amp)
    ema_final = _resolve_bool_flag(args_ns.ema)
    det_final = _resolve_bool_flag(args_ns.deterministic)

    args = TrainArgs(
        variant=args_ns.variant,
        epochs=args_ns.epochs,
        batch=args_ns.batch,
        imgsz=args_ns.imgsz,
        lr0=args_ns.lr0,
        lrf=args_ns.lrf,
        weight_decay=args_ns.weight_decay,
        warmup_steps=args_ns.warmup_steps,
        amp=amp_final,
        ema=ema_final,
        bn_eval_fallback=bool(args_ns.bn_eval_fallback),
        swap_bn_to_gn=bool(args_ns.swap_bn_to_gn),
        val_interval=args_ns.val_interval,
        overlay_every=args_ns.overlay_every,
        pr_curves_every=args_ns.pr_curves_every,
        cm_every=args_ns.cm_every,
        save_period=args_ns.save_period,
        device=args_ns.device,
        seed=args_ns.seed,
        deterministic=det_final,
        verbosity=args_ns.verbosity,
        hud=args_ns.hud,
        conf_thr=args_ns.conf_thr,
        iou_thr=args_ns.iou_thr,
        grad_accum=args_ns.grad_accum,
    )
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
