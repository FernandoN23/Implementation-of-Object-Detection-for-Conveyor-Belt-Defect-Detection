# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: train.py
# Script principal de entrenamiento para YOLOv11. Integra: parser de
# configuraciones, construcción de modelo, dataloaders, bucle de
# entrenamiento con HUD de consola compacto, validación interna
# espaciable, early-stopping, guardado de checkpoints (best/last/periodic),
# AMP/EMA opcionales, métricas (mAP, P/R), y salida a TensorBoard.
#==============================================================

from __future__ import annotations

import os
import sys
import math
import json
import time
import argparse
import datetime as dt
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple, Dict, Any, List

# --- Ajuste de ruta del proyecto (ejecutar desde la raíz del repo) ---
PROJ_MARKERS = ("configs", "models", "utility")
THIS = os.path.abspath(os.path.dirname(__file__))
if not all(os.path.exists(os.path.join(THIS, m)) for m in PROJ_MARKERS):
    cand = os.path.join(THIS, "YOLOv11")
    if all(os.path.exists(os.path.join(cand, m)) for m in PROJ_MARKERS):
        sys.path.insert(0, cand)
        os.chdir(cand)
    else:
        sys.path.insert(0, THIS)
else:
    sys.path.insert(0, THIS)

# --- Silencio de logs/avisos molestos (antes de importar torch) ---
import warnings
os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
# Mitigación: forzar búsqueda de kernels (evita DB SQLite corrupta)
os.environ.setdefault("MIOPEN_FIND_ENFORCE", "3")
warnings.filterwarnings("ignore", message=r".*Cannot set number of intraop threads.*")
warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*before `optimizer\.step\(\)`.*")
# ANSI en Windows (PowerShell/PyCharm) para soporte de \x1b[1A, \x1b[2K
try:
    import colorama  # type: ignore
    colorama.just_fix_windows_console()
except Exception:  # noqa: E722
    pass

# --- Importes de terceros ---
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    from torchvision.ops import nms  # para NMS en validación
except Exception:  # noqa: E722
    nms = None

# Opción de teclas (F8) para salida elegante
try:
    import keyboard  # type: ignore
    _HAS_KEYBOARD = True
except Exception:  # noqa: E722
    _HAS_KEYBOARD = False

# --- Importes del proyecto ---
from models.parser_yaml import ConfigParserYaml  # orquesta configs y construye modelo
from utility.data_loader import build_yolo_dataloader
from utility.losses import YOLOLoss, LossHyperparams
from utility.metrics import DetMetricsYOLOv11
from utility.logger import ExperimentLogger
from utility.weights import WeightsManager

# Visualización/TensorBoard (opcional, tolerante a fallos de API)
try:
    from utility.visualization import TBRefOverlaySession, log_ref_session_epoch  # type: ignore
except Exception:  # noqa: E722
    TBRefOverlaySession, log_ref_session_epoch = None, None


# ========================= Helpers generales ========================= #

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic


def select_device(pref: str | None = None) -> torch.device:
    if pref and pref.lower() in ("cpu", "cuda", "mps"):
        if pref.lower() == "cpu":
            return torch.device("cpu")
        if pref.lower() == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
    # Auto: prioriza CUDA/ROCm, luego MPS, luego CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    """Acceso robusto a objetos o dicts (dataclass/objetos admitidos)."""
    try:
        if isinstance(meta, dict):
            return meta.get(key, default)
        return getattr(meta, key, default)
    except Exception:
        return default


def _to_dict_like(meta: Any) -> Dict[str, Any]:
    """Convierte un objeto (incluido dataclass) en un dict sencillo.
    - dict -> tal cual
    - dataclass -> asdict
    - objeto genérico -> __dict__ / atributos públicos no llamables
    """
    if isinstance(meta, dict):
        return meta
    try:
        if is_dataclass(meta):
            return asdict(meta)
    except Exception:
        pass
    try:
        if hasattr(meta, "__dict__"):
            return dict(meta.__dict__)
        # fallback: recorrer dir()
        out = {}
        for k in dir(meta):
            if k.startswith("_"):
                continue
            try:
                v = getattr(meta, k)
            except Exception:
                continue
            if callable(v):
                continue
            out[k] = v
        return out
    except Exception:
        return {}


class GracefulStopper:
    """Detección de parada elegante por F8 / Ctrl+C."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._requested = False
        self._last_prompt_ts = 0.0

    def arm(self) -> None:
        self._requested = True

    def poll(self) -> bool:
        if not self.enabled:
            return False
        # Ctrl+C será manejado por KeyboardInterrupt fuera
        if _HAS_KEYBOARD:
            try:
                if keyboard.is_pressed("f8"):
                    now = time.time()
                    if now - self._last_prompt_ts < 2.0:
                        self.arm()
                    else:
                        print("[F8] detectado. Presiona F8 nuevamente (<2s) para confirmar salida segura…", flush=True)
                        self._last_prompt_ts = now
            except Exception:
                pass
        return self._requested


# ========================= Programación del LR ========================= #

def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer,
                                  epochs: int,
                                  steps_per_epoch: int,
                                  lr0: float,
                                  lrf: float = 0.1,
                                  warmup_epochs: float = 3.0) -> LambdaLR:
    """LambdaLR: warmup lineal + decaimiento coseno. lrf=factor mínimo (lr_min=lr0*lrf)."""
    total_steps = max(1, int(epochs * steps_per_epoch))
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_min = lr0 * lrf

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return lr_min / lr0 + 0.5 * (1.0 - lr_min / lr0) * (1.0 + math.cos(math.pi * t))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ========================= EMA sencilla opcional ========================= #
class ModelEMA:
    """EMA ligero de parámetros del modelo."""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.ema = self._clone_model(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

# --- Mitigación ROCm/MIOpen: BN en eval() ---

def apply_bn_eval(module: nn.Module, verbose: bool = True) -> int:
    count = 0
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
            m.eval()
            count += 1
    if verbose:
        print(f"[BN-EVAL] Fallback activado: {count} capas BatchNorm en eval().")
    return count


# ========================= Post-proceso (validación) ========================= #
@torch.no_grad()
def scores_boxes_to_dets(scores: torch.Tensor,
                         boxes: torch.Tensor,
                         conf_thr: float = 0.25,
                         iou_thr: float = 0.7,
                         max_det: int = 300) -> List[torch.Tensor]:
    """Convierte (B,N,nc) y (B,N,4) en listas de [x1,y1,x2,y2,conf,cls] por imagen."""
    B, N, nc = scores.shape
    if scores.min() < 0 or scores.max() > 1:
        scores = scores.sigmoid()

    out: List[torch.Tensor] = []
    for b in range(B):
        sb = scores[b]      # (N,nc)
        bb = boxes[b]       # (N,4)
        conf, cls = sb.max(dim=1)
        mask = conf >= conf_thr
        if mask.any():
            bb = bb[mask]
            conf = conf[mask]
            cls = cls[mask]
        else:
            out.append(torch.zeros((0, 6), device=scores.device))
            continue

        if nms is not None and bb.numel() > 0:
            keep = nms(bb, conf, iou_thr)[:max_det]
            bb, conf, cls = bb[keep], conf[keep], cls[keep]
        else:
            k = min(max_det, conf.numel())
            topk = conf.topk(k).indices
            bb, conf, cls = bb[topk], conf[topk], cls[topk]

        det = torch.cat([bb, conf[:, None], cls[:, None].float()], dim=1)
        out.append(det)
    return out


# ========================= Adaptador robusto de salidas del modelo ========================= #
@torch.no_grad()
def _flatten_hw_to_NC(t: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W)->(B,H*W,C) ; (B,4,H,W)->(B,H*W,4). Si ya es (B,N,C) retorna igual."""
    if t.dim() == 4:
        return t.permute(0, 2, 3, 1).contiguous().view(t.size(0), -1, t.size(1))
    if t.dim() == 3:
        # Heurística: si es (B,C,N) -> (B,N,C)
        if t.size(1) in (4, 5, 6, 8) and t.size(2) > t.size(1):
            return t.permute(0, 2, 1).contiguous()
        return t
    raise ValueError("Tensor con dimensionalidad no soportada para _flatten_hw_to_NC")


def _cat_levels_to_BNC(seq: List[torch.Tensor]) -> torch.Tensor:
    parts = [_flatten_hw_to_NC(x) for x in seq]
    # Concat por N (dim=1), asumiendo mismo B y C
    return torch.cat(parts, dim=1)


@torch.no_grad()

def adapt_outputs_to_scores_boxes(outputs: Any, nc: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Intenta extraer (scores[B,N,nc], boxes[B,N,4]) desde múltiples formatos posibles."""
    # 1) Dict con 'scores'/'boxes'
    if isinstance(outputs, dict):
        s = outputs.get("scores", None)
        b = outputs.get("boxes", None)
        if s is not None and b is not None:
            # Manejar listas por nivel
            if isinstance(s, (list, tuple)):
                s = _cat_levels_to_BNC(list(s))
            else:
                s = _flatten_hw_to_NC(s)
            if isinstance(b, (list, tuple)):
                b = _cat_levels_to_BNC(list(b))
            else:
                b = _flatten_hw_to_NC(b)
            # Garantizar shape final
            if s.dim() == 3 and s.size(-1) == nc and b.dim() == 3 and b.size(-1) == 4:
                return s, b
        # 1-bis) Dict con 'cls'/'boxes' (común en nuestros heads decode=True)
        s = outputs.get("cls", None)
        b = outputs.get("boxes", None)
        if s is not None and b is not None:
            if isinstance(s, (list, tuple)):
                s = _cat_levels_to_BNC(list(s))
            else:
                s = _flatten_hw_to_NC(s)
            # (B,nc,N) -> (B,N,nc) si aplica
            if s.dim() == 3 and s.size(-1) != nc and s.size(1) == nc:
                s = s.permute(0, 2, 1).contiguous()
            if isinstance(b, (list, tuple)):
                b = _cat_levels_to_BNC(list(b))
            else:
                b = _flatten_hw_to_NC(b)
            if s.dim() == 3 and s.size(-1) == nc and b.dim() == 3 and b.size(-1) == 4:
                # Aplicar sigmoid a cls si no está en [0,1]
                if (s.min() < 0) or (s.max() > 1):
                    s = s.sigmoid()
                return s, b
        # 2) Dict con 'cls'/'reg'
        s = outputs.get("cls", None)
        r = outputs.get("reg", None)
        if s is not None and r is not None:
            if isinstance(s, (list, tuple)):
                s = _cat_levels_to_BNC(list(s))
            else:
                s = _flatten_hw_to_NC(s)
            if s.size(-1) != nc and s.size(1) == nc:
                # (B,nc,N) -> (B,N,nc)
                s = s.permute(0, 2, 1).contiguous()
            if isinstance(r, (list, tuple)):
                r = _cat_levels_to_BNC(list(r))
            else:
                r = _flatten_hw_to_NC(r)
            if s.dim() == 3 and r.dim() == 3 and r.size(-1) in (4,):
                if (s.min() < 0) or (s.max() > 1):
                    s = s.sigmoid()
                return s, r
    # 3) Tuple/List de 2 tensores -> heurística por canal final 4 vs nc
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2 and all(torch.is_tensor(x) for x in outputs):
        a, b = outputs
        a = _flatten_hw_to_NC(a)
        b = _flatten_hw_to_NC(b)
        if a.size(-1) == nc and b.size(-1) == 4:
            return a, b
        if b.size(-1) == nc and a.size(-1) == 4:
            return b, a
        # Si alguno es (B,N,nc+4)
        if a.size(-1) == nc + 4:
            return a[..., :nc], a[..., nc:]
        if b.size(-1) == nc + 4:
            return b[..., :nc], b[..., nc:]
    # 4) Tensor único (B,N,nc+4) o (B,C,H,W) combinados
    if torch.is_tensor(outputs):
        t = _flatten_hw_to_NC(outputs)
        if t.size(-1) == nc + 4:
            return t[..., :nc], t[..., nc:]
    # 5) Lista de tensores por nivel [ (B,nc,Hi,Wi), (B,nc,Hj,Wj), ... ]
    if isinstance(outputs, (list, tuple)) and all(torch.is_tensor(x) for x in outputs):
        t = _cat_levels_to_BNC(list(outputs))
        if t.size(-1) == nc + 4:
            return t[..., :nc], t[..., nc:]
    return None, None


# ========================= Interfaz / CLI ========================= #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="YOLOv11 — Entrenamiento",
        description="Entrena variantes YOLOv11 con validación interna, HUD compacto, F8 stop, AMP/EMA opcionales.",
    )
    p.add_argument("--variant", type=str, default=None, help="n/s/m/l/xl o dejar None para usar default del parser.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    # Frecuencias
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--pr-curves-every", type=int, default=10)
    p.add_argument("--cm-every", type=int, default=10)
    p.add_argument("--overlay-every", type=int, default=10)
    # Checkpoints / early stop
    p.add_argument("--save-period", type=int, default=10)
    p.add_argument("--keep-checkpoint-max", type=int, default=5)
    p.add_argument("--patience", type=int, default=50, help="en unidades de validación (se multiplica por val_interval)")
    # Umbrales inferencia validación
    p.add_argument("--conf-thr", type=float, default=0.25)
    p.add_argument("--iou-thr", type=float, default=0.70)
    # Reanudar
    p.add_argument("--resume", type=str, default=None, help="ruta a last.pt")
    # Verbosidad HUD
    p.add_argument("--verbosity", type=str, choices=["v0", "v1", "v2"], default="v1")
    # HUD (one/two/off)
    p.add_argument("--hud", type=str, choices=["one", "two", "off"], default="two")
    # Wizard interactivo
    p.add_argument("--interactive", action="store_true", help="Inicia asistente interactivo antes de entrenar")
    # Mitigación ROCm/MIOpen
    p.add_argument("--bn-eval-fallback", action="store_true", help="Fuerza BatchNorm en eval() desde el inicio para evitar fallos MIOpen")
    return p


def interactive_wizard(cfg: Any, args: argparse.Namespace) -> argparse.Namespace:
    """Asistente simple: permite sobreescribir algunos parámetros leídos de configs."""
    def ask_int(prompt: str, default: Optional[int]) -> Optional[int]:
        s = input(f"{prompt} [{default}]: ").strip()
        return int(s) if s else default

    def ask_str(prompt: str, default: Optional[str]) -> Optional[str]:
        s = input(f"{prompt} [{default}]: ").strip()
        return s if s else default

    train_section = _to_dict_like(getattr(cfg, "train", {}))
    tr = _to_dict_like(train_section.get("config", {}))

    print("=== Asistente interactivo de entrenamiento ===")
    args.variant = ask_str("Variante (n/s/m/l/xl)", args.variant or getattr(cfg, "default_variant", None))
    args.imgsz = ask_int("Tamaño de imagen (imgsz)", args.imgsz or tr.get("imgsz", 640))
    args.epochs = ask_int("Épocas", args.epochs or tr.get("epochs", 150))
    args.batch = ask_int("Batch", args.batch or tr.get("batch", 16))
    args.val_interval = ask_int("Validación cada N épocas", args.val_interval)
    args.save_period = ask_int("Guardar checkpoint cada N épocas", args.save_period)
    args.keep_checkpoint_max = ask_int("Máx. checkpoints a mantener", args.keep_checkpoint_max)
    args.patience = ask_int("Paciencia (en validaciones)", args.patience)
    args.verbosity = ask_str("Verbosity (v0/v1/v2)", args.verbosity)
    print("=== Fin asistente ===")
    return args


# ========================= HUD consola ========================= #
class ConsoleHUD:
    def __init__(self, verbosity: str = "v1"):
        self.verbosity = verbosity
        self._last_len1 = 0
        self._last_len2 = 0

    @staticmethod
    def _bar(frac: float, width: int = 10) -> str:
        k = max(0, min(width, int(round(frac * width))))
        return "▰" * k + "▱" * (width - k)

    def live(self, epoch: int, epochs: int, it: int, it_total: int,
             elapsed_s: float, eta_s: float, imgs_per_s: float,
             lr: float, amp: bool, ema: bool, grad_accum: int,
             loss_avg: float, loss_box: float, loss_cls: float, loss_dfl: float,
             mem_gb: float, device_str: str) -> None:
        if self.verbosity == "v0":
            return
        frac = it / max(1, it_total)
        line1 = (f"[EPOCH {epoch}/{epochs}] {self._bar(frac)}  {int(frac*100):2d}% | "
                 f"it {it}/{it_total} | {time.strftime('%H:%M:%S', time.gmtime(elapsed_s))} ⏱ | "
                 f"ETA {time.strftime('%H:%M:%S', time.gmtime(max(0, int(eta_s))))} | "
                 f"imgs/s {imgs_per_s:.0f} | lr {lr:.2e} | AMP {'on' if amp else 'off'} | EMA{'✓' if ema else '×'} | F8=stop")
        line2 = (f"train: loss {loss_avg:.3f} (box {loss_box:.2f}, cls {loss_cls:.2f}, dfl {loss_dfl:.2f}) | "
                 f"grad_acc {grad_accum}x | mem {mem_gb:.1f}GB | dev: {device_str}")
        sys.stdout.write("\r" + line1 + ("\n" if self._last_len2 else ""))
        sys.stdout.write(("\r\x1b[1A\x1b[2K" if self._last_len2 else "") + line2 + "\n")
        sys.stdout.flush()
        self._last_len1, self._last_len2 = len(line1), len(line2)

    def epoch_summary(self, text: str) -> None:
        if self.verbosity != "v0":
            sys.stdout.write("\x1b[2K\r")
        print(text, flush=True)
        self._last_len1 = self._last_len2 = 0


# ========================= HUD compacto (one-line / two-line) ========================= #
class SimpleHUD:
    def __init__(self, verbosity: str = "v1", mode: str = "two"):
        self.verbosity = verbosity
        self.mode = mode  # 'one' una línea, 'two' dos líneas, 'off' desactivar
        self._last_len = 0
        self._ansi_ok = sys.stdout.isatty()

    @staticmethod
    def _bar(frac: float, width: int = 10) -> str:
        k = max(0, min(width, int(round(frac * width))))
        return "▰" * k + "▱" * (width - k)

    def live(self, epoch: int, epochs: int, it: int, it_total: int,
             elapsed_s: float, eta_s: float, imgs_per_s: float,
             lr: float, amp: bool, ema: bool, grad_accum: int,
             loss_avg: float, loss_box: float, loss_cls: float, loss_dfl: float,
             mem_gb: float, device_str: str) -> None:
        if self.verbosity == "v0" or self.mode == "off":
            return
        frac = it / max(1, it_total)
        if self.mode == "one" or not self._ansi_ok:
            line = (f"[EPOCH {epoch}/{epochs}] {self._bar(frac)} {int(frac*100):2d}% | it {it}/{it_total} | "
                    f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_s))} ⏱ | ETA {time.strftime('%H:%M:%S', time.gmtime(max(0,int(eta_s))))} | "
                    f"imgs/s {imgs_per_s:.0f} | lr {lr:.2e} | AMP {'on' if amp else 'off'} | EMA{'✓' if ema else '×'} | "
                    f"loss {loss_avg:.3f} (b {loss_box:.2f}, c {loss_cls:.2f}, d {loss_dfl:.2f}) | mem {mem_gb:.1f}GB | {device_str}")
            pad = max(0, self._last_len - len(line))
            sys.stdout.write("\r" + line + (" " * pad))
            sys.stdout.flush()
            self._last_len = len(line)
            return
        # two-line con ANSI (requiere soporte en terminal)
        line1 = (f"[EPOCH {epoch}/{epochs}] {self._bar(frac)}  {int(frac*100):2d}% | "
                 f"it {it}/{it_total} | {time.strftime('%H:%M:%S', time.gmtime(elapsed_s))} ⏱ | "
                 f"ETA {time.strftime('%H:%M:%S', time.gmtime(max(0, int(eta_s))))} | "
                 f"imgs/s {imgs_per_s:.0f} | lr {lr:.2e} | AMP {'on' if amp else 'off'} | EMA{'✓' if ema else '×'} | F8=stop")
        line2 = (f"train: loss {loss_avg:.3f} (box {loss_box:.2f}, cls {loss_cls:.2f}, dfl {loss_dfl:.2f}) | "
                 f"grad_acc {grad_accum}x | mem {mem_gb:.1f}GB | dev: {device_str}")
        sys.stdout.write("\r" + line1 + "\n")
        sys.stdout.write("\r\x1b[1A\x1b[2K" + line2 + "\n")
        sys.stdout.flush()
        self._last_len = len(line2)

    def epoch_summary(self, text: str) -> None:
        if self.verbosity != "v0":
            sys.stdout.write("\n")
        print(text, flush=True)
        self._last_len = 0

# ========================= Bucle de entrenamiento ========================= #

def train_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 1) Cargar configs
    cfg = ConfigParserYaml().load()
    if args.interactive:
        args = interactive_wizard(cfg, args)

    # 2) Resolver secciones robustamente (soporta dict/objeto/dataclass)
    variant = args.variant or getattr(cfg, "default_variant", None)

    train_section = _to_dict_like(getattr(cfg, "train", {}))
    tr_cfg = _to_dict_like(train_section.get("config", {}))

    imgsz_for_strides = args.imgsz or tr_cfg.get("imgsz", 640)
    model = cfg.build_model(variant=variant, imgsz_for_strides=imgsz_for_strides)

    # Metadata del modelo
    model_meta = getattr(cfg, "model_meta", {})
    nc = _meta_get(model_meta, "nc", 5)
    reg_max = _meta_get(model_meta, "reg_max", 16)
    strides = _meta_get(model_meta, "strides", [8, 16, 32])
    if isinstance(strides, torch.Tensor):
        strides = strides.detach().cpu().tolist()

    # 3) Runtime / device / seed
    runtime_dict = _to_dict_like(getattr(cfg, "runtime", {}))
    device = select_device(args.device or runtime_dict.get("device", None))
    seed_everything(int(runtime_dict.get("seed", 42)), bool(runtime_dict.get("deterministic", False)))
    compile_flag = bool(runtime_dict.get("compile", False))

    model.to(device)
    if compile_flag and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            print("[warn] torch.compile falló; continuando sin compile().")
    # BN eval fallback opcional desde el arranque
    bn_eval_active = False
    if args.bn_eval_fallback:
        apply_bn_eval(model, verbose=True)
        bn_eval_active = True

    # 4) Dataloaders
    imgsz = args.imgsz or tr_cfg.get("imgsz", 640)
    epochs = args.epochs or tr_cfg.get("epochs", 150)
    batch = args.batch or tr_cfg.get("batch", 16)

    dl_cfg = _to_dict_like(tr_cfg.get("dataloader", {}))
    workers = int(dl_cfg.get("workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent = bool(dl_cfg.get("persistent_workers", True))

    train_loader = build_yolo_dataloader("train", imgsz=imgsz, batch=batch,
                                         workers=workers, pin_memory=pin_memory,
                                         persistent_workers=persistent, shuffle=True)
    val_loader = build_yolo_dataloader("val", imgsz=imgsz, batch=batch,
                                       workers=workers, pin_memory=pin_memory,
                                       persistent_workers=persistent, shuffle=False)

    steps_per_epoch = max(1, len(train_loader))

    # 5) Pérdida, optimizador, scheduler, AMP, EMA
    loss_weights = _to_dict_like(tr_cfg.get("loss weights", {"box": 7.5, "cls": 0.5, "dfl": 1.5}))
    hyp = LossHyperparams(
        box=float(loss_weights.get("box", 7.5)),
        cls=float(loss_weights.get("cls", 0.5)),
        dfl=float(loss_weights.get("dfl", 1.5)),
    )
    loss_fn = YOLOLoss(
        nc=int(nc),
        reg_max=int(reg_max),
        strides=tuple(int(s) for s in (strides if isinstance(strides, (list, tuple)) else [8, 16, 32])),
        hyp=hyp,
        safe_fp32=True,
        cls_pos_only=True,
        use_iou_weight=True,
    )
    # Mover criterio al mismo dispositivo que el modelo para alinear buffers (proj, strides_buf, etc.)
    loss_fn = loss_fn.to(device)

    lr0 = float(tr_cfg.get("lr0", 0.0025))
    lrf = float(tr_cfg.get("lrf", 0.1))
    wd = float(tr_cfg.get("weight_decay", 0.01))
    betas_list = tr_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas_list[0]), float(betas_list[1])) if isinstance(betas_list, (list, tuple)) else (0.9, 0.999)

    optimizer = AdamW(model.parameters(), lr=lr0, betas=betas, weight_decay=wd)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                              lr0=lr0, lrf=lrf, warmup_epochs=float(tr_cfg.get("warmup_epochs", 3.0)))

    amp_enabled = bool(args.amp or tr_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ema_enabled = bool(args.ema or tr_cfg.get("ema", True))
    ema_decay = float(tr_cfg.get("ema_decay", 0.9999))
    ema = ModelEMA(model, decay=ema_decay, device=device) if ema_enabled else None

    # 6) Artefactos: logger, weights, TB overlays (tolerante)
    variant_safe = variant or getattr(cfg, "default_variant", "m")
    run_name = f"yolo11_{variant_safe}_train_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    logger = ExperimentLogger(variant=variant_safe, phase="train", run_name=run_name)
    try:
        logger.save_config_json({
            "variant": variant_safe,
            "train": tr_cfg,
            "model_meta": {"nc": nc, "reg_max": reg_max, "strides": strides},
            "runtime": runtime_dict,
        })
    except Exception:
        pass

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        strides_int = [int(s) for s in (strides if isinstance(strides, (list, tuple)) else [8, 16, 32])]
        logger.save_model_summary(model, extra={"total_params": int(total_params), "trainable_params": int(trainable_params), "strides": strides_int, "nc": int(nc), "reg_max": int(reg_max), "device": str(device)})
    except Exception:
        pass

    save_cfg = _to_dict_like(getattr(cfg, "save", {}))
    wm = WeightsManager(variant=variant_safe, phase="train", run_name=run_name)
    # Sobrescribir políticas de guardado (CLI tiene prioridad sobre parser.yaml)
    wm.save_opts.update({
        "save_best": bool(save_cfg.get("save_best", True)),
        "save_last": bool(save_cfg.get("save_last", True)),
        "save_period": int(args.save_period),
        "keep_checkpoint_max": int(args.keep_checkpoint_max),
    })

    tb_session = None
    if TBRefOverlaySession is not None:
        try:
            tb_session = TBRefOverlaySession(variant=variant_safe, phase="train", run_name=run_name)
        except Exception:
            tb_session = None

    # 7) Frecuencias / Early stop / Umbrales val
    val_interval = max(1, int(args.val_interval))
    pr_every = max(1, int(args.pr_curves_every))
    cm_every = max(1, int(args.cm_every))
    overlay_every = max(1, int(args.overlay_every))

    patience = int(args.patience)
    best_score = -1.0
    best_epoch = -1
    since_best = 0  # en unidades de validación

    conf_thr = float(args.conf_thr)
    iou_thr = float(args.iou_thr)

    # 8) HUD y stopper
    hud = SimpleHUD(args.verbosity, mode=args.hud)
    stopper = GracefulStopper(enabled=True)

    # 9) Reanudar si corresponde (tolerante)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                print("[warn] No se pudo cargar estado del optimizador del checkpoint.")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[arg-type]
            except Exception:
                print("[warn] No se pudo cargar estado del scheduler del checkpoint.")
        print(f"[RESUME] Cargado desde {args.resume}")

    # 10) Bucle de entrenamiento
    global_step = 0
    grad_accum = max(1, int(args.grad_accum))

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_loss_box = 0.0
            epoch_loss_cls = 0.0
            epoch_loss_dfl = 0.0

            t0 = time.time()
            last_it_time = t0

            for it, batch_data in enumerate(train_loader, start=1):
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    images, targets = batch_data[0], batch_data[1]
                else:
                    raise RuntimeError("El dataloader debe retornar (images, targets, ...) mínimo.")
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True) if torch.is_tensor(targets) else targets
                # Forward/backward con reintento si MIOpen revienta en BatchNorm
                retried = False
                while True:
                    try:
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            out = model(images)
                            loss, loss_items = loss_fn(out, targets)
                        scaler.scale(loss).backward()
                        break
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if (any(k in msg for k in ("miopen", "sqlite", "evaluateinvokers", "elapsed time"))) and (not retried):
                            print("[MIOpen] RuntimeError detectado (", e, ") → activando BN.eval() y reintentando batch una vez…")
                            apply_bn_eval(model, verbose=True)
                            bn_eval_active = True
                            optimizer.zero_grad(set_to_none=True)
                            retried = True
                            continue
                        raise
                if it % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
                epoch_loss_sum += float(loss.detach())
                epoch_loss_box += float(loss_items.get("box", 0.0))
                epoch_loss_cls += float(loss_items.get("cls", 0.0))
                epoch_loss_dfl += float(loss_items.get("dfl", 0.0))
                if args.verbosity != "v0":
                    now = time.time(); elapsed = now - t0; dt_it = max(1e-6, now - last_it_time); last_it_time = now
                    imgs_per_s = (images.size(0) * grad_accum) / dt_it
                    eta = (steps_per_epoch - it) * (elapsed / max(1, it))
                    mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    device_str = str(device)
                    loss_avg = epoch_loss_sum / it; loss_box_avg = epoch_loss_box / it; loss_cls_avg = epoch_loss_cls / it; loss_dfl_avg = epoch_loss_dfl / it
                    hud.live(epoch, epochs, it, steps_per_epoch, elapsed, eta, imgs_per_s, optimizer.param_groups[0]["lr"], amp_enabled, ema is not None, grad_accum, loss_avg, loss_box_avg, loss_cls_avg, loss_dfl_avg, mem_gb, device_str)
                if stopper.poll():
                    print("[STOP] Parada solicitada. Guardando last.pt y cerrando…")
                    wm.save_epoch(ema.ema if ema is not None else model, epoch, score=None, optimizer=optimizer, scheduler=scheduler, extra={"imgsz": imgsz, "stopped": True, "bn_eval": bn_eval_active})
                    raise KeyboardInterrupt
                global_step += 1

            it_done = max(1, steps_per_epoch)
            train_loss_mean = epoch_loss_sum / it_done
            train_box_mean = epoch_loss_box / it_done
            train_cls_mean = epoch_loss_cls / it_done
            train_dfl_mean = epoch_loss_dfl / it_done

            # Validación interna
            do_val = (epoch % val_interval == 0) or (epoch == epochs)
            val_map5095 = None
            val_precision = None
            val_recall = None

            if do_val:
                eval_model = ema.ema if ema is not None else model
                eval_model.eval()
                # → FIX: pasar nc explícito y usar save_dir para figuras/PR/CM
                metrics = DetMetricsYOLOv11(class_names=None, nc=int(nc), save_dir=os.path.join(logger.logs_dir, "pr_curves"))

                with torch.no_grad():
                    for images, targets, *rest in val_loader:
                        images = images.to(device, non_blocking=True)
                        # Intento principal: decode+concat del modelo
                        outputs = eval_model(images, decode=True, concat=True)
                        scores, boxes = adapt_outputs_to_scores_boxes(outputs, int(nc))

                        # Fallback 1: si el modelo no concatena niveles
                        if scores is None or boxes is None:
                            try:
                                outputs_nc = eval_model(images, decode=True, concat=False)
                                scores, boxes = adapt_outputs_to_scores_boxes(outputs_nc, int(nc))
                            except Exception:
                                pass

                        # Fallback 2: aceptar formatos alternativos (tuplas listas, tensor combinado)
                        if scores is None or boxes is None:
                            # Diagnóstico breve para el usuario en caso de fallo
                            dtype = type(outputs).__name__
                            keys = list(outputs.keys()) if isinstance(outputs, dict) else None
                            raise RuntimeError(f"El modelo no entregó 'scores' y 'boxes' en decode=True. tipo={dtype} keys={keys}")

                        dets = scores_boxes_to_dets(scores, boxes, conf_thr=conf_thr, iou_thr=iou_thr)

                        Bv = images.size(0)
                        for b in range(Bv):
                            if torch.is_tensor(targets) and targets.numel():
                                tmask = targets[:, 0] == b
                                # → FIX: mantener tensores (no numpy) y pasar img_hw como lista de tuplas
                                t_b = targets[tmask][:, 1:].detach().cpu()
                            else:
                                t_b = None
                            metrics.add_batch([dets[b].detach().cpu()], [t_b], img_hw=[(imgsz, imgsz)])

                summary, _ = metrics.finalize()
                val_map5095 = float(summary.map50_95)
                val_precision = float(summary.precision)
                val_recall = float(summary.recall)

                # finalize() ya guarda PR/CM cuando save_dir != None; estas llamadas son tolerantes.
                if (epoch % pr_every) == 0:
                    try:
                        metrics.save_pr_curves()
                    except Exception:
                        pass
                if (epoch % cm_every) == 0:
                    try:
                        metrics.save_confusion_matrix()
                    except Exception:
                        pass

            # Logging por época
            log_row = {
                "train/loss": train_loss_mean,
                "train/loss_box": train_box_mean,
                "train/loss_cls": train_cls_mean,
                "train/loss_dfl": train_dfl_mean,
            }
            if val_map5095 is not None:
                log_row.update({
                    "val/mAP50-95": val_map5095,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                })

            try:
                logger.log_epoch(epoch, log_row, split="train")
            except Exception:
                pass

            # Overlays
            if tb_session is not None and (epoch % overlay_every) == 0:
                try:
                    if log_ref_session_epoch is not None:
                        log_ref_session_epoch(tb_session, epoch=epoch, split="val")
                except Exception:
                    pass

            # Checkpoints y early stop
            score = val_map5095 if val_map5095 is not None else None
            wm.save_epoch(ema.ema if ema is not None else model, epoch, score=score, optimizer=optimizer,
                          scheduler=scheduler, extra={"imgsz": imgsz})

            improved = (score is not None) and (score > best_score)
            if improved:
                best_score = score  # type: ignore[assignment]
                best_epoch = epoch
                since_best = 0
                print(f"[↑] new best mAP50-95: {best_score:.3f} @ ep {best_epoch}")
            elif do_val:
                since_best += 1

            if val_map5095 is not None:
                hud.epoch_summary(
                    f"[✓] ep {epoch} | train_loss {train_loss_mean:.2f} (box {train_box_mean:.2f}, cls {train_cls_mean:.2f}, dfl {train_dfl_mean:.2f}) "
                    f"| val mAP50-95 {val_map5095:.3f} (P {val_precision:.2f}, R {val_recall:.2f}) | best {best_score:.3f}@ep{best_epoch}"
                )
            else:
                hud.epoch_summary(
                    f"[✓] ep {epoch} | train_loss {train_loss_mean:.2f} (box {train_box_mean:.2f}, cls {train_cls_mean:.2f}, dfl {train_dfl_mean:.2f}) | val: —"
                )

            if do_val and since_best >= patience:
                print(f"early-stop: paciencia {patience} validaciones (val_interval={val_interval} ⇒ efectivo {patience*val_interval} épocas)")
                break

    except KeyboardInterrupt:
        print("[EXIT] Entrenamiento detenido por el usuario.")

    finally:
        try:
            logger.close()
        except Exception:
            pass
        print("Rutas clave:")
        print(f"  Pesos  : {wm.weights_dir}")
        print(f"  Logs   : {logger.logs_dir}")
        print(f"  TB/Runs: {logger.runs_dir}")


if __name__ == "__main__":
    train_main()
