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

# ========================= Importes estándar ========================= #
import os, sys, math, json, time, argparse, datetime as dt
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple, Dict, Any, List
import subprocess, socket, webbrowser, shutil
import warnings

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

# ========================= Importes de terceros ========================= #
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

# ========================= Importes del proyecto ========================= #
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
    """Fija semillas (PyTorch/NumPy/Random) y ajusta CUDNN para reproducibilidad.
    deterministic=True fuerza kernels deterministas (más lento); benchmark activa autotuning.
    """
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
    """Selecciona dispositivo: respeta pref (cpu/cuda/mps) o elige automática.
    Prioridad: CUDA/ROCm > MPS > CPU.
    """
    if pref and pref.lower() in ("cpu", "cuda", "mps"):
        if pref.lower() == "cpu":
            return torch.device("cpu")
        if pref.lower() == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
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
    """Convierte un objeto (incl. dataclass) a dict sencillo sin alterar contenido.
    - dict -> tal cual; - dataclass -> asdict; - objeto -> __dict__/atributos públicos.
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
    """Detección de parada elegante por F8 / Ctrl+C sin alterar el bucle principal."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._requested = False
        self._last_prompt_ts = 0.0

    def arm(self) -> None:
        self._requested = True

    def poll(self) -> bool:
        if not self.enabled:
            return False
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
    """LambdaLR: warmup lineal + coseno. lrf=factor mínimo (lr_min=lr0*lrf)."""
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
    """EMA ligero de parámetros del modelo sin cambiar forward/backward."""
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
    """Pone capas BatchNorm en eval() (mitiga fallos MIOpen/SQLite en ROCm Windows)."""
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
    """Convierte (B,N,nc)/(B,N,4) a listas [x1,y1,x2,y2,conf,cls] por imagen (con NMS opcional)."""
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
            bb = bb[mask]; conf = conf[mask]; cls = cls[mask]
        else:
            out.append(torch.zeros((0, 6), device=scores.device)); continue

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
        if t.size(1) in (4, 5, 6, 8) and t.size(2) > t.size(1):
            return t.permute(0, 2, 1).contiguous()
        return t
    raise ValueError("Tensor con dimensionalidad no soportada para _flatten_hw_to_NC")


def _cat_levels_to_BNC(seq: List[torch.Tensor]) -> torch.Tensor:
    """Concatena niveles tras aplanarlos a (B,N,C)."""
    parts = [_flatten_hw_to_NC(x) for x in seq]
    return torch.cat(parts, dim=1)


@torch.no_grad()
def adapt_outputs_to_scores_boxes(outputs: Any, nc: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Intenta extraer (scores[B,N,nc], boxes[B,N,4]) desde formatos comunes (dict/list/tensor)."""
    # 1) Dict con 'scores'/'boxes'
    if isinstance(outputs, dict):
        s = outputs.get("scores", None); b = outputs.get("boxes", None)
        if s is not None and b is not None:
            if isinstance(s, (list, tuple)): s = _cat_levels_to_BNC(list(s))
            else: s = _flatten_hw_to_NC(s)
            if isinstance(b, (list, tuple)): b = _cat_levels_to_BNC(list(b))
            else: b = _flatten_hw_to_NC(b)
            if s.dim() == 3 and s.size(-1) == nc and b.dim() == 3 and b.size(-1) == 4:
                return s, b
        # 1-bis) Dict con 'cls'/'boxes'
        s = outputs.get("cls", None); b = outputs.get("boxes", None)
        if s is not None and b is not None:
            if isinstance(s, (list, tuple)): s = _cat_levels_to_BNC(list(s))
            else: s = _flatten_hw_to_NC(s)
            if s.dim() == 3 and s.size(-1) != nc and s.size(1) == nc: s = s.permute(0, 2, 1).contiguous()
            if isinstance(b, (list, tuple)): b = _cat_levels_to_BNC(list(b))
            else: b = _flatten_hw_to_NC(b)
            if s.dim() == 3 and s.size(-1) == nc and b.dim() == 3 and b.size(-1) == 4:
                if (s.min() < 0) or (s.max() > 1): s = s.sigmoid()
                return s, b
        # 2) Dict con 'cls'/'reg'
        s = outputs.get("cls", None); r = outputs.get("reg", None)
        if s is not None and r is not None:
            if isinstance(s, (list, tuple)): s = _cat_levels_to_BNC(list(s))
            else: s = _flatten_hw_to_NC(s)
            if s.size(-1) != nc and s.size(1) == nc: s = s.permute(0, 2, 1).contiguous()
            if isinstance(r, (list, tuple)): r = _cat_levels_to_BNC(list(r))
            else: r = _flatten_hw_to_NC(r)
            if s.dim() == 3 and r.dim() == 3 and r.size(-1) in (4,):
                if (s.min() < 0) or (s.max() > 1): s = s.sigmoid()
                return s, r
    # 3) Tuple/List de 2 tensores
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2 and all(torch.is_tensor(x) for x in outputs):
        a, b = outputs
        a = _flatten_hw_to_NC(a); b = _flatten_hw_to_NC(b)
        if a.size(-1) == nc and b.size(-1) == 4: return a, b
        if b.size(-1) == nc and a.size(-1) == 4: return b, a
        if a.size(-1) == nc + 4: return a[..., :nc], a[..., nc:]
        if b.size(-1) == nc + 4: return b[..., :nc], b[..., nc:]
    # 4) Tensor único (B,N,nc+4)
    if torch.is_tensor(outputs):
        t = _flatten_hw_to_NC(outputs)
        if t.size(-1) == nc + 4: return t[..., :nc], t[..., nc:]
    # 5) Lista de tensores por nivel
    if isinstance(outputs, (list, tuple)) and all(torch.is_tensor(x) for x in outputs):
        t = _cat_levels_to_BNC(list(outputs))
        if t.size(-1) == nc + 4: return t[..., :nc], t[..., nc:]
    return None, None


# ========================= Interfaz / CLI ========================= #

def build_parser() -> argparse.ArgumentParser:
    """Argumentos de entrenamiento (sin cambiar valores por defecto actuales)."""
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
    # Overrides de optimización/planificación
    p.add_argument("--lr0", type=float, default=None, help="Override LR inicial (AdamW)")
    p.add_argument("--lrf", type=float, default=None, help="Factor mínimo LR (cosine)")
    p.add_argument("--warmup-epochs", type=float, default=None, help="Override warmup epochs del scheduler")
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
    # Warm-up HUD explícito (opción A)
    p.add_argument("--warmup-hud", action="store_true", help="Ejecuta forwards sintéticos y muestra un HUD de warm-up antes del primer batch")
    p.add_argument("--warmup-steps", type=int, default=3, help="Pasos sintéticos de warm-up para compilar kernels")
    p.add_argument("--warmup-batch", type=int, default=1, help="Tamaño de batch usado durante el warm-up")
    p.add_argument("--warmup-imgsz", type=int, default=None, help="imgsz específico para warm-up (por defecto usa --imgsz)")
    p.add_argument("--warmup-only", action="store_true", help="Ejecuta solo el warm-up y sale (smoketest)")
    # TensorBoard / Auto-lanzamiento
    p.add_argument("--tb-auto", dest="tb_auto", action="store_true")
    p.add_argument("--no-tb-auto", dest="tb_auto", action="store_false")
    p.set_defaults(tb_auto=True)
    p.add_argument("--tb-port", type=int, default=None)
    p.add_argument("--tb-host", type=str, default=None)
    p.add_argument("--tb-open-browser", action="store_true", help="Abrir URL en el navegador al lanzar TB")
    # Ajustes de aprendizaje temprano
    p.add_argument("--freeze-epochs", type=int, default=0, help="Congelar backbone los primeros K epochs")
    p.add_argument("--conf-ramp-epochs", type=int, default=5, help="Rampa de confianza en validación hasta --conf-thr")
    return p


def interactive_wizard(cfg: Any, args: argparse.Namespace) -> argparse.Namespace:
    """Asistente simple para sobreescribir parámetros clave leídos de configs.* sin cambiar defaults."""
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
    """HUD de dos líneas con barra de progreso y métricas (sin cambiar lógica)."""
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
    """HUD compacto configurable (one/two/off). No altera parámetros ni salidas existentes."""
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
            sys.stdout.write("\r" + line + (" " * pad)); sys.stdout.flush(); self._last_len = len(line); return
        line1 = (f"[EPOCH {epoch}/{epochs}] {self._bar(frac)}  {int(frac*100):2d}% | "
                 f"it {it}/{it_total} | {time.strftime('%H:%M:%S', time.gmtime(elapsed_s))} ⏱ | "
                 f"ETA {time.strftime('%H:%M:%S', time.gmtime(max(0, int(eta_s))))} | "
                 f"imgs/s {imgs_per_s:.0f} | lr {lr:.2e} | AMP {'on' if amp else 'off'} | EMA{'✓' if ema else '×'} | F8=stop")
        line2 = (f"train: loss {loss_avg:.3f} (box {loss_box:.2f}, cls {loss_cls:.2f}, dfl {loss_dfl:.2f}) | "
                 f"grad_acc {grad_accum}x | mem {mem_gb:.1f}GB | dev: {device_str}")
        sys.stdout.write("\r" + line1 + "\n"); sys.stdout.write("\r\x1b[1A\x1b[2K" + line2 + "\n"); sys.stdout.flush()
        self._last_len = len(line2)

    def epoch_summary(self, text: str) -> None:
        if self.verbosity != "v0":
            sys.stdout.write("\n")
        print(text, flush=True)
        self._last_len = 0


# ========================= HUD de Warm-up (opción A) ========================= #
class WarmupHUD:
    """Una línea compacta para la fase de warm-up (kernel search, cachés, etc.)."""
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    def __init__(self, verbosity: str = "v1") -> None:
        self.verbosity = verbosity
        self._i = 0
        self._last_len = 0
        self._t0 = time.time()
    def _fmt_t(self, t: float) -> str:
        return time.strftime('%M:%S', time.gmtime(int(max(0,t))))
    def live(self, step: int, total: int, dev: str, amp: bool, bn_eval: bool, mem_alloc_gb: float, mem_res_gb: float) -> None:
        if self.verbosity == "v0":
            return
        dt_s = time.time() - self._t0
        spinner = self.FRAMES[self._i % len(self.FRAMES)]; self._i += 1
        line = (f"[WARM-UP] {spinner}  t={self._fmt_t(dt_s)}  mem {mem_alloc_gb:.1f}/{mem_res_gb:.1f}GB  dev: {dev}  AMP {'on' if amp else 'off'}  BN-eval {'✓' if bn_eval else '×'}  step {step}/{total}")
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + (" " * pad)); sys.stdout.flush(); self._last_len = len(line)
    def done(self) -> None:
        if self.verbosity == "v0":
            return
        sys.stdout.write("\n"); sys.stdout.flush(); self._last_len = 0


def _mem_gb(device: torch.device) -> Tuple[float, float]:
    """Devuelve (allocated_GB, reserved_GB) para CUDA/ROCm; zeros en CPU/MPS."""
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / (1024**3)
            r = torch.cuda.memory_reserved() / (1024**3)
            return float(a), float(r)
    except Exception:
        pass
    return 0.0, 0.0

@torch.no_grad()
def do_model_warmup(model: nn.Module, device: torch.device, *, steps: int, batch: int, imgsz: int, in_ch: int, amp: bool, bn_eval_active: bool, verbosity: str) -> float:
    """Ejecuta 'steps' forwards sintéticos con batch/imgsz dados, muestra HUD de warm-up y restaura el estado del modelo.
    Devuelve el tiempo total de warm-up (s)."""
    hud = WarmupHUD(verbosity=verbosity)
    was_train = model.training
    t0 = time.time()
    try:
        model.eval()
        x = torch.randn(batch, in_ch, imgsz, imgsz, device=device, dtype=torch.float32)
        use_cuda_autocast = (device.type == "cuda") and amp
        for i in range(1, steps + 1):
            if use_cuda_autocast:
                with torch.amp.autocast('cuda'):
                    try:
                        _ = model(x, decode=False, concat=False)
                    except TypeError:
                        _ = model(x)
            else:
                try:
                    _ = model(x, decode=False, concat=False)
                except TypeError:
                    _ = model(x)
            if device.type == "cuda":
                try: torch.cuda.synchronize()
                except Exception: pass
            a_gb, r_gb = _mem_gb(device)
            hud.live(step=i, total=steps, dev=device.type, amp=amp, bn_eval=bn_eval_active, mem_alloc_gb=a_gb, mem_res_gb=r_gb)
        hud.done()
    finally:
        model.train(was_train)
    return float(time.time() - t0)

# ========================= Utilidades TensorBoard ========================= #

def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Comprueba si el puerto/host está libre (para auto-lanzar TensorBoard)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def _find_free_port(start: int = 6006, host: str = "127.0.0.1", limit: int = 20) -> int:
    """Busca un puerto libre a partir de 'start' con un máximo de 'limit' intentos."""
    p = start
    for _ in range(limit):
        if _is_port_free(p, host):
            return p
        p += 1
    return start  # fallback


def launch_tensorboard(logdir: str, port: int = 6006, host: str = "127.0.0.1"):
    """Lanza TensorBoard en segundo plano si es posible. Devuelve el proceso o None."""
    try:
        if not os.path.isdir(logdir):
            os.makedirs(logdir, exist_ok=True)
        if not _is_port_free(port, host):
            new_port = _find_free_port(port, host)
            if new_port != port:
                print(f"[TB] Puerto {port} ocupado, usando {new_port}."); port = new_port
        tb_cmd = None
        if shutil.which("tensorboard"): tb_cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--host", host]
        else: tb_cmd = [sys.executable, "-m", "tensorboard", "--logdir", logdir, "--port", str(port), "--host", host]
        proc = subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc, port
    except Exception as e:
        print(f"[TB] Auto-lanzamiento falló: {e}"); return None, port


# ========================= Bucle de entrenamiento (setup actual) ========================= #

def train_main() -> None:
    """Configura entrenamiento y artefactos (esta versión incluye arranque de warm-up robusto)."""
    parser = build_parser(); args = parser.parse_args()

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
        apply_bn_eval(model, verbose=True); bn_eval_active = True

    # --- Warm-up explícito (opción A) antes del primer batch ---
    if getattr(args, "warmup_hud", False):
        warm_steps = int(max(0, args.warmup_steps))
        if warm_steps <= 0:
            print("[WARM-UP] saltado: --warmup-steps ≤ 0")
        else:
            warm_batch = int(max(1, args.warmup_batch))
            warm_imgsz = int(args.warmup_imgsz or imgsz_for_strides or 640)
            in_ch = int(_meta_get(model_meta, "in_channels", 3))
            amp_flag = bool(args.amp or tr_cfg.get("amp", True))
            print(f"[WARM-UP] Inicializando · steps={warm_steps} · batch={warm_batch} · imgsz={warm_imgsz} · in_ch={in_ch} · AMP={'on' if amp_flag else 'off'}")
            dt_warm = do_model_warmup(model, device, steps=warm_steps, batch=warm_batch, imgsz=warm_imgsz, in_ch=in_ch, amp=amp_flag, bn_eval_active=bn_eval_active, verbosity=args.verbosity)
            a_gb, r_gb = _mem_gb(device)
            print(f"[WARM-UP] Listo en {dt_warm:.2f}s · mem {a_gb:.2f}/{r_gb:.2f} GB")
            if args.warmup_only:
                print("[WARM-UP] --warmup-only activado. Saliendo tras smoketest ✔")
                return

    # 4) Dataloaders
    imgsz = args.imgsz or tr_cfg.get("imgsz", 640)
    epochs = args.epochs or tr_cfg.get("epochs", 150)
    batch = args.batch or tr_cfg.get("batch", 16)

    dl_cfg = _to_dict_like(tr_cfg.get("dataloader", {}))
    workers = int(dl_cfg.get("workers", 4)); pin_memory = bool(dl_cfg.get("pin_memory", True)); persistent = bool(dl_cfg.get("persistent_workers", True))

    train_loader = build_yolo_dataloader("train", imgsz=imgsz, batch=batch, workers=workers, pin_memory=pin_memory, persistent_workers=persistent, shuffle=True)
    val_loader = build_yolo_dataloader("val", imgsz=imgsz, batch=batch, workers=workers, pin_memory=pin_memory, persistent_workers=persistent, shuffle=False)

    # --- Dataset summary on-screen ---
    try:
        data_sec = _to_dict_like(getattr(cfg, "data", {}))
        yaml_path = (data_sec.get("config") or data_sec.get("dataset") or data_sec.get("path") or os.path.join("configs","dataset.yaml"))
        yaml_abs = os.path.abspath(yaml_path); verified = os.path.exists(yaml_abs)
        print(f"[DATA] Config (yaml): {yaml_abs}" + ("" if verified else " (no verificado)"))
        tr_len = len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader)
        print(f"[DATA] split=train | imgsz={imgsz} | batch={batch} | samples={tr_len}")
        ds_names = None
        if hasattr(train_loader, "dataset"):
            _ds = train_loader.dataset
            ds_names = getattr(_ds, "names", None) or getattr(_ds, "class_names", None)
        if ds_names is not None:
            try:
                print(f"[DATA] classes (nc={len(ds_names)}): {list(ds_names)}")
            except Exception:
                pass
        else:
            print(f"[DATA] classes (nc={int(nc)}) — desde config/model_meta")
    except Exception as _e:
        print("[DATA] resumen no disponible:", _e)

    steps_per_epoch = max(1, len(train_loader))

    # 5) Pérdida, optimizador, scheduler, AMP, EMA
    loss_weights = _to_dict_like(tr_cfg.get("loss weights", {"box": 7.5, "cls": 0.5, "dfl": 1.5}))
    hyp = LossHyperparams(box=float(loss_weights.get("box", 7.5)), cls=float(loss_weights.get("cls", 0.5)), dfl=float(loss_weights.get("dfl", 1.5)))
    loss_fn = YOLOLoss(nc=int(nc), reg_max=int(reg_max), strides=tuple(int(s) for s in (strides if isinstance(strides, (list, tuple)) else [8, 16, 32])), hyp=hyp, safe_fp32=True, cls_pos_only=True, use_iou_weight=True)
    loss_fn = loss_fn.to(device)

    lr0 = float(args.lr0 if args.lr0 is not None else tr_cfg.get("lr0", 0.0025))
    lrf = float(args.lrf if args.lrf is not None else tr_cfg.get("lrf", 0.1))
    wd = float(tr_cfg.get("weight_decay", 0.01))
    betas_list = tr_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas_list[0]), float(betas_list[1])) if isinstance(betas_list, (list, tuple)) else (0.9, 0.999)

    optimizer = AdamW(model.parameters(), lr=lr0, betas=betas, weight_decay=wd)
    warmup_e = float(args.warmup_epochs if args.warmup_epochs is not None else tr_cfg.get("warmup_epochs", 3.0))
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch, lr0=lr0, lrf=lrf, warmup_epochs=warmup_e)

    amp_enabled = bool(args.amp or tr_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ema_enabled = bool(args.ema or tr_cfg.get("ema", True))
    ema_decay = float(tr_cfg.get("ema_decay", 0.9999))
    ema = ModelEMA(model, decay=ema_decay, device=device) if ema_enabled else None

    # 6) Artefactos: logger, weights, TB overlays (tolerante)
    variant_safe = variant or getattr(cfg, "default_variant", "m")
    run_name = f"yolo11_{variant_safe}_train_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    logger = ExperimentLogger(variant=variant_safe, phase="train", run_name=run_name)
    # --- TensorBoard: auto-lanzamiento y URL cliqueable ---
    tb_port = int(args.tb_port or os.environ.get("TB_PORT") or os.environ.get("TENSORBOARD_PORT") or 6006)
    tb_host = (args.tb_host or os.environ.get("TB_HOST") or "127.0.0.1").strip()
    tb_logdir_run = logger.runs_dir
    tb_logdir_parent = os.path.dirname(tb_logdir_run)
    tb_logdir_parent_rel = os.path.relpath(tb_logdir_parent, start=os.getcwd())
    tb_cmd = f'tensorboard --logdir "{tb_logdir_parent_rel}" --port {tb_port} --host {tb_host}'
    print(f"[TB] comando: {tb_cmd}")

    if getattr(args, "tb_auto", True):
        proc, tb_port = launch_tensorboard(tb_logdir_parent, tb_port, tb_host)
        if proc is None:
            print("[TB] No se pudo iniciar TensorBoard automáticamente; usa el comando anterior.")
        else:
            tb_url = f"http://{tb_host}:{tb_port}/"
            print(f"[TB] TensorBoard · {tb_url}")
            if getattr(args, "tb_open_browser", False):
                try: webbrowser.open(tb_url, new=2, autoraise=False)
                except Exception: pass

    try:
        logger.save_config_json({"variant": variant_safe, "train": tr_cfg, "model_meta": {"nc": nc, "reg_max": reg_max, "strides": strides}, "runtime": runtime_dict})
    except Exception:
        pass

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        strides_int = [int(s) for s in (strides if isinstance(strides, (list, tuple)) else [8, 16, 32])]
        logger.save_model_summary(model, extra={"total_params": int(total_params), "trainable_params": int(trainable_params), "strides": strides_int})
    except Exception:
        pass

    # Nota: El bucle de entrenamiento completo se gestiona en versiones posteriores; aquí nos enfocamos en el warm-up y setup.


if __name__ == "__main__":
    try:
        train_main()
    except KeyboardInterrupt:
        print("[STOP] Entrenamiento interrumpido por el usuario (F8/Ctrl+C)."); import sys; sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {e}"); raise
