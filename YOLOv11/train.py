
# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: train.py
# Script principal de entrenamiento para YOLOv11, versión compacta y modular.
# Define clases para warm-up, evaluación, checkpoints y entrenamiento,
# reduciendo lógica duplicada y manteniendo integridad funcional.
#==============================================================

from __future__ import annotations
# ========================= Importes estándar ========================= #
import os, sys, math, json, time, argparse, datetime as dt
from pathlib import Path
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
# --- Utilidad BN->GN ---
def _swap_bn_to_gn(module: nn.Module, groups: int = 32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            ch = child.num_features
            g = min(groups, ch)
            while ch % g != 0 and g > 1:
                g -= 1
            setattr(module, name, nn.GroupNorm(num_groups=max(1, g), num_channels=ch, eps=1e-5, affine=True))
        else:
            _swap_bn_to_gn(child, groups)

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


# ========================= HUD consola ========================= #
class SimpleHUD:
    """HUD compacto configurable (one/two/off)."""
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
    def live(self, step: int, total: int, dev: str, amp: bool, bn_eval: bool, vram_pct: float, alloc_gb: float, reserved_gb: float) -> None:
        if self.verbosity == "v0":
            return
        dt_s = time.time() - self._t0
        spinner = self.FRAMES[self._i % len(self.FRAMES)]; self._i += 1
        line = (f"[WARM-UP] {spinner}  t={self._fmt_t(dt_s)}  VRAM {vram_pct:.1f}% (alloc {alloc_gb:.2f}GB, resv {reserved_gb:.2f}GB)  dev: {dev}  AMP {'on' if amp else 'off'}  BN-eval {'✓' if bn_eval else '×'}  step {step}/{total}")
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + (" " * pad)); sys.stdout.flush(); self._last_len = len(line)
    def done(self) -> None:
        if self.verbosity == "v0":
            return
        sys.stdout.write("\n"); sys.stdout.flush(); self._last_len = 0


# ========================= Memoria GPU ========================= #
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


def _mem_stats(device: torch.device) -> Tuple[float, float, float, float, float]:
    """(alloc_GB, reserved_GB, total_GB, alloc_pct, reserved_pct)."""
    a_gb, r_gb = _mem_gb(device)
    total_gb = 0.0
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total_gb = float(props.total_memory) / (1024**3)
    except Exception:
        total_gb = 0.0
    alloc_pct = (a_gb / total_gb * 100.0) if total_gb > 0 else 0.0
    resv_pct = (r_gb / total_gb * 100.0) if total_gb > 0 else 0.0
    return a_gb, r_gb, total_gb, alloc_pct, resv_pct


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


# ========================= Clases compactas ========================= #
class WarmupRunner:
    def __init__(self, model: nn.Module, device: torch.device, verbosity: str = "v1"):
        self.model = model
        self.device = device
        self.verbosity = verbosity

    @torch.no_grad()
    def run(self, *, steps: int, batch: int, imgsz: int, in_ch: int, amp: bool, bn_eval_active: bool) -> float:
        hud = WarmupHUD(verbosity=self.verbosity)
        was_train = self.model.training
        t0 = time.time()

        def _forward_once(x: torch.Tensor, use_amp: bool) -> None:
            if use_amp:
                with torch.amp.autocast('cuda'):
                    try:
                        _ = self.model(x, decode=False, concat=False)
                    except TypeError:
                        _ = self.model(x)
            else:
                try:
                    _ = self.model(x, decode=False, concat=False)
                except TypeError:
                    _ = self.model(x)

        try:
            self.model.eval()
            x = torch.randn(batch, in_ch, imgsz, imgsz, device=self.device, dtype=torch.float32)
            use_cuda_autocast = (self.device.type == "cuda") and amp

            for i in range(1, steps + 1):
                step_done = False
                attempts = 0
                while not step_done and attempts < 2:
                    attempts += 1
                    try:
                        _forward_once(x, use_cuda_autocast)
                        step_done = True
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ("miopen" in msg or "evaluateinvokers" in msg or "sqlite" in msg) and not bn_eval_active:
                            print("[MIOpen] RuntimeError detectado (" + e.__class__.__name__ + ") → activando BN.eval() y reintentando batch una vez…")
                            apply_bn_eval(self.model, verbose=True)
                            bn_eval_active = True
                            time.sleep(0.2)
                            continue
                        print(f"[MIOpen] Aviso: fallo en warm-up step {i} (intento {attempts}) → {e}")
                        break

                if self.device.type == "cuda":
                    try: torch.cuda.synchronize()
                    except Exception: pass
                a_gb, r_gb, t_gb, a_pct, r_pct = _mem_stats(self.device)
                hud.live(step=i, total=steps, dev=self.device.type, amp=amp, bn_eval=bn_eval_active,
                         vram_pct=a_pct, alloc_gb=a_gb, reserved_gb=r_gb)
            hud.done()
        finally:
            self.model.train(was_train)
        return float(time.time() - t0)


class Evaluator:
    def __init__(self, imgsz: int, nc: int, conf_thr: float, iou_thr: float):
        self.imgsz = int(imgsz)
        self.nc = int(nc)
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)

    @torch.no_grad()
    def __call__(self, model: nn.Module, loader) -> Dict[str, Optional[float]]:
        metrics = {"mAP50-95": None, "mAP50": None, "precision": None, "recall": None}
        try:
            device_eval = next(model.parameters()).device
            model.eval()
            dm = DetMetricsYOLOv11(nc=self.nc)
            for ims, tgts, *rest in loader:
                ims = ims.to(device_eval, non_blocking=True).float()
                try:
                    out = model(ims, decode=True, concat=True)
                except TypeError:
                    out = model(ims)
                scores, boxes = adapt_outputs_to_scores_boxes(out, self.nc)
                if scores is None or boxes is None:
                    continue
                dets = scores_boxes_to_dets(scores, boxes, conf_thr=self.conf_thr, iou_thr=self.iou_thr, max_det=300)
                if tgts is None:
                    batch_gt = [[] for _ in range(len(ims))]
                else:
                    tgts_np = tgts.detach().cpu().numpy()
                    B = len(ims); batch_gt = [[] for _ in range(B)]
                    for (bi, c, x, y, w, h) in tgts_np:
                        bi = int(bi); batch_gt[bi].append([float(c), float(x), float(y), float(w), float(h)])
                dm.add_batch(preds=[d for d in dets], targets=[(torch.tensor(gt, dtype=torch.float32, device=device_eval) if len(gt) else torch.zeros((0,5), dtype=torch.float32, device=device_eval)) for gt in batch_gt], img_hw=[(self.imgsz, self.imgsz)]*len(dets), labels_is_xywhn=True, conf_min_for_cm=float(self.conf_thr), iou_match_for_cm=float(self.iou_thr))
            res = dm.finalize()
            for k in ("mAP50-95", "map_50_95", "map50_95", "map5095"):
                if k in res: metrics["mAP50-95"] = float(res[k]); break
            for k in ("mAP50", "map50"):
                if k in res: metrics["mAP50"] = float(res[k]); break
            for k in ("precision", "P"):
                if k in res: metrics["precision"] = float(res[k]); break
            for k in ("recall", "R"):
                if k in res: metrics["recall"] = float(res[k]); break
        except Exception as _e:
            print(f"[val] Omitiendo validación (motivo: {_e})")
        return metrics


class Checkpointer:
    def __init__(self, base_dir: str, variant: str, run_name: str, save_period: int = 10, keep_max: int = 5):
        self.dir = os.path.join(base_dir, variant, "train", run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.best_metric: Optional[float] = None
        self.save_period = max(1, int(save_period))
        self.keep_max = max(1, int(keep_max))

    def _prune_old(self) -> None:
        try:
            files = [f for f in os.listdir(self.dir) if f.startswith('VAR_Train_Epoch_') and f.endswith('.pt')]
            def _epoch_of(fname: str) -> int:
                try:
                    return int(fname.split('_')[-1].split('.')[0])
                except Exception:
                    return -1
            files_sorted = sorted(files, key=_epoch_of)
            excess = max(0, len(files_sorted) - self.keep_max)
            for f in files_sorted[:excess]:
                try:
                    os.remove(os.path.join(self.dir, f))
                except Exception:
                    pass
        except Exception:
            pass

    def save(self, *, epoch: int, max_epochs: int, model_state: Dict[str, Any], optimizer_state: Dict[str, Any],
             scaler_state: Optional[Dict[str, Any]], metrics: Dict[str, Optional[float]],
             ref_metric: float) -> None:
        is_best = (self.best_metric is None) or (ref_metric > self.best_metric)
        if is_best:
            self.best_metric = ref_metric
        ckpt = {
            "epoch": epoch,
            "max_epochs": int(max_epochs),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scaler_state": scaler_state,
            "best_metric": self.best_metric,
            "metrics": metrics,
        }
        torch.save(ckpt, os.path.join(self.dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(self.dir, "best.pt"))
        if (epoch % self.save_period) == 0:
            torch.save(ckpt, os.path.join(self.dir, f"VAR_Train_Epoch_{epoch:03d}.pt"))
            self._prune_old()


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # 1) Cargar configs
        self.cfg = ConfigParserYaml().load()
        if args.interactive:
            self.args = interactive_wizard(self.cfg, args)

        # 2) Resolver secciones
        self.variant = self.args.variant or getattr(self.cfg, "default_variant_name", None)
        train_section = _to_dict_like(getattr(self.cfg, "train", {}))
        # Soporta {'config': {...}} y dict plano
        self.tr_cfg = _to_dict_like(train_section.get("config", train_section))

        imgsz_for_strides = self.args.imgsz or self.tr_cfg.get("imgsz", 640)
        self.model = self.cfg.build_model(variant=self.variant, imgsz_for_strides=imgsz_for_strides)
        _miopen_dc = os.environ.get("MIOPEN_DISABLE_CACHE", None)
        _miopen_fe = os.environ.get("MIOPEN_FIND_ENFORCE", None)
        if _miopen_dc or _miopen_fe:
            print(f"[MIOpen-CLI] MIOPEN_DISABLE_CACHE={_miopen_dc if _miopen_dc is not None else '∅'} | MIOPEN_FIND_ENFORCE={_miopen_fe if _miopen_fe is not None else '∅'}")

        # Metadata
        self.model_meta = getattr(self.cfg, "model_meta", {})
        self.nc = int(_meta_get(self.model_meta, "nc", 5))
        self.reg_max = int(_meta_get(self.model_meta, "reg_max", 16))
        self.strides = _meta_get(self.model_meta, "strides", [8, 16, 32])
        if isinstance(self.strides, torch.Tensor):
            self.strides = self.strides.detach().cpu().tolist()

        # 3) Runtime / device / seed
        runtime_dict = _to_dict_like(getattr(self.cfg, "runtime", {}))
        self.device = select_device(self.args.device or runtime_dict.get("device", None))
        seed_everything(int(runtime_dict.get("seed", 42)), bool(runtime_dict.get("deterministic", False)))
        compile_flag = bool(runtime_dict.get("compile", False))
        if getattr(self.args, 'force_gn', False):
            _swap_bn_to_gn(self.model, groups=getattr(self.args, 'gn_groups', 32))
            print(f"[Norm] BatchNorm2d -> GroupNorm(g={getattr(self.args,'gn_groups',32)}) aplicado.")
            # Importante: mover el modelo al dispositivo tras el swap para evitar desajustes CPU/GPU
        self.model.to(self.device)
        if compile_flag and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                print("[warn] torch.compile falló; continuando sin compile().")

        # BN eval fallback opcional
        self.bn_eval_active = False
        if self.args.bn_eval_fallback:
            apply_bn_eval(self.model, verbose=True); self.bn_eval_active = True

        # 4) Data
        self.imgsz = self.args.imgsz or self.tr_cfg.get("imgsz", 640)
        self.epochs = self.args.epochs or self.tr_cfg.get("epochs", 150)
        self.batch = self.args.batch or self.tr_cfg.get("batch", 16)
        self.grad_accum = int(self.args.grad_accum) if (getattr(self.args, "grad_accum", None) is not None) else int(self.tr_cfg.get("grad_accum", 1))
        self.grad_accum = max(1, self.grad_accum)

        dl_cfg = _to_dict_like(self.tr_cfg.get("dataloader", {}))
        if not dl_cfg:
            dl_cfg = {k: self.tr_cfg.get(k) for k in ("workers","pin_memory","persistent_workers","shuffle") if k in self.tr_cfg}
        workers = int(dl_cfg.get("workers", 4))
        pin_memory = bool(dl_cfg.get("pin_memory", True))
        persistent = bool(dl_cfg.get("persistent_workers", True))

        self.train_loader = build_yolo_dataloader(
            "train", imgsz=self.imgsz, batch=self.batch, workers=workers,
            pin_memory=pin_memory, persistent_workers=persistent, shuffle=True
        )
        self.val_loader = build_yolo_dataloader(
            "val", imgsz=self.imgsz, batch=self.batch, workers=workers,
            pin_memory=pin_memory, persistent_workers=persistent, shuffle=False
        )
        self.steps_per_epoch = max(1, len(self.train_loader))

        # 5) Criterio/opt/planificador
        loss_weights = _to_dict_like(self.tr_cfg.get("loss", self.tr_cfg.get("loss weights", {"box": 7.5, "cls": 0.5, "dfl": 1.5})))
        hyp = LossHyperparams(
            box=float(loss_weights.get("box", 7.5)),
            cls=float(loss_weights.get("cls", 0.5)),
            dfl=float(loss_weights.get("dfl", 1.5))
        )
        self.loss_fn = YOLOLoss(
            nc=int(self.nc), reg_max=int(self.reg_max),
            strides=tuple(int(s) for s in (self.strides if isinstance(self.strides, (list, tuple)) else [8, 16, 32])),
            hyp=hyp, safe_fp32=True, cls_pos_only=True, use_iou_weight=True
        ).to(self.device)

        self.lr0 = float(self.args.lr0 if self.args.lr0 is not None else self.tr_cfg.get("lr0", 0.0025))
        self.lrf = float(self.args.lrf if self.args.lrf is not None else self.tr_cfg.get("lrf", 0.1))
        wd = float(self.tr_cfg.get("weight_decay", 0.01))
        betas_list = self.tr_cfg.get("betas", [0.9, 0.999])
        betas = (float(betas_list[0]), float(betas_list[1])) if isinstance(betas_list, (list, tuple)) else (0.9, 0.999)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr0, betas=betas, weight_decay=wd)
        warmup_e = float(self.args.warmup_epochs if self.args.warmup_epochs is not None else self.tr_cfg.get("warmup_epochs", 3.0))
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch,
            lr0=self.lr0, lrf=self.lrf, warmup_epochs=warmup_e
        )

        self.amp_enabled = bool(self.args.amp or self.tr_cfg.get("amp", True))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.ema_enabled = bool(self.args.ema or self.tr_cfg.get("ema", True))
        ema_decay = float(self.tr_cfg.get("ema_decay", 0.9999))
        self.ema = ModelEMA(self.model, decay=ema_decay, device=self.device) if self.ema_enabled else None


        # Políticas de guardado (cfg.save con override por CLI)
        cfg_save = getattr(self.cfg, "save", None)
        def _get_save_attr(name, default):
            if self.args is not None and getattr(self.args, name.replace('-', '_'), None) is not None:
                return getattr(self.args, name.replace('-', '_'))
            return getattr(cfg_save, name, default) if cfg_save is not None else default
        self.save_period = int(_get_save_attr("save_period", 10))
        self.keep_checkpoint_max = int(_get_save_attr("keep_checkpoint_max", 5))

        # 6) Artefactos: logger/TB
        self.variant_safe = self.variant or getattr(self.cfg, "default_variant", "m")
        # Elegir run_name (si reanudamos y la ruta contiene '/<variant>/train/<run>/', reutilizamos ese run)
        resume_path = getattr(self.args, "resume", None)
        run_from_resume = None
        if resume_path:
            try:
                rp = Path(resume_path).resolve()
                # Buscamos patrón .../weights/<variant>/train/<run>/file.pt
                parts = rp.parts
                if 'weights' in parts:
                    idx = parts.index('weights')
                    if idx+3 < len(parts):
                        # variant=parts[idx+1], 'train'=parts[idx+2], run=parts[idx+3]
                        if parts[idx+2] == 'train':
                            run_from_resume = parts[idx+3]
            except Exception:
                pass
        self.run_name = run_from_resume or f"yolo11_{self.variant_safe}_train_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.logger = ExperimentLogger(variant=self.variant_safe, phase="train", run_name=self.run_name)

        tb_port = int(self.args.tb_port or os.environ.get("TB_PORT") or os.environ.get("TENSORBOARD_PORT") or 6006)
        tb_host = (self.args.tb_host or os.environ.get("TB_HOST") or "127.0.0.1").strip()
        tb_logdir_parent = os.path.dirname(self.logger.runs_dir)
        tb_logdir_parent_rel = os.path.relpath(tb_logdir_parent, start=os.getcwd())
        tb_cmd = f'tensorboard --logdir "{tb_logdir_parent_rel}" --port {tb_port} --host {tb_host}'
        print(f"[TB] comando: {tb_cmd}")
        if getattr(self.args, "tb_auto", True):
            proc, tb_port = launch_tensorboard(tb_logdir_parent, tb_port, tb_host)
            if proc is None:
                print("[TB] No se pudo iniciar TensorBoard automáticamente; usa el comando anterior.")
            else:
                tb_url = f"http://{tb_host}:{tb_port}/"
                print(f"[TB] TensorBoard · {tb_url}")
                if getattr(self.args, "tb_open_browser", False):
                    try: webbrowser.open(tb_url, new=2, autoraise=False)
                    except Exception: pass

        try:
            self.logger.save_config_json({
                "variant": self.variant_safe, "train": self.tr_cfg,
                "model_meta": {"nc": self.nc, "reg_max": self.reg_max, "strides": self.strides},
                "runtime": runtime_dict
            })
        except Exception:
            pass
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            strides_int = [int(s) for s in (self.strides if isinstance(self.strides, (list, tuple)) else [8, 16, 32])]
            self.logger.save_model_summary(self.model, extra={
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "strides": strides_int
            })
        except Exception:
            pass

        # 7) Utilidades de ejecución
        self.hud = SimpleHUD(verbosity=self.args.verbosity, mode=self.args.hud)
        self.stopper = GracefulStopper(enabled=True)
        self.checkpointer = Checkpointer(base_dir=str(self.cfg.paths.weights_dir), variant=self.variant_safe, run_name=self.run_name, save_period=self.save_period, keep_max=self.keep_checkpoint_max)

        # 8) Evaluador
        conf_thr_eff = float(self.args.conf_thr) if getattr(self.args, 'conf_thr', None) is not None else float(self.tr_cfg.get('conf_thres', 0.001))
        iou_thr_eff = float(self.args.iou_thr) if getattr(self.args, 'iou_thr', None) is not None else float(self.tr_cfg.get('iou_thres', 0.70))
        self.evaluator = Evaluator(imgsz=self.imgsz, nc=self.nc, conf_thr=conf_thr_eff, iou_thr=iou_thr_eff)

        # 9) Early stop (opcional)
        self.patience = max(0, int(self.tr_cfg.get('patience', self.args.patience)))
        # Reanudar entrenamiento si corresponde
        self.start_epoch = 1
        self.resuming = False
        ckpt_path = getattr(self.args, "resume", None)
        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # Cargar estado del modelo
                state = ckpt.get("model_state", None)
                if state is not None:
                    self.model.load_state_dict(state, strict=False)
                    try:
                        if self.ema is not None:
                            self.ema.ema.load_state_dict(state, strict=False)
                    except Exception:
                        pass
                # Opt/Scaler
                opt_state = ckpt.get("optimizer_state", None)
                if opt_state is not None:
                    try: self.optimizer.load_state_dict(opt_state)
                    except Exception as e: print(f"[RESUME] No se pudo cargar optimizer_state: {e}")
                sc_state = ckpt.get("scaler_state", None)
                if sc_state is not None and hasattr(self, "scaler") and self.scaler is not None:
                    try: self.scaler.load_state_dict(sc_state)
                    except Exception: pass
                # Best metric
                if getattr(self, "checkpointer", None):
                    self.checkpointer.best_metric = ckpt.get("best_metric", None)
                # Epochs
                last_epoch = int(ckpt.get("epoch", 0))
                max_epochs_ckpt = int(ckpt.get("max_epochs", 0) or 0)
                # Si no se especifican --epochs, preferir el del checkpoint; si no, YAML
                if self.args.epochs is None:
                    self.epochs = int(max(max_epochs_ckpt, int(self.epochs)))
                # Definir inicio
                self.start_epoch = last_epoch + 1
                self.resuming = True
                # Ajustar scheduler al número de pasos ya recorridos
                try:
                    if hasattr(self, 'scheduler') and hasattr(self, 'steps_per_epoch'):
                        self.scheduler.last_epoch = int(max(0, (self.start_epoch - 1) * self.steps_per_epoch - 1))
                except Exception:
                    pass
                print(f"[RESUME] Reanudando desde epoch {last_epoch} → {self.start_epoch} / {self.epochs}")
            except Exception as e:
                print(f"[RESUME] No se pudo cargar el checkpoint: {e}")
        self.val_interval = int(self.args.val_interval) if (getattr(self.args, "val_interval", None) is not None) else int(self.tr_cfg.get("val_interval", self.tr_cfg.get("val_period", 1)))
        self.val_interval = max(1, self.val_interval)
        self.best_epoch_seen = 0

    def _compute_loss(self, preds, targets):
        """
        Normaliza la salida de self.loss_fn a: (loss_tensor, lbox_float, lcls_float, ldfl_float).
        Acepta salidas tipo Tensor, tuple/list o dict.
        """
        out = self.loss_fn(preds, targets)

        def _to_float(x):
            import torch
            if isinstance(x, (int, float)):
                return float(x)
            if hasattr(x, 'item'):
                try:
                    return float(x.item())
                except Exception:
                    pass
            try:
                return float(x)
            except Exception:
                return 0.0

        import torch
        lbox = lcls = ldfl = 0.0

        if torch.is_tensor(out):
            loss = out
        elif isinstance(out, (list, tuple)):
            loss = out[0]
            if len(out) > 1 and isinstance(out[1], dict):
                d = out[1]
                lbox = _to_float(d.get('box', d.get('loss_box', 0.0)))
                lcls = _to_float(d.get('cls', d.get('loss_cls', 0.0)))
                ldfl = _to_float(d.get('dfl', d.get('loss_dfl', 0.0)))
            else:
                lbox = _to_float(out[1]) if len(out) > 1 else 0.0
                lcls = _to_float(out[2]) if len(out) > 2 else 0.0
                ldfl = _to_float(out[3]) if len(out) > 3 else 0.0
        elif isinstance(out, dict):
            loss = out.get('loss', out.get('total_loss', out.get('total', None)))
            lbox = _to_float(out.get('box', out.get('loss_box', 0.0)))
            lcls = _to_float(out.get('cls', out.get('loss_cls', 0.0)))
            ldfl = _to_float(out.get('dfl', out.get('loss_dfl', 0.0)))
            if loss is None:
                total = _to_float(lbox) + _to_float(lcls) + _to_float(ldfl)
                loss = torch.tensor(total, dtype=torch.float32, device=self.device, requires_grad=True)
            elif not torch.is_tensor(loss):
                loss = torch.tensor(_to_float(loss), dtype=torch.float32, device=self.device, requires_grad=True)
            else:
                loss = loss.to(self.device)
        else:
            raise TypeError(f"Unsupported loss output type: {type(out)}")

        if hasattr(loss, 'device') and loss.device != self.device:
            loss = loss.to(self.device)

        return loss, lbox, lcls, ldfl
    def warmup_if_needed(self) -> bool:
        steps = int(max(0, getattr(self.args, "warmup_steps", 0)))
        if steps <= 0:
            return True
        runner = WarmupRunner(self.model, self.device, verbosity=self.args.verbosity)
        batch = int(max(1, getattr(self.args, "warmup_batch", self.batch)))
        imgsz = int(self.args.warmup_imgsz or self.imgsz)
        amp_flag = self.amp_enabled if (getattr(self.args, 'warmup_amp', None) is None) else bool(self.args.warmup_amp)
        print(f"[WARM-UP] steps={steps} · batch={batch} · imgsz={imgsz} · AMP={'on' if amp_flag else 'off'}")
        try:
            dt_warm = runner.run(steps=steps, batch=batch, imgsz=imgsz, in_ch=int(_meta_get(self.model_meta, "in_channels", 3)), amp=amp_flag, bn_eval_active=self.bn_eval_active)
        except Exception as e:
            if not getattr(self.args, 'force_gn', False):
                try:
                    _swap_bn_to_gn(self.model, groups=getattr(self.args, 'gn_groups', 32))
                    # Mover los nuevos parámetros de GroupNorm al dispositivo actual
                    self.model.to(self.device)
                    print(f"[Norm] BatchNorm2d -> GroupNorm(g={getattr(self.args,'gn_groups',32)}) aplicado automáticamente tras fallo en warm-up.")
                    runner = WarmupRunner(self.model, self.device, verbosity=self.args.verbosity)
                    dt_warm = runner.run(steps=steps, batch=batch, imgsz=imgsz, in_ch=int(_meta_get(self.model_meta, 'in_channels', 3)), amp=amp_flag, bn_eval_active=self.bn_eval_active)
                except Exception as e2:
                    raise e2
            else:
                raise e

        a_gb, _r_gb, t_gb, a_pct, _r_pct = _mem_stats(self.device)
        print(f"[WARM-UP] {dt_warm:.2f}s · VRAM {a_pct:.2f}% ({a_gb:.2f}GB de {t_gb:.2f}GB)")
        if getattr(self.args, "warmup_only", False):
            print("[WARM-UP] --warmup-only activado. Saliendo tras smoketest ✔")
            return False
        return True

    def train(self) -> None:
        if not self.warmup_if_needed():
            try: self.logger.close()
            except Exception: pass
            print(f"[DONE] Entrenamiento finalizado. Revisa '{self.checkpointer.dir}' (best.pt, last.pt).")
            return

        print("Inicializando entrenamiento (tiempo estimado: 3-7 min)")


        # Loop de épocas
        for epoch in range(int(getattr(self, 'start_epoch', 1)), int(self.epochs) + 1):
            self.model.train()
            t_epoch0 = time.time()
            loss_m = box_m = cls_m = dfl_m = 0.0
            steps = len(self.train_loader); it = 0

            self.optimizer.zero_grad(set_to_none=True)

            for batch_i, batch_data in enumerate(self.train_loader):
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    imgs, targets = batch_data[0], batch_data[1]
                else:
                    imgs, targets = batch_data, None

                imgs = imgs.to(self.device, non_blocking=True).float()

                
                if targets is None:
                    targets = torch.zeros((0, 6), device=self.device, dtype=torch.float32)
                else:
                    try:
                        targets = targets.to(self.device, non_blocking=True).float()
                    except AttributeError:
                        import numpy as _np
                        if isinstance(targets, _np.ndarray):
                            targets = torch.from_numpy(targets).to(self.device).float()
                        else:
                            targets = torch.as_tensor(targets, device=self.device, dtype=torch.float32)
                with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                    # Forward con fallback MIOpen (BN.eval() → GN)
                    for _attempt in range(3):
                        try:
                            try:
                                preds = self.model(imgs, decode=False, concat=False)
                            except TypeError:
                                preds = self.model(imgs)
                            break
                        except RuntimeError as e:
                            _msg = str(e).lower()
                            if ("miopen" in _msg) or ("evaluateinvokers" in _msg) or ("sqlite" in _msg) or ("miopenstatusinternalerror" in _msg):
                                if not getattr(self, 'bn_eval_active', False):
                                    print("[MIOpen] RuntimeError en forward → activando BN.eval() y reintentando…")
                                    try:
                                        apply_bn_eval(self.model, verbose=True)
                                        self.bn_eval_active = True
                                    except Exception:
                                        pass
                                    continue
                                if not getattr(self, '_miopen_auto_gn', False):
                                    _g = int(getattr(self.args, 'gn_groups', 32)) if hasattr(self, 'args') else 32
                                    try:
                                        _swap_bn_to_gn(self.model, groups=_g)
                                        # Asegurar que los nuevos módulos GroupNorm residan en el mismo dispositivo
                                        self.model.to(self.device)
                                        self._miopen_auto_gn = True
                                        print(f"[MIOpen] Persisten fallos → BatchNorm2d → GroupNorm(g={_g}) aplicado (fallback runtime). Reintentando…")
                                    except Exception:
                                        pass
                                    continue
                            raise

                    loss, lbox, lcls, ldfl = self._compute_loss(preds, targets)

                loss_scaled = loss / self.grad_accum
                if self.amp_enabled:
                    self.scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                do_step = ((batch_i + 1) % self.grad_accum == 0) or ((batch_i + 1) == steps)
                if do_step:
                    if self.amp_enabled:
                        self.scaler.step(self.optimizer); self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        try: self.scheduler.step()
                        except Exception: pass
                    if self.ema is not None:
                        try: self.ema.update(self.model)
                        except Exception: pass

                it += 1
                loss_m += float(loss.detach().item())
                box_m += float(lbox); cls_m += float(lcls); dfl_m += float(ldfl)

                elapsed = time.time() - t_epoch0
                imgs_per_s = (it * imgs.size(0)) / max(1e-9, elapsed)
                eta_s = (steps - it) * (elapsed / max(1, it))
                alloc_gb, _resv_gb, total_gb, alloc_pct, _resv_pct = _mem_stats(self.device)
                self.hud.live(
                    epoch=epoch, epochs=self.epochs, it=it, it_total=steps,
                    elapsed_s=elapsed, eta_s=eta_s, imgs_per_s=imgs_per_s,
                    lr=self.optimizer.param_groups[0]["lr"], amp=self.amp_enabled, ema=(self.ema is not None),
                    grad_accum=self.grad_accum,
                    loss_avg=loss_m / it, loss_box=box_m / it, loss_cls=cls_m / it, loss_dfl=dfl_m / it,
                    mem_gb=alloc_gb, device_str=self.device.type
                )

                if self.stopper.poll():
                    print("\n[STOP] Solicitud de parada detectada. Guardando 'last.pt' y saliendo...")
                    state = self.ema.ema.state_dict() if self.ema is not None else self.model.state_dict()
                    ckpt_last = {
                        "epoch": epoch,
                        "model_state": state,
                        "optimizer_state": self.optimizer.state_dict(),
                        "scaler_state": (self.scaler.state_dict() if self.amp_enabled else None),
                        "config": {"variant": self.variant_safe, "imgsz": self.imgsz, "batch": self.batch}
                    }
                    torch.save(ckpt_last, os.path.join(self.checkpointer.dir, "last.pt"))
                    try: self.logger.close()
                    except Exception: pass
                    return

            # Fin de época
            loss_epoch = loss_m / max(1, it)
            box_epoch = box_m / max(1, it)
            cls_epoch = cls_m / max(1, it)
            dfl_epoch = dfl_m / max(1, it)

            # Validación
            val_metrics = {"mAP50-95": None, "mAP50": None, "precision": None, "recall": None}
            if (epoch % self.val_interval) == 0:
                model_eval = (self.ema.ema if self.ema is not None else self.model)
                val_metrics = self.evaluator(model_eval, self.val_loader)

            # Logging
            epoch_log = {
                "epoch": int(epoch),
                "max_epochs": int(self.epochs),
                "loss": float(loss_epoch),
                "loss_box": float(box_epoch),
                "loss_cls": float(cls_epoch),
                "loss_dfl": float(dfl_epoch),
            }
            for k, v in val_metrics.items():
                if v is not None:
                    epoch_log[f"val/{k}"] = float(v)
            try: self.logger.log_epoch(epoch_log, split="train")
            except Exception: pass

            # Checkpoints
            ref_metric = val_metrics["mAP50-95"] if val_metrics["mAP50-95"] is not None else (-loss_epoch)
            model_state = (self.ema.ema.state_dict() if self.ema is not None else self.model.state_dict())
            self.checkpointer.save(
                epoch=epoch, max_epochs=int(self.epochs),
                model_state=model_state, optimizer_state=self.optimizer.state_dict(),
                scaler_state=(self.scaler.state_dict() if self.amp_enabled else None),
                metrics=val_metrics, ref_metric=ref_metric
            )
            if (val_metrics["mAP50-95"] is not None) and (self.checkpointer.best_metric == val_metrics["mAP50-95"]):
                self.best_epoch_seen = epoch

            # HUD resumen
            parts = [f"epoch {epoch}/{self.epochs}",
                     f"loss {loss_epoch:.3f} (b {box_epoch:.2f}, c {cls_epoch:.2f}, d {dfl_epoch:.2f})"]
            if val_metrics["mAP50-95"] is not None:
                parts.append(f"val mAP50-95 {val_metrics['mAP50-95']:.3f}")
            if val_metrics["precision"] is not None and val_metrics["recall"] is not None:
                parts.append(f"P {val_metrics['precision']:.3f} | R {val_metrics['recall']:.3f}")
            self.hud.epoch_summary(" | ".join(parts))

            # Early stopping (si aplica)
            if self.patience > 0 and val_metrics["mAP50-95"] is not None:
                epochs_since_best = epoch - max(1, self.best_epoch_seen)
                if epochs_since_best >= (self.patience * self.val_interval):
                    print(f"[EARLY-STOP] Paciencia agotada (sin mejora por {epochs_since_best} épocas).")
                    break

        try: self.logger.close()
        except Exception: pass
        # Mensaje final de confirmación de entrenamiento completado
        print(f"[DONE] Entrenamiento completado. Revisa '{self.checkpointer.dir}' (best.pt, last.pt).")


# ========================= Interfaz / CLI ========================= #

def build_parser() -> argparse.ArgumentParser:
    """Argumentos de entrenamiento con *ayuda enriquecida* mostrando defaults efectivos
    (leídos desde configs/train.yaml y parser.yaml). Los valores por defecto del CLI
    permanecen en None para permitir que Trainer use los del YAML.
    """
    # Leer configs para informar defaults en -h
    try:
        _cfg = ConfigParserYaml().load()
        _train_section = _to_dict_like(getattr(_cfg, "train", {}))
        _tr = _to_dict_like(_train_section.get("config", _train_section))
        _save = getattr(_cfg, "save", None)
        _runtime = _to_dict_like(getattr(_cfg, "runtime", {}))
    except Exception:
        _tr, _save, _runtime = {}, None, {}

    # Helpers
    def _d(key, fallback):
        return _tr.get(key, fallback)
    _valint = _tr.get("val_interval", _tr.get("val_period", 1))
    _amp = bool(_tr.get("amp", True))
    _ema = bool(_tr.get("ema", True))
    _ema_decay = float(_tr.get("ema_decay", 0.9999))
    _lr0 = float(_tr.get("lr0", 0.0025))
    _lrf = float(_tr.get("lrf", 0.1))
    _wu = float(_tr.get("warmup_epochs", 3.0))
    _batch = int(_tr.get("batch", 16))
    _imgsz = int(_tr.get("imgsz", 640))
    _epochs = int(_tr.get("epochs", 150))
    _ga = int(_tr.get("grad_accum", 1))
    _conf = float(_tr.get("conf_thres", 0.001))
    _iou = float(_tr.get("iou_thres", 0.70))
    _workers = int(_to_dict_like(_tr.get("dataloader", {})).get("workers", _tr.get("workers", 4)))
    _save_period = int(getattr(_save, "save_period", 10) if _save is not None else 10)
    _keep_max = int(getattr(_save, "keep_checkpoint_max", 5) if _save is not None else 5)

    p = argparse.ArgumentParser(
        prog="YOLOv11 — Entrenamiento",
        description=(
            "Entrena variantes YOLOv11 con validación interna, HUD compacto, F8 stop, AMP/EMA opcionales.\n"
            f"Defaults YAML → imgsz={_imgsz}, epochs={_epochs}, batch={_batch}, grad_accum={_ga}, "
            f"val_interval={_valint}, lr0={_lr0}, lrf={_lrf}, warmup_epochs={_wu}, "
            f"amp={'on' if _amp else 'off'}, ema={'on' if _ema else 'off'}."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--variant", type=str, default=None, help=f"n/s/m/l/xl (default YAML: {_cfg.default_variant_name})")
    p.add_argument("--epochs", type=int, default=None, help=f"Épocas totales (default YAML: {_epochs})")
    p.add_argument("--batch", type=int, default=None, help=f"Tamaño de batch (default YAML: {_batch})")
    p.add_argument("--imgsz", type=int, default=None, help=f"Tamaño de imagen cuadrado (default YAML: {_imgsz})")
    p.add_argument("--device", type=str, default=None, help=f"cpu/cuda/mps (default runtime: {_runtime.get('device', 'auto')})")
    p.add_argument("--amp", action="store_true", help=f"Habilitar AMP explícitamente (default YAML: {'on' if _amp else 'off'})")
    p.add_argument("--ema", action="store_true", help=f"Habilitar EMA explícitamente (default YAML: {'on' if _ema else 'off'}, decay={_ema_decay})")
    p.add_argument("--grad-accum", type=int, default=None, help=f"Acumulación de gradiente (default YAML: {_ga})")
    # Frecuencias
    p.add_argument("--val-interval", type=int, default=None, help=f"Validar cada N épocas (default YAML: {_valint})")
    p.add_argument("--pr-curves-every", type=int, default=10, help="Guardar PR-curves cada N val. (default: 10)")
    p.add_argument("--cm-every", type=int, default=10, help="Guardar matriz de confusión cada N val. (default: 10)")
    p.add_argument("--overlay-every", type=int, default=10, help="Guardar overlays GT/Pred cada N val. (default: 10)")
    # Checkpoints / early stop
    p.add_argument("--save-period", type=int, default=None, help=f"Guardar checkpoint cada N épocas (default YAML: {_save_period})")
    p.add_argument("--keep-checkpoint-max", type=int, default=None, help=f"Nº máx de checkpoints periódicos (default YAML: {_keep_max})")
    p.add_argument("--patience", type=int, default=50, help="en unidades de validación (se multiplica por val_interval)")
    # Umbrales inferencia validación
    p.add_argument("--conf-thr", type=float, default=None, help=f"Umbral de confianza (default YAML: {_conf})")
    p.add_argument("--iou-thr", type=float, default=None, help=f"Umbral de IoU para NMS (default YAML: {_iou})")
    # Overrides de optimización/planificación
    p.add_argument("--lr0", type=float, default=None, help=f"Override LR inicial (AdamW) — default YAML: {_lr0}")
    p.add_argument("--lrf", type=float, default=None, help=f"Factor mínimo LR (cosine) — default YAML: {_lrf}")
    p.add_argument("--warmup-epochs", type=float, default=None, help=f"Override warmup epochs del scheduler — default YAML: {_wu}")
    # Reanudar
    p.add_argument("--resume", type=str, default=None, help="Ruta a un checkpoint .pt para reanudar (last.pt o VAR_Train_Epoch_XXX.pt)")
    # Verbosidad HUD
    p.add_argument("--verbosity", type=str, choices=["v0", "v1", "v2"], default="v1", help="Nivel de logs en consola (v1 por defecto)")
    # HUD (one/two/off)
    p.add_argument("--hud", type=str, choices=["one", "two", "off"], default="two", help="Formato del HUD en consola (two por defecto)")
    # Wizard interactivo
    p.add_argument("--interactive", action="store_true", help="Inicia asistente interactivo antes de entrenar")
    # Mitigación ROCm/MIOpen
    p.add_argument("--bn-eval-fallback", action="store_true", help="Fuerza BatchNorm en eval() desde el inicio para evitar fallos MIOpen")

    # MIOpen (variables de entorno controladas por CLI)
    p.add_argument("--miopen-disable-cache", action="store_true",
                   help="Establece MIOPEN_DISABLE_CACHE=1 para esta ejecución")
    p.add_argument("--miopen-find-enforce", type=int, choices=[0,1,2,3], default=None,
                   help="Establece MIOPEN_FIND_ENFORCE={0,1,2,3} (1=fuerza búsqueda, 3=Find+Immediate)")
    p.add_argument('--force-gn', action='store_true', help='Reemplaza BatchNorm2d por GroupNorm')
    p.add_argument('--gn-groups', type=int, default=32, help='Grupos para GroupNorm (default 32)')
    # Warm-up (integrado)
    p.add_argument("--warmup-steps", type=int, default=3, help="Pasos sintéticos de warm-up para compilar kernels (default: 3)")
    p.add_argument("--warmup-batch", type=int, default=1, help=f"Tamaño de batch usado durante el warm-up (default: 1; entrena con batch={_batch})")
    p.add_argument("--warmup-imgsz", type=int, default=None, help=f"imgsz específico para warm-up (por defecto usa --imgsz / YAML: {_imgsz})")
    p.add_argument('--warmup-amp', type=int, choices=[0,1], default=None, help='Forzar AMP en warm-up (1=on, 0=off; default: usa AMP del entrenamiento)')
    p.add_argument("--warmup-only", action="store_true", help="Ejecuta solo el warm-up y sale (smoketest)")
    # TensorBoard / Auto-lanzamiento
    p.add_argument("--tb-auto", dest="tb_auto", action="store_true", help="Auto-lanzar TensorBoard (default: on)")
    p.add_argument("--no-tb-auto", dest="tb_auto", action="store_false", help="No lanzar TensorBoard automáticamente")
    p.add_argument("--tb-port", type=int, default=None, help="Puerto TensorBoard (default: 6006)")
    p.add_argument("--tb-host", type=str, default=None, help="Host TensorBoard (default: 127.0.0.1)")
    p.set_defaults(tb_auto=True)
    p.add_argument("--tb-open-browser", action="store_true", help="Abrir URL en el navegador al lanzar TB")
    # Ajustes de aprendizaje temprano
    p.add_argument("--freeze-epochs", type=int, default=0, help="Congelar backbone los primeros K epochs (no implementado)")
    p.add_argument("--conf-ramp-epochs", type=int, default=5, help="Rampa de confianza en validación hasta --conf-thr (no implementado)")
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


# ========================= Entrypoint ========================= #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()


    # === MIOpen via CLI (aplica variables de entorno para esta ejecución) ===
    if getattr(args, "miopen_disable_cache", False):
        os.environ["MIOPEN_DISABLE_CACHE"] = "1"
    if getattr(args, "miopen_find_enforce", None) is not None:
        os.environ["MIOPEN_FIND_ENFORCE"] = str(int(args.miopen_find_enforce))
    # Construcción + Entrenamiento
    trainer = Trainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[STOP] Entrenamiento interrumpido por el usuario (F8/Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()



# ==============================================================