
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
            dm = DetMetricsYOLOv11()
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
                dm.add_batch(detections=[d.detach().cpu().numpy() for d in dets], ground_truth=batch_gt, imgsz=self.imgsz)
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
    def __init__(self, base_dir: str, variant: str, run_name: str, save_period: int = 10):
        self.dir = os.path.join(base_dir, variant, "train", run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.best_metric: Optional[float] = None
        self.save_period = max(1, int(save_period))

    def save(self, *, epoch: int, model_state: Dict[str, Any], optimizer_state: Dict[str, Any],
             scaler_state: Optional[Dict[str, Any]], metrics: Dict[str, Optional[float]],
             ref_metric: float) -> None:
        is_best = (self.best_metric is None) or (ref_metric > self.best_metric)
        if is_best:
            self.best_metric = ref_metric
        ckpt = {
            "epoch": epoch,
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


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # 1) Cargar configs
        self.cfg = ConfigParserYaml().load()
        if args.interactive:
            self.args = interactive_wizard(self.cfg, args)

        # 2) Resolver secciones
        self.variant = self.args.variant or getattr(self.cfg, "default_variant", None)
        train_section = _to_dict_like(getattr(self.cfg, "train", {}))
        self.tr_cfg = _to_dict_like(train_section.get("config", {}))

        imgsz_for_strides = self.args.imgsz or self.tr_cfg.get("imgsz", 640)
        self.model = self.cfg.build_model(variant=self.variant, imgsz_for_strides=imgsz_for_strides)

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

        dl_cfg = _to_dict_like(self.tr_cfg.get("dataloader", {}))
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
        loss_weights = _to_dict_like(self.tr_cfg.get("loss weights", {"box": 7.5, "cls": 0.5, "dfl": 1.5}))
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

        # 6) Artefactos: logger/TB
        self.variant_safe = self.variant or getattr(self.cfg, "default_variant", "m")
        self.run_name = f"yolo11_{self.variant_safe}_train_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
        self.checkpointer = Checkpointer(base_dir="weights", variant=self.variant_safe, run_name=self.run_name, save_period=int(self.args.save_period))

        # 8) Evaluador
        self.evaluator = Evaluator(imgsz=self.imgsz, nc=self.nc, conf_thr=float(self.args.conf_thr), iou_thr=float(self.args.iou_thr))

        # 9) Early stop (opcional)
        self.patience = max(0, int(self.args.patience))
        self.val_interval = max(1, int(self.args.val_interval))
        self.grad_accum = max(1, int(self.args.grad_accum))
        self.best_epoch_seen = 0

    def _compute_loss(self, preds, targets):
        out = self.loss_fn(preds, targets)
        # Soporte tuple/list/dict
        if isinstance(out, (tuple, list)):
            loss = out[0]; lbox = float(out[1]) if len(out) > 1 else 0.0; lcls = float(out[2]) if len(out) > 2 else 0.0; ldfl = float(out[3]) if len(out) > 3 else 0.0
            return loss, lbox, lcls, ldfl
        if isinstance(out, dict):
            return out.get("loss", None), float(out.get("loss_box", out.get("box", 0.0))), float(out.get("loss_cls", out.get("cls", 0.0))), float(out.get("loss_dfl", out.get("dfl", 0.0)))
        return out, 0.0, 0.0, 0.0

    def warmup_if_needed(self) -> bool:
        steps = int(max(0, getattr(self.args, "warmup_steps", 0)))
        if steps <= 0:
            return True
        runner = WarmupRunner(self.model, self.device, verbosity=self.args.verbosity)
        batch = int(max(1, getattr(self.args, "warmup_batch", self.batch)))
        imgsz = int(getattr(self.args, "warmup_imgsz", self.imgsz))
        amp_flag = bool(self.args.amp or self.tr_cfg.get("amp", True))
        print(f"[WARM-UP] steps={steps} · batch={batch} · imgsz={imgsz} · AMP={'on' if amp_flag else 'off'}")
        dt_warm = runner.run(steps=steps, batch=batch, imgsz=imgsz, in_ch=int(_meta_get(self.model_meta, "in_channels", 3)), amp=amp_flag, bn_eval_active=self.bn_eval_active)
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
            return

        # Loop de épocas
        for epoch in range(1, int(self.epochs) + 1):
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

                with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                    try:
                        preds = self.model(imgs, decode=False, concat=False)
                    except TypeError:
                        preds = self.model(imgs)
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
                epoch=epoch, model_state=model_state, optimizer_state=self.optimizer.state_dict(),
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


# ========================= Interfaz / CLI ========================= #
def build_parser() -> argparse.ArgumentParser:
    """Argumentos de entrenamiento (conserva defaults del proyecto)."""
    p = argparse.ArgumentParser(
        prog="YOLOv11 — Entrenamiento",
        description="Entrena variantes YOLOv11 con validación interna, HUD compacto, F8 stop, AMP/EMA opcionales.",
    )
    p.add_argument("--variant", type=str, default=None, help="n/s/m/l/xl o usar default del parser.yaml")
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
    p.add_argument("--resume", type=str, default=None, help="ruta a last.pt (no implementado en esta versión)")
    # Verbosidad HUD
    p.add_argument("--verbosity", type=str, choices=["v0", "v1", "v2"], default="v1")
    # HUD (one/two/off)
    p.add_argument("--hud", type=str, choices=["one", "two", "off"], default="two")
    # Wizard interactivo
    p.add_argument("--interactive", action="store_true", help="Inicia asistente interactivo antes de entrenar")
    # Mitigación ROCm/MIOpen
    p.add_argument("--bn-eval-fallback", action="store_true", help="Fuerza BatchNorm en eval() desde el inicio para evitar fallos MIOpen")
    # Warm-up (integrado)
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

    # Construcción + Entrenamiento
    trainer = Trainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[STOP] Entrenamiento interrumpido por el usuario (F8/Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
