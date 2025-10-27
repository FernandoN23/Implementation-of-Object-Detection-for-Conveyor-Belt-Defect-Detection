# YOLOv11/train.py
# =============================================================
#  Trabajo de Memoria de Título - Fernando Navarrete (Feña)
#  Script: train.py (robusto)
#  Objetivo: Entrenamiento modular YOLOv11 con:
#   - AMP autocast + GradScaler
#   - Acumulación de gradientes y clipping
#   - Scheduler (cosine/one-cycle) seleccionable
#   - Early Stopping por pérdida de entrenamiento
#   - Checkpointing: last/best + reanudar
#   - CSV + TensorBoard sin duplicados (puerto libre)
#   - Parche GN automático en ROCm si BN falla
#   - Señales SIGINT (Ctrl+C) y tecla F8 opcional
# =============================================================

from __future__ import annotations
import os, sys, gc, time, socket, math, csv, signal, subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Dependencias críticas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from omegaconf import OmegaConf
from tqdm import tqdm

# Backend (headless)
import matplotlib
matplotlib.use("Agg")

# Dependencias opcionales
try:
    import psutil
except Exception:
    psutil = None

try:
    import keyboard  # <- opcional y puede no estar en tu entorno
    KEYBOARD_AVAILABLE = True
except Exception:
    KEYBOARD_AVAILABLE = False

# Proyecto
from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.data_loader import create_dataloader
from utility.losses import YoloLoss
from utility.logger import get_logger
from utility.visualization import TensorboardVisualizer
from utility.weights import save_checkpoint, load_checkpoint


# =============================================================
#  Utilidades de entorno
# =============================================================
def seed_everything(seed: int = 42, deterministic: bool = False):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def limit_vram_usage(device: torch.device, fraction: float = 0.8):
    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            print(f"⚙️  Límite VRAM: {fraction*100:.0f}%")
        except Exception as e:
            print(f"⚠️ No se pudo limitar VRAM: {e}")


def _replace_bn_with_gn(model: nn.Module, groups_default: int = 32) -> int:
    """Reemplaza BatchNorm2d->GroupNorm de forma segura (ROCm)."""
    count = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                c = child.num_features
                # Encuentra un número de grupos que divida c
                groups_eff = next((g for g in (groups_default, 16, 8, 4, 2, 1) if c % g == 0), 1)
                setattr(module, name, nn.GroupNorm(groups_eff, c, affine=True))
                count += 1
    return count


def forward_sanity(model: nn.Module, device: torch.device, imgsz: int = 640):
    """Prueba de forward; si falla por ROCm/miopen aplica GN y reintenta."""
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    model.eval()
    with torch.no_grad():
        try:
            _ = model(x)
            print("✅ Forward OK.")
            return
        except Exception as e:
            msg = str(e).lower()
            is_rocm = hasattr(torch.version, "hip") or (getattr(torch.version, "cuda", "") and "rocm" in torch.version.cuda.lower())
            if is_rocm or "miopen" in msg:
                print("🔧 ROCm/MIOpen detectado. Parchando BN->GN…")
                n = _replace_bn_with_gn(model)
                print(f"🩹 {n} capas BN reemplazadas por GN.")
                _ = model(x)  # reintento
                print("✅ Forward corregido (ROCm).")
            else:
                raise


def ensure_dirs(base: str, variant: str):
    for sub in ("logs", "runs", "weights", "metrics"):
        Path(base, sub, variant, "train").mkdir(parents=True, exist_ok=True)


def tb_singleton(logdir: str, base_port: int = 6006, max_port: int = 6015):
    if psutil is None:
        print("ℹ️ psutil no disponible: no puedo verificar instancias de TensorBoard.")
    else:
        # ¿ya corre TB para este logdir?
        for p in psutil.process_iter(attrs=["name", "cmdline"]):
            try:
                nm = (p.info["name"] or "").lower()
                cmd = " ".join(p.info["cmdline"] or [])
                if "tensorboard" in nm or "tensorboard" in cmd:
                    if logdir in cmd:
                        print(f"ℹ️ TensorBoard ya activo para {logdir}")
                        return
            except Exception:
                pass

    # encuentra puerto libre
    def find_free_port():
        for port in range(base_port, max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return None

    free = find_free_port()
    if free is None:
        print("⚠️ Sin puerto libre para TensorBoard (6006–6015).")
        return
    try:
        subprocess.Popen(
            ["tensorboard", f"--logdir={logdir}", f"--port={free}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"🔗 TensorBoard: http://localhost:{free}")
    except Exception as e:
        print(f"⚠️ No se pudo lanzar TensorBoard: {e}")


# =============================================================
#  Carga de configs y modelo
# =============================================================
def load_configs(variant_override: Optional[str] = None):
    train_cfg = OmegaConf.load("YOLOv11/configs/train.yaml")
    variants_cfg = OmegaConf.load("YOLOv11/configs/model_variants.yaml")
    model_cfg_path = "YOLOv11/configs/yolo11.yaml"

    variant = str(variant_override or train_cfg.get("model_variant", "n")).strip().lower()
    if "variants" not in variants_cfg or variant not in variants_cfg.variants:
        raise ValueError(f"Variante '{variant}' no existe en model_variants.yaml")

    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    nc = int(model_cfg.get("nc", 1))
    return train_cfg, variants_cfg.variants[variant], model_cfg_path, nc, variant


def build_model(model_cfg_path: str, nc: int) -> nn.Module:
    model = YOLOv11(cfg_path=model_cfg_path, num_classes=nc)
    return model


# =============================================================
#  Optimizador y scheduler
# =============================================================
def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float, momentum: float | None = None):
    name = (name or "AdamW").lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum or 0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "radam":
        return optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizador no soportado: {name}")


def build_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, lrf: float = 0.01):
    name = (name or "cosine").lower()
    if name == "cosine":
        # One-cycle estilo YOLO (cosine warmdown simple)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    if name == "linear":
        lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    if name == "none":
        return None
    raise ValueError(f"Scheduler no soportado: {name}")


# =============================================================
#  Entrenamiento por época
# =============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger,
    tb: TensorboardVisualizer,
    global_step: int,
    amp_enabled: bool,
    grad_clip: float,
    accumulate: int,
) -> tuple[float, int]:
    model.train()
    running = 0.0
    nb = len(loader)
    pbar = tqdm(enumerate(loader), total=nb, desc=f"Epoch {epoch+1}", leave=False)

    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in pbar:
        if isinstance(images, torch.Tensor):
            images = images.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss, loss_items = criterion(outputs, labels)

        # acumulación
        loss_scaled = loss / accumulate
        scaler.scale(loss_scaled).backward()

        if (i + 1) % accumulate == 0:
            # clipping
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.detach().item())
        step = global_step + i
        tb.log_metrics({"train/step_loss": float(loss_items.get("total_loss", loss.detach().item()))}, step, phase="train")
        pbar.set_postfix(loss=float(loss.detach().item()))

        # Parada manual con F8 (si está disponible)
        if KEYBOARD_AVAILABLE and keyboard.is_pressed("f8"):
            print("\n⚠️ F8 detectado. Deteniendo con checkpoint seguro…")
            return running / (i + 1), -1  # código de parada

    epoch_loss = running / max(nb, 1)
    logger.info(f"📉 Epoch {epoch+1} | train_loss={epoch_loss:.5f}")
    tb.log_metrics({"train/epoch_loss": epoch_loss}, epoch, phase="train")
    return epoch_loss, nb


# =============================================================
#  Early Stopping simple (por pérdida de entrenamiento)
# =============================================================
class EarlyStopper:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0

    def step(self, value: float) -> bool:
        if value + self.min_delta < self.best:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience


# =============================================================
#  MAIN
# =============================================================
def main(variant_cli: Optional[str] = None):
    BASE = "YOLOv11"
    # Interfaz mínima (CLI opcional)
    variant = (variant_cli or input("Seleccione variante [n/s/m/l/x]: ").strip().lower() or "n")
    if variant not in ("n", "s", "m", "l", "x"):
        print("⚠️ Variante inválida, usando 'n'.")
        variant = "n"

    # Señales (Ctrl+C => guarda last y sale limpio)
    stop_flag = {"stop": False}
    def _sigint_handler(signum, frame):
        print("\n🛑 SIGINT recibido. Deteniendo tras el batch actual…")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    # Configuración base
    train_cfg, variant_cfg, model_cfg_path, nc, variant = load_configs(str(variant))
    ensure_dirs(BASE, variant)
    device = select_device()
    limit_vram_usage(device, fraction=float(train_cfg.get("gpu_memory_fraction", 0.8)))

    logger = get_logger(log_dir=f"{BASE}/logs/{variant}/train", name=f"train_yolo11_{variant}")
    tb = TensorboardVisualizer(log_dir=f"{BASE}/runs/{variant}/train")

    # Semillas
    seed_everything(int(train_cfg.get("seed", 42)), deterministic=bool(train_cfg.get("deterministic", False)))

    # Modelo
    model = build_model(model_cfg_path, nc=nc).to(device)
    # Sanity ROCm
    forward_sanity(model, device, imgsz=int(train_cfg.get("imgsz", 640)))

    # Data
    train_loader = create_dataloader(train_cfg, phase="train")

    # Loss, Optimizer, Scheduler
    criterion = YoloLoss()
    optimizer = build_optimizer(
        model,
        name=str(train_cfg.optimizer.get("name", "AdamW")),
        lr=float(train_cfg.optimizer.get("lr", 1e-3)),
        weight_decay=float(train_cfg.optimizer.get("weight_decay", 0.05)),
        momentum=float(train_cfg.optimizer.get("momentum", 0.9)),
    )
    scheduler = build_scheduler(
        optimizer,
        name=str(train_cfg.get("scheduler", "cosine")),
        epochs=int(train_cfg.get("epochs", 100)),
        lrf=float(train_cfg.get("lrf", 0.01)),
    )

    # AMP + acumulación
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    accumulate = int(max(1, train_cfg.get("accumulate", 1)))
    grad_clip = float(train_cfg.get("grad_clip", 10.0))

    # Checkpoints
    ckpt_dir = f"{BASE}/weights/{variant}/train"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    last_path = Path(ckpt_dir, "last.pt")
    best_path = Path(ckpt_dir, "best.pt")
    start_epoch = 0
    best_loss = float("inf")

    # Reanudar (si resume: true en train.yaml)
    if bool(train_cfg.get("resume", False)):
        try:
            start_epoch = load_checkpoint(model, optimizer, path=ckpt_dir, device=device)
            # Si el loader cambia, no apagues resume global: partimos desde start_epoch
            logger.info(f"🔁 Reanudando desde época {start_epoch}")
            if best_path.exists():
                _, ckpt = load_checkpoint(best_path)
                best_loss = float(ckpt.get("metrics", {}).get("train/epoch_loss", best_loss))
        except FileNotFoundError:
            logger.warning("ℹ️ No se encontró checkpoint previo. Entrenamiento limpio.")

    # CSV metrics
    csv_path = Path(f"{BASE}/metrics/{variant}/train/results.csv")
    if start_epoch == 0 and csv_path.exists():
        csv_path.unlink()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # TensorBoard único
    tb_singleton(logdir=f"{BASE}/runs/{variant}/train")

    # EarlyStoppper
    es = EarlyStopper(
        patience=int(train_cfg.get("early_stop_patience", 15)),
        min_delta=float(train_cfg.get("early_stop_min_delta", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 100))
    global_step = start_epoch * len(train_loader)
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        if stop_flag["stop"]:
            logger.info("🛑 Detención por señal de usuario antes de iniciar la época.")
            break

        # step epoch
        epoch_loss, nbatches = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            logger=logger,
            tb=tb,
            global_step=global_step,
            amp_enabled=amp_enabled,
            grad_clip=grad_clip,
            accumulate=accumulate,
        )

        if nbatches == -1:  # parada con F8
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                       Path(ckpt_dir, f"yolo11_{variant}_interrupted_epoch_{epoch+1}.pt"))
            logger.info("🛑 Entrenamiento detenido por usuario (F8).")
            break

        # scheduler
        if scheduler is not None:
            scheduler.step()

        # CSV logging
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["epoch", "train/epoch_loss", "lr", "time"])
            lr_now = optimizer.param_groups[0]["lr"]
            w.writerow([epoch + 1, f"{epoch_loss:.6f}", f"{lr_now:.6e}", f"{time.time()-t0:.2f}"])

        # Checkpoints (last y best)
        save_checkpoint(
            model, optimizer, epoch + 1, path=ckpt_dir, filename="last.pt",
            extra={"metrics": {"train/epoch_loss": epoch_loss}, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        )
        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            save_checkpoint(
                model, optimizer, epoch + 1, path=ckpt_dir, filename="best.pt",
                extra={"metrics": {"train/epoch_loss": best_loss}, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
            )

        global_step += max(nbatches, 0)

        # Early stop
        if es.step(epoch_loss):
            logger.info(f"⏹️ Early stopping activado en epoch {epoch+1}. Mejor pérdida: {best_loss:.6f}")
            break

        # ¿Ctrl+C solicitado?
        if stop_flag["stop"]:
            logger.info("🛑 Detención por señal de usuario tras cerrar la época.")
            break

        # Limpieza VRAM/CPU moderada
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    tb.close()
    logger.info("✅ Entrenamiento finalizado.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error fatal en entrenamiento: {e}")
        sys.exit(1)
