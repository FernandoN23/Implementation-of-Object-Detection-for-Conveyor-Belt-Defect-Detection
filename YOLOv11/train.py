"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: train.py
=============================================================

Entrenamiento modular del modelo YOLOv11.
Permite alternar variantes (n, s, m, l, x) y ajustar hiperparámetros
de entrenamiento desde configs/*.yaml sin modificar este script.
Los logs, checkpoints y TensorBoard se guardan dentro de YOLOv11/.
=============================================================
"""

import os, sys

# ========= Forzar UTF-8 en consola Windows =========
os.environ["PYTHONIOENCODING"] = "utf-8"

# ========= Rutas para ejecución desde la raíz =========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "YOLOv11"))

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm

# -------------------- Importar utilidades --------------------
from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser

from utility.data_loader import create_dataloader
from utility.losses import YoloLoss
from utility.logger import get_logger
from utility.visualization import TensorboardVisualizer
from utility.weights import save_checkpoint, load_checkpoint
from utility.metrics import evaluate_model, measure_fps
# =============================================================
#                 CONFIGURACIÓN GENERAL
# =============================================================
def setup_environment():
    """Inicializa entorno, logs, dispositivo y visualizador."""
    os.makedirs("YOLOv11/logs", exist_ok=True)
    os.makedirs("YOLOv11/runs", exist_ok=True)
    os.makedirs("YOLOv11/checkpoints", exist_ok=True)
    os.makedirs("YOLOv11/metrics", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(log_dir="YOLOv11/logs", name="train_yolo11")
    tb = TensorboardVisualizer(log_dir="YOLOv11/runs/yolo11_train")

    logger.info(f"📦 Dispositivo en uso: {device}")
    return device, logger, tb


import torch.nn as nn
import traceback

# =============================================================
# 🔩 Funciones auxiliares para GPUs AMD ROCm
# =============================================================
# --- en train.py: reemplaza el parche BN->Identity por BN->GroupNorm ---
def _replace_batchnorm_with_groupnorm(model: nn.Module, groups: int = 32):
    count = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                groups_eff = min(groups, num_channels) or 1
                if num_channels % groups_eff != 0:
                    # ajusta groups a un divisor de num_channels
                    for g in (32, 16, 8, 4, 2, 1):
                        if num_channels % g == 0:
                            groups_eff = g
                            break
                setattr(module, name, nn.GroupNorm(groups_eff, num_channels, affine=True))
                count += 1
    print(f"🩹 {count} capas BatchNorm2d reemplazadas por GroupNorm (compatibilidad ROCm).")
    return count

def try_model_forward_safe(model, dummy, device):
    try:
        model.eval()
        with torch.no_grad():
            _ = model(dummy)
        print("✅ Forward exitoso en GPU ROCm.")
    except Exception as e:
        msg = str(e).lower()
        print("❌ Error detectado en forward inicial.")
        if "miopen" in msg:
            print("🩹 Parcheando modelo → reemplazando BatchNorm2d por GroupNorm...")
            _replace_batchnorm_with_groupnorm(model)
            model.eval()
            with torch.no_grad():
                _ = model(dummy)
            print("✅ Forward corregido y validado en GPU.")
        else:
            raise
# =============================================================
#                 CONFIGURACIÓN DEL MODELO
# =============================================================
def load_model_and_configs():
    """Carga configuraciones YAML (train y modelo) y crea instancia YOLOv11."""
    # Configs
    train_cfg = OmegaConf.load("YOLOv11/configs/train.yaml")
    model_cfg_path = "YOLOv11/configs/yolo11.yaml"

    # Parse del modelo
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()

    num_classes = model_cfg.get("nc", 1)
    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)

    return model, train_cfg, model_cfg


# =============================================================
#                 LOOP PRINCIPAL DE ENTRENAMIENTO
# =============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, tb):
    """Ejecuta una época completa de entrenamiento."""
    model.train()
    epoch_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for i, (images, labels) in enumerate(progress):
        images = images.to(device)
        labels = labels.to(device)  # Placeholder actual (0s)

        # Forward
        outputs = model(images)

        # 🔧 Dummy targets adaptados a cada escala del modelo (sin réplica de anchors)
        targets = []
        if isinstance(outputs, (list, tuple)):
            for o in outputs:
                # Salida: [B, C, H, W] → convertir a [B, H, W, C]
                if o.ndim == 4:
                    t = o.permute(0, 2, 3, 1).contiguous() * 0
                else:
                    t = torch.zeros_like(o, dtype=o.dtype, device=o.device)
                targets.append(t)
        else:
            t = outputs.permute(0, 2, 3, 1).contiguous() * 0
            targets = [t]

        # Loss
        loss, loss_items = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())

        if i % 10 == 0:
            logger.info(f"[Epoch {epoch+1} | Step {i}] Loss total: {loss_items['total_loss']:.4f}")
            tb.log_metrics({"loss": loss_items["total_loss"]}, epoch * len(dataloader) + i, phase="train")

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"📉 Epoch {epoch+1} finalizado | Loss promedio: {avg_loss:.4f}")
    tb.log_metrics({"epoch_loss": avg_loss}, epoch, phase="train")

    return avg_loss


# =============================================================
#                     VALIDACIÓN
# =============================================================
def validate_model(model, device, logger):
    """Validación de métricas básicas con dummy preds/targets (prototipo)."""
    model.eval()
    with torch.no_grad():
        # Dummy data (se reemplazará por post-procesado real)
        preds = [[0.1, 0.1, 0.4, 0.4, 0.9, 0]]
        targets = [[0.1, 0.1, 0.4, 0.4, 1.0, 0]]

        metrics = evaluate_model(preds, targets, save_results=True)
        logger.info(f"📊 Métricas: {metrics}")
        return metrics


# =============================================================
#                 RUTINA PRINCIPAL DE ENTRENAMIENTO
# =============================================================
def main():
    device, logger, tb = setup_environment()

    # Cargar modelo y configs
    model, train_cfg, model_cfg = load_model_and_configs()
    model.to(device)

    # --- Verificación de compatibilidad ROCm ---
    dummy = torch.randn(1, 3, 640, 640).to(device)
    try_model_forward_safe(model, dummy, device)

    # Crear DataLoader (usa estructura de train.yaml)
    train_loader = create_dataloader(train_cfg)

    # Criterio de pérdida y optimizador
    criterion = YoloLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.optimizer.lr, weight_decay=train_cfg.optimizer.weight_decay)

    start_epoch = 0
    if train_cfg.resume and os.path.exists(train_cfg.checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, path=train_cfg.checkpoint_path, device=device)

    # Entrenamiento principal
    num_epochs = train_cfg.epochs
    logger.info(f"🚀 Iniciando entrenamiento por {num_epochs} épocas...")

    for epoch in range(start_epoch, num_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, tb)

        # Validación cada N épocas
        if (epoch + 1) % train_cfg.validate_every == 0:
            metrics = validate_model(model, device, logger)
            tb.log_metrics(metrics, epoch, phase="valid")

        # Guardar checkpoint
        save_checkpoint(model, optimizer, epoch + 1,
                        path="YOLOv11/checkpoints",
                        filename=f"yolo11_epoch_{epoch+1}.pt")

    # Evaluación de rendimiento (FPS)
    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"⚡ FPS promedio del modelo: {fps:.2f}")

    tb.close()
    logger.info("✅ Entrenamiento finalizado correctamente.")


# =============================================================
#                       EJECUCIÓN
# =============================================================
if __name__ == "__main__":
    main()
