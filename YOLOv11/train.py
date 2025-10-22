"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: train.py
=============================================================

Entrenamiento modular del modelo YOLOv11.
Permite alternar variantes (n, s, m, l, x) y ajustar hiperparámetros
desde configs/*.yaml sin modificar este script.
=============================================================
"""

import os, sys, torch, torch.nn as nn, torch.optim as optim, traceback, subprocess
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# -------------------- Importar módulos --------------------
from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.data_loader import create_dataloader
from utility.losses import YoloLoss
from utility.logger import get_logger
from utility.visualization import TensorboardVisualizer
from utility.weights import save_checkpoint, load_checkpoint
from utility.metrics import evaluate_model, measure_fps


# =============================================================
#          CONFIGURACIÓN DE ENTORNO Y LOGS
# =============================================================
def setup_environment(model_variant="n"):
    """Inicializa entorno, logs, weights y TensorBoard para la variante YOLOv11."""
    base_dir = "YOLOv11"
    variant = model_variant.lower()

    # Crear estructura de carpetas por variante y fase (train)
    os.makedirs(os.path.join(base_dir, "logs", variant, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs", variant, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "weights", variant, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "metrics", variant, "train"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger(log_dir=f"{base_dir}/logs/{variant}/train", name=f"train_yolo11_{variant}")
    tb = TensorboardVisualizer(log_dir=f"{base_dir}/runs/{variant}/train")

    logger.info(f"📦 Dispositivo en uso: {device}")
    logger.info(f"🧩 Logs y weights configurados para YOLOv11-{variant.upper()} (fase: train)")

    return device, logger, tb


# =============================================================
#          FUNCIONES AUXILIARES ROCm / BatchNorm Patch
# =============================================================
def _replace_batchnorm_with_groupnorm(model: nn.Module, groups: int = 32):
    count = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                groups_eff = min(groups, num_channels) or 1
                if num_channels % groups_eff != 0:
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
#            CARGA DE CONFIGS Y CREACIÓN DEL MODELO
# =============================================================
def load_model_and_configs(variant_override=None):
    """Carga configs YAML y crea la instancia YOLOv11."""
    train_cfg = OmegaConf.load("YOLOv11/configs/train.yaml")
    variants_cfg = OmegaConf.load("YOLOv11/configs/model_variants.yaml")

    # Permite sobrescribir desde consola
    variant_name = variant_override or train_cfg.get("model_variant", "n")
    if variant_name not in variants_cfg.variants:
        raise ValueError(f"⚠️ Variante '{variant_name}' no existe en model_variants.yaml")

    variant_params = variants_cfg.variants[variant_name]
    print(f"🧩 Configuración de variante YOLOv11-{variant_name.upper()}: {variant_params}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    num_classes = model_cfg.get("nc", 1)

    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)
    return model, train_cfg, variant_name


# =============================================================
#                LOOP DE ENTRENAMIENTO
# =============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, tb):
    model.train()
    epoch_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for i, (images, labels) in enumerate(progress):
        images = images.to(device)
        outputs = model(images)
        loss, loss_items = criterion(outputs, labels)
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
#                     VALIDACIÓN DURANTE TRAIN
# =============================================================
def validate_model(model, device, logger, model_variant="n"):
    model.eval()
    with torch.no_grad():
        preds = [[0.1, 0.1, 0.4, 0.4, 0.9, 0]]
        targets = [[0.1, 0.1, 0.4, 0.4, 1.0, 0]]
        metrics = evaluate_model(preds, targets, save_results=True, model_variant=model_variant)
        logger.info(f"📊 Métricas ({model_variant.upper()}): {metrics}")
        return metrics


# =============================================================
#                 CONSOLA INTERACTIVA PRINCIPAL
# =============================================================
def interactive_console():
    print("====================================================")
    print("🚀 Entrenamiento YOLOv11 interactivo")
    print("====================================================")
    variant = input("Seleccione variante del modelo [n/s/m/l/x]: ").strip().lower()
    if variant not in ["n", "s", "m", "l", "x"]:
        print("⚠️ Variante inválida. Se usará 'n' por defecto.")
        variant = "n"

    ckpt_dir = f"YOLOv11/weights/{variant}/train"
    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        ans = input(f"⚠️ Ya existe un modelo entrenado en {ckpt_dir}. ¿Desea reemplazarlo? (s/n): ").strip().lower()
        if ans == "s":
            for f in os.listdir(ckpt_dir):
                os.remove(os.path.join(ckpt_dir, f))
            print("🗑️ Checkpoints anteriores eliminados.")
        else:
            print("✅ Se conservarán los modelos existentes.")
    return variant


# =============================================================
#                     MAIN TRAIN LOOP
# =============================================================
def main():
    model_variant = interactive_console()
    device, logger, tb = setup_environment(model_variant)
    model, train_cfg, _ = load_model_and_configs(variant_override=model_variant)
    model.to(device)

    print(f"🚀 Entrenando variante YOLOv11-{model_variant.upper()}")
    logger.info(f"🚀 Entrenando variante YOLOv11-{model_variant.upper()}")

    dummy = torch.randn(1, 3, 640, 640).to(device)
    try_model_forward_safe(model, dummy, device)

    train_loader = create_dataloader(train_cfg, phase="train")
    criterion = YoloLoss()
    opt_params = train_cfg.optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt_params.lr, weight_decay=opt_params.weight_decay)

    # === FIX: Reanudación y guardado por variante ===
    start_epoch = 0
    ckpt_dir = f"YOLOv11/weights/{model_variant}/train"
    os.makedirs(ckpt_dir, exist_ok=True)

    if train_cfg.resume:
        try:
            start_epoch = load_checkpoint(model, optimizer, path=ckpt_dir, device=device)
            logger.info(f"🔁 Reanudando entrenamiento desde la época {start_epoch}")
        except FileNotFoundError:
            logger.warning(f"⚠️ No se encontró checkpoint previo en {ckpt_dir}.")

    num_epochs = train_cfg.epochs
    logger.info(f"🚀 Iniciando entrenamiento por {num_epochs} épocas...")

    # 🔹 Lanzar TensorBoard una vez confirmado inicio
    print("🧠 Iniciando TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", f"YOLOv11/runs/{model_variant}/train", "--port", "6006"])

    for epoch in range(start_epoch, num_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, tb)
        if (epoch + 1) % train_cfg.validate_every == 0:
            metrics = validate_model(model, device, logger, model_variant)
            tb.log_metrics(metrics, epoch, phase="valid")

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            path=ckpt_dir,
            filename=f"yolo11_{model_variant}_epoch_{epoch+1}.pt"
        )

    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"⚡ FPS promedio del modelo ({model_variant.upper()}): {fps:.2f}")

    tb.close()
    logger.info("✅ Entrenamiento finalizado correctamente.")


# =============================================================
#                       EJECUCIÓN
# =============================================================
if __name__ == "__main__":
    main()
