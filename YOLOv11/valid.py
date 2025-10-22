"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: valid.py
=============================================================

Validación externa del modelo YOLOv11.
Evalúa métricas de desempeño y curva de pérdida
comparando entrenamiento vs validación.
=============================================================
"""
import os, sys, torch, subprocess, keyboard
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.data_loader import create_dataloader
from utility.logger import get_logger
from utility.metrics import evaluate_model, measure_fps
from utility.weights import load_checkpoint
from utility.losses import YoloLoss
from utility.visualization import TensorboardVisualizer


# =============================================================
# CONFIGURACIÓN DE ENTORNO
# =============================================================
def setup_environment(model_variant="n"):
    base_dir = "YOLOv11"
    variant = model_variant.lower()
    os.makedirs(os.path.join(base_dir, "logs", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "metrics", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs", variant, "valid"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(log_dir=f"{base_dir}/logs/{variant}/valid", name=f"valid_yolo11_{variant}")
    logger.info(f"📦 Dispositivo en uso: {device}")
    return device, logger


# =============================================================
# CARGA DEL MODELO
# =============================================================
def load_model_and_configs():
    valid_cfg = OmegaConf.load("YOLOv11/configs/valid.yaml")
    variants_cfg = OmegaConf.load("YOLOv11/configs/model_variants.yaml")
    variant_name = valid_cfg.get("model_variant", "n")
    variant_params = variants_cfg.variants[variant_name]
    print(f"🧩 Configuración YOLOv11-{variant_name.upper()}: {variant_params}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    num_classes = model_cfg.get("nc", 1)
    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)
    return model, valid_cfg, variant_name


# =============================================================
# VALIDACIÓN PRINCIPAL
# =============================================================
def run_validation(model, dataloader, device, logger, model_variant, tb=None):
    model.eval()
    criterion = YoloLoss()
    all_preds, all_targets = [], []
    total_loss, val_loss_history = 0.0, []

    logger.info("🚀 Iniciando validación externa...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Validando")):
            if keyboard.is_pressed("esc"):
                logger.info("🛑 Validación interrumpida (ESC).")
                return None

            images = images.to(device)
            preds = model(images)
            loss, _ = criterion(preds, labels)
            total_loss += loss.item()
            val_loss_history.append(loss.item())
            all_preds.append(preds)
            all_targets.append(labels)

            if tb:
                tb.log_metrics({"val_loss": loss.item()}, i, phase="valid")

    avg_loss = total_loss / len(dataloader)
    metrics = evaluate_model(all_preds, all_targets, save_results=True,
                             model_variant=model_variant, phase="valid")
    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"📉 Loss validación promedio: {avg_loss:.4f} | ⚡ FPS: {fps:.2f}")
    return metrics, avg_loss, val_loss_history


# =============================================================
# MAIN
# =============================================================
def main():
    base_train_loss_path = "YOLOv11/metrics/train_loss_history.pt"
    model_variant = input("👉 Variante a validar (n/s/m/l/x): ").strip().lower()
    device, logger = setup_environment(model_variant)
    model, valid_cfg, _ = load_model_and_configs()
    model.to(device)

    ckpt_dir = os.path.join("YOLOv11", "weights", model_variant, "train")
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpt_files:
        raise FileNotFoundError(f"⚠️ No hay checkpoints en {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    print(f"📁 Cargando checkpoint: {ckpt_path}")
    load_checkpoint(model, path=ckpt_path, device=device)

    valid_loader = create_dataloader(valid_cfg, phase="valid")
    tb = TensorboardVisualizer(log_dir=f"YOLOv11/runs/{model_variant}/valid")

    results = run_validation(model, valid_loader, device, logger, model_variant, tb)
    if results is None:
        sys.exit(0)

    metrics, avg_loss, val_loss_history = results
    tb.log_metrics({"val_loss_final": avg_loss}, 0, phase="valid")
    tb.close()

    # === Curvas de pérdida ===
    save_dir = f"YOLOv11/metrics/{model_variant}/valid"
    os.makedirs(save_dir, exist_ok=True)
    train_loss_history = torch.load(base_train_loss_path) if os.path.exists(base_train_loss_path) else []

    plt.figure(figsize=(8, 5))
    if train_loss_history:
        plt.plot(train_loss_history, label="Entrenamiento", color="blue")
    plt.plot(val_loss_history, label="Validación", color="orange")
    plt.title(f"Curva de pérdida - YOLOv11-{model_variant.upper()}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_vs_valid_loss.png"))
    plt.close()

    print(f"\n📉 Loss validación promedio: {avg_loss:.4f}")
    print("📊 Métricas:", metrics)
    subprocess.Popen(["tensorboard", "--logdir", f"YOLOv11/runs/{model_variant}", "--port", "6006"])
    print("🔗 Visualiza resultados en: http://localhost:6006")


if __name__ == "__main__":
    main()
