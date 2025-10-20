"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: valid.py
=============================================================

Validación externa del modelo YOLOv11.
Permite evaluar un conjunto independiente (external set)
usando métricas de rendimiento y guardando resultados.
=============================================================
"""

import os, sys, torch
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# -------------------- Módulos del proyecto --------------------
from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.data_loader import create_dataloader
from utility.logger import get_logger
from utility.metrics import evaluate_model, measure_fps
from utility.weights import load_checkpoint
from utility.visualization import TensorboardVisualizer


# =============================================================
#        CONFIGURACIÓN DE ENTORNO Y LOGS PARA VALIDACIÓN
# =============================================================
def setup_environment(model_variant="n"):
    """Inicializa entorno y logs para la validación externa."""
    base_dir = "YOLOv11"
    variant = model_variant.lower()

    os.makedirs(os.path.join(base_dir, "logs", variant), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "metrics", variant), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs", variant), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger(log_dir=f"{base_dir}/logs/{variant}", name=f"valid_yolo11_{variant}")
    tb = TensorboardVisualizer(log_dir=f"{base_dir}/runs/{variant}/yolo11_valid")

    logger.info(f"📦 Dispositivo en uso: {device}")
    logger.info(f"🧩 Validación configurada para YOLOv11-{variant.upper()}")
    return device, logger, tb


# =============================================================
#            CARGA DEL MODELO Y CONFIGURACIONES
# =============================================================
def load_model_and_configs():
    """Carga configuración y modelo para validación externa."""
    valid_cfg = OmegaConf.load("YOLOv11/configs/valid.yaml")
    variants_cfg = OmegaConf.load("YOLOv11/configs/model_variants.yaml")

    variant_name = valid_cfg.get("model_variant", "n")
    if variant_name not in variants_cfg.variants:
        raise ValueError(f"⚠️ Variante '{variant_name}' no existe en model_variants.yaml")

    variant_params = variants_cfg.variants[variant_name]
    print(f"🧩 Configuración de variante YOLOv11-{variant_name.upper()}: {variant_params}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    num_classes = model_cfg.get("nc", 1)

    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)
    return model, valid_cfg, variant_name


# =============================================================
#                FUNCIÓN PRINCIPAL DE VALIDACIÓN
# =============================================================
def run_validation(model, dataloader, device, logger, tb, model_variant):
    model.eval()
    all_preds, all_targets = [], []

    logger.info("🚀 Iniciando validación externa...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validando"):
            images = images.to(device)
            preds = model(images)
            all_preds.extend(preds)
            all_targets.extend(labels)

    # Evaluar métricas globales
    metrics = evaluate_model(all_preds, all_targets, save_results=True, model_variant=model_variant)
    logger.info(f"📊 Resultados finales ({model_variant.upper()}): {metrics}")
    tb.log_metrics(metrics, 0, phase="valid")

    # FPS (rendimiento)
    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"⚡ FPS promedio en validación: {fps:.2f}")

    return metrics


# =============================================================
#                      MAIN LOOP
# =============================================================
def main():
    device, logger, tb = setup_environment()
    model, valid_cfg, model_variant = load_model_and_configs()
    model.to(device)

    # === Carga de checkpoint a validar ===
    ckpt_dir = f"YOLOv11/checkpoints/{model_variant}"
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"❌ No se encontró la carpeta de checkpoints en: {ckpt_dir}")

    last_ckpt = sorted(os.listdir(ckpt_dir))[-1]
    ckpt_path = os.path.join(ckpt_dir, last_ckpt)
    load_checkpoint(model, path=ckpt_path, device=device)
    logger.info(f"✅ Checkpoint cargado desde {ckpt_path}")

    # === Crear dataloader externo ===
    valid_loader = create_dataloader(valid_cfg)

    # === Ejecutar validación ===
    metrics = run_validation(model, valid_loader, device, logger, tb, model_variant)

    tb.close()
    logger.info("✅ Validación externa completada correctamente.")


# =============================================================
#                        EJECUCIÓN
# =============================================================
if __name__ == "__main__":
    main()
