"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: valid.py
=============================================================

Validación externa del modelo YOLOv11 con registro de pérdida
para comparar curvas train vs valid en TensorBoard.
TensorBoard se lanza solo al final de la validación.
=============================================================
"""

import os, sys, torch, subprocess, keyboard
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.data_loader import create_dataloader
from utility.logger import get_logger
from utility.metrics import evaluate_model, measure_fps
from utility.weights import load_checkpoint
from utility.losses import YoloLoss
from utility.visualization import TensorboardVisualizer


# =============================================================
#  CONFIGURACIÓN DE ENTORNO Y LOGS
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
    logger.info(f"🧩 Validación configurada para YOLOv11-{variant.upper()}")
    return device, logger


# =============================================================
#  CARGA DEL MODELO Y CONFIGS
# =============================================================
def load_model_and_configs():
    valid_cfg = OmegaConf.load("YOLOv11/configs/valid.yaml")
    variants_cfg = OmegaConf.load("YOLOv11/configs/model_variants.yaml")

    variant_name = valid_cfg.get("model_variant", "n")
    if variant_name not in variants_cfg.variants:
        raise ValueError(f"⚠️ Variante '{variant_name}' no existe en model_variants.yaml")

    variant_params = variants_cfg.variants[variant_name]
    print(f"🧩 Configuración base YOLOv11-{variant_name.upper()}: {variant_params}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    num_classes = model_cfg.get("nc", 1)

    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)
    return model, valid_cfg, variant_name


# =============================================================
#   SELECCIÓN INTERACTIVA DE VARIANTE
# =============================================================
def select_training_variant():
    base_path = "YOLOv11/weights"
    if not os.path.exists(base_path):
        print("⚠️ No se encontró la carpeta 'YOLOv11/weights'.")
        sys.exit(1)

    variants = sorted([v for v in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, v))])
    if not variants:
        print("⚠️ No hay variantes de entrenamiento disponibles en 'weights/'.")
        sys.exit(1)

    print("\n📂 Variantes disponibles para validación externa:")
    for v in variants:
        print(f"  • {v.upper()}")

    while True:
        choice = input("\n👉 Ingresa la letra de la variante a validar (ej: n, s, m, l, x): ").strip().lower()
        if choice in variants:
            print(f"✅ Variante seleccionada: {choice}")
            return choice
        else:
            print("⚠️ Variante no válida. Intenta nuevamente.")


# =============================================================
#        FUNCIÓN DE VALIDACIÓN (CON LOSS)
# =============================================================
def _to_cpu(x):
    """Convierte recursivamente tensores o colecciones a CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return [_to_cpu(i) for i in x]
    elif isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


def run_validation(model, dataloader, device, logger, model_variant):
    """Ejecuta validación externa con cálculo de pérdida."""
    model.eval()
    criterion = YoloLoss()
    all_preds, all_targets = [], []
    total_loss = 0.0

    logger.info("🚀 Iniciando validación externa...")
    print("\n🧩 Presiona ESC en cualquier momento para detener la validación de forma segura.\n")

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Validando")):
            if keyboard.is_pressed("esc"):
                print("\n🛑 Validación interrumpida manualmente (ESC presionado).")
                logger.info("🛑 Validación interrumpida manualmente por el usuario (ESC).")
                return None

            images = images.to(device, non_blocking=True)
            preds = model(images)
            loss, _ = criterion(preds, labels)
            total_loss += loss.item()

            all_preds.append(_to_cpu(preds))
            all_targets.append(_to_cpu(labels))

    # === Promedio de pérdida y métricas finales ===
    avg_loss = total_loss / len(dataloader)
    logger.info(f"📉 Loss promedio validación: {avg_loss:.4f}")

    metrics = evaluate_model(all_preds, all_targets, save_results=True, model_variant=model_variant, phase="valid")
    logger.info(f"📊 Resultados finales ({model_variant.upper()}): {metrics}")

    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"⚡ FPS promedio en validación: {fps:.2f}")
    return metrics, avg_loss


# =============================================================
#                    MAIN PRINCIPAL
# =============================================================
def main():
    model_variant = select_training_variant()
    device, logger = setup_environment(model_variant)
    model, valid_cfg, _ = load_model_and_configs()
    model.to(device)

    ckpt_dir = os.path.abspath(os.path.join("YOLOv11", "weights", model_variant, "train"))
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpt_files:
        raise FileNotFoundError(f"⚠️ No hay archivos de pesos en {ckpt_dir}")

    print("\n📦 Checkpoints disponibles:")
    for i, f in enumerate(ckpt_files, 1):
        print(f"  [{i}] {f}")

    choice = input("Selecciona el número del checkpoint a validar (Enter para el último): ").strip()
    ckpt_selected = ckpt_files[int(choice)-1] if choice.isdigit() else ckpt_files[-1]

    ckpt_path = os.path.join(ckpt_dir, ckpt_selected)
    print(f"\n📁 Cargando checkpoint: {ckpt_path}")
    load_checkpoint(model, path=ckpt_path, device=device)
    logger.info(f"✅ Pesos cargados correctamente ({ckpt_selected})")

    valid_loader = create_dataloader(valid_cfg, phase="valid")

    # 🔹 Ejecutar validación sin lanzar TensorBoard todavía
    results = run_validation(model, valid_loader, device, logger, model_variant)
    if results is None:
        print("\n🛑 Validación detenida manualmente.\n")
        sys.exit(0)

    metrics, avg_loss = results
    logger.info("✅ Validación externa completada correctamente.")

    # 🔹 Registrar métricas y pérdida en TensorBoard al final
    tb = TensorboardVisualizer(log_dir=f"YOLOv11/runs/{model_variant}/valid")
    tb.log_metrics({"val_loss": avg_loss}, 0, phase="valid")
    for k, v in metrics.items():
        tb.log_metrics({k: v}, 0, phase="valid")
    tb.close()

    # 🔹 Lanzar TensorBoard recién al final
    print("\n🧠 Iniciando TensorBoard con resultados de validación...")
    subprocess.Popen(["tensorboard", "--logdir", f"YOLOv11/runs/{model_variant}", "--port", "6006"])
    print("🔗 Visualiza resultados en: http://localhost:6006")

    print(f"\n📉 Loss validación promedio: {avg_loss:.4f}")
    print("📊 Métricas:", metrics)


if __name__ == "__main__":
    main()
