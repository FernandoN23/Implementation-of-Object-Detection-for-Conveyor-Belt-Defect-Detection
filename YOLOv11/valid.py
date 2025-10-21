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

import os, sys, torch, subprocess
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
    """Inicializa entorno, logs y rutas para validación externa."""
    base_dir = "YOLOv11"
    variant = model_variant.lower()

    os.makedirs(os.path.join(base_dir, "logs", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "metrics", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "weights", variant, "train"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(log_dir=f"{base_dir}/logs/{variant}/valid", name=f"valid_yolo11_{variant}")
    tb = TensorboardVisualizer(log_dir=f"{base_dir}/runs/{variant}/valid")

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
    print(f"🧩 Configuración base YOLOv11-{variant_name.upper()}: {variant_params}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    num_classes = model_cfg.get("nc", 1)

    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)
    return model, valid_cfg, variant_name


# =============================================================
#       FUNCIÓN PARA ELEGIR EL ENTRENAMIENTO A VALIDAR
# =============================================================
def select_training_variant():
    """Permite elegir interactivamente el entrenamiento a validar (por letra)."""
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

    metrics = evaluate_model(all_preds, all_targets, save_results=True, model_variant=model_variant)
    logger.info(f"📊 Resultados finales ({model_variant.upper()}): {metrics}")
    tb.log_metrics(metrics, 0, phase="valid")

    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"⚡ FPS promedio en validación: {fps:.2f}")

    return metrics


# =============================================================
#                      MAIN LOOP
# =============================================================
def main():
    model_variant = select_training_variant()
    device, logger, tb = setup_environment(model_variant)
    model, valid_cfg, _ = load_model_and_configs()
    model.to(device)

    # === Carga de pesos ===
    ckpt_dir = os.path.abspath(os.path.join("YOLOv11", "weights", model_variant, "train"))
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"❌ No se encontró la carpeta de pesos: {ckpt_dir}")

    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpt_files:
        raise FileNotFoundError(f"⚠️ No hay archivos de pesos en {ckpt_dir}")

    print("\n📦 Checkpoints disponibles:")
    for i, f in enumerate(ckpt_files, 1):
        print(f"  [{i}] {f}")

    choice = input("Selecciona el número del checkpoint a validar (Enter para el último): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(ckpt_files):
        ckpt_selected = ckpt_files[int(choice) - 1]
    else:
        ckpt_selected = ckpt_files[-1]

    ckpt_path = os.path.normpath(os.path.join(ckpt_dir, ckpt_selected))
    print(f"\n📁 Cargando checkpoint: {ckpt_path}")

    # 🔧 Cargar desde carpeta base (sin warning)
    logger.info(f"🧠 Cargando pesos desde carpeta base: {ckpt_dir}")
    load_checkpoint(model, path=ckpt_dir, device=device)
    logger.info(f"✅ Pesos cargados correctamente ({ckpt_selected})")

    # === Crear dataloader externo ===
    valid_loader = create_dataloader(valid_cfg)

    # 🔹 Lanzar TensorBoard justo antes de iniciar la validación
    print("🧠 Iniciando TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", f"YOLOv11/runs/{model_variant}/valid", "--port", "6006"])
    print("🔗 Visualiza métricas en: http://localhost:6006")
    logger.info(f"📊 TensorBoard activo y registrando en: YOLOv11/runs/{model_variant}/valid")

    # === Ejecutar validación ===
    metrics = run_validation(model, valid_loader, device, logger, tb, model_variant)

    tb.close()
    logger.info("✅ Validación externa completada correctamente.")


# =============================================================
#                        EJECUCIÓN
# =============================================================
if __name__ == "__main__":
    main()
