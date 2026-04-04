# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/valid.py
# Descripción: Script de entrada (CLI) para validación de DETR.
#              Carga modelo, pesos (con búsqueda automática) y
#              ejecuta el reporte completo de métricas.
# ==============================================================

import argparse
import yaml
import sys
import os
from pathlib import Path

# --- CONFIGURACIÓN DE ENTORNO ROCm/MIOPEN ---
FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parent
if str(DETR_ROOT) not in sys.path:
    sys.path.append(str(DETR_ROOT))

from engine.bootstrap_miopen import bootstrap, MIOpenConfig


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    print(f"[valid.py] Iniciando script de validación...")
    parser = argparse.ArgumentParser(description="DETR Validation CLI")
    parser.add_argument("--cfg-valid", type=str, default="DETR/configs/valid.yaml")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # 1. Cargar configuraciones
    valid_cfg = load_yaml(args.cfg_valid)
    dataset_cfg = load_yaml(valid_cfg['paths']['dataset_cfg'])
    variants_cfg = load_yaml(valid_cfg['paths']['variants_cfg'])

    # 2. Aplicar Preset
    if args.preset:
        if args.preset in valid_cfg.get('presets', {}):
            overrides = valid_cfg['presets'][args.preset].get('overrides', {})
            for section, values in overrides.items():
                valid_cfg[section].update(values)
            print(f"[valid.py] Preset '{args.preset}' aplicado correctamente.")
        else:
            print(f"[valid.py] ERROR FATAL: El preset '{args.preset}' no existe en el archivo YAML.")
            sys.exit(1)

    # 3. Bootstrap MIOpen
    mi_cfg = valid_cfg['miopen']
    bootstrap(MIOpenConfig(
        find_mode=mi_cfg['find_mode'],
        user_db_path=mi_cfg['user_db_path'],
        disable_cache=True,
        verbose=mi_cfg['verbose']
    ))

    # --- INICIALIZACIÓN (DESPUÉS DEL BOOTSTRAP) ---
    import torch
    from engine.warnings import install_global_warning_filters
    from engine.Validator import Validator
    from utility.data_loader import build_dataloader

    install_global_warning_filters()

    # 4. Preparar argumentos del modelo (Namespace Dummy para evitar AttributeErrors)
    v_name = valid_cfg['validation']['variant']
    run_name = valid_cfg['validation']['run_name']
    v_params = variants_cfg['variants'][v_name]

    # Diccionario base con todos los atributos que DETR podría pedir internamente
    base_args = {
        'lr_backbone': 0,  # No importa en validación, pero debe existir
        'masks': False,
        'frozen_weights': None,
        'aux_loss': False,
        'set_cost_class': 1.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,
        'bbox_loss_coef': 5.0,
        'giou_loss_coef': 2.0,
        'eos_coef': 0.1,
        'dataset_file': 'coco',
        'device': args.device or valid_cfg['validation']['device']
    }

    # Actualizar con los parámetros específicos de la variante (Transformer, Backbone, etc.)
    base_args.update(v_params)

    # Convertir a Namespace
    model_args = argparse.Namespace(**base_args)

    # 5. Auto-descubrimiento de Pesos
    weights_path = args.weights or valid_cfg['validation']['weights']
    if not weights_path:
        auto_path = DETR_ROOT / "runs" / v_name / "train" / run_name / "weights" / "best.pt"
        if auto_path.exists():
            weights_path = str(auto_path)
            print(f"[valid.py] Pesos cargados: {weights_path}")
        else:
            print(f"[valid.py] ERROR: No se encontraron pesos en: {auto_path}")
            return

    if not os.path.exists(weights_path):
        print(f"[valid.py] ERROR: La ruta de pesos no existe: {weights_path}")
        return

    # 6. Construir Modelo y Cargar Pesos
    from models import build_model
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig

    print(f"[valid.py] Cargando modelo {v_name}...")
    model, criterion, postprocessors = build_model(model_args)

    hidden_dim = model.transformer.d_model
    model.class_embed = torch.nn.Linear(hidden_dim, dataset_cfg['nc'] + 1)

    # [CORRECCIÓN]: Aplicar el parche BN2GN ANTES de cargar los pesos
    if valid_cfg['bn2gn']['policy'] == 'on':
        replace_bn_with_gn(model, BN2GNConfig(policy='on'))

    # Cargar pesos en la arquitectura ya parcheada
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    device = torch.device(model_args.device)
    model.to(device)

    # 7. Ejecutar Reporte de Validación
    val_loader = build_dataloader(valid_cfg['validation']['phase'], valid_cfg['validation']['batch_size'])
    save_dir = DETR_ROOT / "metrics" / "detect" / v_name / valid_cfg['validation']['phase'] / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    validator = Validator(model, criterion, postprocessors, device)
    print(f"[valid.py] --- Iniciando Reporte de Validación: {run_name} ---")
    class_names = list(dataset_cfg['names'].values())

    # Ejecutar reporte leyendo parámetros del YAML
    metrics = validator.run_full_report(
        val_loader,
        save_dir,
        class_names,
        plot_ratio=valid_cfg['validation'].get('plot_ratio', 0.20),
        max_images=valid_cfg['validation'].get('max_images', 50)
    )

    with open(save_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    print(f"[valid.py] Reporte finalizado. Resultados en: {save_dir}")


if __name__ == "__main__":
    main()