# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/valid.py
# Descripción: Script de entrada (CLI) para validación de DETR.
#              Carga modelo, pesos (con auto-descubrimiento) y
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
    if args.preset and args.preset in valid_cfg.get('presets', {}):
        overrides = valid_cfg['presets'][args.preset].get('overrides', {})
        for section, values in overrides.items():
            valid_cfg[section].update(values)

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

    # 4. Preparar argumentos del modelo
    v_name = valid_cfg['validation']['variant']
    run_name = valid_cfg['validation']['run_name']
    v_params = variants_cfg['variants'][v_name]
    model_args = argparse.Namespace(**v_params)

    model_args.bbox_loss_coef = 5.0
    model_args.giou_loss_coef = 2.0
    model_args.eos_coef = 0.1
    model_args.aux_loss = False
    model_args.masks = False
    model_args.dataset_file = 'coco'
    model_args.device = args.device or valid_cfg['validation']['device']

    # 5. Auto-descubrimiento de Pesos
    weights_path = args.weights or valid_cfg['validation']['weights']
    if not weights_path:
        auto_path = DETR_ROOT / "runs" / v_name / "train" / run_name / "weights" / "best.pt"
        if auto_path.exists():
            weights_path = str(auto_path)
            print(f"[valid.py] Auto-descubierto peso: {weights_path}")
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

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    if valid_cfg['bn2gn']['policy'] == 'on':
        replace_bn_with_gn(model, BN2GNConfig(policy='on'))

    device = torch.device(model_args.device)
    model.to(device)

    # 7. Ejecutar Reporte de Validación
    val_loader = build_dataloader(valid_cfg['validation']['phase'], valid_cfg['validation']['batch_size'])
    save_dir = DETR_ROOT / "metrics" / "detect" / v_name / valid_cfg['validation']['phase'] / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    validator = Validator(model, criterion, postprocessors, device)
    print(f"[valid.py] --- Iniciando Reporte de Validación: {run_name} ---")
    class_names = list(dataset_cfg['names'].values())
    metrics = validator.run_full_report(val_loader, save_dir, class_names)

    with open(save_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    print(f"[valid.py] Reporte finalizado. Resultados en: {save_dir}")


if __name__ == "__main__":
    main()