# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/train.py
# Descripción: Punto de entrada principal para entrenamiento DETR.
#              Orquestador de CLI, configuración YAML y parches ROCm.
# ==============================================================

import argparse
import yaml
import sys
import os
from pathlib import Path

# --- PASO 1: BOOTSTRAP MIOPEN (Antes de importar torch) ---
# Necesario para estabilidad en Windows ROCm Preview
FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parent
if str(DETR_ROOT) not in sys.path:
    sys.path.append(str(DETR_ROOT))

from engine.bootstrap_miopen import bootstrap, MIOpenConfig


def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DETR Training CLI - Conveyor Belt Defects")
    parser.add_argument("--cfg-train", type=str, default="DETR/configs/train.yaml")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # 2. Cargar configuraciones maestros
    train_cfg = load_yaml(args.cfg_train)
    dataset_cfg = load_yaml(train_cfg['paths']['dataset_cfg'])
    variants_cfg = load_yaml(train_cfg['paths']['variants_cfg'])

    # 3. Aplicar Preset si se solicita
    if args.preset and args.preset in train_cfg.get('presets', {}):
        overrides = train_cfg['presets'][args.preset].get('overrides', {})
        for section, values in overrides.items():
            train_cfg[section].update(values)

    # 4. Ejecutar Bootstrap de MIOpen
    mi_cfg = train_cfg['miopen']
    bootstrap(MIOpenConfig(
        find_mode=mi_cfg['find_mode'],
        user_db_path=mi_cfg['user_db_path'],
        disable_cache=True,
        verbose=mi_cfg['verbose']
    ))

    # --- PASO 2: IMPORTAR TORCH Y MOTOR (DESPUÉS DEL BOOTSTRAP) ---
    import torch
    from engine.warnings import install_global_warning_filters
    from engine.Trainer import Trainer, TrainerConfig

    install_global_warning_filters()

    # 5. Preparar argumentos arquitectónicos para el submódulo DETR
    variant_name = train_cfg['training']['variant']
    v_params = variants_cfg['variants'][variant_name]

    # Construir Namespace compatible con detr/models/detr.py::build
    model_args = argparse.Namespace(**v_params)
    for k in ['bbox_loss_coef', 'giou_loss_coef', 'eos_coef', 'aux_loss']:
        setattr(model_args, k, train_cfg['training'][k])

    model_args.set_cost_class = train_cfg['training'].get('set_cost_class', 1.0)
    model_args.set_cost_bbox = train_cfg['training'].get('set_cost_bbox', 5.0)
    model_args.set_cost_giou = train_cfg['training'].get('set_cost_giou', 2.0)

    model_args.masks = False  # Enfocado en detección de cajas
    model_args.dataset_file = 'coco'
    model_args.device = args.device or train_cfg['training']['device']

    # 6. Configurar Trainer
    cfg = TrainerConfig(
        variant=variant_name,
        run_name=train_cfg['training']['run_name'],
        epochs=train_cfg['training']['epochs'],
        batch_size=train_cfg['training']['batch_size'],
        lr=train_cfg['training']['lr'],
        lr_backbone=train_cfg['training']['lr_backbone'],
        weight_decay=train_cfg['training']['weight_decay'],
        lr_drop=train_cfg['training']['lr_drop'],
        clip_max_norm=train_cfg['training']['clip_max_norm'],
        pretrain_weights=train_cfg['training']['pretrain_weights'],
        nc=dataset_cfg['nc'],
        device=model_args.device,
        model_args=model_args,
        bn2gn_policy=train_cfg['bn2gn']['policy']
    )

    trainer = Trainer(cfg)

    print(f"\n[train.py] Setup completado. Modelo listo para entrenar con {cfg.nc} clases.")
    print("Siguiente paso: Implementar utility/data_loader.py")
    # trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()