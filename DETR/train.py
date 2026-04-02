# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/train.py
# Descripción: Punto de entrada CLI para el entrenamiento de DETR.
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
    print(f"[train.py] Iniciando script de entrenamiento...")
    parser = argparse.ArgumentParser(description="DETR Training CLI - Belt Defects")
    parser.add_argument("--cfg-train", type=str, default="DETR/configs/train.yaml")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                        help="Reanudar entrenamiento. Si se usa sin valor, intenta autodescubrir el último run.")
    args = parser.parse_args()

    # 1. Cargar configuraciones
    train_cfg = load_yaml(args.cfg_train)
    dataset_cfg = load_yaml(train_cfg['paths']['dataset_cfg'])
    variants_cfg = load_yaml(train_cfg['paths']['variants_cfg'])

    # 2. Aplicar Overrides de Preset
    if args.preset and args.preset in train_cfg.get('presets', {}):
        overrides = train_cfg['presets'][args.preset].get('overrides', {})
        for section, values in overrides.items():
            train_cfg[section].update(values)

    # 3. Bootstrap MIOpen
    mi_cfg = train_cfg['miopen']
    bootstrap(MIOpenConfig(
        find_mode=mi_cfg['find_mode'],
        user_db_path=mi_cfg['user_db_path'],
        disable_cache=True,
        verbose=mi_cfg['verbose']
    ))

    # --- INICIALIZACIÓN DE MOTOR ---
    import torch
    from engine.warnings import install_global_warning_filters
    from engine.Trainer import Trainer, TrainerConfig

    install_global_warning_filters()

    # 4. Preparar argumentos del modelo
    v_name = train_cfg['training']['variant']
    v_params = variants_cfg['variants'][v_name]

    model_args = argparse.Namespace(**v_params)
    for k in ['bbox_loss_coef', 'giou_loss_coef', 'eos_coef', 'aux_loss', 'lr_backbone']:
        setattr(model_args, k, train_cfg['training'][k])

    model_args.set_cost_class = train_cfg['training'].get('set_cost_class', 1.0)
    model_args.set_cost_bbox = train_cfg['training'].get('set_cost_bbox', 5.0)
    model_args.set_cost_giou = train_cfg['training'].get('set_cost_giou', 2.0)
    model_args.frozen_weights = None
    model_args.masks = False
    model_args.dataset_file = 'coco'
    model_args.device = train_cfg['training']['device']

    resume_val = args.resume if args.resume is not None else train_cfg['training'].get('resume', False)

    # 5. Instanciar TrainerConfig
    cfg = TrainerConfig(
        variant=v_name,
        run_name=train_cfg['training']['run_name'],
        epochs=train_cfg['training']['epochs'],
        batch_size=train_cfg['training']['batch_size'],
        lr=train_cfg['training']['lr'],
        lr_backbone=train_cfg['training']['lr_backbone'],
        weight_decay=train_cfg['training']['weight_decay'],
        lr_drop=train_cfg['training']['lr_drop'],
        lr_gamma=train_cfg['training'].get('lr_gamma', 0.1),
        clip_max_norm=train_cfg['training']['clip_max_norm'],
        pretrain_weights=str(Path(train_cfg['training']['pretrain_weights']).resolve()),
        nc=dataset_cfg['nc'],
        device=model_args.device,
        model_args=model_args,
        bn2gn_policy=train_cfg['bn2gn']['policy'],
        exist_ok=train_cfg['training'].get('exist_ok', False),
        metrics_root=Path(train_cfg['paths']['metrics_dir']).resolve(),
        resume=resume_val
    )

    # 6. Ejecutar Entrenamiento
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()