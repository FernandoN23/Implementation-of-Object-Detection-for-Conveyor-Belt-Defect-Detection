# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/train.py
# Descripción: Punto de entrada CLI para el entrenamiento de DINO.
# ==============================================================

import argparse
import yaml
import sys
import os
from pathlib import Path

# --- CONFIGURACIÓN DE ENTORNO ROCm/MIOPEN ---
FILE = Path(__file__).resolve()
DINO_ROOT = FILE.parent
if str(DINO_ROOT) not in sys.path:
    sys.path.append(str(DINO_ROOT))

from engine.bootstrap_miopen import bootstrap, MIOpenConfig

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


class Dict2Obj:
    """Convierte un diccionario en un objeto para inicialización segura de DINO."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        return None


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    print(f"[train.py] Iniciando script de entrenamiento DINO...")
    parser = argparse.ArgumentParser(description="DINO Training CLI - Belt Defects")
    parser.add_argument("--cfg-train", type=str, default="DINO/configs/train.yaml")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                        help="Reanudar entrenamiento.")
    args = parser.parse_args()

    # 1. Cargar configuraciones
    train_cfg = load_yaml(args.cfg_train)
    dataset_cfg = load_yaml(train_cfg['paths']['dataset_cfg'])
    variants_cfg = load_yaml(train_cfg['paths']['variants_cfg'])

    # 2. Aplicar Overrides de Preset
    if args.preset:
        if args.preset in train_cfg.get('presets', {}):
            overrides = train_cfg['presets'][args.preset].get('overrides', {})
            for section, values in overrides.items():
                train_cfg[section].update(values)
            print(f"[train.py] Preset '{args.preset}' aplicado correctamente.")
        else:
            print(f"[train.py] ERROR FATAL: El preset '{args.preset}' no existe.")
            sys.exit(1)

    use_coco128 = train_cfg['training'].get('use_coco128', False)
    if use_coco128:
        print(f"[train.py] Bandera 'use_coco128' detectada. Sobrescribiendo dataset a 80 clases COCO.")
        dataset_cfg['nc'] = 80
        dataset_cfg['names'] = {i: name for i, name in enumerate(COCO_CLASSES)}

    # 3. Bootstrap MIOpen y Hardware
    mi_cfg = train_cfg['miopen']
    hw_cfg = train_cfg['hardware']
    bootstrap(MIOpenConfig(
        find_mode=mi_cfg['find_mode'],
        user_db_path=mi_cfg['user_db_path'],
        disable_cache=True,
        expandable_segments=hw_cfg['expandable_segments'],
        verbose=mi_cfg['verbose']
    ))

    # --- INICIALIZACIÓN DE MOTOR ---
    from engine.warnings import install_global_warning_filters
    from engine.Trainer import Trainer, TrainerConfig

    install_global_warning_filters()

    # 4. Preparar argumentos del modelo
    v_name = train_cfg['training']['variant']
    v_params = variants_cfg['variants'][v_name]

    base_args = v_params.copy()

    # Extraer coeficientes de pérdida y configuraciones específicas de DINO
    for k in ['bbox_loss_coef', 'giou_loss_coef', 'cls_loss_coef', 'focal_alpha', 'aux_loss', 'lr_backbone']:
        if k in train_cfg['training']:
            base_args[k] = train_cfg['training'][k]

    base_args['set_cost_class'] = train_cfg['training'].get('set_cost_class', 2.0)
    base_args['set_cost_bbox'] = train_cfg['training'].get('set_cost_bbox', 5.0)
    base_args['set_cost_giou'] = train_cfg['training'].get('set_cost_giou', 2.0)
    base_args['frozen_weights'] = None
    base_args['masks'] = False
    base_args['dataset_file'] = 'coco'
    base_args['device'] = train_cfg['training']['device']
    base_args['num_classes'] = dataset_cfg['nc']

    model_args = Dict2Obj(base_args)

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
        class_names=list(dataset_cfg['names'].values()),
        device=model_args.device,
        model_args=model_args,
        bn2gn_policy=train_cfg['bn2gn']['policy'],
        exist_ok=train_cfg['training'].get('exist_ok', False),
        metrics_root=Path(train_cfg['paths']['metrics_dir']).resolve(),
        resume=resume_val,
        use_coco128=use_coco128,
        empty_cache_freq=hw_cfg['empty_cache_freq'],
        use_amp=hw_cfg.get('use_amp', True),
        ema_decay=train_cfg['training'].get('ema_decay', 0.9997)
    )

    # 6. Ejecutar Entrenamiento
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()