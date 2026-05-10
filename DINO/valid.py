# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/valid.py
# Descripción: Script de entrada (CLI) para validación de DINO.
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
DINO_ROOT = FILE.parent
if str(DINO_ROOT) not in sys.path:
    sys.path.append(str(DINO_ROOT))

from engine.bootstrap_miopen import bootstrap, MIOpenConfig

# Clases estándar de COCO (80 clases)
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
    print(f"[valid.py] Iniciando script de validación DINO...")
    parser = argparse.ArgumentParser(description="DINO Validation CLI")
    parser.add_argument("--cfg-valid", type=str, default="DINO/configs/valid.yaml")
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

    use_coco128 = valid_cfg['validation'].get('use_coco128', False)
    if use_coco128:
        print(f"[valid.py] Bandera 'use_coco128' detectada. Sobrescribiendo dataset a 80 clases COCO.")
        dataset_cfg['nc'] = 80
        dataset_cfg['names'] = {i: name for i, name in enumerate(COCO_CLASSES)}

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

    base_args = {
        'lr_backbone': 0, 'masks': False, 'frozen_weights': None,
        'aux_loss': False, 'set_cost_class': 1.0, 'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0, 'bbox_loss_coef': 5.0, 'giou_loss_coef': 2.0,
        'cls_loss_coef': 1.0, 'focal_alpha': 0.25,
        'dataset_file': 'coco', 'device': args.device or valid_cfg['validation']['device'],
        'num_classes': dataset_cfg['nc']
    }
    base_args.update(v_params)
    model_args = Dict2Obj(base_args)

    # 5. Auto-descubrimiento de Pesos
    weights_path = args.weights or valid_cfg['validation']['weights']
    if not weights_path:
        auto_path = DINO_ROOT / "runs" / v_name / "train" / run_name / "weights" / "best.pt"
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

    if valid_cfg['bn2gn']['policy'] == 'on':
        replace_bn_with_gn(model, BN2GNConfig(policy='on'))

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Intentar cargar el modelo EMA si existe
    if 'ema_model' in checkpoint:
        print("[valid.py] Cargando pesos suavizados (EMA)...")
        model.load_state_dict(checkpoint['ema_model'], strict=False)
    else:
        model.load_state_dict(checkpoint['model'], strict=False)

    device = torch.device(model_args.device)
    model.to(device)

    # 7. Ejecutar Reporte de Validación
    class_names = list(dataset_cfg['names'].values())

    val_loader = build_dataloader(
        valid_cfg['validation']['phase'],
        valid_cfg['validation']['batch_size'],
        use_coco128=use_coco128,
        class_names=class_names
    )

    save_dir = DINO_ROOT / "metrics" / "detect" / v_name / valid_cfg['validation']['phase'] / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    validator = Validator(model, criterion, postprocessors, device)
    print(f"[valid.py] --- Iniciando Reporte de Validación: {run_name} ---")

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