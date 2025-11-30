# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/ssd_check_params.py
# Descripción: Herramienta de diagnóstico de arquitectura.
#              Carga la configuración del modelo y calcula la
#              cantidad total de parámetros, desglosando por
#              componentes (Backbone, Extras, Heads).
# ==============================================================

from __future__ import annotations

import argparse
import sys
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

# --------------------------------------------------------------
# Rutas base
# --------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parents[1]  # .../SSD
PROJECT_ROOT = SSD_ROOT.parent
CONFIGS_ROOT = SSD_ROOT / "configs"
SSD_MODEL_PATH = SSD_ROOT / "ssd" / "ssd.py"


# --------------------------------------------------------------
# Utilidad de carga dinámica (Robusta)
# --------------------------------------------------------------

def _load_module_from(path: Path, name: str):
    """Carga dinámica de un módulo Python inyectando el contexto."""
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    # Contexto para imports relativos (necesario para ssd.py)
    module_dir = str(path.parent)
    sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if module_dir in sys.path:
            sys.path.remove(module_dir)
    return module


# --------------------------------------------------------------
# Funciones de Conteo
# --------------------------------------------------------------

def count_params(module: nn.Module) -> int:
    """Cuenta parámetros totales en un módulo."""
    return sum(p.numel() for p in module.parameters())


def count_trainable(module: nn.Module) -> int:
    """Cuenta parámetros entrenables (requires_grad=True)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_num(num: int) -> str:
    """Formatea números con separador de miles."""
    return f"{num:,}"


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Resumen de parámetros del modelo SSD")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIGS_ROOT / "train.yaml"),
        help="Ruta al archivo de configuración de entrenamiento"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ssd300_default",
        help="Nombre del preset a analizar"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Cargar Configuración
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Error] No se encontró config: {config_path}")
        return

    with config_path.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    if args.preset not in full_cfg["presets"]:
        print(f"[Error] Preset '{args.preset}' no encontrado en {config_path}")
        return

    preset_cfg = full_cfg["presets"][args.preset]
    exp_cfg = full_cfg.get("experiment", {})

    # 2. Obtener datos del Dataset (para num_classes)
    dataset_config_rel = full_cfg.get("paths", {}).get("dataset_config", "SSD/configs/dataset.yaml")
    dataset_config_path = (PROJECT_ROOT / dataset_config_rel).resolve()

    with dataset_config_path.open("r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    num_classes = len(ds_cfg["names"]) + 1  # +1 background
    img_dim = preset_cfg.get("img_dim", 300)
    variant = exp_cfg.get("variant", "ssd300")

    # 3. Cargar Modelo SSD
    print(f"[Info] Cargando definición del modelo desde: {SSD_MODEL_PATH}")
    try:
        ssd_mod = _load_module_from(SSD_MODEL_PATH, "ssd_model")
        build_ssd = ssd_mod.build_ssd
        # Instanciar (pesos aleatorios, solo nos importa la arquitectura)
        model = build_ssd("train", img_dim, num_classes)
    except Exception as e:
        print(f"[Error] Fallo al instanciar el modelo: {e}")
        return

    # 4. Calcular Parámetros
    total_params = count_params(model)
    trainable_params = count_trainable(model)

    # Desglose por componentes principales de SSD
    # SSD tiene: vgg (backbone), extras (capas extra), loc (head), conf (head)
    vgg_params = count_params(model.vgg)
    extras_params = count_params(model.extras)
    loc_params = count_params(model.loc)
    conf_params = count_params(model.conf)
    l2norm_params = count_params(model.L2Norm)

    # 5. Imprimir Reporte
    print("\n" + "=" * 60)
    print(f" REPORTE DE ARQUITECTURA: {variant.upper()}")
    print("=" * 60)

    print(f"\n[1] Configuración General")
    print(f"    Preset:          {args.preset}")
    print(f"    Input Dim:       {img_dim}x{img_dim}")
    print(f"    Clases:          {num_classes} (1 Background + {num_classes - 1} Objetos)")
    print(f"    Batch Size:      {preset_cfg.get('batch_size', 'N/A')}")
    print(f"    Optimizador:     {preset_cfg.get('opt', 'N/A')}")
    print(f"    Learning Rate:   {preset_cfg.get('lr', 'N/A')}")

    print(f"\n[2] Desglose de Parámetros")
    print(f"    {'-' * 46}")
    print(f"    {'Componente':<25} | {'Parámetros':>18}")
    print(f"    {'-' * 46}")
    print(f"    {'Backbone (VGG16 Base)':<25} | {format_num(vgg_params):>18}")
    print(f"    {'L2 Norm Scale':<25} | {format_num(l2norm_params):>18}")
    print(f"    {'Extra Layers':<25} | {format_num(extras_params):>18}")
    print(f"    {'Localization Head':<25} | {format_num(loc_params):>18}")
    print(f"    {'Confidence Head':<25} | {format_num(conf_params):>18}")
    print(f"    {'-' * 46}")
    print(f"    {'TOTAL':<25} | {format_num(total_params):>18}")
    print(f"    {'-' * 46}")

    print(f"\n[3] Resumen de Entrenamiento")
    print(f"    Total Parámetros:       {format_num(total_params)}")
    print(f"    Parámetros Entrenables: {format_num(trainable_params)}")
    print(f"    Tamaño estimado (MB):   {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()