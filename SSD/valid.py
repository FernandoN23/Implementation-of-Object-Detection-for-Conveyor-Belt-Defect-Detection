# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/valid.py
# Descripción: Script de entrada (CLI) para validación de SSD.
#              Carga modelo, pesos y ejecuta ValidatorSSD.
# ==============================================================

from __future__ import annotations

import argparse
import sys
import os
import types
import yaml
import torch
import importlib.util
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------
# Rutas base del proyecto SSD
# --------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parent  # .../SSD
PROJECT_ROOT = SSD_ROOT.parent  # raíz del proyecto
CONFIGS_ROOT = SSD_ROOT / "configs"

VALIDATOR_PATH = SSD_ROOT / "engine" / "Validator.py"
DATA_LOADER_PATH = SSD_ROOT / "utility" / "data_loader.py"
SSD_MODEL_PATH = SSD_ROOT / "ssd" / "ssd.py"


# --------------------------------------------------------------
# Utilidad de carga dinámica
# --------------------------------------------------------------

def _load_module_from(path: Path, name: str):
    """Carga dinámica de módulos internos."""
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    # Contexto para imports relativos
    module_dir = str(path.parent)
    sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if module_dir in sys.path:
            sys.path.remove(module_dir)
    return module


# --------------------------------------------------------------
# Mocking Legacy (Igual que en train.py)
# --------------------------------------------------------------

def _mock_legacy_coco_dependency():
    """Neutraliza dependencia de COCO para evitar errores de importación."""

    class Dummy: pass

    mock = types.ModuleType("data.coco")
    mock.COCODetection = Dummy
    mock.COCOAnnotationTransform = Dummy
    mock.COCO_CLASSES = []
    mock.COCO_ROOT = ""
    sys.modules["data.coco"] = mock
    sys.modules["ssd.data.coco"] = mock


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Validación de modelo SSD")

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Ruta al archivo .pth de pesos. Si no se indica, busca 'best.pth' en runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIGS_ROOT / "valid.yaml"),
        help="Ruta a SSD/configs/valid.yaml"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ssd300_default",
        help="Nombre del preset en valid.yaml"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo (cuda, cpu)"
    )
    return parser.parse_args()


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def main():
    args = _parse_args()

    # 1. Preparar entorno
    _mock_legacy_coco_dependency()

    # 2. Cargar Configuración YAML
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    preset_cfg = full_cfg["presets"][args.preset]
    exp_cfg = full_cfg["experiment"]

    # Construir objeto de configuración simple para pasar al Validator/DataLoader
    # Combinamos experiment + preset
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, types.SimpleNamespace(**v))
                else:
                    setattr(self, k, v)

    # Aplanar configuración para facilitar acceso
    combined_cfg = {**exp_cfg, **preset_cfg}
    # Rutas absolutas
    combined_cfg["data_config"] = (PROJECT_ROOT / full_cfg["paths"]["dataset_config"]).resolve()
    combined_cfg["weights_root"] = (PROJECT_ROOT / full_cfg["paths"]["weights_root"]).resolve()
    combined_cfg["runs_root"] = (PROJECT_ROOT / full_cfg["paths"]["runs_root"]).resolve()
    combined_cfg["metrics_root"] = (PROJECT_ROOT / full_cfg["paths"]["metrics_root"]).resolve()

    cfg = Config(combined_cfg)

    # 3. Determinar Pesos
    if args.weights:
        weights_path = Path(args.weights).resolve()
    else:
        # Intentar inferir ruta: weights_root / task / variant / phase / run_name / best.pth
        # Nota: La fase en train suele ser 'train', aquí estamos en 'valid'.
        # Asumimos que el usuario quiere validar lo que entrenó en 'train'.
        # Ajuste manual de ruta común:
        train_run_dir = cfg.weights_root / cfg.task / cfg.variant / "train" / cfg.run_name.replace("_validation", "")
        weights_path = train_run_dir / "best.pth"

        if not weights_path.exists():
            # Fallback a last.pth
            weights_path = train_run_dir / "last.pth"

    if not weights_path.exists():
        print(f"[Error] No se encontraron pesos en: {weights_path}")
        print("Por favor especifique --weights explícitamente.")
        return 1

    print(f"[SSD/valid] Usando pesos: {weights_path}")

    # 4. Cargar Módulos Dinámicos
    # Data Loader
    dl_mod = _load_module_from(DATA_LOADER_PATH, "ssd_data_loader")
    build_dataloaders = dl_mod.build_dataloaders
    load_dataset_config = dl_mod.load_dataset_config

    # Modelo SSD
    ssd_mod = _load_module_from(SSD_MODEL_PATH, "ssd_model")
    build_ssd = ssd_mod.build_ssd

    # Validator
    val_mod = _load_module_from(VALIDATOR_PATH, "ssd_validator")
    ValidatorSSD = val_mod.ValidatorSSD

    # 5. Construir DataLoader
    # build_dataloaders espera un objeto con atributos data_config, img_dim, batch_size, num_workers
    # cfg ya cumple con esto.
    _, val_loader = build_dataloaders(cfg)

    # Obtener nombres de clases del dataset config
    ds_cfg_dict = load_dataset_config(cfg.data_config)
    class_names = list(ds_cfg_dict["names"].values())
    # SSD requiere background en index 0, pero nuestra lista class_names es solo foreground.
    # El Validator maneja la lógica de índices.
    num_classes = len(class_names) + 1

    # 6. Construir Modelo
    print(f"[SSD/valid] Construyendo SSD300 (clases={num_classes})...")
    # Phase='test' es CRÍTICO para que SSD active self.detect y devuelva predicciones decodificadas
    model = build_ssd("test", cfg.img_dim, num_classes)

    # Cargar estado
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(weights_path, map_location=device)

    # Manejo de DataParallel o claves 'model_state_dict'
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Limpiar prefijo 'module.' si existe
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 7. Ejecutar Validación
    # Definir directorio de salida específico para esta validación
    save_dir = cfg.metrics_root / cfg.task / cfg.variant / cfg.phase / cfg.run_name
    print(f"[SSD/valid] Guardando resultados en: {save_dir}")

    validator = ValidatorSSD(
        model=model,
        val_loader=val_loader,
        cfg=cfg,
        class_names=class_names,
        save_dir=save_dir
    )

    metrics = validator.run()

    # Guardar métricas numéricas en YAML simple
    metrics_file = save_dir / "metrics.yaml"
    with metrics_file.open("w") as f:
        yaml.dump(metrics, f)

    print("[SSD/valid] Validación completada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())