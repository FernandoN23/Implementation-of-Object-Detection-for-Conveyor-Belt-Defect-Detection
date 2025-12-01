# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/train.py
# Descripción: Script de entrada (CLI) para el entrenamiento de SSD.
#              Orquesta el bootstrap MIOpen (opcional), la carga de
#              configuración y la ejecución de TrainerSSD.
# ==============================================================

from __future__ import annotations

import argparse
import sys
import os
import types
from dataclasses import replace
from pathlib import Path
from typing import Optional
import importlib.util

# --------------------------------------------------------------
# Rutas base del proyecto SSD
# --------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parent  # .../SSD
PROJECT_ROOT = SSD_ROOT.parent  # raíz del proyecto
CONFIGS_ROOT = SSD_ROOT / "configs"  # SSD/configs

TRAINER_PATH = SSD_ROOT / "engine" / "Trainer.py"
MIOPEN_BOOTSTRAP_PATH = SSD_ROOT / "engine" / "bootstrap_miopen.py"


# --------------------------------------------------------------
# Utilidad de carga dinámica de módulos
# --------------------------------------------------------------


def _load_module_from(path: Path, name: str):
    """Carga dinámica de un módulo Python desde un path arbitrario."""
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo requerido en: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para módulo: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    module_dir = str(path.parent)
    sys.path.insert(0, module_dir)

    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except Exception:
        if name in sys.modules:
            del sys.modules[name]
        raise
    finally:
        if module_dir in sys.path:
            sys.path.remove(module_dir)

    return module


# --------------------------------------------------------------
# Bootstrap MIOpen (opcional, antes de importar torch)
# --------------------------------------------------------------

# Variable global para guardar la clase MuteStderr si se carga
_MuteStderr = None

def _maybe_bootstrap_miopen(enable: bool = True) -> None:
    """Ejecuta bootstrap MIOpen si el módulo está disponible."""
    global _MuteStderr

    if not enable:
        return
    if not MIOPEN_BOOTSTRAP_PATH.is_file():
        return

    try:
        mod = _load_module_from(MIOPEN_BOOTSTRAP_PATH, "ssd_bootstrap_miopen")
        MIOpenConfig = mod.MIOpenConfig  # type: ignore[attr-defined]
        bootstrap = mod.bootstrap  # type: ignore[attr-defined]
        _MuteStderr = getattr(mod, "MuteStderr", None) # Guardar referencia

        cfg = MIOpenConfig()
        exported = bootstrap(cfg)

        verbose = getattr(cfg, "verbose", 0)
        if verbose:
            keys = ", ".join(sorted(exported.keys())) if exported else "(sin cambios)"
            print(f"[SSD/train] bootstrap_miopen aplicado. Variables: {keys}")
    except Exception as exc:
        print(f"[SSD/train] Advertencia: fallo bootstrap_miopen: {exc}", file=sys.stderr)


# --------------------------------------------------------------
# Utilidad de Mocking (Legacy Fix)
# --------------------------------------------------------------

def _mock_legacy_coco_dependency():
    """Crea un módulo 'fake' para data.coco y ssd.data.coco."""
    class DummyDataset:
        def __init__(self, *args, **kwargs): pass

    class DummyTransform:
        def __init__(self, *args, **kwargs): pass

    mock_coco = types.ModuleType("data.coco")
    mock_coco.COCODetection = DummyDataset
    mock_coco.COCOAnnotationTransform = DummyTransform
    mock_coco.COCO_CLASSES = []
    mock_coco.COCO_ROOT = ""
    mock_coco.get_label_map = lambda x: {}

    sys.modules["data.coco"] = mock_coco
    sys.modules["ssd.data.coco"] = mock_coco


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de SSD sobre dataset de fallas."
    )

    parser.add_argument(
        "--train-config",
        type=str,
        default=str(CONFIGS_ROOT / "train.yaml"),
        help="Ruta al archivo SSD/configs/train.yaml.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ssd300_default",
        help="Nombre del preset dentro de train.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo PyTorch (cuda:0, cpu).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Sobrescribe experiment.run_name.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="",
        help="Sobrescribe experiment.phase.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Ruta a checkpoint .pth para reanudar.",
    )
    parser.add_argument(
        "--no-bootstrap-miopen",
        action="store_true",
        help="Desactiva el bootstrap MIOpen.",
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------
# Punto de entrada principal
# --------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    _mock_legacy_coco_dependency()
    _maybe_bootstrap_miopen(enable=not args.no_bootstrap_miopen)

    trainer_mod = _load_module_from(TRAINER_PATH, "ssd_trainer")
    TrainerConfigSSD = trainer_mod.TrainerConfigSSD  # type: ignore[attr-defined]
    TrainerSSD = trainer_mod.TrainerSSD  # type: ignore[attr-defined]

    cfg = TrainerConfigSSD.from_yaml(args.train_config, preset=args.preset)

    overrides = {}
    if args.device: overrides["device"] = args.device
    if args.run_name: overrides["run_name"] = args.run_name
    if args.phase: overrides["phase"] = args.phase
    if args.resume: overrides["resume"] = Path(args.resume).expanduser().resolve()

    if overrides:
        cfg = replace(cfg, **overrides)

    if not cfg.exist_ok and not cfg.resume:
        variant_name = cfg.preset_name if cfg.is_test else cfg.variant
        subdir = Path(cfg.task) / variant_name / cfg.phase / cfg.run_name
        run_dir = cfg.runs_root / subdir
        if run_dir.exists():
            base = cfg.run_name
            i = 1
            while True:
                candidate = f"{base}_{i}"
                candidate_subdir = Path(cfg.task) / variant_name / cfg.phase / candidate
                if not (cfg.runs_root / candidate_subdir).exists():
                    print(f"[SSD/train] Advertencia: run_name '{cfg.run_name}' ya existe; se usará '{candidate}'.")
                    cfg = replace(cfg, run_name=candidate)
                    break
                i += 1

    # FIX: Usar MuteStderr para silenciar advertencias de MIOpen durante la inicialización
    # del modelo y el inicio del entrenamiento.
    if _MuteStderr:
        print("[SSD/train] Silenciando stderr (MIOpen warnings) durante inicialización...")
        with _MuteStderr():
            trainer = TrainerSSD(cfg)
            trainer.fit()
    else:
        trainer = TrainerSSD(cfg)
        trainer.fit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())