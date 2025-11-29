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
    """Carga dinámica de un módulo Python desde un path arbitrario.

    Se utiliza para importar módulos internos (Trainer, bootstrap_miopen)
    sin requerir instalación del proyecto como paquete.

    FIX (2025-05):
    1. Registro temprano en sys.modules (fix dataclasses/pickle).
    2. Inyección temporal de path.parent en sys.path para resolver
       imports relativos implícitos (legacy imports como 'from layers import *').
    """
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo requerido en: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para módulo: {path}")

    module = importlib.util.module_from_spec(spec)

    # 1) Registro temprano en sys.modules
    sys.modules[name] = module

    # 2) Context Manager para sys.path:
    # Agregamos el directorio del archivo a sys.path para que pueda encontrar
    # sus dependencias hermanas (ej: ssd.py encontrando layers/)
    module_dir = str(path.parent)
    sys.path.insert(0, module_dir)

    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except Exception:
        # Limpieza en caso de fallo crítico
        if name in sys.modules:
            del sys.modules[name]
        raise
    finally:
        # 3) Limpieza de sys.path para no contaminar el resto de la ejecución
        if module_dir in sys.path:
            sys.path.remove(module_dir)

    return module


# --------------------------------------------------------------
# Bootstrap MIOpen (opcional, antes de importar torch)
# --------------------------------------------------------------


def _maybe_bootstrap_miopen(enable: bool = True) -> None:
    """Ejecuta bootstrap MIOpen si el módulo está disponible.

    Debe llamarse *antes* de importar cualquier módulo que traiga `torch`
    (por eso el Trainer se importa de forma diferida en `main`).
    """

    if not enable:
        return
    if not MIOPEN_BOOTSTRAP_PATH.is_file():
        return

    try:
        mod = _load_module_from(MIOPEN_BOOTSTRAP_PATH, "ssd_bootstrap_miopen")
        MIOpenConfig = mod.MIOpenConfig  # type: ignore[attr-defined]
        bootstrap = mod.bootstrap  # type: ignore[attr-defined]

        cfg = MIOpenConfig()  # usa valores por defecto definidos en el módulo
        exported = bootstrap(cfg)

        verbose = getattr(cfg, "verbose", 0)
        if verbose:
            keys = ", ".join(sorted(exported.keys())) if exported else "(sin cambios)"
            print(f"[SSD/train] bootstrap_miopen aplicado. Variables: {keys}")
    except Exception as exc:  # pragma: no cover - defensivo
        print(f"[SSD/train] Advertencia: fallo bootstrap_miopen: {exc}", file=sys.stderr)


# --------------------------------------------------------------
# Utilidad de Mocking (Legacy Fix)
# --------------------------------------------------------------

def _mock_legacy_coco_dependency():
    """Crea un módulo 'fake' para data.coco y ssd.data.coco.

    El código original importa 'data.coco' y ejecuta código que busca
    archivos físicos (coco_labels.txt). Si el usuario no tiene COCO,
    esto rompe la ejecución incluso si solo quiere usar VOC o su propio dataset.

    Este mock inyecta las clases necesarias en sys.modules para satisfacer
    la importación sin ejecutar lógica de I/O.
    """

    # Definir clases dummy que emulan la interfaz esperada por ssd/data/__init__.py
    class DummyDataset:
        def __init__(self, *args, **kwargs): pass

    class DummyTransform:
        def __init__(self, *args, **kwargs): pass

    # Crear el módulo mock
    mock_coco = types.ModuleType("data.coco")
    mock_coco.COCODetection = DummyDataset
    mock_coco.COCOAnnotationTransform = DummyTransform
    mock_coco.COCO_CLASSES = []
    mock_coco.COCO_ROOT = ""
    mock_coco.get_label_map = lambda x: {}

    # Inyectar en sys.modules bajo los nombres posibles que use el legacy code
    # El traceback indica que 'ssd.py' añade su carpeta a path, así que
    # se importa como 'data.coco'
    sys.modules["data.coco"] = mock_coco

    # Por seguridad, inyectar también rutas completas si fuesen usadas
    sys.modules["ssd.data.coco"] = mock_coco

    # print("[SSD/train] Mock inyectado: 'data.coco' (Dependencia legacy neutralizada).")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento de SSD (Single Shot Multibox Detector) sobre "
            "dataset de fallas en correas transportadoras."
        )
    )

    parser.add_argument(
        "--train-config",
        type=str,
        default=str(CONFIGS_ROOT / "train.yaml"),
        help="Ruta al archivo SSD/configs/train.yaml a utilizar.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ssd300_default",
        help=(
            "Nombre del preset dentro de train.yaml (clave en 'presets'). "
            "Por ejemplo: ssd300_default."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo PyTorch a utilizar (cuda:0, cpu, etc.). Si se omite, se auto-detecta.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Sobrescribe experiment.run_name definido en train.yaml. "
            "Útil para etiquetar ejecuciones distintas con la misma config."
        ),
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="",
        help="Sobrescribe experiment.phase (train, finetune, etc.) si se desea.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Ruta a checkpoint .pth para reanudar entrenamiento (opcional).",
    )
    parser.add_argument(
        "--no-bootstrap-miopen",
        action="store_true",
        help=(
            "Desactiva el bootstrap MIOpen incluso si engine/bootstrap_miopen.py "
            "está disponible."
        ),
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------
# Punto de entrada principal
# --------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # 1) FIX (2025-05): Neutralizar dependencia COCO antes de cualquier carga.
    _mock_legacy_coco_dependency()

    # 2) Bootstrap MIOpen/ROCm (si corresponde) ANTES de importar Trainer/torch
    _maybe_bootstrap_miopen(enable=not args.no_bootstrap_miopen)

    # 3) Importar Trainer dinámicamente (esto traerá torch dentro)
    trainer_mod = _load_module_from(TRAINER_PATH, "ssd_trainer")
    TrainerConfigSSD = trainer_mod.TrainerConfigSSD  # type: ignore[attr-defined]
    TrainerSSD = trainer_mod.TrainerSSD  # type: ignore[attr-defined]

    # 4) Construir configuración base desde YAML/preset
    cfg = TrainerConfigSSD.from_yaml(args.train_config, preset=args.preset)

    # 5) Aplicar overrides simples desde CLI
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.run_name:
        overrides["run_name"] = args.run_name
    if args.phase:
        overrides["phase"] = args.phase
    if args.resume:
        overrides["resume"] = Path(args.resume).expanduser().resolve()

    if overrides:
        cfg = replace(cfg, **overrides)

    # 6) Si exist_ok=False y la carpeta de ejecución ya existe, ajustar run_name
    # FIX: Solo renombrar si NO estamos reanudando (resume es None/False)
    if not cfg.exist_ok and not cfg.resume:
        # Ajustar nombre de variante si es un test (lógica replicada de Trainer para check previo)
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
                    print(
                        f"[SSD/train] Advertencia: run_name '{cfg.run_name}' ya existe; "
                        f"se usará '{candidate}'."
                    )
                    cfg = replace(cfg, run_name=candidate)
                    break
                i += 1

    # 7) Ejecutar entrenamiento
    trainer = TrainerSSD(cfg)
    trainer.fit()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())