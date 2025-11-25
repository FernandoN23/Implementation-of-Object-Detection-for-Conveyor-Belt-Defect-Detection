# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/train.py
# Descripción: Punto de entrada principal para el entrenamiento YOLO.
#              Orquestador de CLI, configuración YAML, presets y Trainer.
#==============================================================

"""Punto de entrada CLI para ejecutar entrenamientos YOLO.

Este script:
- Resuelve rutas base del proyecto (YOLO/, configs/, runs/, etc.).
- Carga un archivo de configuración `train.yaml` (opción `--cfg-train`).
- Apoya una estructura de configuración basada en secciones:
  - `paths`, `training`, `miopen`, `bn2gn`, `presets`.
- Permite seleccionar **presets** definidos en `train.yaml` mediante el
  argumento `--preset`, aplicando overrides sobre las secciones
  `paths` y `training` antes de construir el `TrainerConfig`.
- Aplica, si corresponde, el `bootstrap` de MIOpen **antes** de importar
  cualquier módulo que inicialice PyTorch.
- Construye un `TrainerConfig` (incluyendo la política BN→GN) y ejecuta
  `Trainer.fit()`.

Notas de diseño (iteración actual)
---------------------------------
- Se asume que `train.yaml` sigue la estructura definida en
  YOLO/configs/train.yaml (paths, training, miopen, bn2gn, presets).
- La sección `presets` se consume opcionalmente vía CLI con `--preset`.
  Cada preset define un bloque `overrides` para `paths.training` y
  otras secciones relevantes.
- MIOpen se configura de forma que el caché quede **siempre
  deshabilitado**, independientemente de lo que diga el YAML.
- La configuración BN→GN se pasa al Trainer, que es responsable de
  aplicar el patch sobre YOLOv5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Rutas base de proyecto (derivadas de este archivo)
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parent                 # .../YOLO
PROJECT_ROOT = YOLO_ROOT.parent         # raíz del proyecto (nivel superior a YOLO)
CONFIGS_ROOT = YOLO_ROOT / "configs"    # YOLO/configs

if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

# Import seguro: no depende de torch
from engine.bootstrap_miopen import MIOpenConfig, bootstrap  # type: ignore


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _resolve_path(path: str | Path, base: Path) -> Path:
    """Resuelve una ruta relativa contra `base`, dejando rutas absolutas intactas."""

    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _load_train_yaml(cfg_path: Path) -> Dict[str, Any]:
    """Carga el archivo YAML de configuración de entrenamiento.

    Lanza un error legible si el archivo no existe o es inválido.
    """

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración de entrenamiento: {cfg_path}"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El archivo YAML {cfg_path} debe contener un objeto mapeo (dict) en la raíz.")
    return data


def _apply_preset(cfg_yaml: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Aplica un preset definido en `cfg_yaml['presets']` sobre el YAML base.

    El preset debe tener la forma:

    presets:
      nombre_preset:
        overrides:
          paths:
            ...
          training:
            ...

    Las claves de `overrides` se fusionan superficialmente con las
    secciones correspondientes del YAML raíz, dando prioridad a los
    valores del preset.
    """

    presets = cfg_yaml.get("presets") or {}
    if not isinstance(presets, dict):
        raise ValueError("La sección 'presets' en train.yaml debe ser un mapeo (dict).")

    if preset_name not in presets:
        available = ", ".join(presets.keys()) or "<ninguno>"
        raise KeyError(
            f"Preset '{preset_name}' no encontrado en train.yaml. "
            f"Presets disponibles: {available}"
        )

    preset = presets[preset_name] or {}
    if not isinstance(preset, dict):
        raise ValueError(f"El preset '{preset_name}' debe ser un mapeo (dict).")

    overrides = preset.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"El preset '{preset_name}.overrides' debe ser un mapeo (dict).")

    # Fusionar superficialmente secciones overrideadas
    for section_name, section_over in overrides.items():
        if not isinstance(section_over, dict):
            continue
        base_section = cfg_yaml.get(section_name) or {}
        if not isinstance(base_section, dict):
            base_section = {}
        merged = {**base_section, **section_over}
        cfg_yaml[section_name] = merged

    return cfg_yaml


# ---------------------------------------------------------------------------
# Construcción de configuraciones de alto nivel
# ---------------------------------------------------------------------------


def _build_miopen_config(cfg_yaml: Dict[str, Any]) -> Optional[MIOpenConfig]:
    """Construye un `MIOpenConfig` a partir de la sección `miopen` del YAML.

    Si la sección no existe, retorna `None`.

    Nota importante: por política de proyecto, el caché de MIOpen se
    desactiva siempre (`disable_cache=True`), independientemente del
    valor indicado en el YAML.
    """

    miopen_cfg = cfg_yaml.get("miopen") or {}
    if not isinstance(miopen_cfg, dict):
        miopen_cfg = {}

    # Valor en YAML (sólo para referencia, no se respeta si es False)
    _yaml_disable_cache = bool(miopen_cfg.get("disable_cache", True))

    return MIOpenConfig(
        find_mode=miopen_cfg.get("find_mode", "FAST"),
        user_db_path=miopen_cfg.get("user_db_path"),
        disable_cache=True,  # política: siempre deshabilitado
        log_level=int(miopen_cfg.get("log_level", 0)),
        extra_env=miopen_cfg.get("extra_env") or {},
        strict_before_torch=bool(miopen_cfg.get("strict_before_torch", True)),
        verbose=int(miopen_cfg.get("verbose", 1)),
    )


def _build_trainer_config(
    cfg_yaml: Dict[str, Any], args: argparse.Namespace, miopen_cfg: Optional[MIOpenConfig]
):
    """Construye un `TrainerConfig` a partir de `train.yaml` y overrides de CLI.

    Esta función no importa PyTorch ni YOLOv5. Sólo prepara los parámetros
    de alto nivel que luego usará `engine.Trainer.Trainer`.
    """

    from engine.Trainer import TrainerConfig  # import diferido (no depende de torch)

    try:  # bn2gn puede no estar disponible en etapas tempranas
        from engine.bn2gn_patch import BN2GNConfig  # type: ignore
    except Exception:  # pragma: no cover
        BN2GNConfig = None  # type: ignore

    paths = cfg_yaml.get("paths") or {}
    training = cfg_yaml.get("training") or {}
    bn2gn_yaml = cfg_yaml.get("bn2gn") or {}

    # ---------------------------
    # Identidad del experimento
    # ---------------------------
    task = training.get("task", "detect")
    variant = args.variant or training.get("variant", "s")
    run_name = args.run_name or training.get("run_name", "exp")

    # ---------------------------
    # Rutas de dataset/config
    # ---------------------------
    dataset_cfg_raw = paths.get("dataset_cfg", str(CONFIGS_ROOT / "dataset.yaml"))
    data_config = _resolve_path(dataset_cfg_raw, base=PROJECT_ROOT)

    # ---------------------------
    # Pesos iniciales y hyps
    # ---------------------------
    if args.weights is not None:
        # Si el usuario entrega un alias de YOLOv5 (p.ej. "yolov5s.pt"), lo
        # dejamos tal cual para que lo resuelva internamente el repo.
        weights = args.weights
    else:
        pretrain_raw = training.get("pretrain_weights", "YOLO/yolov5/yolov5s.pt")
        if pretrain_raw:
            # Interpretar como ruta relativa al PROJECT_ROOT
            weights = str(_resolve_path(pretrain_raw, base=PROJECT_ROOT))
        else:
            # Cadena vacía → entrenamiento desde cero
            weights = ""

    hyp_raw = training.get("hyp_yaml")
    hyp = _resolve_path(hyp_raw, base=PROJECT_ROOT) if hyp_raw else None

    # ---------------------------
    # Hiperparámetros básicos
    # ---------------------------
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 100))
    batch_size = args.batch_size if args.batch_size is not None else int(training.get("batch_size", 16))
    imgsz = args.imgsz if args.imgsz is not None else int(training.get("imgsz", 640))
    workers = int(training.get("workers", 0)) or 4

    # Dispositivo y semilla
    device = args.device if args.device is not None else str(training.get("device", ""))
    seed = int(training.get("seed", 0))

    # ---------------------------
    # Logging y guardado
    # ---------------------------
    save_period = int(training.get("save_period", -1))
    ndjson_console = bool(training.get("ndjson_console", False))
    ndjson_file = bool(training.get("ndjson_file", False))

    # exist_ok a nivel de entrenamiento (si no está, por defecto False)
    exist_ok = bool(training.get("exist_ok", False))

    # ---------------------------
    # Configuración BN2GN
    # ---------------------------
    bn2gn_cfg = None
    if BN2GNConfig is not None and isinstance(bn2gn_yaml, dict):
        policy = str(bn2gn_yaml.get("policy", "on")).lower()
        if policy != "off":  # sólo construimos config si la política no es "off"
            bn2gn_cfg = BN2GNConfig(
                policy=policy,
                max_groups=int(bn2gn_yaml.get("max_groups", 32)),
                min_channels_per_group=int(bn2gn_yaml.get("min_channels_per_group", 1)),
                verbose=int(bn2gn_yaml.get("verbose", 1)),
            )

    # ---------------------------
    # Construcción del objeto TrainerConfig
    # ---------------------------
    cfg = TrainerConfig(
        task=task,
        variant=variant,
        run_name=run_name,
        data_config=data_config,
        hyp=hyp,
        weights=weights,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        workers=workers,
        device=device,
        save_period=save_period,
        seed=seed,
        exist_ok=exist_ok,
        ndjson_console=ndjson_console,
        ndjson_file=ndjson_file,
        miopen=miopen_cfg,
        bn2gn=bn2gn_cfg,
    )

    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comando para el entrenamiento YOLO."""

    parser = argparse.ArgumentParser(
        prog="YOLO.train",
        description=(
            "Punto de entrada principal para entrenar modelos YOLO basados en YOLOv5 "
            "utilizando una configuración centralizada en YOLO/configs/train.yaml."
        ),
    )

    # Configuración YAML
    parser.add_argument(
        "--cfg-train",
        type=str,
        default=None,
        help="Ruta al archivo train.yaml (por defecto: YOLO/configs/train.yaml)",
    )

    # Selección de preset
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Nombre de un preset definido en train.yaml (sección 'presets'). "
            "Aplica overrides antes de construir TrainerConfig."
        ),
    )

    # Overrides rápidos
    parser.add_argument("--variant", type=str, default=None, help="Variante del modelo (n, s, m, l, x)")
    parser.add_argument("--run-name", type=str, default=None, help="Nombre lógico del experimento")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo: '', '0', '0,1', 'cpu', etc.")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas (override de train.yaml)")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch (override de train.yaml)")
    parser.add_argument("--imgsz", type=int, default=None, help="Tamaño de imagen (override de train.yaml)")
    parser.add_argument("--weights", type=str, default=None, help="Pesos iniciales (ruta .pt o alias de YOLOv5)")

    # Control de bootstrap MIOpen
    parser.add_argument(
        "--no-bootstrap-miopen",
        action="store_true",
        help="Desactiva el bootstrap MIOpen incluso si hay sección miopen en train.yaml",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Entrypoint de entrenamiento YOLO.

    Flujo:
    1) Parseo de CLI.
    2) Carga de `train.yaml`.
    3) Aplicación opcional de `--preset` (overrides sobre paths/training).
    4) Bootstrap MIOpen (si corresponde) **antes** de importar Torch/YOLOv5.
    5) Construcción de `TrainerConfig` (incluyendo BN2GN).
    6) Ejecución de `Trainer.fit()`.
    """

    args = parse_args(argv)

    # 1) Resolver ruta a train.yaml
    cfg_train_path = Path(args.cfg_train) if args.cfg_train else (CONFIGS_ROOT / "train.yaml")
    cfg_train_path = cfg_train_path.resolve()

    # 2) Cargar YAML de entrenamiento
    cfg_yaml = _load_train_yaml(cfg_train_path)

    # 3) Aplicar preset si se especifica
    if args.preset is not None:
        cfg_yaml = _apply_preset(cfg_yaml, args.preset)

    # 4) Bootstrap MIOpen antes de cualquier importación de Torch/YOLOv5
    miopen_cfg = None
    if not args.no_bootstrap_miopen and "miopen" in cfg_yaml:
        miopen_cfg = _build_miopen_config(cfg_yaml)
        if miopen_cfg is not None:
            bootstrap(miopen_cfg)

    # 5) Importar Trainer una vez configurado el entorno MIOpen
    from engine.Trainer import Trainer  # type: ignore

    trainer_cfg = _build_trainer_config(cfg_yaml, args, miopen_cfg)

    # 6) Ejecutar entrenamiento
    trainer = Trainer(trainer_cfg)
    trainer.fit()


if __name__ == "__main__":  # pragma: no cover
    main()
