# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/valid.py
# Descripción: Punto de entrada principal para la validación YOLO.
#              Orquestador de CLI, configuración YAML, presets y Validator.
#==============================================================

"""Punto de entrada CLI para ejecutar campañas de validación YOLO.

Este script:
- Resuelve rutas base del proyecto (YOLO/, configs/, runs/, metrics, etc.).
- Carga un archivo de configuración `valid.yaml` (opción `--cfg-valid`).
- Apoya una estructura de configuración basada en secciones:
  - `paths`, `validation`, `miopen`, `bn2gn`, `presets`.
- Permite seleccionar **presets** definidos en `valid.yaml` mediante el
  argumento `--preset`, aplicando overrides sobre las secciones
  `paths`, `validation` y otras relevantes.
- Aplica, si corresponde, el `bootstrap` de MIOpen **antes** de importar
  cualquier módulo que inicialice PyTorch.
- Construye un `ValidatorConfig` (incluyendo la política BN→GN) y
  ejecuta `Validator.run()`.

Notas de diseño (iteración actual)
---------------------------------
- Se asume que `valid.yaml` sigue la estructura definida en
  YOLO/configs/valid.yaml (paths, validation, miopen, bn2gn, presets).
- La sección `presets` se consume opcionalmente vía CLI con `--preset`.
  Cada preset define un bloque `overrides` para `paths`, `validation` y
  otras secciones relevantes.
- MIOpen se configura de forma que el caché quede **siempre
  deshabilitado**, independientemente de lo que diga el YAML.
- La configuración BN→GN se pasa al Validator, que es responsable de
  exponerla al backend de validación (por ejemplo, a YOLOv5/val.py).
- La fase lógica del experimento (val/test) se lee desde `validation`
  en `valid.yaml` y se propaga a `ValidatorConfig` para estructurar la
  jerarquía de directorios y seleccionar el split correcto del
  `dataset.yaml` (data[phase]).
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


def _load_valid_yaml(cfg_path: Path) -> Dict[str, Any]:
    """Carga el archivo YAML de configuración de validación.

    Lanza un error legible si el archivo no existe o es inválido.
    """

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración de validación: {cfg_path}"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"El archivo YAML {cfg_path} debe contener un objeto mapeo (dict) en la raíz."
        )
    return data


def _apply_preset(cfg_yaml: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Aplica un preset definido en `cfg_yaml['presets']` sobre el YAML base.

    El preset debe tener la forma:

    presets:
      nombre_preset:
        overrides:
          paths:
            ...
          validation:
            ...

    Las claves de `overrides` se fusionan superficialmente con las
    secciones correspondientes del YAML raíz, dando prioridad a los
    valores del preset.
    """

    presets = cfg_yaml.get("presets") or {}
    if not isinstance(presets, dict):
        raise ValueError("La sección 'presets' en valid.yaml debe ser un mapeo (dict).")

    if preset_name not in presets:
        available = ", ".join(presets.keys()) or "<ninguno>"
        raise KeyError(
            f"Preset '{preset_name}' no encontrado en valid.yaml. "
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


def _build_validator_config(
    cfg_yaml: Dict[str, Any], args: argparse.Namespace, miopen_cfg: Optional[MIOpenConfig]
):
    """Construye un `ValidatorConfig` a partir de `valid.yaml` y overrides de CLI.

    Esta función no importa PyTorch ni YOLOv5. Sólo prepara los
    parámetros de alto nivel que luego usará `engine.Validator.Validator`.
    """

    from engine.Validator import ValidatorConfig  # import diferido (no depende de torch)

    try:  # bn2gn puede no estar disponible en etapas tempranas
        from engine.bn2gn_patch import BN2GNConfig  # type: ignore
    except Exception:  # pragma: no cover
        BN2GNConfig = None  # type: ignore

    paths = cfg_yaml.get("paths") or {}
    validation = cfg_yaml.get("validation") or {}
    bn2gn_yaml = cfg_yaml.get("bn2gn") or {}

    # ---------------------------
    # Tipo de modelo y fase lógica
    # ---------------------------
    task_model = str(validation.get("task_model", "detect"))
    phase = str(args.phase) if getattr(args, "phase", None) else str(validation.get("phase", "val"))

    # ---------------------------
    # Identidad del experimento
    # ---------------------------
    variant = args.variant or validation.get("variant", "s")
    run_name = args.run_name or validation.get("run_name", "val_default")

    # ---------------------------
    # Rutas de dataset/config
    # ---------------------------
    dataset_cfg_raw = paths.get("dataset_cfg", str(CONFIGS_ROOT / "dataset.yaml"))
    data_config = _resolve_path(dataset_cfg_raw, base=PROJECT_ROOT)

    # ---------------------------
    # Pesos a validar
    # ---------------------------
    if args.weights is not None:
        weights = args.weights
    else:
        weights_raw = validation.get("weights", "")
        if weights_raw:
            weights = str(_resolve_path(weights_raw, base=PROJECT_ROOT))
        else:
            weights = ""

    # ---------------------------
    # Hiperparámetros de validación
    # ---------------------------
    batch_size = (
        args.batch_size if args.batch_size is not None else int(validation.get("batch_size", 4))
    )
    imgsz = args.imgsz if args.imgsz is not None else int(validation.get("imgsz", 640))

    conf_thres = (
        args.conf_thres if getattr(args, "conf_thres", None) is not None
        else float(validation.get("conf_thres", 0.001))
    )
    iou_thres = (
        args.iou_thres if getattr(args, "iou_thres", None) is not None
        else float(validation.get("iou_thres", 0.6))
    )
    max_det = (
        args.max_det if getattr(args, "max_det", None) is not None
        else int(validation.get("max_det", 300))
    )

    # Recursos
    device = args.device if args.device is not None else str(validation.get("device", ""))
    workers = int(validation.get("workers", 4))

    # Flags de salida
    save_txt = bool(validation.get("save_txt", False))
    save_hybrid = bool(validation.get("save_hybrid", False))
    save_conf = bool(validation.get("save_conf", False))
    save_json = bool(validation.get("save_json", False))
    plots = bool(validation.get("plots", True))
    exist_ok = bool(validation.get("exist_ok", False))

    # Logging NDJSON
    ndjson_console = bool(validation.get("ndjson_console", False))
    ndjson_file = bool(validation.get("ndjson_file", False))

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
    # Construcción del objeto ValidatorConfig
    # ---------------------------
    cfg = ValidatorConfig(
        task_model=task_model,
        variant=variant,
        run_name=run_name,
        phase=phase,
        data_config=data_config,
        weights=weights,
        batch_size=batch_size,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        device=device,
        workers=workers,
        save_txt=save_txt,
        save_hybrid=save_hybrid,
        save_conf=save_conf,
        save_json=save_json,
        plots=plots,
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
    """Define y parsea los argumentos de línea de comando para la validación YOLO."""

    parser = argparse.ArgumentParser(
        prog="YOLO.valid",
        description=(
            "Punto de entrada principal para validar modelos YOLO basados en YOLOv5 "
            "utilizando una configuración centralizada en YOLO/configs/valid.yaml."
        ),
    )

    # Configuración YAML
    parser.add_argument(
        "--cfg-valid",
        type=str,
        default=None,
        help="Ruta al archivo valid.yaml (por defecto: YOLO/configs/valid.yaml)",
    )

    # Selección de preset
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Nombre de un preset definido en valid.yaml (sección 'presets'). "
            "Aplica overrides antes de construir ValidatorConfig."
        ),
    )

    # Overrides rápidos
    parser.add_argument("--variant", type=str, default=None, help="Variante del modelo (n, s, m, l, x)")
    parser.add_argument("--run-name", type=str, default=None, help="Nombre lógico del experimento de validación")
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Split lógico del dataset a validar: 'val' o 'test' (override de valid.yaml)",
    )
    parser.add_argument("--device", type=str, default=None, help="Dispositivo: '', '0', '0,1', 'cpu', etc.")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch (override de valid.yaml)")
    parser.add_argument("--imgsz", type=int, default=None, help="Tamaño de imagen (override de valid.yaml)")
    parser.add_argument("--weights", type=str, default=None, help="Pesos a validar (ruta .pt o alias de YOLOv5)")
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=None,
        help="Umbral de confianza para NMS (override de valid.yaml)",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=None,
        help="Umbral IoU para NMS (override de valid.yaml)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="Máx. detecciones por imagen (override de valid.yaml)",
    )

    # Control de bootstrap MIOpen
    parser.add_argument(
        "--no-bootstrap-miopen",
        action="store_true",
        help="Desactiva el bootstrap MIOpen incluso si hay sección miopen en valid.yaml",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Entrypoint de validación YOLO.

    Flujo:
    1) Parseo de CLI.
    2) Carga de `valid.yaml`.
    3) Aplicación opcional de `--preset` (overrides sobre paths/validation).
    4) Bootstrap MIOpen (si corresponde) **antes** de importar Torch/YOLOv5.
    5) Instalación de filtros globales de warnings/logging del proyecto.
    6) Construcción de `ValidatorConfig` (incluyendo BN2GN).
    7) Ejecución de `Validator.run()`.
    """

    args = parse_args(argv)

    # 1) Resolver ruta a valid.yaml
    cfg_valid_path = Path(args.cfg_valid) if args.cfg_valid else (CONFIGS_ROOT / "valid.yaml")
    cfg_valid_path = cfg_valid_path.resolve()

    # 2) Cargar YAML de validación
    cfg_yaml = _load_valid_yaml(cfg_valid_path)

    # 3) Aplicar preset si se especifica
    if args.preset is not None:
        cfg_yaml = _apply_preset(cfg_yaml, args.preset)

    # 4) Bootstrap MIOpen antes de cualquier importación de Torch/YOLOv5
    miopen_cfg = None
    if not args.no_bootstrap_miopen and "miopen" in cfg_yaml:
        miopen_cfg = _build_miopen_config(cfg_yaml)
        if miopen_cfg is not None:
            bootstrap(miopen_cfg)

    # 5) Instalar sistema global de warnings/logging del proyecto
    try:
        from engine.warnings import install_global_warning_filters  # type: ignore
    except Exception:  # pragma: no cover - entorno sin módulo de warnings del proyecto
        install_global_warning_filters = None  # type: ignore

    if install_global_warning_filters is not None:
        install_global_warning_filters(force=False)

    # 6) Importar Validator una vez configurado el entorno MIOpen y los warnings
    from engine.Validator import Validator  # type: ignore

    validator_cfg = _build_validator_config(cfg_yaml, args, miopen_cfg)

    # 7) Ejecutar validación
    validator = Validator(validator_cfg)
    validator.run()


if __name__ == "__main__":  # pragma: no cover
    main()
