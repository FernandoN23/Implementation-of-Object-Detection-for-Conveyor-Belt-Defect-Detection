# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/engine/Validator.py
# Descripción: Orquestador principal de validación YOLO.
#              Interfaz entre YOLO/valid.py, dataset.yaml y
#              los backends de validación de YOLOv5 (detección
#              y clasificación).
#==============================================================

"""Módulo de validación de alto nivel para experimentos YOLO.

Este `Validator` actúa como puente entre:
- La configuración externa (`YOLO/configs/valid.yaml`).
- El entrypoint CLI `YOLO/valid.py`.
- Los backends de validación de Ultralytics YOLOv5:
  - Detección:  `YOLO/yolov5/val.py`.
  - Clasificación: `YOLO/yolov5/classify/val.py`.

Responsabilidades principales
----------------------------
- Resolver rutas canónicas de salida para runs y métricas:
  `YOLO/runs/<task_model>/<variant>/<phase>/<run_name>` y
  `YOLO/metrics/<task_model>/<variant>/<phase>/<run_name>`.
- Construir un conjunto de opciones (`argparse.Namespace`) compatible
  con los scripts `val.py` de YOLOv5.
- Ejecutar la validación de detección o clasificación según
  `ValidatorConfig.task_model`.
- Localizar la carpeta de run efectiva utilizada por YOLOv5 (incluyendo
  posibles sufijos de auto-numeración) y sincronizar métricas clave a
  `YOLO/metrics/...`.
- Propagar la configuración BN→GN (BatchNorm→GroupNorm), cuando exista,
  hacia los backends de validación de YOLOv5 mediante flags explícitos.

Notas
-----
- El bootstrap de MIOpen (entorno ROCm) se ejecuta en `YOLO/valid.py`
  **antes** de importar este módulo, por lo que aquí se asume que el
  entorno ya está inicializado.
- Este módulo no importa directamente Torch; delega toda la carga de
  modelos e inferencia a los backends de YOLOv5.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Rutas base de proyecto
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parents[1]           # .../YOLO
PROJECT_ROOT = YOLO_ROOT.parent       # raíz del proyecto (nivel superior a YOLO)
CONFIGS_ROOT = YOLO_ROOT / "configs"  # YOLO/configs
RUNS_ROOT = YOLO_ROOT / "runs"        # YOLO/runs
METRICS_ROOT = YOLO_ROOT / "metrics"  # YOLO/metrics
YOLOV5_ROOT = YOLO_ROOT / "yolov5"    # copia local de YOLOv5 oficial

if str(YOLOV5_ROOT) not in sys.path:
    # Necesario para que imports internos de YOLOv5 (models, utils, etc.)
    # funcionen correctamente cuando cargamos val.py como módulo.
    sys.path.append(str(YOLOV5_ROOT))

# Import de tipos para integración con MIOpen/BN2GN (opcionales)
try:  # pragma: no cover
    from engine.bn2gn_patch import BN2GNConfig  # type: ignore
except Exception:  # pragma: no cover
    BN2GNConfig = object  # type: ignore

try:  # pragma: no cover
    from engine.bootstrap_miopen import MIOpenConfig  # type: ignore
except Exception:  # pragma: no cover
    MIOpenConfig = object  # type: ignore


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _load_yolov5_detect_val_module() -> Any:
    """Carga `YOLO/yolov5/val.py` como módulo.

    Se fuerza explícitamente la ruta mediante `importlib.util` para
    evitar conflictos con otros módulos llamados `val.py` en el
    proyecto.
    """

    val_path = YOLOV5_ROOT / "val.py"
    if not val_path.is_file():
        raise RuntimeError(
            f"No se encontró 'val.py' en el repositorio YOLOv5 esperado: {val_path}"
        )

    spec = importlib.util.spec_from_file_location("yolov5_val", val_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo crear el spec para cargar yolov5.val")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yolov5_classify_val_module() -> Any:
    """Carga `YOLO/yolov5/classify/val.py` como módulo.

    Este backend se utiliza para validación de modelos de clasificación
    (`task_model='classify'`).
    """

    val_path = YOLOV5_ROOT / "classify" / "val.py"
    if not val_path.is_file():
        raise RuntimeError(
            f"No se encontró 'classify/val.py' en el repositorio YOLOv5 esperado: {val_path}"
        )

    spec = importlib.util.spec_from_file_location("yolov5_classify_val", val_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo crear el spec para cargar yolov5.classify.val")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_latest_run_dir(task_model: str, variant: str, phase: str, run_name: str) -> Optional[Path]:
    """Resuelve la carpeta de run efectiva usada por YOLOv5.

    YOLOv5 auto-numera runs cuando `exist_ok=False`, generando
    directorios como `run_name`, `run_name2`, `run_name3`, etc. Esta
    utilidad busca en `YOLO/runs/<task_model>/<variant>/<phase>` todos
    los directorios cuyo nombre coincida con `run_name` o comience con
    `run_name` y selecciona el más recientemente modificado.
    """

    base_parent = RUNS_ROOT / task_model / variant / phase
    if not base_parent.exists():
        return None

    candidates = []
    prefix = run_name
    for d in base_parent.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name == prefix or name.startswith(prefix):
            candidates.append(d)

    if not candidates:
        return None

    # Directorio con mayor mtime → run efectivo más reciente
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


# ---------------------------------------------------------------------------
# Configuración de alto nivel del Validator
# ---------------------------------------------------------------------------


@dataclass
class ValidatorConfig:
    """Configuración de alto nivel para el `Validator`.

    Esta capa intermedia se diseña como puente entre:
    - Los parámetros conceptuales del proyecto (tipo de modelo, variante,
      dataset, política BN2GN, etc.).
    - Los argumentos concretos esperados por los backends de validación
      de YOLOv5 (`val.py` de detección y clasificación).

    La estructura puede extenderse sin romper la API pública mientras se
    respeten los nombres actuales de los campos.
    """

    # Identidad del experimento y tipo de modelo
    task_model: str = "detect"      # "detect" (YOLOv5) o "classify" (YOLOv5-cls)
    variant: str = "s"              # n, s, m, l, x (ej. YOLOv5/11-s)
    run_name: str = "val_default"   # nombre lógico del experimento de validación
    phase: str = "val"              # split lógico del dataset: "val" o "test"

    # Datos y rutas
    data_config: Path = field(default_factory=lambda: CONFIGS_ROOT / "dataset.yaml")
    weights: str = ""               # ruta o alias a pesos .pt

    # Hiperparámetros de validación (detección)
    batch_size: int = 4
    imgsz: int = 640
    conf_thres: float = 0.001
    iou_thres: float = 0.6
    max_det: int = 300

    # Recursos
    device: str = ""                # "" → auto, "0", "0,1", "cpu", etc.
    workers: int = max(os.cpu_count() - 1, 1) if os.cpu_count() else 2

    # Flags de salida
    save_txt: bool = False
    save_hybrid: bool = False
    save_conf: bool = False
    save_json: bool = False
    plots: bool = True
    exist_ok: bool = False

    # Logging NDJSON
    ndjson_console: bool = False
    ndjson_file: bool = False

    # Integración con MIOpen/BN2GN
    miopen: Optional["MIOpenConfig"] = None      # inicializado en YOLO/valid.py
    bn2gn: Optional["BN2GNConfig"] = None        # política BN->GN a propagar al backend

    def as_dict(self) -> Dict[str, Any]:
        """Retorna la configuración en formato dict estándar (útil para logs/meta)."""

        return asdict(self)


# ---------------------------------------------------------------------------
# Clase Validator
# ---------------------------------------------------------------------------


class Validator:
    """Orquestador de validación YOLO basado en backends YOLOv5.

    Responsabilidades principales:
    - Resolver rutas de `project` y `name` para los scripts de
      validación de YOLOv5 (detección y clasificación).
    - Construir un `argparse.Namespace` compatible con `yolov5/val.py` y
      `yolov5/classify/val.py`.
    - Ejecutar la validación según `cfg.task_model`.
    - Sincronizar métricas clave hacia `YOLO/metrics`.
    """

    def __init__(self, cfg: ValidatorConfig) -> None:
        self.cfg = cfg

        # Subcarpeta relativa "canónica" del experimento: p.ej.
        # detect/s/val/val_default o classify/s/val/val_default
        self.subdir = Path(self.cfg.task_model) / self.cfg.variant / self.cfg.phase / self.cfg.run_name

        # Carpeta "canónica" del run; la carpeta efectiva se resolverá
        # tras la ejecución del backend YOLOv5.
        self.save_dir = RUNS_ROOT / self.subdir
        self.metrics_dir = METRICS_ROOT / self.subdir

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Ejecuta una campaña de validación completa.

        Flujo:
        1) Crear estructura mínima de directorios de salida.
        2) Construir opciones (`Namespace`) para el backend adecuado.
        3) Cargar el módulo de validación de YOLOv5 (detección o
           clasificación) y ejecutar su función principal.
        4) Resolver la carpeta de run efectiva (auto-sufijos).
        5) Sincronizar métricas clave a `YOLO/metrics`.
        """

        # 1) Crear estructura mínima de directorios
        self._ensure_directories()

        # 2) Construir opciones para el backend adecuado
        task_model = self.cfg.task_model.lower()
        if task_model == "detect":
            opt = self._build_yolov5_detect_val_opt()
            module = _load_yolov5_detect_val_module()
        elif task_model == "classify":
            opt = self._build_yolov5_classify_val_opt()
            module = _load_yolov5_classify_val_module()
        else:
            raise ValueError(f"task_model desconocido en ValidatorConfig: {self.cfg.task_model!r}")

        # 3) Ejecutar backend YOLOv5
        if hasattr(module, "main"):
            module.main(opt)  # type: ignore[arg-type]
        elif hasattr(module, "run"):
            module.run(**vars(opt))  # type: ignore[call-arg]
        else:  # pragma: no cover
            raise RuntimeError(
                "El módulo de validación de YOLOv5 no expone ni 'main(opt)' ni 'run(**kwargs)'."
            )

        # 4) Resolver directorio de run efectivo (incluyendo auto-sufijos)
        self._resolve_effective_run_dirs()

        # 5) Sincronizar métricas a la jerarquía del proyecto
        self._sync_metrics()

    # ------------------------------------------------------------------
    # Construcción de opciones para YOLOv5 (detección y clasificación)
    # ------------------------------------------------------------------

    def _build_yolov5_detect_val_opt(self) -> argparse.Namespace:
        """Construye un `Namespace` compatible con `yolov5/val.py` (detección).

        La idea es replicar los argumentos de alto nivel esperados por
        `parse_opt()` en YOLOv5, alimentándolos desde `ValidatorConfig`.
        Sólo se cubre el subconjunto más relevante; el resto puede
        ampliarse sin romper la API.
        """

        # Rutas de proyecto/name según convención de YOLOv5
        # p.ej. YOLO/runs/detect/s/val
        project = RUNS_ROOT / "detect" / self.cfg.variant / self.cfg.phase
        name = self.cfg.run_name

        # Configuración BN2GN a propagar al backend de detección
        if isinstance(self.cfg.bn2gn, BN2GNConfig):  # type: ignore[arg-type]
            bn2gn_policy = getattr(self.cfg.bn2gn, "policy", "off")
            bn2gn_max_groups = int(getattr(self.cfg.bn2gn, "max_groups", 32))
            bn2gn_min_channels_per_group = int(getattr(self.cfg.bn2gn, "min_channels_per_group", 1))
            bn2gn_verbose = int(getattr(self.cfg.bn2gn, "verbose", 1))
        else:
            bn2gn_policy = "off"
            bn2gn_max_groups = 32
            bn2gn_min_channels_per_group = 1
            bn2gn_verbose = 1

        # Construir diccionario base de opciones para detección
        opt_dict = dict(
            # Pesos y dataset
            weights=self.cfg.weights,
            data=str(self.cfg.data_config),

            # Hiperparámetros de validación (detección)
            batch_size=self.cfg.batch_size,
            imgsz=self.cfg.imgsz,
            conf_thres=self.cfg.conf_thres,
            iou_thres=self.cfg.iou_thres,
            max_det=self.cfg.max_det,

            # Control de dataset/split
            task=self.cfg.phase,  # "val" o "test" → data[task]

            # Recursos
            device=self.cfg.device,
            workers=self.cfg.workers,

            # Flags de salida y comportamiento
            save_txt=self.cfg.save_txt,
            save_hybrid=self.cfg.save_hybrid,
            save_conf=self.cfg.save_conf,
            save_json=self.cfg.save_json,
            project=str(project),
            name=name,
            exist_ok=self.cfg.exist_ok,
            half=False,        # por defecto; se puede exponer en valid.yaml si se requiere
            dnn=False,
            plots=self.cfg.plots,

            # Parámetros adicionales con valores razonables por defecto
            single_cls=False,
            augment=False,
            verbose=True,
            rect=False,
            classes=None,
            agnostic_nms=False,
            max_frames=0,

            # Logging NDJSON (consumido opcionalmente por un fork de YOLOv5)
            ndjson_console=self.cfg.ndjson_console,
            ndjson_file=self.cfg.ndjson_file,

            # Configuración BN2GN a nivel de backend de detección
            bn2gn_policy=bn2gn_policy,
            bn2gn_max_groups=bn2gn_max_groups,
            bn2gn_min_channels_per_group=bn2gn_min_channels_per_group,
            bn2gn_verbose=bn2gn_verbose,
        )

        return argparse.Namespace(**opt_dict)

    def _build_yolov5_classify_val_opt(self) -> argparse.Namespace:
        """Construye un `Namespace` compatible con `yolov5/classify/val.py`.

        Este backend se centra en métricas de clasificación (top-1,
        top-5). Muchos parámetros de detección (conf_thres, iou_thres,
        max_det) no aplican y se ignoran.
        """

        # Rutas de proyecto/name para clasificación; se separan de los
        # runs de detección para mantener claridad en la jerarquía.
        project = RUNS_ROOT / "classify" / self.cfg.variant / self.cfg.phase
        name = self.cfg.run_name

        # Configuración BN2GN a propagar al backend de clasificación
        if isinstance(self.cfg.bn2gn, BN2GNConfig):  # type: ignore[arg-type]
            bn2gn_policy = getattr(self.cfg.bn2gn, "policy", "off")
            bn2gn_max_groups = int(getattr(self.cfg.bn2gn, "max_groups", 32))
            bn2gn_min_channels_per_group = int(getattr(self.cfg.bn2gn, "min_channels_per_group", 1))
            bn2gn_verbose = int(getattr(self.cfg.bn2gn, "verbose", 1))
        else:
            bn2gn_policy = "off"
            bn2gn_max_groups = 32
            bn2gn_min_channels_per_group = 1
            bn2gn_verbose = 1

        opt_dict = dict(
            # Pesos y dataset (en clasificación, `data` suele ser ruta a
            # un ImageFolder o a un .yaml específico de clasificación).
            weights=self.cfg.weights,
            data=str(self.cfg.data_config),

            # Hiperparámetros básicos
            batch_size=self.cfg.batch_size,
            imgsz=self.cfg.imgsz,

            # Recursos
            device=self.cfg.device,
            workers=self.cfg.workers,

            # Flags de salida
            project=str(project),
            name=name,
            exist_ok=self.cfg.exist_ok,
            half=False,
            dnn=False,
            plots=self.cfg.plots,

            # Logging NDJSON (si el backend extendido lo soporta)
            ndjson_console=self.cfg.ndjson_console,
            ndjson_file=self.cfg.ndjson_file,

            # Configuración BN2GN (en caso de que el backend la utilice)
            bn2gn_policy=bn2gn_policy,
            bn2gn_max_groups=bn2gn_max_groups,
            bn2gn_min_channels_per_group=bn2gn_min_channels_per_group,
            bn2gn_verbose=bn2gn_verbose,
        )

        return argparse.Namespace(**opt_dict)

    # ------------------------------------------------------------------
    # Resolución de directorios efectivos y sincronización de métricas
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        """Crea las carpetas base necesarias para la validación.

        Importante: **no** crea la carpeta leaf del run (`save_dir`), ya
        que esto provocaría que YOLOv5 la detecte como existente y
        auto-numere el run (añadiendo sufijos). En su lugar, se asegura
        únicamente de que existan:

        - `YOLO/runs/<task_model>/<variant>/<phase>`
        - `YOLO/metrics`
        """

        runs_task_variant_phase = RUNS_ROOT / self.cfg.task_model / self.cfg.variant / self.cfg.phase

        for p in (RUNS_ROOT, runs_task_variant_phase, METRICS_ROOT):
            p.mkdir(parents=True, exist_ok=True)

    def _resolve_effective_run_dirs(self) -> None:
        """Resuelve y actualiza las rutas de run/métricas tras la validación.

        - Localiza la carpeta de run efectiva utilizada por YOLOv5
          (incluyendo sufijos numéricos si los hay).
        - Actualiza `self.save_dir` para apuntar a esa carpeta real.
        - Ajusta `self.metrics_dir` para mantener la misma subestructura
          relativa (`detect/s/val/<run_name_effectivo>`, por ejemplo).
        """

        latest = _resolve_latest_run_dir(
            self.cfg.task_model, self.cfg.variant, self.cfg.phase, self.cfg.run_name
        )
        if latest is None:
            # No se encontró carpeta efectiva; se mantiene la canónica.
            return

        self.save_dir = latest

        # Mantener espejo de la estructura de runs dentro de YOLO/metrics
        try:
            rel_subdir = self.save_dir.relative_to(RUNS_ROOT)
        except ValueError:
            # Si por alguna razón la ruta no es relativa a RUNS_ROOT,
            # se vuelve al comportamiento canónico.
            rel_subdir = self.subdir

        self.metrics_dir = METRICS_ROOT / rel_subdir

    def _sync_metrics(self) -> None:
        """Copia métricas y gráficos clave desde `save_dir` a `YOLO/metrics`.

        Se asume la convención estándar de YOLOv5:
        - `results.csv`, `results.png` generados por el logger principal.
        - Curvas PR/F1/P/R y matriz de confusión en detección.
        - Para clasificación, al menos `results.csv` y gráficos
          asociados si están disponibles.
        """

        if not self.save_dir.exists():
            return

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            "results.csv",
            "results.png",
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "confusion_matrix.png",
            "labels.jpg",
            "labels_correlogram.jpg",
        ]

        for name in candidates:
            src = self.save_dir / name
            if src.is_file():
                dst = self.metrics_dir / name
                shutil.copy2(src, dst)


__all__ = ["ValidatorConfig", "Validator"]
