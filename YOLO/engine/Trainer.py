# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/engine/Trainer.py
# Descripción: Entrenador principal del modelo YOLO.
#              Orquestador de runs, rutas y delegación a YOLOv5.
#==============================================================

"""Entrenador principal para experimentos YOLO basados en YOLOv5.

Iteración actual del módulo `Trainer`, responsable de:

- Definir rutas canónicas del proyecto (YOLO/runs, YOLO/weights, YOLO/metrics).
- Construir una configuración de entrenamiento compatible con `yolov5/train.py`.
- Delegar el entrenamiento a YOLOv5 (detección) manteniendo el proyecto funcional.
- Sincronizar pesos y artefactos clave a las rutas del proyecto.
- Gestionar la política BN→GN (BatchNorm → GroupNorm) a nivel de
  configuración, dejando su aplicación efectiva al backend YOLOv5.

Notas importantes
-----------------
- El bootstrap de MIOpen (desactivando la caché y configurando el
  entorno ROCm) se realiza en `YOLO/train.py` **antes** de importar este
  módulo y cualquier dependencia de PyTorch.
- Este `Trainer` asume que dicho bootstrap se ejecutó correctamente y
  no vuelve a modificar el entorno MIOpen.
- La política BN→GN se controla desde `YOLO/configs/train.yaml` (sección
  `bn2gn`) y se expone en `TrainerConfig.bn2gn`. La aplicación concreta
  sobre el modelo YOLOv5 deberá realizarse en el backend (por ejemplo,
  dentro del propio repo YOLOv5) una vez que exista una instancia de
  modelo.
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
PROJECT_ROOT = YOLO_ROOT.parent       # carpeta raíz del proyecto (nivel superior a YOLO)
CONFIGS_ROOT = YOLO_ROOT / "configs"  # YOLO/configs
WEIGHTS_ROOT = YOLO_ROOT / "weights"  # YOLO/weights
RUNS_ROOT = YOLO_ROOT / "runs"        # YOLO/runs
METRICS_ROOT = YOLO_ROOT / "metrics"  # YOLO/metrics
DATASET_ROOT = PROJECT_ROOT / "Dataset"  # Proyecto/Dataset
YOLOV5_ROOT = YOLO_ROOT / "yolov5"    # copia local de YOLOv5 oficial

if str(YOLOV5_ROOT) not in sys.path:
    # Necesario para que imports internos de YOLOv5 (models, utils, etc.)
    # funcionen correctamente cuando cargamos train.py como módulo.
    sys.path.append(str(YOLOV5_ROOT))

# Import de contexto para integración con BN2GN y MIOpen
try:  # pragma: no cover - se usa como hint de diseño y puede no estar en etapas tempranas
    from engine.bn2gn_patch import BN2GNConfig  # type: ignore
except Exception:  # pragma: no cover - tolerar falta del módulo en etapas tempranas
    BN2GNConfig = object  # type: ignore

try:  # pragma: no cover
    from engine.bootstrap_miopen import MIOpenConfig  # type: ignore
except Exception:  # pragma: no cover
    MIOpenConfig = object  # type: ignore


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _load_yolov5_train_module() -> Any:
    """Carga el script oficial de entrenamiento de YOLOv5 como módulo.

    Se fuerza explícitamente la ruta `YOLO/yolov5/train.py` mediante
    `importlib.util.spec_from_file_location` para evitar conflictos con
    el propio `YOLO/train.py` del proyecto.
    """

    train_path = YOLOV5_ROOT / "train.py"
    if not train_path.is_file():
        raise RuntimeError(
            f"No se encontró 'train.py' en el repositorio YOLOv5 esperado: {train_path}"
        )

    spec = importlib.util.spec_from_file_location("yolov5_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo crear el spec para cargar yolov5.train")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Configuración de alto nivel del Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Configuración de alto nivel para el `Trainer`.

    Esta capa se diseña como puente entre:
    - Los parámetros conceptuales del proyecto (variant, nombre de run,
      dataset, política BN2GN, etc.).
    - Los argumentos concretos esperados por `yolov5/train.py`.

    La estructura puede extenderse sin romper la API pública mientras se
    respeten los nombres actuales de los campos.
    """

    # Identidad del experimento
    task: str = "detect"         # por ahora: "detect" (detección). Futuro: "classify".
    variant: str = "s"           # n, s, m, l, x (ej. YOLOv11-s)
    run_name: str = "exp"        # nombre lógico del experimento

    # Datos y rutas
    data_config: Path = field(default_factory=lambda: CONFIGS_ROOT / "dataset.yaml")
    hyp: Optional[Path] = None          # ruta a hyp.yaml (hiperparámetros YOLOv5)
    weights: str = ""                  # pesos iniciales (yolov5s.pt, ruta a .pt, o "")

    # Hiperparámetros esenciales
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    workers: int = max(os.cpu_count() - 1, 1) if os.cpu_count() else 2

    # Dispositivo y opciones de entrenamiento
    device: str = ""                   # "" → auto, "0", "0,1", "cpu", etc.
    save_period: int = -1               # guarda epoch-k si > 0
    seed: int = 0
    exist_ok: bool = False              # reutilizar carpeta si existe

    # Opciones de logging
    ndjson_console: bool = False
    ndjson_file: bool = False

    # Esqueleto para integración con MIOpen/BN2GN
    miopen: Optional["MIOpenConfig"] = None  # se aplica en YOLO/train.py antes de importar torch
    bn2gn: Optional["BN2GNConfig"] = None    # configuración BN2GN (aplicación en backend YOLOv5)

    def as_dict(self) -> Dict:
        """Retorna la configuración en formato dict estándar (útil para logs/meta)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Clase Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orquestador de entrenamiento YOLO basado en YOLOv5.

    Responsabilidades principales:
    - Resolver rutas `save_dir`, `weights_dir` y `metrics_dir` según las
      convenciones del proyecto.
    - Construir un `argparse.Namespace` compatible con `yolov5/train.py`.
    - Delegar el entrenamiento al script oficial de YOLOv5.
    - Sincronizar pesos y artefactos clave a `YOLO/weights` y `YOLO/metrics`.
    - Exponer la configuración BN→GN (si está configurada) para que el
      backend YOLOv5 pueda aplicarla donde corresponda (modelo ya
      instanciado).
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg

        # Subcarpeta relativa del experimento: p.ej. detect/s/exp, detect/m/belt_A, etc.
        self.subdir = Path(self.cfg.task) / self.cfg.variant / self.cfg.run_name

        # Carpeta principal del run (donde YOLOv5 escribirá todo).
        self.save_dir = RUNS_ROOT / self.subdir

        # Carpeta de pesos asociada al run.
        self.weights_dir = self.save_dir / "weights"

        # Carpeta de métricas agregadas del proyecto.
        self.metrics_dir = METRICS_ROOT / self.subdir

        # Carpeta de pesos globales (catálogo) para este tipo de tarea.
        self.global_weights_dir = WEIGHTS_ROOT / self.cfg.task

        # Nota: la aplicación efectiva de BN→GN se delega al backend
        # YOLOv5, dado que es allí donde se construye el modelo. Aquí
        # sólo mantenemos la configuración en `self.cfg.bn2gn`.

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Ejecuta un entrenamiento completo delegando en `yolov5/train.py`.

        Flujo:
        1) Asegura la existencia de carpetas de salida.
        2) Construye un `Namespace` de opciones para YOLOv5.
        3) Importa y llama a `yolov5.train.main(opt)` para ejecutar el
           entrenamiento.
        4) Sincroniza pesos y artefactos clave a `YOLO/weights` y
           `YOLO/metrics`.

        Notas:
        - Se asume que MIOpen ya fue inicializado externamente mediante
          `bootstrap_miopen` en `YOLO/train.py` antes de instanciar este
          `Trainer`.
        - La política BN→GN se encuentra disponible en `self.cfg.bn2gn`,
          pero su aplicación se debe realizar dentro del backend YOLOv5
          una vez que exista el modelo.
        """

        self._ensure_directories()
        opt = self._build_yolov5_opt()

        # Import explícito del script de entrenamiento de YOLOv5 desde
        # YOLO/yolov5/train.py, evitando colisiones con YOLO/train.py.
        yolov5_train = _load_yolov5_train_module()

        if not hasattr(yolov5_train, "main"):
            raise RuntimeError(
                "El módulo cargado desde YOLO/yolov5/train.py no expone una función 'main(opt)'."
            )

        # Ejecutar entrenamiento principal de YOLOv5
        yolov5_train.main(opt)

        # Sincronizar artefactos a la jerarquía del proyecto
        self._sync_weights()
        self._sync_metrics()

    # ------------------------------------------------------------------
    # Construcción de opciones para YOLOv5
    # ------------------------------------------------------------------

    def _build_yolov5_opt(self) -> argparse.Namespace:
        """Construye un `argparse.Namespace` compatible con `yolov5/train.py`.

        La idea es replicar los argumentos esperados por `parse_opt()` en
        YOLOv5, pero alimentándolos desde `TrainerConfig`.
        Sólo se cubre el subconjunto más relevante para esta etapa; el
        resto puede añadirse posteriormente sin romper la API.
        """

        # Rutas de proyecto/name según convención de YOLOv5
        project = self.save_dir.parent  # .../YOLO/runs/detect/s
        name = self.save_dir.name       # run_name

        # Configuración BN2GN para propagarla al backend YOLOv5
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

        # Construir diccionario base de opciones
        opt_dict = dict(
            # Pesos y modelo
            weights=self.cfg.weights or "yolov5s.pt",
            cfg="",  # se usará el modelo por defecto asociado a los pesos

            # Dataset y hyps
            data=str(self.cfg.data_config),
            hyp=str(self.cfg.hyp) if self.cfg.hyp is not None else str(YOLOV5_ROOT / "data/hyps/hyp.scratch-low.yaml"),

            # Hiperparámetros básicos
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            imgsz=self.cfg.imgsz,

            # Opciones de entrenamiento (por ahora, valores razonables por defecto)
            rect=False,
            resume=False,
            nosave=False,
            noval=False,
            noautoanchor=False,
            noplots=False,
            evolve=None,
            bucket="",
            cache="ram",
            image_weights=False,
            device=self.cfg.device,
            multi_scale=False,
            single_cls=False,
            optimizer="SGD",
            sync_bn=False,
            workers=self.cfg.workers,

            # Rutas de salida
            project=str(project),
            name=name,
            exist_ok=self.cfg.exist_ok,

            # Extras y estabilidad
            quad=False,
            cos_lr=False,
            label_smoothing=0.0,
            patience=50,
            freeze=[0],
            save_period=self.cfg.save_period,
            seed=self.cfg.seed,
            local_rank=-1,
            entity=None,
            upload_dataset=False,
            bbox_interval=-1,
            artifact_alias="latest",

            # Logging NDJSON
            ndjson_console=self.cfg.ndjson_console,
            ndjson_file=self.cfg.ndjson_file,

            # Configuración BN2GN (consumida en YOLOv5/train.py)
            bn2gn_policy=bn2gn_policy,
            bn2gn_max_groups=bn2gn_max_groups,
            bn2gn_min_channels_per_group=bn2gn_min_channels_per_group,
            bn2gn_verbose=bn2gn_verbose,
        )

        return argparse.Namespace(**opt_dict)

    # ------------------------------------------------------------------
    # Sincronización de artefactos
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        """Crea las carpetas necesarias para el experimento si no existen."""

        for p in (self.save_dir, self.weights_dir, self.metrics_dir, self.global_weights_dir):
            p.mkdir(parents=True, exist_ok=True)

    def _sync_weights(self) -> None:
        """Copia los pesos `best.pt` y `last.pt` al catálogo global `YOLO/weights`.

        Convención de nombres:
        - best → `<variant>_<run_name>_best.pt`
        - last → `<variant>_<run_name>_last.pt`
        """

        best_src = self.weights_dir / "best.pt"
        last_src = self.weights_dir / "last.pt"

        self.global_weights_dir.mkdir(parents=True, exist_ok=True)

        if best_src.is_file():
            best_dst = self.global_weights_dir / f"{self.cfg.variant}_{self.cfg.run_name}_best.pt"
            shutil.copy2(best_src, best_dst)

        if last_src.is_file():
            last_dst = self.global_weights_dir / f"{self.cfg.variant}_{self.cfg.run_name}_last.pt"
            shutil.copy2(last_src, last_dst)

    def _sync_metrics(self) -> None:
        """Copia métricas y gráficos clave desde `save_dir` a `YOLO/metrics`.

        Esta función asume la convención estándar de YOLOv5:
        - `results.csv`, `results.png` generados por el logger principal.
        - Curvas PR/F1/P/R, matriz de confusión, etc., cuando están
          disponibles en la carpeta del experimento.
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


__all__ = ["TrainerConfig", "Trainer"]
