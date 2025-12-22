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
# ==============================================================

"""Entrenador principal para experimentos YOLO basados en YOLOv5.

Iteración actual del módulo `Trainer`, responsable de:

- Definir rutas canónicas del proyecto (YOLO/runs, YOLO/weights, YOLO/metrics).
- Construir una configuración de entrenamiento compatible con `yolov5/train.py`.
- Delegar el entrenamiento a YOLOv5 (detección) manteniendo el proyecto funcional.
- Sincronizar pesos y artefactos clave a las rutas del proyecto.
- Gestionar la política BN→GN (BatchNorm → GroupNorm) a nivel de
  configuración, dejando su aplicación efectiva al backend YOLOv5.
- Gestionar la fase lógica del experimento (train/val/test) para
  estructurar la jerarquía de runs, métricas y pesos.
- Exponer un flag de alto nivel para activar/desactivar Albumentations
  en la copia local de YOLOv5 mediante una variable de entorno.
- Gestionar la reanudación (resume) de entrenamientos interrumpidos,
  ya sea por ruta explícita o autodescubrimiento del último run.

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
- La fase lógica (`phase = train/val/test`) se define en
  `YOLO/configs/train.yaml` y se utiliza para construir rutas del tipo
  `YOLO/runs/<task>/<variant>/<phase>/<run_name>`.
- El flag `use_albumentations` se expone también desde
  `YOLO/configs/train.yaml` y se traduce en la variable de entorno
  `YOLO_DISABLE_ALBUMENTATIONS` para que la copia local de YOLOv5 pueda
  decidir si construye o no el pipeline de Albumentations.

Cambios relevantes en esta iteración
------------------------------------
- Se evita crear la carpeta de run (`save_dir`) **antes** de llamar a
  YOLOv5 para no forzar la auto-numeración (`...run`, `...run2`, etc.).
- Tras el entrenamiento, se resuelve dinámicamente cuál fue la carpeta
  de run efectiva utilizada por YOLOv5 (incluyendo posibles sufijos
  numéricos) y se sincronizan pesos y métricas desde ahí.
- La estructura de directorios de salida se extiende a:
  `YOLO/runs/<task>/<variant>/<phase>/<run_name>` y espejo en
  `YOLO/metrics`, mientras que los pesos consolidados se almacenan en
  `YOLO/weights/<task>/<variant>/<phase>/`.
- Se introduce un flag de configuración de alto nivel para controlar la
  activación de Albumentations en YOLOv5 vía la variable de entorno
  `YOLO_DISABLE_ALBUMENTATIONS`.
- Implementación de lógica de `resume` inteligente.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Rutas base de proyecto
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parents[1]  # .../YOLO
PROJECT_ROOT = YOLO_ROOT.parent  # carpeta raíz del proyecto (nivel superior a YOLO)
CONFIGS_ROOT = YOLO_ROOT / "configs"  # YOLO/configs
WEIGHTS_ROOT = YOLO_ROOT / "weights"  # YOLO/weights
RUNS_ROOT = YOLO_ROOT / "runs"  # YOLO/runs
METRICS_ROOT = YOLO_ROOT / "metrics"  # YOLO/metrics
DATASET_ROOT = PROJECT_ROOT / "Dataset"  # Proyecto/Dataset
YOLOV5_ROOT = YOLO_ROOT / "yolov5"  # copia local de YOLOv5 oficial

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


def _resolve_latest_run_dir(task: str, variant: str, phase: str, run_name: str) -> Optional[Path]:
    """Resuelve la carpeta de run efectiva usada por YOLOv5.

    YOLOv5 puede auto-numerar runs cuando `exist_ok=False`, generando
    directorios como `run`, `run2`, `run3`, etc. Esta utilidad busca en
    `YOLO/runs/<task>/<variant>/<phase>` todos los directorios cuyo
    nombre coincida con `run_name` o comience con `run_name` y
    selecciona el más recientemente modificado.
    """

    base_parent = RUNS_ROOT / task / variant / phase
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

    # Seleccionar el directorio con mayor mtime como run "activo" más reciente
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


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
    task: str = "detect"  # por ahora: "detect" (detección). Futuro: "classify".
    variant: str = "s"  # n, s, m, l, x (ej. YOLOv11-s)
    run_name: str = "exp"  # nombre lógico del experimento
    phase: str = "train"  # fase lógica: train / val / test

    # Datos y rutas
    data_config: Path = field(default_factory=lambda: CONFIGS_ROOT / "dataset.yaml")
    hyp: Optional[Path] = None  # ruta a hyp.yaml (hiperparámetros YOLOv5)
    weights: str = ""  # pesos iniciales (yolov5s.pt, ruta a .pt, o "")

    # Hiperparámetros esenciales
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    workers: int = max(os.cpu_count() - 1, 1) if os.cpu_count() else 2

    # Dispositivo y opciones de entrenamiento
    device: str = ""  # "" → auto, "0", "0,1", "cpu", etc.
    save_period: int = -1  # guarda epoch-k si > 0
    seed: int = 0
    exist_ok: bool = False  # reutilizar carpeta si existe

    # Reanudación de entrenamiento
    resume: bool | str = False  # False, True (auto), o ruta explícita

    # Opciones de logging
    ndjson_console: bool = False
    ndjson_file: bool = False

    # Control de augmentations externas (Albumentations)
    use_albumentations: bool = False

    # Esqueleto para integración con MIOpen/BN2GN
    miopen: Optional["MIOpenConfig"] = None  # se aplica en YOLO/train.py antes de importar torch
    bn2gn: Optional["BN2GNConfig"] = None  # configuración BN2GN (aplicación en backend YOLOv5)

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
    - Exponer la fase lógica del experimento para estructurar la
      jerarquía de directorios de salida.
    - Propagar la política de uso de Albumentations a YOLOv5 mediante
      la variable de entorno `YOLO_DISABLE_ALBUMENTATIONS`.
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg

        # Subcarpeta relativa "canónica" del experimento: p.ej.
        # detect/s/train/exp. Esta ruta se usa como intención de diseño,
        # pero la carpeta efectiva de YOLOv5 puede auto-numerarse.
        self.subdir = Path(self.cfg.task) / self.cfg.variant / self.cfg.phase / self.cfg.run_name

        # Carpeta "canónica" del run (intención). La carpeta efectiva
        # usada por YOLOv5 se resolverá luego de la ejecución.
        self.save_dir = RUNS_ROOT / self.subdir

        # Estas rutas se actualizarán tras `_resolve_effective_run_dirs`
        # para reflejar la carpeta real utilizada por YOLOv5.
        self.weights_dir = self.save_dir / "weights"
        self.metrics_dir = METRICS_ROOT / self.subdir

        # Carpeta de pesos globales (catálogo) para este tipo de tarea,
        # variante y fase (train/val/test).
        self.global_weights_dir = WEIGHTS_ROOT / self.cfg.task / self.cfg.variant / self.cfg.phase

        # Nota: la aplicación efectiva de BN→GN se delega al backend
        # YOLOv5, dado que es allí donde se construye el modelo. Aquí
        # sólo mantenemos la configuración en `self.cfg.bn2gn`.

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Ejecuta un entrenamiento completo delegando en `yolov5/train.py`.

        Flujo:
        1) Configura la variable de entorno para controlar el uso de
           Albumentations en YOLOv5.
        2) Asegura la existencia de carpetas base de salida (sin forzar
           la creación del run leaf para no interferir con YOLOv5).
        3) Construye un `Namespace` de opciones para YOLOv5.
        4) Importa y llama a `yolov5.train.main(opt)` para ejecutar el
           entrenamiento.
        5) Resuelve la carpeta de run efectiva de YOLOv5.
        6) Sincroniza pesos y artefactos clave a `YOLO/weights` y
           `YOLO/metrics`.

        Notas:
        - Se asume que MIOpen ya fue inicializado externamente mediante
          `bootstrap_miopen` en `YOLO/train.py` antes de instanciar este
          `Trainer`.
        - La política BN→GN se encuentra disponible en `self.cfg.bn2gn`,
          pero su aplicación se debe realizar dentro del backend YOLOv5
          una vez que exista el modelo.
        """

        # 1) Control de Albumentations vía variable de entorno
        if not self.cfg.use_albumentations:
            os.environ["YOLO_DISABLE_ALBUMENTATIONS"] = "1"
        else:
            os.environ.pop("YOLO_DISABLE_ALBUMENTATIONS", None)

        # 2) Crear estructura mínima de directorios
        self._ensure_directories()

        # 3) Construir opciones para YOLOv5
        opt = self._build_yolov5_opt()

        # 4) Import explícito del script de entrenamiento de YOLOv5 desde
        # YOLO/yolov5/train.py, evitando colisiones con YOLO/train.py.
        yolov5_train = _load_yolov5_train_module()

        if not hasattr(yolov5_train, "main"):
            raise RuntimeError(
                "El módulo cargado desde YOLO/yolov5/train.py no expone una función 'main(opt)'."
            )

        # Ejecutar entrenamiento principal de YOLOv5
        yolov5_train.main(opt)

        # 5) Resolver directorios efectivos (incluyendo auto-sufijos de YOLOv5)
        self._resolve_effective_run_dirs()

        # 6) Sincronizar artefactos a la jerarquía del proyecto
        self._sync_weights()
        self._sync_metrics()

    # ------------------------------------------------------------------
    # Resolución de Resume
    # ------------------------------------------------------------------

    def _resolve_resume_path(self) -> Union[str, bool]:
        """Resuelve la ruta de reanudación (resume) o devuelve False.

        Lógica:
        - Si self.cfg.resume es False -> retorna False.
        - Si es str -> retorna la ruta absoluta (si existe o no, YOLOv5 lo validará).
        - Si es True -> intenta autodescubrir el último run basado en task/variant/phase/run_name.
          Si encuentra 'last.pt', retorna esa ruta. Si no, advierte y retorna False.
        """
        if not self.cfg.resume:
            return False

        # Caso 1: Ruta explícita
        if isinstance(self.cfg.resume, str):
            # Convertir a absoluta para evitar ambigüedades en YOLOv5
            return str(Path(self.cfg.resume).resolve())

        # Caso 2: Autodescubrimiento (resume=True)
        if self.cfg.resume is True:
            latest_dir = _resolve_latest_run_dir(
                self.cfg.task, self.cfg.variant, self.cfg.phase, self.cfg.run_name
            )
            if latest_dir:
                last_pt = latest_dir / "weights" / "last.pt"
                if last_pt.is_file():
                    print(f"[Trainer] Auto-resume: encontrado checkpoint en {last_pt}")
                    return str(last_pt.resolve())

            print(f"[Trainer] ADVERTENCIA: --resume activado pero no se encontró 'last.pt' "
                  f"para el experimento '{self.cfg.run_name}'. Se iniciará entrenamiento desde cero.")
            return False

        return False

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
        project = RUNS_ROOT / self.cfg.task / self.cfg.variant / self.cfg.phase  # p.ej. YOLO/runs/detect/s/train
        name = self.cfg.run_name  # nombre lógico del experimento

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

        # Resolver lógica de resume
        resume_arg = self._resolve_resume_path()

        # Resolver ruta de pesos a utilizar
        # Si estamos reanudando, YOLOv5 prioriza 'resume', pero es buena práctica
        # alinear 'weights' con el checkpoint de reanudación si es explícito.
        if isinstance(resume_arg, str):
            weights_arg = resume_arg
        elif self.cfg.weights:
            weights_arg = str(self.cfg.weights)
        else:
            # Ruta por defecto del modelo base yolov5s.pt dentro de YOLO/weights
            weights_arg = str(WEIGHTS_ROOT / "yolov5s.pt")

        # Construir diccionario base de opciones
        opt_dict = dict(
            # Pesos y modelo
            weights=weights_arg,
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
            resume=resume_arg,  # False o ruta str
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
    # Resolución de directorios efectivos y sincronización de artefactos
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        """Crea las carpetas base necesarias para el experimento.

        Importante: **no** crea la carpeta leaf del run (`save_dir`), ya
        que esto provocaría que YOLOv5 la detecte como existente y
        auto-numere el run (añadiendo sufijos `2`, `3`, ...). En su
        lugar, se asegura únicamente de que existan:

        - `YOLO/runs`
        - `YOLO/runs/<task>/<variant>/<phase>`
        - `YOLO/weights/<task>/<variant>/<phase>`
        - `YOLO/metrics`
        """

        runs_task_variant_phase = RUNS_ROOT / self.cfg.task / self.cfg.variant / self.cfg.phase

        for p in (RUNS_ROOT, runs_task_variant_phase, WEIGHTS_ROOT, self.global_weights_dir, METRICS_ROOT):
            p.mkdir(parents=True, exist_ok=True)

    def _resolve_effective_run_dirs(self) -> None:
        """Resuelve y actualiza las rutas de run/weights/metrics tras YOLOv5.

        - Localiza la carpeta de run efectiva utilizada por YOLOv5
          (incluyendo sufijos numéricos si los hay).
        - Actualiza `self.save_dir` y `self.weights_dir` para apuntar a
          esa carpeta real.
        - Ajusta `self.metrics_dir` para mantener la misma sub-estructura
          relativa (`detect/s/train/<run_name_effectivo>`, por ejemplo).
        """

        latest = _resolve_latest_run_dir(self.cfg.task, self.cfg.variant, self.cfg.phase, self.cfg.run_name)
        if latest is None:
            # No se encontró carpeta efectiva; se mantiene la canónica.
            return

        self.save_dir = latest
        self.weights_dir = self.save_dir / "weights"

        # Mantener espejo de la estructura de runs dentro de YOLO/metrics
        try:
            rel_subdir = self.save_dir.relative_to(RUNS_ROOT)
        except ValueError:
            # Si por alguna razón la ruta no es relativa a RUNS_ROOT,
            # se vuelve al comportamiento canónico.
            rel_subdir = self.subdir

        self.metrics_dir = METRICS_ROOT / rel_subdir

    def _sync_weights(self) -> None:
        """Copia los pesos `best.pt` y `last.pt` al catálogo global `YOLO/weights`.

        Convención de nombres:
        - best → `<variant>_<run_name>_best.pt`
        - last → `<variant>_<run_name>_last.pt`

        Nota: el nombre de archivo utiliza el `run_name` lógico de la
        configuración, no el nombre final auto-numerado del directorio.
        Esto permite sobreescribir explícitamente el catálogo global con
        la versión más reciente de ese experimento lógico.
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