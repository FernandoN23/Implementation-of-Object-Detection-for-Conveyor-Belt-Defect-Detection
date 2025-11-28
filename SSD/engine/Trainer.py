# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/engine/Trainer.py
# Descripción: Entrenador principal del modelo SSD.
#              Orquestador de entrenamiento/validación para SSD300
#              sobre dataset con etiquetas en formato YOLO.
#==============================================================

from __future__ import annotations

import csv
import os
import random
import sys
import time
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------
# Rutas base del proyecto SSD
# --------------------------------------------------------------

FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parents[1]           # .../SSD
PROJECT_ROOT = SSD_ROOT.parent       # raíz del proyecto
CONFIGS_ROOT = SSD_ROOT / "configs"  # SSD/configs


# --------------------------------------------------------------
# Carga dinámica de módulos internos (evita problemas de paquetes)
# --------------------------------------------------------------


def _load_module_from(path: Path, name: str):
    """Carga dinámica de un módulo Python desde un path arbitrario.

    Esto permite usar los submódulos del repo original de SSD
    (ubicados bajo `SSD/ssd` y `SSD/utility`) sin requerir que
    todo el proyecto esté instalado como paquete en el entorno.
    """
    path = path.resolve()
    if not path.is_file():
        raise ImportError(f"No se encontró el módulo requerido en: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para módulo: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


# Módulo de datos (adaptador YOLO → SSD)
_DATA_LOADER_PATH = SSD_ROOT / "utility" / "data_loader.py"
_data_loader = _load_module_from(_DATA_LOADER_PATH, "ssd_data_loader")
load_dataset_config = _data_loader.load_dataset_config
build_dataloaders = _data_loader.build_dataloaders

# Módulo principal del modelo SSD
_SSD_MODEL_PATH = SSD_ROOT / "ssd" / "ssd.py"
_ssd_model = _load_module_from(_SSD_MODEL_PATH, "ssd_model")
# Función de construcción del modelo (convención del repo SSD original)
build_ssd = _ssd_model.build_ssd  # type: ignore[attr-defined]

# Módulo de pérdida Multibox
_MBL_PATH = SSD_ROOT / "ssd" / "layers" / "modules" / "multibox_loss.py"
_mbl = _load_module_from(_MBL_PATH, "ssd_multibox_loss")
MultiBoxLoss = _mbl.MultiBoxLoss  # type: ignore[attr-defined]

# Módulo de métricas (implementado en SSD/ssd/utils/metrics.py)
_METRICS_PATH = SSD_ROOT / "ssd" / "utils" / "metrics.py"
_metrics_mod = _load_module_from(_METRICS_PATH, "ssd_metrics")
fitness = _metrics_mod.fitness  # type: ignore[attr-defined]
ap_per_class = _metrics_mod.ap_per_class  # type: ignore[attr-defined]
ConfusionMatrix = _metrics_mod.ConfusionMatrix  # type: ignore[attr-defined]

# Parche BatchNorm → GroupNorm (si está disponible)
_BN2GN_PATH = SSD_ROOT / "engine" / "bn2gn_patch.py"
if _BN2GN_PATH.is_file():
    _bn2gn_mod = _load_module_from(_BN2GN_PATH, "ssd_bn2gn_patch")
    apply_bn2gn_patch = getattr(_bn2gn_mod, "apply_bn2gn_patch", None)
else:  # pragma: no cover - entorno sin parche
    apply_bn2gn_patch = None


# ==============================================================
# Configuración de entrenamiento
#==============================================================


@dataclass
class TrainerConfigSSD:
    """Configuración de alto nivel para `TrainerSSD`.

    Normaliza el contenido de `SSD/configs/train.yaml` para que el
    entrenador trabaje con tipos explícitos y rutas absolutas.
    """

    # Identidad del experimento
    task: str
    variant: str
    run_name: str
    phase: str

    # Paths base (absolutos)
    train_config_path: Path
    data_config: Path
    weights_root: Path
    runs_root: Path
    metrics_root: Path
    base_weights: Optional[Path]

    # Hiperparámetros de optimización
    opt: str
    lr: float
    momentum: float
    weight_decay: float
    gamma: float
    lr_steps: Tuple[int, ...]

    # Configuración de entrenamiento
    img_dim: int
    batch_size: int
    max_iter: int
    num_workers: int
    device: str
    resume: Optional[Path]
    save_period: int
    seed: int
    exist_ok: bool

    # Parámetros de pérdida y matching
    overlap_thresh: float
    neg_pos_ratio: int
    neg_overlap: float

    # Inference / NMS (para futuras extensiones y evaluación)
    conf_thresh: float
    nms_thresh: float
    top_k: int

    # Augmentations (se usa principalmente `mean`; el resto se respeta
    # para consistencia con el YAML)
    mean: Tuple[float, float, float]
    brightness_delta: int
    contrast_range: Tuple[float, float]
    saturation_range: Tuple[float, float]
    hue_delta: int
    random_mirror_prob: float

    # Logging
    ndjson_console: bool
    ndjson_file: bool

    # Configuración opcional BN→GN (se mantiene como dict genérico; el
    # módulo `bn2gn_patch` se encarga de interpretarlo)
    bn2gn_cfg: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(
        cls,
        train_config_path: str | Path,
        preset: str = "ssd300_default",
    ) -> "TrainerConfigSSD":
        """Construye `TrainerConfigSSD` a partir de SSD/configs/train.yaml.

        Parameters
        ----------
        train_config_path:
            Ruta al YAML de entrenamiento (por defecto SSD/configs/train.yaml).
        preset:
            Nombre del preset dentro de la clave `presets:` (ej. `ssd300_default`).
        """

        import yaml

        train_config_path = Path(train_config_path).resolve()
        with train_config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        exp_cfg = raw.get("experiment", {}) or {}
        paths_cfg = raw.get("paths", {}) or {}
        presets_cfg = raw.get("presets", {}) or {}
        if preset not in presets_cfg:
            raise KeyError(f"Preset '{preset}' no encontrado en {train_config_path}")
        p = presets_cfg[preset] or {}

        loss_cfg = p.get("loss", {}) or {}
        inf_cfg = p.get("inference", {}) or {}
        aug_cfg = p.get("augmentation", {}) or {}
        log_cfg = p.get("logging", {}) or {}
        bn2gn_cfg = p.get("bn2gn", {}) or None

        # Paths base relativos a la raíz del proyecto
        dataset_config_rel = Path(paths_cfg.get("dataset_config", "SSD/configs/dataset.yaml"))
        weights_root_rel = Path(paths_cfg.get("weights_root", "SSD/weights"))
        runs_root_rel = Path(paths_cfg.get("runs_root", "SSD/runs"))
        metrics_root_rel = Path(paths_cfg.get("metrics_root", "SSD/metrics"))

        data_config = (PROJECT_ROOT / dataset_config_rel).resolve()
        weights_root = (PROJECT_ROOT / weights_root_rel).resolve()
        runs_root = (PROJECT_ROOT / runs_root_rel).resolve()
        metrics_root = (PROJECT_ROOT / metrics_root_rel).resolve()

        # Pesos base VGG16 (opcionales)
        base_weights_rel = paths_cfg.get("base_weights", None)
        base_weights: Optional[Path] = None
        if base_weights_rel:
            base_weights = (PROJECT_ROOT / Path(base_weights_rel)).resolve()

        # Derivados/por defecto
        def _cpu_workers_default() -> int:
            n = os.cpu_count() or 2
            return max(n - 1, 1)

        lr_steps_raw = p.get("lr_steps", [80000, 100000, 120000])
        lr_steps = tuple(int(x) for x in lr_steps_raw)

        mean = tuple(aug_cfg.get("mean", [104, 117, 123]))  # type: ignore[assignment]
        contrast_range = tuple(aug_cfg.get("contrast_range", [0.5, 1.5]))  # type: ignore[assignment]
        saturation_range = tuple(aug_cfg.get("saturation_range", [0.5, 1.5]))  # type: ignore[assignment]

        resume_path = p.get("resume")
        resume: Optional[Path] = None
        if resume_path:
            resume = Path(resume_path).expanduser().resolve()

        return cls(
            # Identidad
            task=str(exp_cfg.get("task", "detect")),
            variant=str(exp_cfg.get("variant", "ssd300")),
            run_name=str(exp_cfg.get("run_name", "ssd300_experiment")),
            phase=str(exp_cfg.get("phase", "train")),

            # Paths
            train_config_path=train_config_path,
            data_config=data_config,
            weights_root=weights_root,
            runs_root=runs_root,
            metrics_root=metrics_root,
            base_weights=base_weights,

            # Optimización
            opt=str(p.get("opt", "SGD")),
            lr=float(p.get("lr", 1e-3)),
            momentum=float(p.get("momentum", 0.9)),
            weight_decay=float(p.get("weight_decay", 5e-4)),
            gamma=float(p.get("gamma", 0.1)),
            lr_steps=lr_steps,

            # Entrenamiento
            img_dim=int(p.get("img_dim", 300)),
            batch_size=int(p.get("batch_size", 32)),
            max_iter=int(p.get("max_iter", 120000)),
            num_workers=int(p.get("num_workers", _cpu_workers_default())),
            device=str(p.get("device", "")),
            resume=resume,
            save_period=int(p.get("save_period", 5000)),
            seed=int(p.get("seed", 0)),
            exist_ok=bool(p.get("exist_ok", False)),

            # Loss / matching
            overlap_thresh=float(loss_cfg.get("overlap_thresh", 0.5)),
            neg_pos_ratio=int(loss_cfg.get("neg_pos_ratio", 3)),
            neg_overlap=float(loss_cfg.get("neg_overlap", 0.5)),

            # Inference
            conf_thresh=float(inf_cfg.get("conf_thresh", 0.01)),
            nms_thresh=float(inf_cfg.get("nms_thresh", 0.45)),
            top_k=int(inf_cfg.get("top_k", 200)),

            # Augmentations
            mean=mean,  # type: ignore[arg-type]
            brightness_delta=int(aug_cfg.get("brightness_delta", 32)),
            contrast_range=contrast_range,  # type: ignore[arg-type]
            saturation_range=saturation_range,  # type: ignore[arg-type]
            hue_delta=int(aug_cfg.get("hue_delta", 18)),
            random_mirror_prob=float(aug_cfg.get("random_mirror_prob", 0.5)),

            # Logging
            ndjson_console=bool(log_cfg.get("ndjson_console", False)),
            ndjson_file=bool(log_cfg.get("ndjson_file", False)),

            # BN→GN
            bn2gn_cfg=bn2gn_cfg,
        )


# ==============================================================
# Entrenador principal
#==============================================================


class TrainerSSD:
    """Entrenador de modelo SSD (SSD300 por defecto).

    Responsabilidades principales:

    - Cargar configuraciones YAML (train/dataset) mediante TrainerConfigSSD.
    - Construir dataloaders a través de `SSD/utility/data_loader.py`.
    - Instanciar el modelo SSD (`build_ssd`) y aplicar parches BN→GN si
      están disponibles.
    - Configurar criterio de pérdida `MultiBoxLoss` y optimizador (SGD).
    - Ejecutar ciclo de entrenamiento/control basado en iteraciones
      (`max_iter`), con validación por época.
    - Registrar métricas de pérdidas en CSV y almacenar checkpoints de
      pesos (`best` y `last`).

    Notas
    -----
    - La mitigación MIOpen/ROCm *no* se ejecuta desde este módulo; debe
      hacerse antes de importar `torch` utilizando `engine/bootstrap_miopen.py`.
    - Las métricas avanzadas de detección (P/R/F1/mAP y curvas) pueden
      integrarse posteriormente utilizando `ssd.utils.metrics` a partir
      de las salidas de validación; aquí se deja un hook para esa lógica.
    """

    def __init__(self, cfg: TrainerConfigSSD) -> None:
        self.cfg = cfg

        # Rutas de salida organizadas por task/variant/phase/run_name
        subdir = Path(cfg.task) / cfg.variant / cfg.phase / cfg.run_name
        self.save_dir = cfg.runs_root / subdir
        self.weights_dir = cfg.weights_root / subdir
        self.metrics_dir = cfg.metrics_root / subdir
        self.results_csv_path = self.metrics_dir / "results.csv"

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Inicialización de dispositivo
        self.device = self._select_device(cfg.device)

        # Fijar semillas para reproducibilidad básica
        self._set_seeds(cfg.seed)

        # Configuración de dataset
        self.dataset_cfg = load_dataset_config(cfg.data_config)
        self.class_names = self._extract_class_names(self.dataset_cfg)
        self.num_classes = len(self.class_names)

        # Estado de entrenamiento
        self.iteration = 0
        self.epoch = 0
        self.best_metric = float("inf")  # menor pérdida de validación
        self._current_lr = cfg.lr
        self._lr_step_index = 0

        # Construcción de componentes principales
        self._build_model_and_loss()
        self._build_dataloaders()

        # Inicializar CSV de resultados
        self._init_results_csv()

    # ----------------------------------------------------------
    # Utilidades internas
    # ----------------------------------------------------------

    @staticmethod
    def _select_device(device_str: str) -> torch.device:
        if device_str:
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _extract_class_names(dataset_cfg: Dict[str, Any]) -> List[str]:
        names = dataset_cfg.get("names", {})
        if isinstance(names, dict):
            # Se asume que las claves son índices de clase (0,1,2,...) en str o int
            return [v for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if isinstance(names, (list, tuple)):
            return list(names)
        raise ValueError("Formato de 'names' no soportado en dataset.yaml")

    # ----------------------------------------------------------
    # Construcción de modelo, dataloaders y pérdidas
    # ----------------------------------------------------------

    def _build_model_and_loss(self) -> None:
        """Instancia el modelo SSD y configura el criterio de pérdida.

        - Usa `build_ssd('train', img_dim, num_classes)` del repo original.
        - Carga pesos base VGG16 si `base_weights` está definido y no
          se está reanudando desde un checkpoint.
        - Aplica parche BN→GN si `bn2gn_cfg.enabled=True` y el módulo
          `bn2gn_patch` está disponible.
        - Configura `MultiBoxLoss` con parámetros canónicos para SSD.
        """

        # 1) Modelo SSD (arquitectura)
        self.model: nn.Module = build_ssd("train", self.cfg.img_dim, self.num_classes)  # type: ignore[call-arg]

        # 2) Cargar pesos base VGG16 preentrenados si corresponde
        if self.cfg.resume is None and self.cfg.base_weights:
            try:
                vgg_state = torch.load(self.cfg.base_weights, map_location="cpu")
                if hasattr(self.model, "vgg"):
                    # El repo original expone el backbone como `ssd_net.vgg`
                    self.model.vgg.load_state_dict(vgg_state)  # type: ignore[attr-defined]
                    print(
                        f"[TrainerSSD] Pesos VGG16 preentrenados cargados desde: {self.cfg.base_weights}"
                    )
                else:
                    print(
                        "[TrainerSSD] Advertencia: el modelo SSD no expone atributo 'vgg'; "
                        "no se aplicaron pesos base."
                    )
            except Exception as exc:  # pragma: no cover - defensivo
                print(f"[TrainerSSD] Advertencia: no se pudieron cargar pesos base VGG16: {exc}")

        # 3) BN → GN si está disponible y configurado
        if apply_bn2gn_patch is not None and self.cfg.bn2gn_cfg:
            try:
                self.model = apply_bn2gn_patch(self.model, self.cfg.bn2gn_cfg)  # type: ignore[call-arg]
                print("[TrainerSSD] Parche BN→GN aplicado al modelo.")
            except Exception as exc:  # pragma: no cover - defensivo
                print(f"[TrainerSSD] Advertencia: no se pudo aplicar BN→GN: {exc}")

        # 4) Enviar modelo al dispositivo
        self.model.to(self.device)

        # 5) Criterio de pérdida (MultiBoxLoss clásico de SSD)
        # La signatura del repo original suele ser:
        #   MultiBoxLoss(num_classes, overlap_thresh, prior_for_matching,
        #                bkg_label, neg_pos, neg_overlap, encode_target,
        #                use_gpu=True)
        self.criterion = MultiBoxLoss(
            self.num_classes,
            self.cfg.overlap_thresh,
            True,
            0,
            self.cfg.neg_pos_ratio,
            self.cfg.neg_overlap,
            False,
            True,
        )

        # 6) Optimizador (SGD por defecto)
        if self.cfg.opt.upper() != "SGD":
            print(f"[TrainerSSD] Advertencia: por ahora sólo se soporta SGD; se ignorará opt={self.cfg.opt}.")
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        # 7) Puntos de cambio de LR (se aplican por iteración, no por época)
        self.lr_steps = list(self.cfg.lr_steps)

        # 8) Reanudar entrenamiento si corresponde
        if self.cfg.resume and self.cfg.resume.is_file():
            self._load_checkpoint(self.cfg.resume)

    def _build_dataloaders(self) -> None:
        """Construye DataLoaders de entrenamiento y validación."""

        train_loader, val_loader = build_dataloaders(self.cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader

    # ----------------------------------------------------------
    # Gestión de LR, checkpoints y logging CSV
    # ----------------------------------------------------------

    def _maybe_adjust_lr(self) -> None:
        """Aplica el decaimiento de LR cuando `iteration` supera `lr_steps`."""
        if self._lr_step_index >= len(self.lr_steps):
            return
        if self.iteration >= self.lr_steps[self._lr_step_index]:
            self._current_lr *= self.cfg.gamma
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._current_lr
            self._lr_step_index += 1
            print(f"[TrainerSSD] LR actualizado a {self._current_lr:.6f} (iter={self.iteration}).")

    def _checkpoint_paths(self) -> Tuple[Path, Path]:
        last = self.weights_dir / "last.pth"
        best = self.weights_dir / "best.pth"
        return last, best

    def _save_checkpoint(self, is_best: bool) -> None:
        """Guarda checkpoint `last.pth` y opcionalmente `best.pth`."""
        last_path, best_path = self._checkpoint_paths()

        state = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": asdict(self.cfg),
            "current_lr": self._current_lr,
        }
        torch.save(state, last_path)
        if is_best:
            torch.save(state, best_path)

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        """Carga un checkpoint existente para reanudar entrenamiento."""
        print(f"[TrainerSSD] Reanudando desde checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = int(ckpt.get("epoch", 0))
        self.iteration = int(ckpt.get("iteration", 0))
        self._current_lr = float(ckpt.get("current_lr", self.cfg.lr))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._current_lr
        print(
            f"[TrainerSSD] Estado restaurado: epoch={self.epoch}, iter={self.iteration}, LR={self._current_lr:.6f}."
        )

    def _init_results_csv(self) -> None:
        if not self.results_csv_path.is_file():
            header = [
                "epoch",
                "iteration",
                "lr",
                "train_loss_loc",
                "train_loss_conf",
                "train_loss_total",
                "val_loss_loc",
                "val_loss_conf",
                "val_loss_total",
            ]
            with self.results_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def _append_results_csv(
        self,
        train_stats: Dict[str, float],
        val_stats: Dict[str, float],
    ) -> None:
        row = [
            self.epoch,
            self.iteration,
            self._current_lr,
            train_stats.get("loss_loc", float("nan")),
            train_stats.get("loss_conf", float("nan")),
            train_stats.get("loss_total", float("nan")),
            val_stats.get("loss_loc", float("nan")),
            val_stats.get("loss_conf", float("nan")),
            val_stats.get("loss_total", float("nan")),
        ]
        with self.results_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ----------------------------------------------------------
    # Ciclo de entrenamiento / validación
    # ----------------------------------------------------------

    def fit(self) -> None:
        """Ejecuta el bucle principal de entrenamiento hasta `max_iter`."""

        max_iter = self.cfg.max_iter
        num_batches = len(self.train_loader)

        print(
            f"[TrainerSSD] Inicio entrenamiento SSD: max_iter={max_iter}, "
            f"batch_size={self.cfg.batch_size}, batches/epoch={num_batches}."
        )

        while self.iteration < max_iter:
            self.epoch += 1
            start_time = time.time()

            train_stats = self._train_one_epoch(max_iter)
            val_stats = self._validate_one_epoch()

            # Métrica de referencia: pérdida total de validación
            val_loss = val_stats.get("loss_total", float("inf"))
            is_best = val_loss < self.best_metric
            if is_best:
                self.best_metric = val_loss

            # Logging y checkpoints
            self._append_results_csv(train_stats, val_stats)
            self._save_checkpoint(is_best=is_best)

            elapsed = time.time() - start_time
            print(
                f"[TrainerSSD] Epoch {self.epoch:03d} | iter={self.iteration:06d}/{max_iter} | "
                f"LR={self._current_lr:.6f} | "
                f"train_loss={train_stats['loss_total']:.4f} | "
                f"val_loss={val_stats['loss_total']:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if self.iteration >= max_iter:
                print("[TrainerSSD] Se alcanzó max_iter; entrenamiento finalizado.")
                break

    def _train_one_epoch(self, max_iter: int) -> Dict[str, float]:
        """Ejecuta una época de entrenamiento y retorna pérdidas medias."""

        self.model.train()

        loss_loc_sum = 0.0
        loss_conf_sum = 0.0
        n_batches = 0

        for images, targets in self.train_loader:
            if self.iteration >= max_iter:
                break

            images = images.to(self.device, non_blocking=True)
            targets = [t.to(self.device) for t in targets]

            # Forward
            out = self.model(images)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                loc, conf, priors = out
            else:  # pragma: no cover - defensivo
                raise RuntimeError(
                    "La salida del modelo SSD en fase 'train' debe ser (loc, conf, priors)."
                )

            # Cálculo de pérdida
            loss_loc, loss_conf = self.criterion((loc, conf, priors), targets)
            loss = loss_loc + loss_conf

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Actualizar LR en base a iteraciones
            self.iteration += 1
            self._maybe_adjust_lr()

            # Acumuladores
            loss_loc_sum += float(loss_loc.detach().item())
            loss_conf_sum += float(loss_conf.detach().item())
            n_batches += 1

        if n_batches == 0:
            return {"loss_loc": float("nan"), "loss_conf": float("nan"), "loss_total": float("nan")}

        loss_loc_mean = loss_loc_sum / n_batches
        loss_conf_mean = loss_conf_sum / n_batches
        loss_total_mean = loss_loc_mean + loss_conf_mean

        return {
            "loss_loc": loss_loc_mean,
            "loss_conf": loss_conf_mean,
            "loss_total": loss_total_mean,
        }

    @torch.no_grad()
    def _validate_one_epoch(self) -> Dict[str, float]:
        """Evalúa el modelo en el conjunto de validación (sólo pérdidas).

        La integración con métricas de detección (P/R/F1/mAP) se puede
        añadir posteriormente reutilizando `ssd.utils.metrics` y las
        predicciones de la fase de test del modelo SSD.
        """

        self.model.eval()

        loss_loc_sum = 0.0
        loss_conf_sum = 0.0
        n_batches = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = [t.to(self.device) for t in targets]

            out = self.model(images)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                loc, conf, priors = out
            else:  # pragma: no cover - defensivo
                raise RuntimeError(
                    "La salida del modelo SSD en fase 'train' debe ser (loc, conf, priors)."
                )

            loss_loc, loss_conf = self.criterion((loc, conf, priors), targets)

            loss_loc_sum += float(loss_loc.detach().item())
            loss_conf_sum += float(loss_conf.detach().item())
            n_batches += 1

        if n_batches == 0:
            return {"loss_loc": float("nan"), "loss_conf": float("nan"), "loss_total": float("nan")}

        loss_loc_mean = loss_loc_sum / n_batches
        loss_conf_mean = loss_conf_sum / n_batches
        loss_total_mean = loss_loc_mean + loss_conf_mean

        return {
            "loss_loc": loss_loc_mean,
            "loss_conf": loss_conf_mean,
            "loss_total": loss_total_mean,
        }


__all__ = ["TrainerConfigSSD", "TrainerSSD"]
