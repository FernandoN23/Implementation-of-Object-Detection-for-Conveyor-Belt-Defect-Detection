# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: parser_yaml.py
# Carga y orquesta las configuraciones YAML del proyecto:
#  - configs/parser.yaml    (rutas, runtime, save-policy, punteros a otros YAML)
#  - configs/train.yaml     (hiperparámetros de entrenamiento)
#  - configs/yolo11.yaml    (parámetros del modelo)
#  - configs/model_variants.yaml (escala por variante d,w,mc) [opcional]
#  - configs/dataset.yaml   (rutas dataset, names, nc) [opcional]
# Expone: ConfigParserYaml con utilidades para:
#  - Resolver variante (d,w,mc) y construir YOLOv11
#  - Verificar coherencia de nc con dataset
#  - Crear carpetas de trabajo (runs, logs, metrics, weights)
#==============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import torch

# Dependencias internas del proyecto
from .yolo11 import YOLOv11, build_model, VARIANTS as VARIANTS_FALLBACK


# ------------------------------
# Utilidades básicas
# ------------------------------
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML no encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_path(base: Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p).resolve()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _auto_device() -> str:
    # En ROCm para Windows, torch expone 'cuda' cuando hay GPU AMD disponible.
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # macOS
        return "mps"
    return "cpu"


# ------------------------------
# Data classes de salida
# ------------------------------
@dataclass
class PathsCfg:
    project_root: Path
    configs_dir: Path
    dataset_yaml: Path
    yolo_yaml: Path
    variants_yaml: Optional[Path]
    train_yaml: Path
    runs_dir: Path
    logs_dir: Path
    metrics_dir: Path
    weights_dir: Path


@dataclass
class RuntimeCfg:
    device: str
    seed: int
    deterministic: bool
    compile: bool
    cudnn_benchmark: bool


@dataclass
class SavePolicy:
    save_best: bool
    save_last: bool
    save_period: int
    keep_checkpoint_max: int


@dataclass
class ModelMeta:
    nc: int
    in_channels: int
    reg_max: int
    use_dw_for_cls: bool
    strides: Tuple[int, int, int]


@dataclass
class ResolvedVariant:
    name: str
    d: float
    w: float
    mc: int


# ------------------------------
# Parser principal
# ------------------------------
class ConfigParserYaml:
    """
    Parser centralizado para los YAML de configuración del proyecto YOLOv11.
    """

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        parser_yaml_path: Optional[str | Path] = None,
    ) -> None:
        # Resolver raíz del proyecto:
        # 1) argumento explícito
        # 2) variable env YOLOV11_ROOT
        # 3) carpeta dos niveles arriba de este archivo (…/YOLOv11)
        default_root = Path(__file__).resolve().parents[1]
        env_root = Path(Path.home() / "YOLOV11_ROOT") if False else None  # placeholder (no usar .env aquí)
        root = Path(project_root) if project_root else default_root
        self.root = root

        # Ubicación por defecto de parser.yaml
        self.parser_yaml = (
            Path(parser_yaml_path) if parser_yaml_path
            else (self.root / "configs" / "parser.yaml")
        ).resolve()

        self._loaded = False

        # Contenedores
        self.paths: PathsCfg | None = None
        self.runtime: RuntimeCfg | None = None
        self.save: SavePolicy | None = None
        self.train_cfg: Dict[str, Any] = {}
        self.model_meta: ModelMeta | None = None
        self.variants_map: Dict[str, Dict[str, Any]] = {}
        self.default_variant_name: str = "m"

    # -------- Carga y resolución ----------
    def load(self) -> "ConfigParserYaml":
        p = _read_yaml(self.parser_yaml)

        # ----- Rutas y punteros a YAML -----
        configs_dir = (self.root / "configs").resolve()
        data_cfg_rel = p.get("data", {}).get("config", "configs/dataset.yaml")
        yolo_cfg_rel = p.get("model", {}).get("yolo_cfg", "configs/yolo11.yaml")
        var_cfg_rel = p.get("model", {}).get("variants_cfg", "configs/model_variants.yaml")
        train_cfg_rel = p.get("train", {}).get("config", "configs/train.yaml")

        runs_rel = p.get("project", {}).get("dirs", {}).get("runs", "runs")
        logs_rel = p.get("project", {}).get("dirs", {}).get("logs", "logs")
        metrics_rel = p.get("project", {}).get("dirs", {}).get("metrics", "metrics")
        weights_rel = p.get("project", {}).get("dirs", {}).get("weights", "weights")

        self.paths = PathsCfg(
            project_root=self.root,
            configs_dir=configs_dir,
            dataset_yaml=_to_path(self.root, data_cfg_rel),
            yolo_yaml=_to_path(self.root, yolo_cfg_rel),
            variants_yaml=_to_path(self.root, var_cfg_rel) if var_cfg_rel else None,
            train_yaml=_to_path(self.root, train_cfg_rel),
            runs_dir=_to_path(self.root, runs_rel),
            logs_dir=_to_path(self.root, logs_rel),
            metrics_dir=_to_path(self.root, metrics_rel),
            weights_dir=_to_path(self.root, weights_rel),
        )

        # ----- Runtime / Save -----
        runtime = p.get("runtime", {})
        device = runtime.get("device", "auto")
        if device == "auto":
            device = _auto_device()

        self.runtime = RuntimeCfg(
            device=device,
            seed=int(runtime.get("seed", 42)),
            deterministic=bool(runtime.get("deterministic", False)),
            compile=bool(runtime.get("compile", False)),
            cudnn_benchmark=bool(runtime.get("cudnn_benchmark", True)),
        )

        save = p.get("save", {})
        self.save = SavePolicy(
            save_best=bool(save.get("save_best", True)),
            save_last=bool(save.get("save_last", True)),
            save_period=int(save.get("save_period", 10)),
            keep_checkpoint_max=int(save.get("keep_checkpoint_max", 5)),
        )

        # ----- Train -----
        self.train_cfg = _read_yaml(self.paths.train_yaml)

        # ----- Modelo (yolo11.yaml) -----
        yolo = _read_yaml(self.paths.yolo_yaml).get("model", {})
        self.model_meta = ModelMeta(
            nc=int(yolo.get("nc", 5)),
            in_channels=int(yolo.get("in_channels", 3)),
            reg_max=int(yolo.get("reg_max", 16)),
            use_dw_for_cls=bool(yolo.get("use_dw_for_cls", True)),
            strides=tuple(yolo.get("strides", [8, 16, 32])),
        )

        # ----- Variantes -----
        self.default_variant_name = p.get("model", {}).get("default_variant", "m")

        # Si existe model_variants.yaml úsalo; si no, usa fallback del código
        self.variants_map = VARIANTS_FALLBACK.copy()
        try:
            if self.paths.variants_yaml and self.paths.variants_yaml.exists():
                loaded = _read_yaml(self.paths.variants_yaml) or {}
                # Se espera formato: {n:{depth_multiple, width_multiple, max_channels}, ...}
                # Normalizamos llaves a d,w,mc
                normalized = {}
                for k, v in (loaded or {}).items():
                    if not isinstance(v, dict):
                        continue
                    d = float(v.get("depth_multiple", v.get("d", 1.0)))
                    w = float(v.get("width_multiple", v.get("w", 1.0)))
                    mc = int(v.get("max_channels", v.get("mc", 1024)))
                    normalized[str(k)] = {"d": d, "w": w, "mc": mc}
                if normalized:
                    self.variants_map = normalized
        except Exception:
            # Silencioso: si algo falla, mantenemos el fallback del código
            pass

        # ----- Dataset (opcional) para validaciones -----
        try:
            ds = _read_yaml(self.paths.dataset_yaml)
            ds_nc = int(ds.get("nc", self.model_meta.nc))
            names = ds.get("names", {})
            if isinstance(names, dict):
                ds_nc_from_names = len(names)
                if ds_nc_from_names != ds_nc:
                    print(f"[parser_yaml] Aviso: dataset.yaml nc={ds_nc} pero names tiene {ds_nc_from_names} clases.")
            if ds_nc != self.model_meta.nc:
                print(f"[parser_yaml] Aviso: nc de modelo ({self.model_meta.nc}) difiere de dataset.nc ({ds_nc}).")
        except Exception:
            # dataset.yaml no es estrictamente necesario para construir el modelo
            pass

        # ----- Crear carpetas de trabajo -----
        for d in [self.paths.runs_dir, self.paths.logs_dir, self.paths.metrics_dir, self.paths.weights_dir]:
            _ensure_dir(d)

        self._loaded = True
        return self

    # -------- Resolución de variante ----------
    def resolve_variant(self, *, variant: Optional[str] = None,
                        d: Optional[float] = None, w: Optional[float] = None, mc: Optional[int] = None) -> ResolvedVariant:
        if not self._loaded:
            self.load()

        if all(v is not None for v in (d, w, mc)):
            return ResolvedVariant(name=variant or "custom", d=float(d), w=float(w), mc=int(mc))

        vname = (variant or self.default_variant_name)
        if vname not in self.variants_map:
            raise KeyError(f"Variante '{vname}' no encontrada. Disponibles: {list(self.variants_map.keys())}")
        vd = self.variants_map[vname]
        return ResolvedVariant(name=vname, d=float(vd["d"]), w=float(vd["w"]), mc=int(vd["mc"]))

    # -------- Construcción del modelo ----------
    def build_model(self, *, variant: Optional[str] = None,
                    d: Optional[float] = None, w: Optional[float] = None, mc: Optional[int] = None,
                    imgsz_for_strides: int = 640) -> YOLOv11:
        if not self._loaded:
            self.load()

        rv = self.resolve_variant(variant=variant, d=d, w=w, mc=mc)
        m = build_model(
            variant=rv.name,
            nc=self.model_meta.nc,
            d=rv.d, w=rv.w, mc=rv.mc,
            in_ch=self.model_meta.in_channels,
            reg_max=self.model_meta.reg_max,
            imgsz_for_strides=imgsz_for_strides,
        )
        return m

    # -------- Resumen ----------
    def summary(self) -> str:
        if not self._loaded:
            self.load()
        assert self.paths and self.runtime and self.save and self.model_meta
        s = []
        s.append("=== ConfigParserYaml Summary ===")
        s.append(f"Project root : {self.paths.project_root}")
        s.append(f"Configs dir  : {self.paths.configs_dir}")
        s.append(f"YAML parser  : {self.parser_yaml}")
        s.append(f"dataset.yaml : {self.paths.dataset_yaml}")
        s.append(f"yolo11.yaml  : {self.paths.yolo_yaml}")
        s.append(f"variants.yaml: {self.paths.variants_yaml} (n={len(self.variants_map)})")
        s.append(f"train.yaml   : {self.paths.train_yaml}")
        s.append(f"runs/logs    : {self.paths.runs_dir} | {self.paths.logs_dir}")
        s.append(f"metrics/weights: {self.paths.metrics_dir} | {self.paths.weights_dir}")
        s.append("-- Runtime --")
        s.append(f"device={self.runtime.device}, seed={self.runtime.seed}, deterministic={self.runtime.deterministic}, "
                 f"compile={self.runtime.compile}, cudnn_benchmark={self.runtime.cudnn_benchmark}")
        s.append("-- Model --")
        s.append(f"nc={self.model_meta.nc}, in_channels={self.model_meta.in_channels}, reg_max={self.model_meta.reg_max}, "
                 f"use_dw_for_cls={self.model_meta.use_dw_for_cls}, strides={self.model_meta.strides}")
        s.append("-- Variants --")
        s.append(", ".join([f"{k}(d={v['d']},w={v['w']},mc={v['mc']})" for k, v in self.variants_map.items()]))
        return "\n".join(s)


# ------------------------------
# Ejecución directa (prueba)
# ------------------------------
if __name__ == "__main__":
    # Ajusta la raíz explícita si ejecutas fuera del repositorio
    # root = Path(r"C:\Users\memorista\Desktop\Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection\YOLOv11")
    root = None  # usa detección automática (dos niveles arriba de este archivo)

    parser = ConfigParserYaml(project_root=root).load()
    print(parser.summary())

    # Construir el modelo con la variante por defecto de parser.yaml
    model = parser.build_model()
    print(f"\nModelo construido (variant={parser.default_variant_name}):")
    print("Strides:", model.strides.tolist())

    # Forward seco para validar shapes
    x = torch.zeros(1, parser.model_meta.in_channels, 640, 640)
    out = model(x)
    for i, (c, r) in enumerate(zip(out["cls"], out["reg"])):
        print(f"P{i+3} -> cls={tuple(c.shape)}, reg={tuple(r.shape)}")
