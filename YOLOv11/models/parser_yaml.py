# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: parser_yaml.py
# Carga y orquesta las configuraciones YAML del proyecto:
#  - configs/parser.yaml          (rutas, runtime, save-policy, punteros a otros YAML)
#  - configs/train.yaml           (hiperparámetros de entrenamiento)
#  - configs/yolo11.yaml          (parámetros del modelo)
#  - configs/model_variants.yaml  (escala por variante d,w,mc) [opcional]
#  - configs/dataset.yaml         (rutas dataset, names, nc)    [opcional]
# Expone: ConfigParserYaml con utilidades para:
#  - Resolver variante (d,w,mc) y construir YOLOv11
#  - Verificar coherencia de nc con dataset
#  - Crear carpetas de trabajo (runs, logs, metrics, weights)
#  - TEST integrado: selectivo y detallado (ver sección CLI al final)
#==============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Iterable

import sys
import yaml
import json
import torch

# --------------------------------------------------------------
# Import del modelo con fallback seguro
#   - Como paquete: `python -m YOLOv11.models.parser_yaml`
#   - Como script : `python YOLOv11/models/parser_yaml.py`
# --------------------------------------------------------------
try:  # ejecución como paquete (import relativo)
    from .yolo11 import YOLOv11, build_model, VARIANTS as VARIANTS_FALLBACK
except Exception:  # ejecución directa como script (fallback)
    _ROOT = Path(__file__).resolve().parents[1]  # .../YOLOv11
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from models.yolo11 import YOLOv11, build_model, VARIANTS as VARIANTS_FALLBACK  # type: ignore


# ==============================
# Utilidades básicas
# ==============================

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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS
        return "mps"
    return "cpu"


def _yaml_dump(d: Dict[str, Any]) -> str:
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, indent=2)


def _print_title(t: str) -> None:
    bar = "=" * len(t)
    print(f"{t}{bar}")


def _kv(rows: Dict[str, Any]) -> str:
    return "".join([f"  - {k}: {v}" for k, v in rows.items()])


def _alias(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return default


# ==============================
# Data classes de salida
# ==============================

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


# ==============================
# Parser principal
# ==============================
class ConfigParserYaml:
    """Parser centralizado para los YAML de configuración del proyecto YOLOv11."""

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        parser_yaml_path: Optional[str | Path] = None,
    ) -> None:
        # Resolver raíz del proyecto:
        # 1) argumento explícito  2) carpeta dos niveles arriba de este archivo (…/YOLOv11)
        default_root = Path(__file__).resolve().parents[1]
        root = Path(project_root) if project_root else default_root
        self.root = root

        # Ubicación por defecto de parser.yaml
        self.parser_yaml = (
            Path(parser_yaml_path) if parser_yaml_path else (self.root / "configs" / "parser.yaml")
        ).resolve()

        self._loaded = False

        # Contenedores
        self.paths: PathsCfg | None = None
        self.runtime: RuntimeCfg | None = None
        self.save: SavePolicy | None = None
        self.train_cfg: Dict[str, Any] = {}
        self.model_meta: ModelMeta | None = None
        self.variants_map: Dict[str, Dict[str, Any]] = {}
        self.default_variant_name: str = "n"

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
        raw_train = _read_yaml(self.paths.train_yaml) or {}
        if isinstance(raw_train, dict):
            # Construir subdict de dataloader si no existe
            if "dataloader" not in raw_train or not isinstance(raw_train.get("dataloader"), dict):
                dl_keys = ("workers", "pin_memory", "persistent_workers", "shuffle")
                dl = {k: raw_train[k] for k in dl_keys if k in raw_train}
                if dl:
                    raw_train["dataloader"] = dl
            # Alias para compatibilidad con versiones previas
            if "loss" in raw_train and "loss weights" not in raw_train:
                raw_train["loss weights"] = raw_train.get("loss", {})
            # Alias val_interval desde val_period (YAML) si aplica
            if "val_period" in raw_train and "val_interval" not in raw_train:
                raw_train["val_interval"] = raw_train.get("val_period")
        self.train_cfg = {"config": raw_train, "normalized": self._normalize_train(raw_train)}

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
        # 1) Fallback por defecto definido en código
        self.variants_map = {k: {"d": v.get("d"), "w": v.get("w"), "mc": v.get("mc")} for k, v in VARIANTS_FALLBACK.items()}

        # 2) Si existe model_variants.yaml, úsalo (aplanando raíz 'variants' si aparece)
        try:
            if self.paths.variants_yaml and self.paths.variants_yaml.exists():
                raw = _read_yaml(self.paths.variants_yaml) or {}
                if isinstance(raw, dict):
                    vroot = raw.get("variants", raw)
                    if isinstance(vroot, dict):
                        normalized: Dict[str, Dict[str, Any]] = {}
                        for k, v in vroot.items():
                            if not isinstance(v, dict):
                                continue
                            d = float(v.get("depth_multiple", v.get("d", 1.0)))
                            w = float(v.get("width_multiple", v.get("w", 1.0)))
                            mc = int(v.get("max_channels", v.get("mc", 1024)))
                            normalized[str(k).lower()] = {"d": d, "w": w, "mc": mc}
                        if normalized:
                            self.variants_map = normalized
        except Exception:
            # Silencioso: si algo falla, se mantiene el fallback del código
            pass

        # 3) Variante por defecto desde parser.yaml (lower-case y validación)
        self.default_variant_name = str(p.get("model", {}).get("default_variant", "m")).lower()
        if self.default_variant_name not in self.variants_map:
            self.default_variant_name = "m" if "m" in self.variants_map else next(iter(self.variants_map.keys()))

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

    # -------- Normalización de train.yaml ----------
    def _normalize_train(self, t: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(t, dict):
            return {}
        # Aliases y defaults
        conf_thr = _alias(t, "conf_thr", "conf_thres", default=None)
        iou_thr = _alias(t, "iou_thr", "iou_thres", default=None)
        warmup = _alias(t, "warmup_steps", "warmup_epochs", default=None)
        loss = t.get("loss", {}) if isinstance(t.get("loss"), dict) else {}
        dl = t.get("dataloader", {}) if isinstance(t.get("dataloader"), dict) else {}
        aug_keys = [
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
            "fliplr", "flipud", "mosaic", "mixup", "copy_paste", "erasing",
        ]
        aug = {k: t.get(k) for k in aug_keys if k in t}
        save_blk = t.get("save", {}) if isinstance(t.get("save"), dict) else {}
        norm = {
            "data": {"imgsz": t.get("imgsz"), "rect": t.get("rect"), "cache": t.get("cache")},
            "optim": {
                "optimizer": t.get("optimizer"), "lr0": t.get("lr0"), "lrf": t.get("lrf"), "weight_decay": t.get("weight_decay"),
                "momentum": t.get("momentum"), "betas": t.get("betas"), "scheduler": t.get("scheduler"),
                "epochs": t.get("epochs"), "batch": t.get("batch"), "warmup": warmup
            },
            "dataloader": {k: dl.get(k) for k in ("workers", "pin_memory", "persistent_workers", "shuffle")},
            "train": {
                "amp": t.get("amp"), "ema": t.get("ema"), "ema_decay": t.get("ema_decay"), "grad_accum": t.get("grad_accum"),
                "max_grad_norm": t.get("max_grad_norm"), "label_smoothing": t.get("label_smoothing"), "patience": t.get("patience"),
                "val_interval": t.get("val_interval"), "overlay_every": t.get("overlay_every"), "pr_curves_every": t.get("pr_curves_every"),
                "cm_every": t.get("cm_every"), "verbosity": t.get("verbosity"), "hud": t.get("hud"), "bn_eval_fallback": t.get("bn_eval_fallback")
            },
            "loss": {"box": loss.get("box"), "cls": loss.get("cls"), "dfl": loss.get("dfl")},
            "augment": aug,
            "eval": {"conf_thr": conf_thr, "iou_thr": iou_thr, "plots": t.get("plots")},
            "logging": {"tensorboard": t.get("tensorboard"), "wandb": t.get("wandb")},
            "save": {
                "save_best": save_blk.get("save_best"), "save_last": save_blk.get("save_last"),
                "save_period": save_blk.get("save_period"), "keep_checkpoint_max": save_blk.get("keep_checkpoint_max")
            },
        }
        return norm

    # -------- Propiedades de conveniencia ----------
    @property
    def default_variant(self) -> str:
        return getattr(self, "default_variant_name", "n")

    @property
    def train(self) -> Dict[str, Any]:  # compat con utilidades externas
        return self.train_cfg

    # -------- Resolución de variante ----------
    def resolve_variant(
        self,
        *,
        variant: Optional[str] = None,
        d: Optional[float] = None,
        w: Optional[float] = None,
        mc: Optional[int] = None,
    ) -> ResolvedVariant:
        if not self._loaded:
            self.load()

        if all(v is not None for v in (d, w, mc)):
            return ResolvedVariant(name=(variant or "custom").lower(), d=float(d), w=float(w), mc=int(mc))

        vname = (variant or self.default_variant_name or "m").lower()

        vmap = self.variants_map
        if "variants" in vmap and isinstance(vmap["variants"], dict):
            vmap = vmap["variants"]  # aplanar si quedó anidado por error
            self.variants_map = vmap  # type: ignore[assignment]

        if vname not in vmap:
            raise KeyError(f"Variante '{vname}' no encontrada. Disponibles: {list(vmap.keys())}")

        vd = vmap[vname]
        return ResolvedVariant(name=vname, d=float(vd["d"]), w=float(vd["w"]), mc=int(vd["mc"]))

    # -------- Construcción del modelo ----------
    def build_model(
        self,
        *,
        variant: Optional[str] = None,
        d: Optional[float] = None,
        w: Optional[float] = None,
        mc: Optional[int] = None,
        imgsz_for_strides: int = 640,
    ) -> YOLOv11:
        if not self._loaded:
            self.load()

        rv = self.resolve_variant(variant=variant, d=d, w=w, mc=mc)
        m = build_model(
            variant=rv.name,
            nc=self.model_meta.nc,
            d=rv.d,
            w=rv.w,
            mc=rv.mc,
            in_ch=self.model_meta.in_channels,
            reg_max=self.model_meta.reg_max,
            imgsz_for_strides=imgsz_for_strides,
        )

        # Actualiza metadatos con strides reales del modelo, si existen
        strides = tuple(getattr(m, "strides", (8, 16, 32)))
        self.model_meta = ModelMeta(
            nc=self.model_meta.nc,
            in_channels=self.model_meta.in_channels,
            reg_max=self.model_meta.reg_max,
            use_dw_for_cls=self.model_meta.use_dw_for_cls,
            strides=strides,
        )
        return m

    # -------- Resumen ----------
    def summary(self) -> str:
        if not self._loaded:
            self.load()
        assert self.paths and self.runtime and self.save and self.model_meta
        s: List[str] = []
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
        s.append(
            f"device={self.runtime.device}, seed={self.runtime.seed}, deterministic={self.runtime.deterministic}, "
            f"compile={self.runtime.compile}, cudnn_benchmark={self.runtime.cudnn_benchmark}"
        )
        s.append("-- Model --")
        s.append(
            f"nc={self.model_meta.nc}, in_channels={self.model_meta.in_channels}, reg_max={self.model_meta.reg_max}, "
            f"use_dw_for_cls={self.model_meta.use_dw_for_cls}, strides={self.model_meta.strides}"
        )
        s.append("-- Variants --")
        s.append(
            ", ".join([f"{k}(d={v['d']},w={v['w']},mc={v['mc']})" for k, v in self.variants_map.items()])
        )
        return "".join(s)


# ==============================
# Utilidades de test / reporte
# ==============================

def _summarize_yaml_keys(d: Dict[str, Any]) -> str:
    if not d:
        return "<vacío>"
    keys = list(d.keys())
    return ", ".join(keys[:12]) + (" …" if len(keys) > 12 else "")


def _report_parser_yaml(cfg: ConfigParserYaml, verbosity: str) -> None:
    p_yaml = _read_yaml(cfg.parser_yaml)
    _print_title("parser.yaml")
    print(f"  - path: {cfg.parser_yaml}")
    if verbosity == "full":
        print(_yaml_dump(p_yaml))
        return
    rows = {
        "model.yolo_cfg": p_yaml.get("model", {}).get("yolo_cfg"),
        "model.variants_cfg": p_yaml.get("model", {}).get("variants_cfg"),
        "model.default_variant": p_yaml.get("model", {}).get("default_variant"),
        "runtime": p_yaml.get("runtime", {}),
        "project.dirs": p_yaml.get("project", {}).get("dirs"),
        "save": p_yaml.get("save", {}),
    }
    print(_kv(rows))


def _report_train_yaml(cfg: ConfigParserYaml, verbosity: str) -> None:
    _print_title("train.yaml")
    print(f"  - path: {cfg.paths.train_yaml}")
    t_yaml = _read_yaml(cfg.paths.train_yaml)
    if verbosity == "full":
        print(_yaml_dump(t_yaml))
        return
    if verbosity == "detailed":
        n = cfg.train_cfg.get("normalized", {})
        print("[Datos]")
        print(_kv(n.get("data", {})))
        print("[Optimizador y LR]")
        print(_kv(n.get("optim", {})))
        print("[Dataloader]")
        print(_kv(n.get("dataloader", {})))
        print("[Entrenamiento]")
        print(_kv(n.get("train", {})))
        print("[Pérdidas]")
        print(_kv(n.get("loss", {})))
        print("[Aumentación]")
        print(_kv(n.get("augment", {})))
        print("[Evaluación]")
        print(_kv(n.get("eval", {})))
        print("[Guardado/Monitoreo]")
        print(_kv(n.get("logging", {})))
        print(_kv({"save": n.get("save", {})}))
        return
    # summary clásico
    rows = {
        "keys": _summarize_yaml_keys(t_yaml),
        "imgsz": t_yaml.get("imgsz"),
        "epochs": t_yaml.get("epochs"),
        "batch": t_yaml.get("batch"),
        "optimizer": t_yaml.get("optimizer"),
        "scheduler": t_yaml.get("scheduler"),
        "dataloader": t_yaml.get("dataloader", {}),
    }
    print(_kv(rows))


def _report_yolo_yaml(cfg: ConfigParserYaml, verbosity: str) -> None:
    _print_title("yolo11.yaml")
    print(f"  - path: {cfg.paths.yolo_yaml}")
    y_yaml = _read_yaml(cfg.paths.yolo_yaml)
    if verbosity == "full":
        print(_yaml_dump(y_yaml))
        return
    model = y_yaml.get("model", {})
    rows = {
        "nc": model.get("nc"),
        "in_channels": model.get("in_channels"),
        "reg_max": model.get("reg_max"),
        "strides": model.get("strides"),
        "use_dw_for_cls": model.get("use_dw_for_cls"),
        "backbone.keys": _summarize_yaml_keys(y_yaml.get("backbone", {})),
        "neck.keys": _summarize_yaml_keys(y_yaml.get("neck", {})),
        "head.keys": _summarize_yaml_keys(y_yaml.get("head", {})),
    }
    print(_kv(rows))


def _report_dataset_yaml(cfg: ConfigParserYaml, verbosity: str) -> None:
    _print_title("dataset.yaml")
    print(f"  - path: {cfg.paths.dataset_yaml}")
    try:
        d_yaml = _read_yaml(cfg.paths.dataset_yaml)
    except Exception as e:
        print(f"  - No disponible: {e}")
        return
    if verbosity == "full":
        print(_yaml_dump(d_yaml))
        return
    names = d_yaml.get("names", {})
    extra_keys = {k: v for k, v in d_yaml.items() if k not in {"path", "train", "val", "test", "nc", "names"}}
    rows = {
        "path": d_yaml.get("path"),
        "train": d_yaml.get("train"),
        "val": d_yaml.get("val"),
        "test": d_yaml.get("test"),
        "nc": d_yaml.get("nc"),
        "names(len)": len(names) if isinstance(names, dict) else None,
        "classes": ", ".join([f"{k}:{v}" for k, v in list(names.items())[:10]]) + (" …" if isinstance(names, dict) and len(names) > 10 else ""),
        "extras": extra_keys,
    }
    print(_kv(rows))


def _report_variants_yaml(cfg: ConfigParserYaml, verbosity: str, *, max_items: int = 8) -> None:
    _print_title("model_variants.yaml")
    print(f"  - path: {cfg.paths.variants_yaml}")
    if cfg.paths.variants_yaml and cfg.paths.variants_yaml.exists():
        v_yaml = _read_yaml(cfg.paths.variants_yaml)
        if verbosity == "full":
            print(_yaml_dump(v_yaml))
            return
        vroot = v_yaml.get("variants", v_yaml)
        try:
            n = len(vroot)
        except Exception:
            n = 0
        print(f"  - variants definidos: {n}")
        shown = 0
        for k, v in vroot.items():
            if not isinstance(v, dict):
                continue
            d = v.get("depth_multiple", v.get("d"))
            w = v.get("width_multiple", v.get("w"))
            mc = v.get("max_channels", v.get("mc"))
            print(f"    * {k}: d={d}, w={w}, mc={mc}")
            shown += 1
            if shown >= max_items:
                break
    else:
        print("  - No encontrado (se usa fallback interno del código).")


def quick_configs_report(cfg: ConfigParserYaml, tests: Iterable[str], *, verbosity: str = "detailed", max_variants: int = 8) -> None:
    """Imprime un reporte selectivo leyendo las configuraciones solicitadas.

    tests: subconjunto de {parser, train, yolo, dataset, variants, all}
    verbosity: "summary" | "detailed" | "full"
    """
    tests = set([t.lower() for t in tests])
    if "all" in tests:
        tests = {"parser", "train", "yolo", "dataset", "variants"}

    if "parser" in tests:
        _report_parser_yaml(cfg, verbosity)
    if "train" in tests:
        _report_train_yaml(cfg, verbosity)
    if "yolo" in tests:
        _report_yolo_yaml(cfg, verbosity)
    if "dataset" in tests:
        _report_dataset_yaml(cfg, verbosity)
    if "variants" in tests:
        _report_variants_yaml(cfg, verbosity, max_items=max_variants)

    

def collect_snapshot(cfg: ConfigParserYaml, tests: Iterable[str]) -> Dict[str, Any]:
    tests = set([t.lower() for t in tests])
    snap: Dict[str, Any] = {
        "paths": {
            "project_root": str(cfg.paths.project_root), "configs_dir": str(cfg.paths.configs_dir),
            "dataset_yaml": str(cfg.paths.dataset_yaml), "yolo_yaml": str(cfg.paths.yolo_yaml),
            "variants_yaml": str(cfg.paths.variants_yaml) if cfg.paths.variants_yaml else None,
            "train_yaml": str(cfg.paths.train_yaml),
        },
        "runtime": cfg.runtime.__dict__,
        "save": cfg.save.__dict__,
        "model_meta": cfg.model_meta.__dict__,
        "default_variant": cfg.default_variant_name,
    }
    if "train" in tests or "all" in tests:
        snap["train"] = {
            "raw": _read_yaml(cfg.paths.train_yaml),
            "normalized": cfg.train_cfg.get("normalized", {}),
        }
    if "parser" in tests or "all" in tests:
        snap["parser_yaml"] = _read_yaml(cfg.parser_yaml)
    if "yolo" in tests or "all" in tests:
        snap["yolo11_yaml"] = _read_yaml(cfg.paths.yolo_yaml)
    if "dataset" in tests or "all" in tests:
        try:
            snap["dataset_yaml"] = _read_yaml(cfg.paths.dataset_yaml)
        except Exception as e:
            snap["dataset_yaml_error"] = str(e)
    if "variants" in tests or "all" in tests:
        if cfg.paths.variants_yaml and cfg.paths.variants_yaml.exists():
            snap["variants_yaml"] = _read_yaml(cfg.paths.variants_yaml)
        else:
            snap["variants_yaml"] = {"fallback_from_code": cfg.variants_map}
    return snap


# ==============================
# Ejecución directa (CLI de test)
# ==============================
if __name__ == "__main__":
    import argparse

    parser_cli = argparse.ArgumentParser(
        prog="parser_yaml.py",
        description=(
            "Resumen y test selectivo de YAMLs del proyecto YOLOv11."
            "- Por defecto imprime un resumen detallado."
            "- Con --test seleccionas qué YAMLs leer (parser/train/yolo/dataset/variants/all)."
            "- Usa --verbosity full para hacer dump completo del YAML seleccionado."
        ),
    )
    parser_cli.add_argument("--project-root", type=str, default=None, help="Raíz del proyecto (si no se detecta automáticamente).")

    # Selección de tests (admite múltiples)
    parser_cli.add_argument("--test", nargs="+", default=None, choices=["parser", "train", "yolo", "dataset", "variants", "all"], help="Selecciona YAMLs a validar (puede ser múltiple). Ej.: --test train yolo")

    # Atajos equivalentes a --test ... (incluye alias como pidió el usuario)
    parser_cli.add_argument("--test-configs", action="store_true", help="Atajo a --test all (summary).")
    parser_cli.add_argument("--tc-parser", "--test-configs-parser", action="store_true", help="Atajo a --test parser")
    parser_cli.add_argument("--tc-train", "--test-configs-train", "--test-configs-train.yaml", action="store_true", help="Atajo a --test train (alias incluye 'train.yaml')")
    parser_cli.add_argument("--tc-yolo", "--test-configs-yolo", "--test-configs-yolo11.yaml", action="store_true", help="Atajo a --test yolo (alias incluye 'yolo11.yaml')")
    parser_cli.add_argument("--tc-dataset", "--test-configs-dataset", "--test-configs-dataset.yaml", action="store_true", help="Atajo a --test dataset (alias incluye 'dataset.yaml')")
    parser_cli.add_argument("--tc-variants", "--test-configs-variants", "--test-configs-model_variants.yaml", action="store_true", help="Atajo a --test variants (alias incluye 'model_variants.yaml')")

    # Opciones de salida
    parser_cli.add_argument("--verbosity", choices=["summary", "detailed", "full"], default="detailed", help="Nivel de detalle del reporte.")
    parser_cli.add_argument("--max-variants", type=int, default=8, help="Máximo de variantes a mostrar en modo resumen/detallado.")

    # Exportar snapshot JSON
    parser_cli.add_argument("--export-json", type=str, default=None, help="Ruta para exportar snapshot JSON de las configs seleccionadas (use '-' para stdout).")

    # Estricto / construcción de modelo
    parser_cli.add_argument("--strict", action="store_true", help="Falla si hay inconsistencias detectables.")
    parser_cli.add_argument("--no-model", action="store_true", help="No construir el modelo (solo lectura de YAMLs).")
    parser_cli.add_argument("--imgsz", type=int, default=640, help="Tamaño usado para calcular strides cuando se construye el modelo.")

    args = parser_cli.parse_args()

    cfg = ConfigParserYaml(project_root=args.project_root).load()
    # 1) Resumen breve del parser (desactivado a petición del usuario)
    # print(cfg.summary())

    # 2) Resolver conjunto de tests
    selected: List[str] = []
    if args.test_configs or (args.test and "all" in args.test):
        selected = ["all"]
    else:
        if args.test:
            selected.extend(args.test)
        if args.tc_parser:
            selected.append("parser")
        if args.tc_train:
            selected.append("train")
        if args.tc_yolo:
            selected.append("yolo")
        if args.tc_dataset:
            selected.append("dataset")
        if args.tc_variants:
            selected.append("variants")

    # 3) Test selectivo de YAMLs (lectura + validaciones ligeras)
    if selected:
        quick_configs_report(cfg, tests=selected, verbosity=args.verbosity, max_variants=args.max_variants)

    # 4) Export opcional a JSON (snapshot)
    if args.export_json:
        tests_for_snap = selected if selected else ["all"]
        snap = collect_snapshot(cfg, tests_for_snap)
        if args.export_json.strip() == "-":
            print(json.dumps(snap, ensure_ascii=False, indent=2))
        else:
            outp = Path(args.export_json).resolve()
            with outp.open("w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            print(f"[OK] Snapshot JSON exportado en: {outp}")

    # 5) Construcción opcional del modelo y sanity forward
    if not args.no_model:
        model = cfg.build_model(variant=cfg.default_variant_name, imgsz_for_strides=args.imgsz)
        _print_title(f"Modelo construido (variant={cfg.default_variant_name})")
        try:
            print("  - Strides:", getattr(model, "strides", [8, 16, 32]))
        except Exception:
            pass
        x = torch.zeros(1, cfg.model_meta.in_channels, args.imgsz, args.imgsz)
        out = model(x)
        for i, (c, r) in enumerate(zip(out["cls"], out["reg"])):
            print(f"  - P{i+3} -> cls={tuple(c.shape)}, reg={tuple(r.shape)}")

    # 6) Validación estricta (si se solicita)
    if args.strict:
        errors: List[str] = []
        # dataset vs model nc
        try:
            dy = _read_yaml(cfg.paths.dataset_yaml)
            ds_nc = int(dy.get("nc", cfg.model_meta.nc))
            names = dy.get("names", {})
            if isinstance(names, dict) and len(names) != ds_nc:
                errors.append(f"names (len={len(names)}) difiere de nc={ds_nc} en dataset.yaml")
            if ds_nc != cfg.model_meta.nc:
                errors.append(f"dataset.nc={ds_nc} difiere de model.nc={cfg.model_meta.nc} (yolo11.yaml)")
        except Exception as e:
            errors.append(f"dataset.yaml no accesible: {e}")
        # existencia de archivos clave
        for pth, tag in [
            (cfg.parser_yaml, "parser.yaml"),
            (cfg.paths.train_yaml, "train.yaml"),
            (cfg.paths.yolo_yaml, "yolo11.yaml"),
        ]:
            if not Path(pth).exists():
                errors.append(f"Falta {tag}: {pth}")
        # variantes opcional
        if cfg.paths.variants_yaml and not cfg.paths.variants_yaml.exists():
            errors.append(f"Ruta variants inexistente: {cfg.paths.variants_yaml}")
        if errors:
            _print_title("STRICT: inconsistencias")
            for e in errors:
                print("  -", e)
            sys.exit(2)

    # --------------------------------------------------------------
    # Ejemplos de uso (PowerShell)
    # --------------------------------------------------------------
    # [A] LECTURA ÍNTEGRA (full dump de cada YAML, sin construir modelo)
    #   python YOLOv11/models/parser_yaml.py --tc-parser   --verbosity full --no-model
    #   python YOLOv11/models/parser_yaml.py --tc-train    --verbosity full --no-model
    #   python YOLOv11/models/parser_yaml.py --tc-yolo     --verbosity full --no-model
    #   python YOLOv11/models/parser_yaml.py --tc-dataset  --verbosity full --no-model
    #   python YOLOv11/models/parser_yaml.py --tc-variants --verbosity full --no-model
    #
    # [B] SUMARIOS (detallados por defecto; usa --verbosity summary para compactar)
    #   python YOLOv11/models/parser_yaml.py --tc-train              # detallado (secciones data/optim/dataloader/...)
    #   python YOLOv11/models/parser_yaml.py --tc-dataset            # detallado (paths, nc, names, extras)
    #   python YOLOv11/models/parser_yaml.py --test yolo train       # dos sumarios a la vez
    #   python YOLOv11/models/parser_yaml.py --tc-yolo --verbosity summary
    #   python YOLOv11/models/parser_yaml.py --test parser dataset --verbosity summary
    #
    # [C] TESTS / SANITY (opcionales; verificar recepción y coherencia básica)
    #   # Todo (detallado) sin construir modelo
    #   python YOLOv11/models/parser_yaml.py --test all --no-model
    #   # Modo estricto: falla (exit code 2) si hay inconsistencias de claves/nc
    #   python YOLOv11/models/parser_yaml.py --test all --strict --no-model
    #   # Exportar snapshot JSON consolidado para consumo externo
    #   python YOLOv11/models/parser_yaml.py --test parser train yolo dataset variants --export-json - --no-model
    #   # Limitar cuántas variantes se listan en resumen
    #   python YOLOv11/models/parser_yaml.py --tc-variants --verbosity summary --max-variants 5
    #==============================================================
