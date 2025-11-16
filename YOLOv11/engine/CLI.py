# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/engine/CLI.py
# Descripción: CLI profesional y modular para YOLOv11. Implementa una
#  clase constructora con atributos, presets documentados (--test-A, etc.) y
#  parseo en dos etapas con precedencia: CLI explícito > preset > YAML > defaults.
#  Integración torch‑free con importador ligero de configs de YOLOv11/configs/*.
#==============================================================

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # lectura YAML opcional; el CLI funciona con defaults seguros

__all__ = [
    # API pública (wrappers sobre la clase)
    "build_common_parser",
    "add_train_args",
    "merge_with_yaml",
    "apply_preset",
    "resolve_and_validate",
    "parse_args_two_stage",
]

# ==============================================================
# Importador seguro de configuraciones (sin torch)
# Lee configs/parser.yaml y configs/train.yaml para generar defaults tipados
# y SIN None, respetando el dominio esperado por train.py
# ==============================================================

@dataclass
class ConfigDefaultsLoader:
    yolo_root: Path

    # --- helpers ---
    def _read_yaml(self, p: Path) -> Dict[str, Any]:
        if yaml is None:
            return {}
        try:
            if not p.exists() or not p.is_file():
                return {}
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    @staticmethod
    def _get(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            if k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    @staticmethod
    def _b(x: Any, default: bool = False) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
            # valores tipo 'one'/'two' para HUD → True
            if s in {"one", "two"}:
                return True
        return default

    @staticmethod
    def _i(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _f(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _amp_from_yaml(v: Any) -> str:
        # YAML puede traer bool o string
        if isinstance(v, bool):
            return "auto" if v else "off"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"auto", "off", "fp16", "bf16"}:
                return s
            if s in {"true", "1", "on"}:
                return "auto"
            if s in {"false", "0", "off"}:
                return "off"
        return "auto"

    def _resolve_path(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        P = Path(p)
        if P.is_absolute():
            return str(P)
        return str((self.yolo_root / P).resolve())

    # --- carga principal ---
    def load(self, parser_yaml_path: Optional[str]) -> Dict[str, Any]:
        # Defaults base (nunca None)
        out: Dict[str, Any] = {
            # rutas
            "data": None, "model": None, "parser": None,
            # escala/modelo
            "variant": "n",
            # training
            "epochs": 150, "batch": 16, "imgsz": 640, "workers": 4,
            # runtime
            "device": "auto", "seed": 42, "compile": False,
            # optim
            "lr": 0.001, "wd": 0.0, "clip_norm": 0.0, "clip_mode": "norm",
            # amp/ema/mitigaciones
            "amp": "auto", "ema": True, "bn2gn": "on_error",
            "warmup": "off", "warmup_epochs": 0,
            # hud/logging
            "hud": None,  # se resolverá a TTY si sigue None
            # proyecto/guardado
            "project": "runs/train", "name": None, "exist_ok": False, "time_limit": 0.0,
            # val_int
            "dl_info": False,
            "val_int_interval": 5, "val_int_max_batches": 1, "val_int_use_train_subset": False,
            "val_int_conf": 0.25, "val_int_split": "val", "val_int_pivots": True,
            "val_int_tb": True, "val_int_tb_nrow": 3, "val_int_tb_conf": 0.25, "val_int_tb_topk": 5,
            "dataset_base": None,
            # modo
            "test": False,
        }

        # --- parser.yaml ---
        p_path = Path(parser_yaml_path) if parser_yaml_path else (self.yolo_root / "configs" / "parser.yaml")
        p_yaml = self._read_yaml(p_path)
        if p_yaml:
            # rutas de otros YAML
            data_cfg = self._get(p_yaml, ("data", "config"), None)
            yolo_cfg = self._get(p_yaml, ("model", "yolo_cfg"), None)
            out["data"] = data_cfg or out["data"]
            out["model"] = yolo_cfg or out["model"]
            out["parser"] = str(p_path)

            # variant por defecto
            dv = self._get(p_yaml, ("model", "default_variant"), None)
            if isinstance(dv, str) and dv:
                out["variant"] = dv.strip().lower()

            # runtime
            dev = self._get(p_yaml, ("runtime", "device"), None)
            if isinstance(dev, str) and dev:
                out["device"] = dev
            out["seed"] = self._i(self._get(p_yaml, ("runtime", "seed"), out["seed"]), out["seed"])
            out["compile"] = self._b(self._get(p_yaml, ("runtime", "compile"), out["compile"]), out["compile"])

            # project
            runs_dir = self._get(p_yaml, ("project", "dirs", "runs"), None)
            if isinstance(runs_dir, str) and runs_dir:
                out["project"] = runs_dir

        # --- train.yaml ---
        train_cfg_rel = self._get(p_yaml, ("train", "config"), "configs/train.yaml")
        t_path = (self.yolo_root / train_cfg_rel) if not Path(str(train_cfg_rel)).is_absolute() else Path(str(train_cfg_rel))
        t_yaml = self._read_yaml(t_path)
        if t_yaml:
            out["imgsz"] = self._i(t_yaml.get("imgsz"), out["imgsz"])
            out["epochs"] = self._i(t_yaml.get("epochs"), out["epochs"])
            out["batch"] = self._i(t_yaml.get("batch"), out["batch"])

            # dataloader.workers
            if isinstance(t_yaml.get("dataloader"), dict):
                out["workers"] = self._i(t_yaml["dataloader"].get("workers"), out["workers"])

            # optim
            out["lr"] = self._f(t_yaml.get("lr0"), out["lr"])
            out["wd"] = self._f(t_yaml.get("weight_decay"), out["wd"])
            out["clip_norm"] = self._f(t_yaml.get("max_grad_norm"), out["clip_norm"])
            out["clip_mode"] = t_yaml.get("clip_mode", out["clip_mode"]) if t_yaml.get("clip_mode") in {"norm", "value"} else out["clip_mode"]

            # amp/ema
            out["amp"] = self._amp_from_yaml(t_yaml.get("amp", out["amp"]))
            out["ema"] = self._b(t_yaml.get("ema", out["ema"]), out["ema"])

            # warmup
            if "warmup_steps" in t_yaml:
                out["warmup_epochs"] = self._i(t_yaml.get("warmup_steps"), out["warmup_epochs"])
            if "warmup_epochs" in t_yaml:
                out["warmup_epochs"] = self._i(t_yaml.get("warmup_epochs"), out["warmup_epochs"])

            # val_int (hereda valores afines del YAML)
            if "val_interval" in t_yaml:
                out["val_int_interval"] = self._i(t_yaml.get("val_interval"), out["val_int_interval"])
            if "conf_thr" in t_yaml:
                out["val_int_conf"] = self._f(t_yaml.get("conf_thr"), out["val_int_conf"])
                out["val_int_tb_conf"] = self._f(t_yaml.get("conf_thr"), out["val_int_tb_conf"])

            # HUD
            hud_yaml = t_yaml.get("hud", None)
            if hud_yaml is not None:
                out["hud"] = self._b(hud_yaml, None)  # puede quedar None → se resolverá por TTY

        # resolver rutas relativas (solo data/model/parser; project se ancla en resolve_and_validate)
        for k in ("data", "model", "parser"):
            out[k] = self._resolve_path(out.get(k)) or out.get(k)

        return out


# --------------------------------------------------------------
# Presets (definidos como dataclass para claridad y validación simple)
# --------------------------------------------------------------

@dataclass
class Preset:
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_defaults(self) -> Dict[str, Any]:
        return dict(self.overrides)

# Presets solicitados: smoketests y forwards previos a entrenamientos largos
PRESETS: Dict[str, Preset] = {
    # Equivalente a tu comando largo de pruebas iniciales (assembly + warmup sanity)
    "test-A": Preset(
        name="test-A",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "dl_info": True,
            "variant": "s",
            "batch": 4,
            "epochs": 0,
            "imgsz": 640,
            "test": True,
            "warmup_epochs": 1,
            "warmup": "sanity",
            "bn2gn": "on",              # TODO: todo GN
            "amp": "fp16",
            "hud": True,
            "miopen_disable_cache": True,  # TODO: cache OFF en smoketest
        },
    ),
    # Warmup más agresivo para medir estabilidad MIOpen/AMP rápidamente
    "test-B": Preset(
        name="test-B",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "dl_info": True,
            "variant": "s",
            "batch": 4,
            "epochs": 0,
            "imgsz": 640,
            "test": True,
            "warmup_epochs": 1,
            "warmup": "fast",           # más iters
            "bn2gn": "on",              # TODO: todo GN
            "amp": "auto",              # intenta bf16 si disponible
            "hud": True,
            "miopen_disable_cache": True,
        },
    ),
    # Smoke corto de entrenamiento real (1 época) sin test, para verificar loop completo
    "smoke-1ep": Preset(
        name="smoke-1ep",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "dl_info": True,
            "variant": "s",
            "batch": 4,
            "epochs": 1,
            "imgsz": 640,
            "test": False,
            "warmup_epochs": 0,
            "warmup": "off",
            "bn2gn": "on",              # TODO: todo GN
            "amp": "fp16",
            "hud": True,
            # --- validación interna siempre activa en smoketest ---
            "val_int_interval": 1,          # corre en cada época (aquí 1)
            "val_int_use_train_subset": True,
            "val_int_max_batches": 1,       # sólo 1 batch de TRAIN para val_int
            "val_int_tb": False,            # sin TensorBoard en este smoketest
            "miopen_disable_cache": True,
        },
    ),
    # Entrenamiento corto (3 épocas) para revisar pérdidas/EMA/scheduler
    "smoke-3ep": Preset(
        name="smoke-3ep",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "dl_info": True,
            "variant": "s",
            "batch": 8,
            "epochs": 3,
            "imgsz": 640,
            "test": False,
            "warmup_epochs": 1,
            "warmup": "sanity",
            "bn2gn": "on",              # TODO: todo GN
            "amp": "fp16",
            "hud": True,
            # --- validación interna siempre activa en smoketest ---
            "val_int_interval": 1,          # val_int en cada época
            "val_int_use_train_subset": True,
            "val_int_max_batches": 1,       # 1 batch de TRAIN por val_int
            "val_int_tb": True,             # aquí sí, para ver métricas en TB
            "miopen_disable_cache": True,
        },
    ),
    # Forward real del primer minibatch del dataloader (sin warmup)
    "forward": Preset(
        name="forward",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "dl_info": True,
            "variant": "s",
            "batch": 4,
            "epochs": 1,
            "imgsz": 640,
            "test": True,               # assembly + forward
            "warmup_epochs": 0,
            "warmup": "off",
            "bn2gn": "on",              # TODO: todo GN
            "amp": "fp16",
            "hud": True,
            "miopen_disable_cache": True,
        },
    ),
    # Stress de warmup (full) con 2 bucles para cacheo/perfilado de kernels
    "stress-warmup": Preset(
        name="stress-warmup",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "variant": "s",
            "batch": 4,
            "epochs": 1,
            "imgsz": 640,
            "test": True,
            "warmup_epochs": 2,
            "warmup": "full",
            "bn2gn": "on",
            "amp": "fp16",
            "hud": True,
            "miopen_disable_cache": True,
        },
    ),
}


# --------------------------------------------------------------
# Clase principal de construcción/parseo
# --------------------------------------------------------------

@dataclass
class CLIBuilder:
    presets: Dict[str, Preset] = field(default_factory=lambda: PRESETS)

    # --- Catálogo de claves conocidas (debe reflejar train.py) ---
    known_keys: Iterable[str] = field(default_factory=lambda: (
        "data","model","parser","variant","epochs","batch","imgsz","workers",
        "device","seed","lr","wd","clip_norm","clip_mode",
        "warmup","warmup_epochs","bn2gn","amp","ema","compile",
        "hud","resume","project","name","exist_ok","time_limit",
        "dl_info",
        # val_int
        "val_int_interval","val_int_max_batches","val_int_use_train_subset",
        "val_int_conf","val_int_split","val_int_pivots","val_int_tb",
        "val_int_tb_nrow","val_int_tb_conf","val_int_tb_topk","dataset_base",
        # modo
        "test",
    ))

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _yolo_root() -> Path:
        return Path(__file__).resolve().parents[1]

    def _filter_known(self, candidates: Dict[str, Any]) -> Dict[str, Any]:
        ks = set(self.known_keys)
        return {k: v for k, v in candidates.items() if k in ks}

    # ---------------------------
    # Parser
    # ---------------------------
    def build_common_parser(self) -> argparse.ArgumentParser:

        p = argparse.ArgumentParser(
            prog="YOLOv11 CLI",
            description=(
                "CLI modular (clase) para YOLOv11: presets, fusión con YAML y "
                "normalización de rutas, sin dependencias a torch."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        # Ayuda minimalista (permite two-stage parse)
        p.add_argument('-h', '--help', action='store_true', help='Mostrar ayuda y salir')

        # Rutas / datos / modelo
        p.add_argument('--data', type=str, help='Ruta a dataset.yaml')
        p.add_argument('--model', type=str, help='Ruta a yolo11.yaml')
        p.add_argument('--parser', type=str, help='Ruta a parser.yaml (opcional)')

        # Variantes y params clave
        p.add_argument('--variant', type=str, choices=['n','s','m','l','x'])
        p.add_argument('--epochs', type=int)
        p.add_argument('--batch', type=int)
        p.add_argument('--imgsz', type=int)
        p.add_argument('--workers', type=int)

        # Sistema/semilla/device
        p.add_argument('--device', type=str)
        p.add_argument('--seed', type=int)

        # Optim básicos (overrides desde parser.yaml)
        p.add_argument('--lr', type=float)
        p.add_argument('--wd', type=float)
        p.add_argument('--clip-norm', dest='clip_norm', type=float)
        p.add_argument('--clip-mode', dest='clip_mode', type=str, choices=['norm','value'])

        # Mitigaciones / precisión / ema / compile
        p.add_argument('--warmup', type=str, choices=['off','sanity','fast','full'])
        p.add_argument('--warmup-epochs', dest='warmup_epochs', type=int)
        p.add_argument('--bn2gn', type=str, choices=['off','on','on_error'])
        p.add_argument('--amp', type=str, choices=['auto','off','fp16','bf16'])

        # Flags booleanos
        p.add_argument('--ema', dest='ema', action='store_true')
        p.add_argument('--no-ema', dest='ema', action='store_false')
        p.add_argument('--compile', dest='compile', action='store_true')
        p.add_argument('--hud', dest='hud', action='store_true')
        p.add_argument('--no-hud', dest='hud', action='store_false')

        p.add_argument('--resume', type=str)
        p.add_argument('--project', type=str)
        p.add_argument('--name', type=str)
        p.add_argument('--exist-ok', dest='exist_ok', action='store_true')
        p.add_argument('--time-limit', dest='time_limit', type=float)

        # Info de data_loader
        p.add_argument('--dl-info', dest='dl_info', action='store_true')

        # Validación interna (val_int)
        p.add_argument('--val-int-interval', dest='val_int_interval', type=int)
        p.add_argument('--val-int-max-batches', dest='val_int_max_batches', type=int)
        p.add_argument('--val-int-use-train-subset', dest='val_int_use_train_subset', action='store_true')
        p.add_argument('--val-int-conf', dest='val_int_conf', type=float)
        p.add_argument('--val-int-split', dest='val_int_split', type=str, choices=['train','val'])
        p.add_argument('--val-int-pivots', dest='val_int_pivots', action='store_true')
        p.add_argument('--no-val-int-pivots', dest='val_int_pivots', action='store_false')
        p.add_argument('--val-int-tb', dest='val_int_tb', action='store_true')
        p.add_argument('--no-val-int-tb', dest='val_int_tb', action='store_false')
        p.add_argument('--val-int-tb-nrow', dest='val_int_tb_nrow', type=int)
        p.add_argument('--val-int-tb-conf', dest='val_int_tb_conf', type=float)
        p.add_argument('--val-int-tb-topk', dest='val_int_tb_topk', type=int)
        p.add_argument('--dataset-base', dest='dataset_base', type=str)

        # Modo prueba
        p.add_argument('--test', action='store_true')

        # Presets (genérico + alias directos)
        self._add_preset_flags(p)
        return p

    def _add_preset_flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--preset', type=str, choices=sorted(self.presets.keys()), help='Selecciona un preset predefinido')
        for name in self.presets.keys():
            parser.add_argument(f'--{name}', dest='_preset', action='store_const', const=name,
                                help=f'--{name}')

    # ---------------------------
    # YAML y presets
    # ---------------------------
    def merge_with_yaml(self, parser_yaml_path: Optional[str]) -> Dict[str, Any]:
        # Conservado por compatibilidad: ahora preferimos ConfigDefaultsLoader
        if not parser_yaml_path:
            return {}
        p = Path(parser_yaml_path)
        if not p.exists() or not p.is_file() or yaml is None:
            return {}
        try:
            data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        except Exception:
            return {}
        base = data.get('train', data) if isinstance(data, dict) else {}
        return self._filter_known(base)

    def apply_preset(self, ns: argparse.Namespace) -> Dict[str, Any]:
        chosen = getattr(ns, 'preset', None) or getattr(ns, '_preset', None)
        if not chosen:
            return {}
        preset = self.presets.get(chosen)
        return preset.to_defaults() if preset else {}

    # ---------------------------
    # Normalización final
    # ---------------------------
    def resolve_and_validate(self, ns: argparse.Namespace) -> argparse.Namespace:
        # HUD auto por TTY si no se especifica
        if getattr(ns, 'hud', None) is None:
            ns.hud = sys.stdout.isatty()

        # Anclar project relativo a YOLOv11/ si no es absoluto
        proj = getattr(ns, 'project', None)
        if proj:
            p = Path(proj)
            if not p.is_absolute():
                ns.project = str((self._yolo_root() / p).resolve())

        # Normalizar booleanos recibidos como texto
        for key in ("ema","compile","dl_info","val_int_use_train_subset","val_int_pivots","val_int_tb","test","exist_ok"):
            val = getattr(ns, key, None)
            if isinstance(val, str):
                low = val.strip().lower()
                if low in ("1","true","yes","on"): setattr(ns, key, True)
                elif low in ("0","false","no","off"): setattr(ns, key, False)

        # Clip mode seguro
        if getattr(ns, 'clip_mode', None) not in (None, 'norm', 'value'):
            ns.clip_mode = 'norm'

        # Valores seguros si algo quedó en None (evita TypeError aguas arriba)
        safe_numbers: List[Tuple[str, Any]] = [
            ("epochs", 150), ("batch", 16), ("imgsz", 640), ("workers", 4),
            ("lr", 0.001), ("wd", 0.0), ("clip_norm", 0.0), ("time_limit", 0.0),
            ("val_int_interval", 5), ("val_int_max_batches", 1), ("val_int_conf", 0.25),
            ("val_int_tb_nrow", 3), ("val_int_tb_conf", 0.25), ("val_int_tb_topk", 5), ("warmup_epochs", 0),
        ]
        for k, dv in safe_numbers:
            if getattr(ns, k, None) is None:
                setattr(ns, k, dv)

        safe_strings: List[Tuple[str, str]] = [
            ("variant", "n"), ("device", "auto"), ("clip_mode", "norm"), ("val_int_split", "val"), ("warmup", "off"),
        ]
        for k, dv in safe_strings:
            if getattr(ns, k, None) in (None, ""):
                setattr(ns, k, dv)

        safe_bools: List[Tuple[str, bool]] = [
            ("ema", True), ("compile", False), ("dl_info", False), ("val_int_use_train_subset", False),
            ("val_int_pivots", True), ("val_int_tb", True), ("exist_ok", False), ("test", False),
        ]
        for k, dv in safe_bools:
            if getattr(ns, k, None) is None:
                setattr(ns, k, dv)

        # AMP normalizado (si viniera None)
        if getattr(ns, 'amp', None) in (None, ""):
            ns.amp = 'auto'

        # Exponer nombre del preset aplicado (si lo hay) para banner aguas arriba
        chosen = getattr(ns, 'preset', None) or getattr(ns, '_preset', None)
        if chosen:
            setattr(ns, 'preset_applied', chosen)

        return ns

    # ---------------------------
    # Parseo en dos etapas
    # ---------------------------
    def parse_args_two_stage(self, argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
        parser = self.build_common_parser()
        prelim, _ = parser.parse_known_args(argv)

        # 1) Determinar preset y parser.yaml efectivo ANTES de cargar YAML
        chosen = getattr(prelim, 'preset', None) or getattr(prelim, '_preset', None)
        preset_defaults = self.apply_preset(prelim) if chosen else {}
        parser_path = getattr(prelim, 'parser', None) or preset_defaults.get('parser') or str(self._yolo_root() / 'configs' / 'parser.yaml')

        # 2) Cargar defaults tipados desde YAMLs (sin None)
        yaml_defaults = ConfigDefaultsLoader(self._yolo_root()).load(parser_path)

        # 3) Establecer defaults en el parser respetando precedencia declarada
        #    YAML primero, luego PRESET (CLI explícito anula ambos en el parse final)
        if yaml_defaults:
            parser.set_defaults(**self._filter_known(yaml_defaults))
        if preset_defaults:
            parser.set_defaults(**self._filter_known(preset_defaults))

        if getattr(prelim, 'help', False):
            parser.print_help()
            sys.exit(0)

        # 4) Parse final y resolución/saneamiento
        final = parser.parse_args(argv)
        final = self.resolve_and_validate(final)
        return final


# --------------------------------------------------------------
# Wrappers de compatibilidad (para train.py actual)
# --------------------------------------------------------------

_BUILDER = CLIBuilder()


def build_common_parser() -> argparse.ArgumentParser:  # pragma: no cover
    return _BUILDER.build_common_parser()


def add_train_args(parser: argparse.ArgumentParser) -> None:  # pragma: no cover
    return None  # ya incluidos en build_common_parser


def merge_with_yaml(parser: argparse.ArgumentParser, parser_yaml_path: Optional[str]) -> Dict[str, Any]:  # pragma: no cover
    # parser no es necesario; se mantiene la firma por compatibilidad
    return _BUILDER.merge_with_yaml(parser_yaml_path)


def apply_preset(ns: argparse.Namespace) -> Dict[str, Any]:  # pragma: no cover
    return _BUILDER.apply_preset(ns)


def resolve_and_validate(ns: argparse.Namespace) -> argparse.Namespace:  # pragma: no cover
    return _BUILDER.resolve_and_validate(ns)


def parse_args_two_stage(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:  # pragma: no cover
    return _BUILDER.parse_args_two_stage(argv)
