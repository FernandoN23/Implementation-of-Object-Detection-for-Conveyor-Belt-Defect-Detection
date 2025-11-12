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
#  Sin dependencias a torch/ROCm.
#==============================================================

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # lectura YAML es opcional; si no hay PyYAML, se ignora con warning

__all__ = [
    # API pública (wrappers sobre la clase)
    "build_common_parser",
    "add_train_args",
    "merge_with_yaml",
    "apply_preset",
    "resolve_and_validate",
    "parse_args_two_stage",
]

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
            "epochs": 1,
            "imgsz": 640,
            "test": True,
            "warmup_epochs": 1,
            "warmup": "sanity",
            "bn2gn": "on",
            "amp": "fp16",
            "hud": True,
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
            "epochs": 1,
            "imgsz": 640,
            "test": True,
            "warmup_epochs": 1,
            "warmup": "fast",  # más iters
            "bn2gn": "on_error",
            "amp": "auto",     # intenta bf16 si disponible
            "hud": True,
        },
    ),
    # Smoke corto de entrenamiento real (1 época) sin test, para verificar loop completo
    "smoke-1ep": Preset(
        name="smoke-1ep",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "variant": "s",
            "batch": 4,
            "epochs": 1,
            "imgsz": 640,
            "test": False,
            "warmup_epochs": 0,
            "warmup": "off",
            "bn2gn": "on_error",
            "amp": "fp16",
            "hud": True,
            "val_int_interval": 5,         # prácticamente no corre en 1 epoca
            "val_int_tb": False,
        },
    ),
    # Entrenamiento corto (3 épocas) para revisar pérdidas/EMA/scheduler
    "smoke-3ep": Preset(
        name="smoke-3ep",
        overrides={
            "data": "configs/dataset.yaml",
            "model": "configs/yolo11.yaml",
            "parser": "configs/parser.yaml",
            "variant": "s",
            "batch": 8,
            "epochs": 3,
            "imgsz": 640,
            "test": False,
            "warmup_epochs": 1,
            "warmup": "sanity",
            "bn2gn": "on_error",
            "amp": "fp16",
            "hud": True,
            "val_int_interval": 3,
            "val_int_tb": True,
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
            "test": True,        # assembly + forward
            "warmup_epochs": 0,
            "warmup": "off",
            "bn2gn": "on_error",
            "amp": "fp16",
            "hud": True,
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

    @staticmethod
    def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                out.update(CLIBuilder._flatten(v, key))
            else:
                out[key] = v
        return out

    def _filter_known(self, candidates: Dict[str, Any]) -> Dict[str, Any]:
        ks = set(self.known_keys)
        return {k: v for k, v in candidates.items() if k in ks}

    # ---------------------------
    # Parser
    # ---------------------------
    def build_common_parser(self) -> argparse.ArgumentParser:
        epilog_lines: List[str] = [
            "Presets disponibles:",
        ]
        for name, preset in self.presets.items():
            epilog_lines.append(f"  --{name:<14} |  --preset {name:<14} → {preset.overrides}")
        epilog_lines.append("\nPrecedencia: CLI explícito > preset > YAML > defaults")

        p = argparse.ArgumentParser(
            prog="YOLOv11 CLI",
            description=(
                "CLI modular (clase) para YOLOv11: presets, fusión con YAML y "
                "normalización de rutas, sin dependencias a torch."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog="\n".join(epilog_lines),
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
                                help=f'Alias para --preset {name}')

    # ---------------------------
    # YAML y presets
    # ---------------------------
    def merge_with_yaml(self, parser_yaml_path: Optional[str]) -> Dict[str, Any]:
        if not parser_yaml_path:
            return {}
        p = Path(parser_yaml_path)
        if not p.exists() or not p.is_file() or yaml is None:
            return {}
        try:
            data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        except Exception:
            return {}
        # Preferir sección train si existe
        base = data.get('train', data) if isinstance(data, dict) else {}
        base_known = self._filter_known(base)
        return base_known

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

        # Normalizar booleanos (por si vienen como strings)
        for key in ("ema","compile","dl_info","val_int_use_train_subset","val_int_pivots","val_int_tb","test","exist_ok"):
            val = getattr(ns, key, None)
            if isinstance(val, str):
                low = val.strip().lower()
                if low in ("1","true","yes","on"): setattr(ns, key, True)
                elif low in ("0","false","no","off"): setattr(ns, key, False)

        # Clip mode seguro
        if getattr(ns, 'clip_mode', None) not in (None, 'norm', 'value'):
            ns.clip_mode = 'norm'

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

        yaml_defaults = self.merge_with_yaml(getattr(prelim, 'parser', None))
        preset_defaults = self.apply_preset(prelim)

        if yaml_defaults:
            parser.set_defaults(**yaml_defaults)
        if preset_defaults:
            parser.set_defaults(**preset_defaults)

        if getattr(prelim, 'help', False):
            parser.print_help()
            sys.exit(0)

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
