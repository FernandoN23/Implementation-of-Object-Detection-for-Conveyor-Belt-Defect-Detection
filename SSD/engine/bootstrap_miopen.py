# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/bootstrap_miopen.py
# Descripción: Inicialización temprana de variables y paths MIOpen/ROCm
#              (debe ejecutarse ANTES de importar torch) y utilidades
#              de verificación/log para entornos Windows ROCm Preview.
# ==============================================================

"""
Bootstrap MIOpen/ROCm (Windows Preview) ⚙️
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional
import contextlib

__all__ = [
    "MIOpenConfig",
    "bootstrap",
    "export_env",
    "prepare_user_db",
    "torch_already_imported",
    "load_config_from_json",
    "MuteStderr",
]

# -------------------------------
# Configuración y validaciones
# -------------------------------

_FIND_MODES = {"FAST": "FAST", "NORMAL": "NORMAL", "HYBRID": "HYBRID"}


@dataclass
class MIOpenConfig:
    find_mode: str = "FAST"
    user_db_path: Optional[str] = None
    disable_cache: bool = False
    log_level: int = 0
    extra_env: Dict[str, str] = None  # type: ignore[assignment]
    strict_before_torch: bool = True
    verbose: int = 1

    def __post_init__(self) -> None:
        self.find_mode = _normalize_find_mode(self.find_mode)
        if self.extra_env is None:
            self.extra_env = {}
        if self.user_db_path:
            self.user_db_path = str(Path(self.user_db_path).expanduser().resolve())
        if not isinstance(self.disable_cache, bool):
            self.disable_cache = str(self.disable_cache).strip().lower() in {"1", "true", "yes"}
        if not isinstance(self.log_level, int) or self.log_level < 0:
            self.log_level = 0
        if self.verbose not in (0, 1, 2):
            self.verbose = 1


# -------------------------------
# Utilidades de logging local
# -------------------------------

_DEF_PREFIX = "[bootstrap_miopen]"


def _log(msg: str, *, level: int, cfg: Optional[MIOpenConfig]) -> None:
    if cfg is None:
        v = 1
    else:
        v = cfg.verbose
    if v >= level:
        print(f"{_DEF_PREFIX} {msg}")


# -------------------------------
# Normalización/chequeos
# -------------------------------

def _normalize_find_mode(mode: str) -> str:
    m = (mode or "").strip().upper()
    if m not in _FIND_MODES:
        return "FAST"
    return _FIND_MODES[m]


def torch_already_imported() -> bool:
    return "torch" in sys.modules


# -------------------------------
# Preparación de carpeta DB usuario
# -------------------------------

def prepare_user_db(path: Optional[str], *, cfg: Optional[MIOpenConfig] = None) -> Optional[str]:
    if not path:
        _log("Sin user_db_path explícito (se usará ubicación por defecto de MIOpen).", level=2, cfg=cfg)
        return None

    p = Path(path).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"No se pudo crear la carpeta para MIOpen DB: {p} -> {e}")

    if not os.access(str(p), os.W_OK):
        raise PermissionError(f"Sin permisos de escritura en: {p}")

    _log(f"MIOpen USER DB preparado en: {p}", level=1, cfg=cfg)
    return str(p.resolve())


# -------------------------------
# Exportación de entorno
# -------------------------------

def export_env(cfg: MIOpenConfig) -> Dict[str, str]:
    exported: Dict[str, str] = {}

    os.environ["MIOPEN_FIND_MODE"] = cfg.find_mode
    exported["MIOPEN_FIND_MODE"] = cfg.find_mode

    if cfg.user_db_path:
        user_db = prepare_user_db(cfg.user_db_path, cfg=cfg)
        if user_db:
            os.environ["MIOPEN_USER_DB_PATH"] = user_db
            exported["MIOPEN_USER_DB_PATH"] = user_db

    os.environ["MIOPEN_DISABLE_CACHE"] = "1" if cfg.disable_cache else "0"
    exported["MIOPEN_DISABLE_CACHE"] = os.environ["MIOPEN_DISABLE_CACHE"]

    os.environ["MIOPEN_LOG_LEVEL"] = str(cfg.log_level)
    exported["MIOPEN_LOG_LEVEL"] = str(cfg.log_level)

    # FIX: Variables adicionales para silenciar MIOpen
    os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
    exported["MIOPEN_ENABLE_LOGGING"] = "0"
    os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "0" # Comando extra
    exported["MIOPEN_ENABLE_LOGGING_CMD"] = "0"
    os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
    exported["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

    for k, v in (cfg.extra_env or {}).items():
        os.environ[k] = v
        exported[k] = v

    _log(f"MIOPEN_FIND_MODE={exported.get('MIOPEN_FIND_MODE')} | "
         f"MIOPEN_DISABLE_CACHE={exported.get('MIOPEN_DISABLE_CACHE')} | "
         f"LOG_LEVEL={exported.get('MIOPEN_LOG_LEVEL')}", level=1, cfg=cfg)

    if "MIOPEN_USER_DB_PATH" in exported:
        _log(f"DB usuario: {exported['MIOPEN_USER_DB_PATH']}", level=1, cfg=cfg)

    return exported


# -------------------------------
# Utilidad para silenciar STDERR (C++ noise)
# -------------------------------

class MuteStderr:
    """Context manager para silenciar stderr a nivel de descriptor de archivo (C++)."""
    def __enter__(self):
        self._original_stderr_fd = sys.stderr.fileno()
        self._saved_stderr_fd = os.dup(self._original_stderr_fd)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        # Reemplazar stderr con devnull
        os.dup2(self._devnull, self._original_stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurar stderr
        os.dup2(self._saved_stderr_fd, self._original_stderr_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._devnull)


# -------------------------------
# Punto principal de bootstrap
# -------------------------------

def bootstrap(cfg: MIOpenConfig) -> Dict[str, str]:
    if cfg.strict_before_torch and torch_already_imported():
        raise RuntimeError(
            "`torch` ya ha sido importado. Ejecute engine.bootstrap_miopen.bootstrap() "
            "ANTES de cualquier import de PyTorch para que las variables MIOpen surtan efecto."
        )

    _log("Inicializando bootstrap MIOpen…", level=1, cfg=cfg)
    exported = export_env(cfg)
    _log("Bootstrap MIOpen completado.", level=1, cfg=cfg)
    return exported


# -------------------------------
# Lectura de config simple desde JSON/YAML (opcional)
# -------------------------------

def load_config_from_json(path: str) -> MIOpenConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return MIOpenConfig(**data)


# -------------------------------
# CLI
# -------------------------------

def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="engine.bootstrap_miopen")
    ap.add_argument("--miopen-find", dest="find_mode", default="FAST")
    ap.add_argument("--miopen-user-db", dest="user_db_path", default=None)
    ap.add_argument("--miopen-disable-cache", dest="disable_cache", default="false")
    ap.add_argument("--miopen-log-level", dest="log_level", type=int, default=0)
    ap.add_argument("--extra-env", dest="extra_env", default=None)
    ap.add_argument("--strict-before-torch", dest="strict_before_torch", default="true")
    ap.add_argument("--verbose", dest="verbose", type=int, default=1)
    return ap


def _main(argv: Optional[list[str]] = None) -> int:
    ap = _build_argparser()
    ns = ap.parse_args(argv)

    extra_env: Dict[str, str] = {}
    if ns.extra_env:
        try:
            extra_env = json.loads(ns.extra_env)
            extra_env = {str(k): str(v) for k, v in extra_env.items()}
        except Exception:
            return 2

    cfg = MIOpenConfig(
        find_mode=ns.find_mode,
        user_db_path=ns.user_db_path,
        disable_cache=_parse_bool(ns.disable_cache),
        log_level=int(ns.log_level),
        extra_env=extra_env,
        strict_before_torch=_parse_bool(ns.strict_before_torch),
        verbose=int(ns.verbose),
    )

    try:
        exported = bootstrap(cfg)
    except Exception as e:
        print(f"{_DEF_PREFIX} ERROR: {e}", file=sys.stderr)
        return 1

    print(json.dumps({"exported": exported, "config": asdict(cfg)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())