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

Este módulo configura variables de entorno y rutas necesarias para mitigar
problemas típicos de MIOpen antes de inicializar PyTorch/HIP.

✔ Debe ejecutarse ANTES de `import torch`.
✔ Expone una API programática (`bootstrap`) y una CLI mínima.
✔ Soporta políticas de modo de búsqueda de kernels, path de DB de usuario,
  desactivación de caché y verificación estricta si `torch` ya fue importado.

Notas prácticas (ROCm Windows Preview 6.4.4):
- MIOPEN_FIND_MODE: "FAST" | "NORMAL" | "HYBRID" (recomendado: FAST en primeras corridas)
- MIOPEN_USER_DB_PATH: carpeta con permisos de escritura del usuario
- MIOPEN_DISABLE_CACHE: 0/1 (usar 0; habilitar 1 sólo para troubleshooting)
- MIOPEN_LOG_LEVEL: 0/1/2 (0 por defecto)

Uso programático
----------------
from engine.bootstrap_miopen import bootstrap, MIOpenConfig
cfg = MIOpenConfig(find_mode="FAST", user_db_path="C:/Users/you/.miopen", disable_cache=False)
bootstrap(cfg)

CLI
---
python -m engine.bootstrap_miopen --miopen-find FAST \
    --miopen-user-db C:/Users/you/.miopen --miopen-disable-cache false \
    --strict-before-torch true --verbose 1

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "MIOpenConfig",
    "bootstrap",
    "export_env",
    "prepare_user_db",
    "torch_already_imported",
    "load_config_from_json",
]

# -------------------------------
# Configuración y validaciones
# -------------------------------

_FIND_MODES = {"FAST": "FAST", "NORMAL": "NORMAL", "HYBRID": "HYBRID"}


@dataclass
class MIOpenConfig:
    """Configuración de bootstrap MIOpen.

    Atributos
    ---------
    find_mode: str
        Modo de búsqueda de kernels (FAST/NORMAL/HYBRID).
    user_db_path: Optional[str]
        Carpeta de base de datos de usuario de MIOpen (se crea si no existe).
    disable_cache: bool
        Si `True`, desactiva la caché de MIOpen (usar sólo para troubleshooting).
    log_level: int
        Nivel de log de MIOpen (0 silencioso por defecto).
    extra_env: Dict[str, str]
        Variables extra para exportar (HIP_VISIBLE_DEVICES, HSA_OVERRIDE_GFX_VERSION, etc.).
    strict_before_torch: bool
        Si `True`, lanza error si `torch` ya fue importado.
    verbose: int
        0: silencioso, 1: info mínima, 2: detallado.
    """

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
            # Permitir strings "true"/"false" provenientes de CLI
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
        # fallback a FAST si inválido
        return "FAST"
    return _FIND_MODES[m]


def torch_already_imported() -> bool:
    """Retorna True si el intérprete ya tiene `torch` cargado.

    Debe usarse para prevenir importaciones prematuras antes del bootstrap.
    """
    return "torch" in sys.modules


# -------------------------------
# Preparación de carpeta DB usuario
# -------------------------------

def prepare_user_db(path: Optional[str], *, cfg: Optional[MIOpenConfig] = None) -> Optional[str]:
    """Crea/verifica la carpeta de DB de usuario de MIOpen.

    Si `path` es None, retorna None (se omitirá el seteo de MIOPEN_USER_DB_PATH).
    """
    if not path:
        _log("Sin user_db_path explícito (se usará ubicación por defecto de MIOpen).", level=2, cfg=cfg)
        return None

    p = Path(path).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"No se pudo crear la carpeta para MIOpen DB: {p} -> {e}")

    # En Windows no ajustamos permisos POSIX; basta con existencia/escritura
    if not os.access(str(p), os.W_OK):
        raise PermissionError(f"Sin permisos de escritura en: {p}")

    _log(f"MIOpen USER DB preparado en: {p}", level=1, cfg=cfg)
    return str(p.resolve())


# -------------------------------
# Exportación de entorno
# -------------------------------

def export_env(cfg: MIOpenConfig) -> Dict[str, str]:
    """Construye y exporta variables de entorno MIOpen/ROCm.

    Retorna un dict con las claves exportadas (para logging/tests).
    """
    exported: Dict[str, str] = {}

    # MIOpen core
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

    # FIX: Forzar silencio de MIOpen
    os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
    exported["MIOPEN_ENABLE_LOGGING"] = "0"
    os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
    exported["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

    # Variables extra opcionales
    for k, v in (cfg.extra_env or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("extra_env debe ser Dict[str, str]")
        os.environ[k] = v
        exported[k] = v

    # Mensajería
    _log(f"MIOPEN_FIND_MODE={exported.get('MIOPEN_FIND_MODE')} | "
         f"MIOPEN_DISABLE_CACHE={exported.get('MIOPEN_DISABLE_CACHE')} | "
         f"LOG_LEVEL={exported.get('MIOPEN_LOG_LEVEL')}", level=1, cfg=cfg)

    if "MIOPEN_USER_DB_PATH" in exported:
        _log(f"DB usuario: {exported['MIOPEN_USER_DB_PATH']}", level=1, cfg=cfg)

    if cfg.verbose >= 2 and cfg.extra_env:
        _log("extra_env=" + json.dumps(cfg.extra_env, ensure_ascii=False), level=2, cfg=cfg)

    return exported


# -------------------------------
# Punto principal de bootstrap
# -------------------------------

def bootstrap(cfg: MIOpenConfig) -> Dict[str, str]:
    """Ejecuta el bootstrap MIOpen y valida el estado previo a torch.

    - Verifica que `torch` no esté importado si `strict_before_torch=True`.
    - Exporta variables MIOpen/ROCm y prepara DB de usuario.
    - Retorna dict de variables exportadas.
    """
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
    """Carga configuración desde JSON sencillo.

    Estructura esperada:
    {
      "find_mode": "FAST",
      "user_db_path": "C:/Users/you/.miopen",
      "disable_cache": false,
      "log_level": 0,
      "extra_env": {"HIP_VISIBLE_DEVICES": "0"},
      "strict_before_torch": true,
      "verbose": 1
    }
    """
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
    ap = argparse.ArgumentParser(
        prog="engine.bootstrap_miopen",
        description=(
            "Bootstrap de variables MIOpen/ROCm (ejecutar ANTES de importar torch)."
        ),
    )
    ap.add_argument("--miopen-find", dest="find_mode", default="FAST",
                    choices=["FAST", "NORMAL", "HYBRID"], help="Modo de búsqueda de kernels MIOpen")
    ap.add_argument("--miopen-user-db", dest="user_db_path", default=None,
                    help="Ruta a carpeta de DB de usuario MIOpen (se crea si no existe)")
    ap.add_argument("--miopen-disable-cache", dest="disable_cache", default="false",
                    help="true/false para desactivar la caché de MIOpen (solo troubleshooting)")
    ap.add_argument("--miopen-log-level", dest="log_level", type=int, default=0,
                    help="Nivel de log MIOpen (0 por defecto)")
    ap.add_argument("--extra-env", dest="extra_env", default=None,
                    help="JSON con variables extra (p. ej., {\"HIP_VISIBLE_DEVICES\": \"0\"})")
    ap.add_argument("--strict-before-torch", dest="strict_before_torch", default="true",
                    help="Falla si torch ya fue importado (true/false)")
    ap.add_argument("--verbose", dest="verbose", type=int, default=1, choices=[0, 1, 2],
                    help="Nivel de verbosidad del bootstrap")
    return ap


def _main(argv: Optional[list[str]] = None) -> int:
    ap = _build_argparser()
    ns = ap.parse_args(argv)

    extra_env: Dict[str, str] = {}
    if ns.extra_env:
        try:
            extra_env = json.loads(ns.extra_env)
            if not isinstance(extra_env, dict):
                raise ValueError
            # Forzar str->str
            extra_env = {str(k): str(v) for k, v in extra_env.items()}
        except Exception:
            print(
                f"{_DEF_PREFIX} ERROR: --extra-env debe ser JSON objeto, ej. '{{\"HIP_VISIBLE_DEVICES\": \"0\"}}'",
                file=sys.stderr,
            )
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

    # Resumen mínimo en salida estándar
    print(json.dumps({"exported": exported, "config": asdict(cfg)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())