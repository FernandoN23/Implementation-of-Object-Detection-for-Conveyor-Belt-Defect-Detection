# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/engine/bootstrap_miopen.py
# Descripción: Inicialización de variables de entorno MIOpen/ROCm.
#              Incluye optimización de memoria para evitar OOM.
#              *VERSIÓN SEGURA PARA WINDOWS*
# ==============================================================

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MIOpenConfig:
    find_mode: str = "FAST"
    user_db_path: Optional[str] = "C:/Users/memorista/.miopen_db"
    disable_cache: bool = True
    expandable_segments: bool = True
    verbose: int = 1


class MuteStderr:
    """
    Context manager seguro para silenciar stderr en Windows.
    [MODIFICADO]: Evita el uso de os.dup() que causa crashes silenciosos
    al interactuar con el DataLoader en entornos Windows/ROCm.
    """
    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        self._old_stderr = sys.stderr
        sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._old_stderr
        self._devnull.close()


def bootstrap(cfg: MIOpenConfig):
    """Configura el entorno MIOpen y PyTorch Allocator para estabilidad en ROCm."""

    if "torch" in sys.modules:
        raise RuntimeError(
            "[bootstrap_miopen] ERROR: 'torch' ya ha sido importado. "
            "El bootstrap debe ocurrir antes de inicializar el backend HIP."
        )

    os.environ["MIOPEN_FIND_MODE"] = cfg.find_mode

    if cfg.user_db_path:
        db_path = Path(cfg.user_db_path).expanduser().resolve()
        db_path.mkdir(parents=True, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = str(db_path)

    os.environ["MIOPEN_DISABLE_CACHE"] = "1" if cfg.disable_cache else "0"

    if cfg.expandable_segments:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    os.environ["AMD_LOG_LEVEL"] = "0"
    os.environ["MIOPEN_LOG_LEVEL"] = "0"
    os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
    os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

    if cfg.verbose > 0:
        print(f"[bootstrap_miopen] MIOpen configurado: FIND_MODE={cfg.find_mode}, "
              f"EXPANDABLE_SEGMENTS={cfg.expandable_segments}")
        print(f"[bootstrap_miopen] INFO: Usando backend de PyTorch puro para Deformable Attention.")