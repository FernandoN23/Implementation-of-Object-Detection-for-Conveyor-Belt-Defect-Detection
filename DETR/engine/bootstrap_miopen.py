# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/bootstrap_miopen.py
# Descripción: Inicialización de variables de entorno MIOpen/ROCm.
#              Incluye utilidad MuteStderr para silenciar ruido C++.
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
    disable_cache: bool = True  # Política de proyecto: Siempre True
    verbose: int = 1


class MuteStderr:
    """Context manager para silenciar stderr a nivel de descriptor de archivo (C++ noise)."""
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


def bootstrap(cfg: MIOpenConfig):
    """Configura el entorno MIOpen para evitar fallos de kernel en ROCm."""

    if "torch" in sys.modules:
        raise RuntimeError(
            "[bootstrap_miopen] ERROR: 'torch' ya ha sido importado. "
            "El bootstrap debe ocurrir antes de inicializar el backend HIP."
        )

    # 1. Modo de búsqueda de kernels
    os.environ["MIOPEN_FIND_MODE"] = cfg.find_mode

    # 2. Base de datos de usuario (asegurar existencia)
    if cfg.user_db_path:
        db_path = Path(cfg.user_db_path).expanduser().resolve()
        db_path.mkdir(parents=True, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = str(db_path)

    # 3. Desactivar caché (Crucial para estabilidad en Windows Preview)
    os.environ["MIOPEN_DISABLE_CACHE"] = "1" if cfg.disable_cache else "0"

    # 4. Silenciar logs de bajo nivel del driver AMD/MIOpen
    os.environ["AMD_LOG_LEVEL"] = "0"
    os.environ["MIOPEN_LOG_LEVEL"] = "0"
    os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
    os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "0"
    os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

    if cfg.verbose > 0:
        print(f"[bootstrap_miopen] MIOpen configurado: FIND_MODE={cfg.find_mode}, "
              f"DISABLE_CACHE={cfg.disable_cache}")