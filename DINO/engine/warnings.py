# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/engine/warnings.py
# Descripción: Configuración centralizada de warnings y mensajes
#              de logging ruidosos del proyecto.
#              - Filtra avisos cosméticos (pin_memory, AMP).
#              - Deduplica warnings conocidos de YOLOv5/Ultralytics
#                (torchvision preview, settings, etc.).
#              - Mantiene un comportamiento seguro: nunca oculta
#                errores desconocidos.
#              - Silencia advertencias cosméticas de MIOpen/ROCm.
#              - [NUEVO] Silencia deprecaciones de torchvision.
#==============================================================

from __future__ import annotations

import warnings
import logging
from typing import Callable, Dict, Optional, Set

try:  # Integración opcional con YOLOv5 (LOGGER)
    from yolov5.utils.general import LOGGER  # type: ignore
except Exception:  # pragma: no cover - entorno sin paquete yolov5
    LOGGER = None  # type: ignore

__all__ = ["install_global_warning_filters", "configure_warnings"]


# ---------------------------------------------------------------------------
# Estado interno
# ---------------------------------------------------------------------------

# Para envolver una sola vez warnings.showwarning
_original_showwarning: Optional[Callable[..., None]] = None
_showwarning_installed: bool = False

# Para envolver LOGGER.warning de YOLOv5 y deduplicar mensajes ruidosos
_original_logger_warning: Optional[Callable[..., None]] = None
_logger_installed: bool = False
_seen_logger_keys: Set[str] = set()


# Patrones de mensajes conocidos (substrings simples, seguros)
KNOWN_WARNING_PATTERNS: Dict[str, str] = {
    # Compatibilidad preview ROCm/Windows: no se puede cambiar torchvision,
    # por lo que se muestra sólo una vez por ejecución.
    "torchvision_mismatch": "torchvision==0.24 is incompatible with torch==2.8",

    # Settings de Ultralytics cuando crea/lee el JSON por primera vez.
    "ultralytics_settings": "Error decoding JSON from Ultralytics settings",
}


# ---------------------------------------------------------------------------
# Filtro de Logging para MIOpen
# ---------------------------------------------------------------------------
class MIOpenNoiseFilter(logging.Filter):
    """Filtra mensajes cosméticos de MIOpen que llegan a través del módulo logging."""
    def filter(self, record):
        msg = record.getMessage()
        noise_patterns = [
            "IsEnoughWorkspace",
            "GetSolutionsFallback",
            "provided ptr: 0000000000000000",
            "Solver <",
            "Invalid elapsed time detected",
            "AllocatorConfig.cpp",
            "PYTORCH_ALLOC_CONF"
        ]
        if any(pattern in msg for pattern in noise_patterns):
            return False
        return True


# ---------------------------------------------------------------------------
# Configuración de warnings del módulo `warnings` estándar
# ---------------------------------------------------------------------------

def _configure_python_warnings(force: bool = False) -> None:
    """Configura filtros a nivel del módulo `warnings` estándar.

    - Ignora el warning cosmético de pin_memory/set_num_threads.
    - Muestra una sola vez por ejecución los FutureWarning de AMP
      (`torch.cuda.amp.autocast`, `torch.cuda.amp.GradScaler`).
    - Silencia advertencias de MIOpen/ROCm.
    - Silencia deprecaciones de torchvision.
    - No intercepta ni modifica errores reales.
    """
    global _original_showwarning, _showwarning_installed

    if _showwarning_installed and not force:
        return

    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning

    # Warning interno de pin_memory (cosmético, no funcional)
    warnings.filterwarnings(
        "ignore",
        message=(
            r".*Cannot set number of intraop threads after parallel work has started "
            r"or after set_num_threads call when using native parallel backend.*"
        ),
        category=UserWarning,
        module=r"torch\.utils\.data\._utils\.pin_memory",
    )

    # FutureWarning de AMP: se muestran como mucho una vez
    warnings.filterwarnings(
        "once",
        message=r".*torch\.cuda\.amp\.autocast.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "once",
        message=r".*torch\.cuda\.amp\.GradScaler.*",
        category=FutureWarning,
    )

    # Filtros Regex para MIOpen/ROCm
    warnings.filterwarnings("ignore", message=".*IsEnoughWorkspace.*")
    warnings.filterwarnings("ignore", message=".*GetSolutionsFallback.*")
    warnings.filterwarnings("ignore", message=".*provided ptr: 0000000000000000.*")
    warnings.filterwarnings("ignore", message=".*Solver <.*>.*")

    # [NUEVO] Filtros Regex para deprecaciones de torchvision (código legado de DINO)
    warnings.filterwarnings("ignore", message=".*The parameter 'pretrained' is deprecated.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*", category=UserWarning)

    def _y11_showwarning(message, category, filename, lineno, file=None, line=None):  # type: ignore[override]
        """Wrapper fino sobre showwarning.

        - Colapsa algunos mensajes extremadamente verbosos (p.ej. MIOpen
          internos o scheduler LR) a una línea compacta.
        - Deja pasar todo lo que no reconozca.
        """

        text = str(message)

        # MIOpen: elapsed time inválido (ruido interno de backend)
        if "Invalid elapsed time detected in EvaluateInvokers" in text:
            return

        # MIOpen: Workspace y Fallbacks (ruido interno de backend)
        miopen_noise = ["IsEnoughWorkspace", "GetSolutionsFallback", "provided ptr: 0000000000000000", "Solver <"]
        if any(noise in text for noise in miopen_noise):
            return

        # Scheduler LR: orden de llamadas step (cosmético, no funcional)
        if "Detected call of `lr_scheduler.step()` before `optimizer.step()`" in text:
            return

        # Fallback al comportamiento original
        if _original_showwarning is not None:
            _original_showwarning(message, category, filename, lineno, file=file, line=line)
        else:  # pragma: no cover - caso extremadamente raro
            try:
                warnings.showwarning(message, category, filename, lineno, file=file, line=line)
            except Exception:
                print(f"WARNING: {category.__name__}: {message}")

    warnings.showwarning = _y11_showwarning  # type: ignore[assignment]
    _showwarning_installed = True


# ---------------------------------------------------------------------------
# Configuración del LOGGER de YOLOv5 (deduplicación de warnings)
# ---------------------------------------------------------------------------

def _configure_yolov5_logger(force: bool = False) -> None:
    """Envuelve LOGGER.warning de YOLOv5 para deduplicar mensajes ruidosos.

    - Sólo aplica si el paquete yolov5 y LOGGER están disponibles.
    - Para cada patrón en KNOWN_WARNING_PATTERNS se permite como máximo
      una emisión por ejecución.
    - Mensajes no reconocidos pasan íntegros al logger original.
    """
    global _original_logger_warning, _logger_installed

    if LOGGER is None:  # entorno sin YOLOv5 importable
        return

    if _logger_installed and not force:
        return

    if _original_logger_warning is None:
        _original_logger_warning = LOGGER.warning

    def _wrapped_warning(msg, *args, **kwargs):  # type: ignore[override]
        text = str(msg)
        key: Optional[str] = None

        for k, pattern in KNOWN_WARNING_PATTERNS.items():
            if pattern in text:
                key = k
                break

        if key is not None:
            if key in _seen_logger_keys:
                # Ya se mostró antes este tipo de warning → suprimir
                return
            _seen_logger_keys.add(key)

        # Mensaje no conocido o primera vez que aparece → se deja pasar
        if _original_logger_warning is not None:
            _original_logger_warning(msg, *args, **kwargs)
        else:  # pragma: no cover
            print(f"LOGGER WARNING: {text}")

    LOGGER.warning = _wrapped_warning  # type: ignore[assignment]
    _logger_installed = True


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def install_global_warning_filters(force: bool = False) -> None:
    """Instala todos los filtros de warnings/logging del proyecto.

    Se puede llamar múltiples veces de forma segura (idempotente). Si
    ``force=True`` fuerza la reinstalación de los wrappers.
    """
    # Añadir filtro al logger raíz de Python
    logging.getLogger().addFilter(MIOpenNoiseFilter())

    _configure_python_warnings(force=force)
    _configure_yolov5_logger(force=force)


def configure_warnings(force: bool = False) -> None:
    """Alias retrocompatible para código legado.

    Históricamente se usaba ``configure_warnings`` como entrada; ahora la
    función recomendada es :func:`install_global_warning_filters`, pero
    mantenemos esta API para no romper imports existentes.
    """

    install_global_warning_filters(force=force)


# Instalación por defecto al importar el módulo (se puede repetir en train.py)
install_global_warning_filters(force=False)