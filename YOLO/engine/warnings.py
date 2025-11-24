# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/warnings.py
# Descripción: Configuración centralizada de warnings del proyecto.
#              Aplica filtros para avisos conocidos (pin_memory,
#              MIOpen, lr_scheduler) y un wrapper idempotente
#              sobre `warnings.showwarning` para limpiar la
#              salida de consola en entrenamiento/validación.
#==============================================================

from __future__ import annotations

import warnings
from typing import Callable, Optional

__all__ = ["configure_warnings"]


# Estado interno para evitar envolver múltiples veces showwarning
_original_showwarning: Optional[Callable[..., None]] = None
_installed: bool = False


def configure_warnings(force: bool = False) -> None:
    """Configura filtros y manejador global de warnings para YOLOv11.

    - Filtra el warning cosmético de `pin_memory` relacionado con
      `set_num_threads`.
    - Reescribe mensajes ruidosos de MIOpen y del scheduler LR en
      un formato compacto y entendible.
    - Es idempotente: si ya fue instalado, no vuelve a envolver
      `warnings.showwarning` a menos que `force=True`.
    """
    global _installed, _original_showwarning

    if _installed and not force:
        return

    # Guardar handler original sólo la primera vez
    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning

    # Filtro específico para warning interno de pin_memory (cosmético, no funcional)
    warnings.filterwarnings(
        "ignore",
        message=(
            r".*Cannot set number of intraop threads after parallel work has started "
            r"or after set_num_threads call when using native parallel backend.*"
        ),
        category=UserWarning,
        module=r"torch\.utils\.data\._utils\.pin_memory",
    )

    def _y11_showwarning(message, category, filename, lineno, file=None, line=None):  # type: ignore[override]
        text = str(message)

        # MIOpen: elapsed time inválido (ruido interno de backend)
        if "Invalid elapsed time detected in EvaluateInvokers" in text:
            print("[Warning]: MIOpen elapsed", flush=True)
            return

        # Scheduler LR: orden de llamadas step (cosmético, no funcional)
        if "Detected call of `lr_scheduler.step()` before `optimizer.step()`" in text:
            print("[Warning]: LRSched orden step", flush=True)
            return

        # Fallback al comportamiento original
        if _original_showwarning is not None:
            _original_showwarning(message, category, filename, lineno, file=file, line=line)
        else:  # pragma: no cover - caso extremadamente raro
            try:
                warnings.showwarning(message, category, filename, lineno, file=file, line=line)
            except Exception:
                # Si todo falla, al menos imprimir algo razonable
                print(f"WARNING: {category.__name__}: {message}")

    warnings.showwarning = _y11_showwarning  # type: ignore[assignment]
    _installed = True


# Instalación por defecto al importar el módulo (se puede repetir en train.py)
configure_warnings(force=False)
