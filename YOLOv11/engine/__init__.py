# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/engine/__init__.py
# Descripción: Inicializador del paquete `engine` para YOLOv11.
#  Define la API pública del paquete y un mecanismo de importación
#  perezosa (lazy) para submódulos pesados (amp/optim/ema/etc.),
#  de forma que `torch` y dependencias GPU sólo se carguen después
#  del bootstrap ROCm/MIOpen realizado en train.py.
#==============================================================

from __future__ import annotations

import importlib
from typing import Any, Dict, List

# --------------------------------------------------------------
# API pública del paquete
# --------------------------------------------------------------
# Nota: `bootstrap_miopen` y `CLI` se exponen aquí sólo para
# comodidad (p.ej. import YOLOv11.engine as eng; eng.CLI). El
# flujo crítico de train.py sigue importándolos como submódulos:
#   from YOLOv11.engine.bootstrap_miopen import bootstrap, MIOpenConfig
#   from YOLOv11.engine.CLI import parse_args_two_stage
# Por diseño, este __init__ es *torch‑free*.

__all__ = [
    # submódulos perezosos principales
    "amp",
    "optim",
    "ema",
    "callbacks",
    "validator",
    "warmup.py",
    "utils",
    "bn2gn_patch",
    "hud",
    "warnings",
    "Trainer",
    # submódulos auxiliares (no necesariamente pesados)
    "bootstrap_miopen",
    "CLI",
]


# Mapa de submódulos a importar perezosamente cuando se acceden
# como atributos del paquete (PEP 562: __getattr__).
_LAZY_MODULES: Dict[str, str] = {
    # núcleo de entrenamiento
    "amp": ".amp",
    "optim": ".optim",
    "ema": ".ema",
    "callbacks": ".callbacks",
    "validator": ".validator",
    "warmup_sanity": ".warmup_sanity",
    "utils": ".utils",
    "bn2gn_patch": ".bn2gn_patch",
    "hud": ".hud",
    "warnings": ".warnings",
    "Trainer": ".Trainer",
    # utilidades torch‑free (seguimos usando lazy para evitar
    # efectos secundarios en importaciones tempranas)
    "bootstrap_miopen": ".bootstrap_miopen",
    "CLI": ".CLI",
}


def _load_submodule(name: str) -> Any:
    """Importa dinámicamente el submódulo `name` y lo cachea.

    Este helper garantiza que cada submódulo se importe sólo una vez
    y se almacene en `globals()` para accesos posteriores. No realiza
    ningún tipo de lógica adicional: la gestión de `torch`/GPU queda
    en cada submódulo concreto.
    """
    full = _LAZY_MODULES.get(name)
    if full is None:
        raise AttributeError(f"módulo '{__name__}' no expone atributo '{name}'")
    module = importlib.import_module(full, __name__)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - usado en tiempo de ejecución
    """Soporta acceso perezoso a submódulos del paquete.

    Ejemplos
    --------
    >>> from YOLOv11 import engine
    >>> engine.amp   # dispara import interno de YOLOv11.engine.amp
    """
    if name in _LAZY_MODULES:
        return _load_submodule(name)
    raise AttributeError(f"módulo '{__name__}' no tiene atributo '{name}'")


def __dir__() -> List[str]:  # pragma: no cover - función de introspección
    """Incluye los símbolos públicos en `dir(engine)` sin importarlos.

    Esto mejora la autocompletación en IDEs (PyCharm/VSCode) sin
    forzar la carga de `torch` ni de los submódulos pesados.
    """
    std = set(globals().keys())
    std.update(__all__)
    return sorted(std)
