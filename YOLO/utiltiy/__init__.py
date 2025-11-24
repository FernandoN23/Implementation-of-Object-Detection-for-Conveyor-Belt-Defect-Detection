# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/utility/__init__.py
# Descripción: Inicializador del paquete `utility` para YOLOv11.
#  Define la API pública de utilidades (data_loader, logger, losses,
#  weights, etc.) y un mecanismo de importación perezosa (lazy) para
#  evitar cargas pesadas innecesarias y mantener el paquete torch‑free.
#==============================================================

from __future__ import annotations

import importlib
from typing import Any, Dict, List

# --------------------------------------------------------------
# API pública del paquete
# --------------------------------------------------------------
# Se exponen únicamente los módulos de biblioteca. Los cleaners y
# scripts de test permanecen accesibles como submódulos directos:
#   python -m YOLOv11.utility.clean_logs_runs
# pero no forman parte de la API oficial (`__all__`).

__all__ = [
    "check_dataset",
    "data_loader",
    "logger",
    "losses",
    "metrics",
    "visualization",
    "weights",
]


# Módulos que se importan de forma perezosa cuando se acceden como
# atributos del paquete (PEP 562: __getattr__).
_LAZY_MODULES: Dict[str, str] = {
    "check_dataset": ".check_dataset",
    "data_loader": ".data_loader",
    "logger": ".logger",
    "losses": ".losses",
    "metrics": ".metrics",
    "visualization": ".visualization",
    "weights": ".weights",
}


def _load_submodule(name: str) -> Any:
    """Importa dinámicamente el submódulo `name` y lo cachea.

    Cada submódulo se importa sólo una vez y se almacena en `globals()`
    para accesos posteriores. Este inicializador permanece torch‑free;
    cualquier lógica de GPU/torch vive en los submódulos.
    """
    full = _LAZY_MODULES.get(name)
    if full is None:
        raise AttributeError(f"módulo '{__name__}' no expone atributo '{name}'")
    module = importlib.import_module(full, __name__)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Soporta acceso perezoso a submódulos del paquete.

    Ejemplos
    --------
    >>> from YOLOv11 import utility
    >>> utility.data_loader  # dispara import interno de YOLOv11.utility.data_loader
    """
    if name in _LAZY_MODULES:
        return _load_submodule(name)
    raise AttributeError(f"módulo '{__name__}' no tiene atributo '{name}'")


def __dir__() -> List[str]:  # pragma: no cover
    """Incluye los símbolos públicos en `dir(utility)` sin importarlos.

    Mejora la autocompletación en IDEs (PyCharm/VSCode) sin forzar la
    carga de submódulos pesados.
    """
    std = set(globals().keys())
    std.update(__all__)
    return sorted(std)
