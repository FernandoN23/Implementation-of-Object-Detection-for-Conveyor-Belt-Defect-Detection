# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/warnings.py
# Descripción: Filtros globales de advertencias ruidosas.
#              Optimizado para PyTorch 2.9.1 y backend MIOpen.
# ==============================================================

import warnings
import logging


def install_global_warning_filters():
    """Configura filtros para silenciar advertencias cosméticas de ROCm/Torch."""

    # 1. Advertencias de PyTorch (pin_memory y multiprocesamiento)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data._utils.pin_memory")

    # 2. Advertencias de Scheduler y AMP (frecuentes en ROCm)
    warnings.filterwarnings("once", category=FutureWarning, message=".*torch.cuda.amp.*")
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step().*before.*optimizer.step().*")

    # 3. Advertencias de obsolescencia de torchvision en DETR original
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

    # 4. [NUEVO] Filtros para ruido de MIOpen (Workspace y Solvers)
    warnings.filterwarnings("ignore", message=".*IsEnoughWorkspace.*")
    warnings.filterwarnings("ignore", message=".*GetSolutionsFallback.*")
    warnings.filterwarnings("ignore", message=".*Solver <.*>.*")

    # 5. Filtro de logs de MIOpen (errores de 'elapsed time' ruidosos)
    class MIOpenNoiseFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Filtrar mensajes de tiempo inválido y avisos de workspace
            if "Invalid elapsed time detected" in msg:
                return False
            if "provided ptr: 0000000000000000 size: 0" in msg:
                return False
            return True

    # Aplicar el filtro al root logger de Python
    logging.getLogger().addFilter(MIOpenNoiseFilter())

    print("[warnings] Filtros de advertencias instalados correctamente.")