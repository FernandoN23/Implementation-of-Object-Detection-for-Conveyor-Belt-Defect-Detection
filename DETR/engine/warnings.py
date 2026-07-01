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

    # 2. Advertencias de Scheduler y AMP
    warnings.filterwarnings("once", category=FutureWarning, message=".*torch.cuda.amp.*")
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step().*before.*optimizer.step().*")

    # 3. Advertencias de obsolescencia de torchvision y variables de entorno
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF is deprecated.*")

    # 4. Filtros para ruido de MIOpen (Workspace y Solvers)
    warnings.filterwarnings("ignore", message=".*IsEnoughWorkspace.*")
    warnings.filterwarnings("ignore", message=".*GetSolutionsFallback.*")
    warnings.filterwarnings("ignore", message=".*Solver <.*>.*")

    # 5. [NUEVO] Filtro para expandable_segments no soportado en Windows/ROCm
    warnings.filterwarnings("ignore", message=".*expandable_segments not supported on this platform.*")

    # 6. Filtro de logs de MIOpen y Allocator (ruido C++)
    class MIOpenNoiseFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Filtrar mensajes de tiempo inválido, punteros nulos y avisos de configuración de memoria
            if "Invalid elapsed time detected" in msg:
                return False
            if "provided ptr: 0000000000000000 size: 0" in msg:
                return False
            if "AllocatorConfig.cpp" in msg or "PYTORCH_ALLOC_CONF" in msg:
                return False
            return True

    # Aplicar el filtro al root logger de Python
    logging.getLogger().addFilter(MIOpenNoiseFilter())

    print("[warnings] Filtros de advertencias instalados correctamente.")