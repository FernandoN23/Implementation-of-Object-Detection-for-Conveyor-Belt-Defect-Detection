# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/engine/warnings.py
# Descripción: Filtros globales de advertencias ruidosas.
#              Optimizado para PyTorch 2.9.1 y backend MIOpen.
# ==============================================================

import warnings
import logging


def install_global_warning_filters():
    """Configura filtros para silenciar advertencias cosméticas de ROCm/Torch."""

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data._utils.pin_memory")
    warnings.filterwarnings("once", category=FutureWarning, message=".*torch.cuda.amp.*")
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step().*before.*optimizer.step().*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF is deprecated.*")
    warnings.filterwarnings("ignore", message=".*IsEnoughWorkspace.*")
    warnings.filterwarnings("ignore", message=".*GetSolutionsFallback.*")
    warnings.filterwarnings("ignore", message=".*Solver <.*>.*")
    warnings.filterwarnings("ignore", message=".*expandable_segments not supported on this platform.*")

    class MIOpenNoiseFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if "Invalid elapsed time detected" in msg:
                return False
            if "provided ptr: 0000000000000000 size: 0" in msg:
                return False
            if "AllocatorConfig.cpp" in msg or "PYTORCH_ALLOC_CONF" in msg:
                return False
            return True

    logging.getLogger().addFilter(MIOpenNoiseFilter())

    print("[warnings] Filtros de advertencias instalados correctamente.")