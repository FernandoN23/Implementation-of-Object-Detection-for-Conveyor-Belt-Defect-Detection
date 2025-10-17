"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: test_model.py
Prueba rápida del modelo YOLOv11 para verificar ejecución,
estructura y compatibilidad con GPU ROCm (AMD).
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Bloques principales:
#   - Parche de entorno MIOpen (evita errores ROCm en Radeon)
#   - _replace_batchnorm_with_identity(): parchea BatchNorm2d
#   - _try_gpu_then_cpu_forward(): prueba GPU y fallback a CPU
#   - test_model(): realiza forward con input simulado
#
# Conexión:
#   Script auxiliar para depuración y diagnóstico del modelo
#   antes del entrenamiento. No modifica pesos ni logs.
# -------------------------------------------------------------


# =====================================================
# 🔧 BLOQUE DE CONFIGURACIÓN AMD ROCm / MIOpen
# (debe ir ANTES de importar torch)
# =====================================================
import os

# Evita compilaciones problemáticas en GPUs Radeon integradas (gfx11+)
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["MIOPEN_DEBUG_CONV_FFT"] = "0"
os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "0"
os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "0"
os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "0"
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
# =====================================================

import sys
import traceback
import torch
import torch.nn as nn

# Desactiva backend MIOpen equivalente a cuDNN
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)

print(f"⚙️  Backends -> cudnn.enabled={torch.backends.cudnn.enabled}")

# =====================================================
# Añadir la raíz del proyecto al sys.path
# =====================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("DEBUG: sys.path añadido ->", sys.path[-1])
# =====================================================

from YOLOv11.models.yolo11 import YOLOv11


# =====================================================
# 🔩 Funciones auxiliares para manejo de errores ROCm
# =====================================================
def _replace_batchnorm_with_identity(model: nn.Module):
    """Reemplaza todas las capas BatchNorm2d por nn.Identity."""
    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
                replaced += 1
    return replaced


def _try_gpu_then_cpu_forward(model: nn.Module, dummy: torch.Tensor):
    """Ejecuta el forward en GPU; si falla, aplica parches y reintenta."""
    try:
        print("🚀 Probando forward en GPU (ROCm/MIOpen desactivado)...")
        model.eval()
        out = model(dummy)
        print("✅ Forward completado correctamente en GPU.")
        return out
    except Exception as e:
        msg = str(e)
        print("❌ Error en forward GPU.")
        traceback.print_exc(limit=2)

        # Detectar si es error típico de MIOpen/inline ASM
        if ("miopen" in msg.lower()) or ("inline asm" in msg.lower()) or ("buildocl" in msg.lower()):
            print("🩹 Detectado error MIOpen/ASM → reemplazando BatchNorm2d por Identity...")
            replaced = _replace_batchnorm_with_identity(model)
            print(f"   ↪ {replaced} capas BatchNorm2d reemplazadas.")
            try:
                model.eval()
                out = model(dummy)
                print("✅ Forward exitoso tras parche BatchNorm.")
                return out
            except Exception as e2:
                print("❌ Aún falla tras parche BatchNorm, migrando a CPU...")
                traceback.print_exc(limit=2)

        # Fallback final: CPU
        print("↩️  Cambiando a CPU (modo seguro)...")
        model_cpu = model.to("cpu").eval()
        dummy_cpu = dummy.to("cpu")
        out = model_cpu(dummy_cpu)
        print("✅ Forward completado en CPU.")
        return out


# =====================================================
# 🧪 Ejecución principal
# =====================================================
def test_model(device):
    print("🚀 Probando modelo YOLOv11...")
    model = YOLOv11().to(device).eval()
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    _ = _try_gpu_then_cpu_forward(model, dummy_input)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧩 Dispositivo actual:", device)
    test_model(device)


if __name__ == "__main__":
    main()
