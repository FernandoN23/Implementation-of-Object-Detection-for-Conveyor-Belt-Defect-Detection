# -*- coding: utf-8 -*-
"""
Test de integración PyTorch + AMD ROCm + TensorBoard
-----------------------------------------------------
1. Detecta backend GPU (CUDA/ROCm/xpu/DML o CPU fallback)
2. Ejecuta una operación tensorial de prueba
3. Registra datos ficticios en TensorBoard para verificar logging
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import platform

print("=" * 70)
print("Prueba de entorno PyTorch + AMD ROCm + TensorBoard")
print("=" * 70)
print(f"Sistema operativo : {platform.system()} {platform.release()}")
print(f"Versión de Python  : {platform.python_version()}")
print(f"Versión de PyTorch : {torch.__version__}")
print("-" * 70)

# ------------------------------------------------------------
# 1️⃣ Detección del backend activo
# ------------------------------------------------------------
cuda_ok = torch.cuda.is_available()
xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
dml_ok = getattr(torch, "_dml_available", False)
hip_ver = getattr(torch.version, "hip", None)

if cuda_ok:
    device = torch.device("cuda")
    print("✅ Backend activo: CUDA (ROCm emulado en Windows Preview)")
    print("Nombre del dispositivo :", torch.cuda.get_device_name(0))
elif xpu_ok:
    device = torch.device("xpu")
    print("✅ Backend activo: XPU (AMD ROCm/DirectML unificado)")
    try:
        print("Nombre del dispositivo :", torch.xpu.get_device_name(0))
    except Exception:
        print("Nombre del dispositivo : (no disponible)")
elif dml_ok:
    device = torch.device("dml")
    print("✅ Backend activo: DirectML (fallback con PyTorch Preview)")
else:
    device = torch.device("cpu")
    print("⚠️  No se detectó GPU compatible, usando CPU.")

print("-" * 70)

# ------------------------------------------------------------
# 2️⃣ Operación de prueba en el dispositivo detectado
# ------------------------------------------------------------
try:
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f"Operación matricial realizada en: {z.device}")
except Exception as e:
    print(f"❌ Error al ejecutar operación en {device}: {e}")

print("-" * 70)

# ------------------------------------------------------------
# 3️⃣ TensorBoard Test
# ------------------------------------------------------------
log_dir = Path("runs/test_tensorboard_amd")
log_dir.mkdir(parents=True, exist_ok=True)

print(f"Inicializando SummaryWriter en: {log_dir.resolve()}")
writer = SummaryWriter(log_dir=str(log_dir))

print("Registrando 20 valores de ejemplo en TensorBoard...")
for n_iter in range(20):
    loss = 1.0 / (n_iter + 1)
    acc = n_iter / 20.0
    writer.add_scalar("Loss/train", loss, n_iter)
    writer.add_scalar("Accuracy/train", acc, n_iter)
writer.close()

print("\n✅ Logs creados correctamente.")
print("Para visualizar en TensorBoard, ejecuta en otra terminal:")
print(f"tensorboard --logdir={log_dir}")
print("\n📈 Actualización en tiempo real (cada 5 s):")
print(f"tensorboard --logdir={log_dir} --reload_interval 5")
print("Luego abre: http://localhost:6006")
print("-" * 70)
print(f"torch.version.hip: {hip_ver}")
print("=" * 70)
time.sleep(1)
