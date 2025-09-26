# -*- coding: utf-8 -*-
"""
Script que:
1. Comprueba GPU AMD con PyTorch (ROCm).
2. Registra datos ficticios en TensorBoard para probar integración.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import time

print("="*60)
print("Prueba PyTorch + ROCm + TensorBoard")
print("="*60)

# --- Info PyTorch + GPU ---
print("Versión de PyTorch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.hip:", getattr(torch.version, "hip", None))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("ID de dispositivo actual:", torch.cuda.current_device())
    print("Nombre del dispositivo:", torch.cuda.get_device_name(0))
    # Operación de prueba en GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print("Operación matricial realizada en:", z.device)
else:
    device = torch.device("cpu")
    print("No se detectó GPU compatible con ROCm/CUDA. Usando CPU.")

# --- TensorBoard ---
print("\nInicializando SummaryWriter de TensorBoard...")
writer = SummaryWriter()  # logs en carpeta 'runs/'

print("Registrando 20 valores de ejemplo en TensorBoard...")
for n_iter in range(20):
    loss = 1.0 / (n_iter + 1)
    acc = n_iter / 20.0
    writer.add_scalar("Loss/train", loss, n_iter)
    writer.add_scalar("Accuracy/train", acc, n_iter)
writer.close()

print("\nListo. Para visualizar en TensorBoard, en otra terminal ejecuta:")
print("tensorboard --logdir=runs")
print("\nNota: para que actualice cada 5s los datos, se debe ejecutar:")
print("tensorboard --logdir=runs --reload_interval 5")
print("Luego, abre el navegador en la URL indicada (por defecto http://localhost:6006).")
print("="*60)
time.sleep(1)
