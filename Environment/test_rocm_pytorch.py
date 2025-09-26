# -*- coding: utf-8 -*-
"""
Script de prueba para verificar:
1. Que PyTorch (compilado con ROCm) detecta y usa tu GPU AMD en Windows (ver administrador de tareas).
2. Que TensorBoard recibe datos desde PyTorch sin romper compatibilidad.

Requisitos previos:
pip install tensorboard==2.17.0 torch-tb-profiler
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import time

print("="*60)
print("Prueba PyTorch + ROCm + TensorBoard")
print("="*60)

# --- SECCIÓN 1: Información sobre PyTorch y GPU ---
print("Versión de PyTorch:", torch.__version__)
# Verifica si PyTorch cree que hay un dispositivo tipo CUDA (en ROCm también responde True)
print("torch.cuda.is_available():", torch.cuda.is_available())
# Si PyTorch se compiló con ROCm, este atributo indica la versión de HIP/ROCm
print("torch.version.hip:", getattr(torch.version, "hip", None))

# Seleccionamos dispositivo: GPU si hay, CPU si no
if torch.cuda.is_available():
    device = torch.device("cuda")  # En ROCm se expone como 'cuda'
    print("ID de dispositivo actual:", torch.cuda.current_device())
    print("Nombre del dispositivo:", torch.cuda.get_device_name(0))

    # --- Operación de prueba en GPU ---
    # Creamos dos tensores grandes directamente en GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    # Multiplicación matricial para forzar cálculo en GPU
    z = torch.matmul(x, y)
    # Imprimimos el dispositivo donde está el resultado
    print("Operación matricial realizada en:", z.device)
else:
    # Si no hay GPU compatible, usamos CPU para no fallar
    device = torch.device("cpu")
    print("No se detectó GPU compatible con ROCm/CUDA. Usando CPU.")

# --- SECCIÓN 2: Integración con TensorBoard ---
print("\nInicializando SummaryWriter de TensorBoard...")
# SummaryWriter crea un directorio 'runs/' donde guardará los logs
writer = SummaryWriter()

print("Registrando 20 valores de ejemplo en TensorBoard...")
# Este bucle simula entrenamiento: pérdida decrece, accuracy crece
for n_iter in range(20):
    loss = 1.0 / (n_iter + 1)     # pérdida decreciente
    acc = n_iter / 20.0           # exactitud creciente
    # Guardamos ambos valores como 'scalars' en TensorBoard
    writer.add_scalar("Loss/train", loss, n_iter)
    writer.add_scalar("Accuracy/train", acc, n_iter)
# Cerramos el writer para que se escriban los datos en disco
writer.close()

# --- INSTRUCCIONES AL USUARIO ---
print("\nListo. Para visualizar en TensorBoard, en otra terminal ejecuta:")
print("  tensorboard --logdir=runs")
print("\nListo. Para visualizar en TensorBoard con intervalos de actualización de 5s, en otra terminal ejecuta:")
print("  tensorboard --logdir=runs --reload_interval 5")
print("y abre el navegador en la URL indicada (por defecto http://localhost:6006).")
print("="*60)

# Pausa breve para asegurar que se impriman bien los mensajes antes de terminar
time.sleep(1)
