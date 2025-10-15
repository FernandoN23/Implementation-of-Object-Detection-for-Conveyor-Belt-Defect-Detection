"""
test_model.py
---------------------------------
Script de testeo rápido para verificar la correcta ejecución de un modelo YOLOv11.
Permite comprobar estructura, dimensiones y forward pass sin errores.
"""

import torch
from models.yolo11 import YOLOv11  # Asegúrate que el modelo esté accesible
from utility.metrics import measure_fps

def test_model(device="cpu", input_size=(1, 3, 640, 640)):
    print("🚀 Probando modelo YOLOv11...")
    model = YOLOv11(num_classes=80).to(device)

    dummy_input = torch.randn(input_size).to(device)
    output = model(dummy_input)

    print(f"✅ Forward pass exitoso | Output shape: {output.shape}")

    fps = measure_fps(model, dummy_input, device=device)
    print(f"⚡ Rendimiento estimado: {fps:.2f} FPS")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model(device)
