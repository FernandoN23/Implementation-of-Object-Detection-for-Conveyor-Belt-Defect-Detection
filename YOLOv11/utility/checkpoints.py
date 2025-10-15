"""
checkpoints.py
---------------------------------
Gestión de checkpoints (guardado y carga de pesos del modelo YOLOv11).
Permite reanudar entrenamiento, guardar mejores modelos y manejar directorios.
"""

import os
import torch

def save_checkpoint(model, optimizer, epoch, path="checkpoints", filename="yolo11_epoch.pt"):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    print(f"✅ Checkpoint guardado en {save_path}")


def load_checkpoint(model, optimizer=None, path="checkpoints/yolo11_epoch.pt", device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró el checkpoint en {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"🔄 Checkpoint cargado desde {path}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch']
