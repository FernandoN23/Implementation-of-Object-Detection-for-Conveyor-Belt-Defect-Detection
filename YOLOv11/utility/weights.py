"""
weights.py
---------------------------------
Gestión de checkpoints (guardado y carga de pesos del modelo YOLOv11).
Permite reanudar entrenamiento, guardar mejores modelos y manejar directorios.
"""

import os
import torch
import glob

def save_checkpoint(model, optimizer, epoch, path="checkpoints", filename=None):
    os.makedirs(path, exist_ok=True)
    if filename is None:
        filename = f"yolo11_epoch_{epoch}.pt"
    save_path = os.path.join(path, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)

    # ✅ Guardar también como "latest.pt" para continuar automáticamente
    latest_path = os.path.join(path, "latest.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, latest_path)

    print(f"✅ Checkpoint guardado en {save_path} y actualizado latest.pt")


def load_checkpoint(model, optimizer=None, path="checkpoints", device="cpu"):
    """Carga el último checkpoint disponible (latest o el más reciente numérico)."""
    ckpt_path = None

    # 1️⃣ Si existe latest.pt → usarlo
    latest = os.path.join(path, "latest.pt")
    if os.path.exists(latest):
        ckpt_path = latest
    else:
        # 2️⃣ Buscar el último por número de época
        ckpts = sorted(glob.glob(os.path.join(path, "yolo11_epoch_*.pt")))
        if ckpts:
            ckpt_path = ckpts[-1]

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ No se encontró ningún checkpoint en {path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"🔄 Checkpoint cargado desde {ckpt_path}, epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]
