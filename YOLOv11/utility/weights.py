# -*- coding: utf-8 -*-
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: weights.py
Manejo de weights del modelo YOLOv11.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Funciones principales:
#   - save_checkpoint(): guarda pesos y estado del optimizador.
#   - load_checkpoint(): restaura entrenamiento desde último punto.
#
# Características:
#   • Guarda 'latest.pt' automáticamente para reanudación rápida.
#   • Permite manejo de múltiples weights por época.
#
# Conexión:
#   Usado directamente por train.py durante entrenamiento
#   y recuperación de pesos para validación o testeo.
# -------------------------------------------------------------

import os
import torch
import glob


def save_checkpoint(model, optimizer, epoch, path="weights", filename=None):
    """Guarda el modelo y el optimizador en un checkpoint."""
    os.makedirs(path, exist_ok=True)
    if filename is None:
        filename = f"yolo11_epoch_{epoch}.pt"
    save_path = os.path.join(path, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)

    # ✅ Guardar también como "latest.pt" para reanudación rápida
    latest_path = os.path.join(path, "latest.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, latest_path)

    print(f"✅ Checkpoint guardado en {save_path} y actualizado latest.pt")


def load_checkpoint(model, optimizer=None, path="weights", device="cpu"):
    """
    Carga un checkpoint de YOLOv11.
    - Si 'path' es una carpeta: busca latest.pt o el último por número de época.
    - Si 'path' es un archivo (.pt): carga directamente ese checkpoint.
    """
    # Si 'path' es un archivo .pt, cargar directamente
    if os.path.isfile(path) and path.endswith(".pt"):
        ckpt_path = path
    else:
        ckpt_path = None
        latest = os.path.join(path, "latest.pt")
        if os.path.exists(latest):
            ckpt_path = latest
        else:
            ckpts = sorted(glob.glob(os.path.join(path, "yolo11_epoch_*.pt")))
            if ckpts:
                ckpt_path = ckpts[-1]

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ No se encontró ningún checkpoint en {path}")

    # Cargar checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"🔄 Checkpoint cargado desde {ckpt_path}, epoch {epoch}")
    return epoch
