# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/metrics.py
# Descripción: Utilidades para visualización de historial de entrenamiento.
#              Procesa results.csv y genera curvas organizadas en carpetas.
# ==============================================================

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración de estilo global similar a YOLOv5
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "seaborn-whitegrid")

# Paleta de colores estándar YOLO
COLOR_TRAIN = '#1f77b4'  # Azul
COLOR_VAL = '#ff7f0e'  # Naranja
COLOR_MAP50 = '#1f77b4'
COLOR_MAP95 = '#ff7f0e'


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Suaviza la curva y usando una ventana de convolución (Box filter)."""
    if len(y) == 0:
        return y
    nf = round(len(y) * f * 2) // 2 * 2 + 1
    nf = max(nf, 1)

    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def plot_results(file: str | Path = "results.csv", save_dir: str | Path = "", model_name: str = "SSD300") -> None:
    """
    Lee results.csv y genera gráficos organizados con estilo YOLO.
    """
    file = Path(file)
    save_dir = Path(save_dir) if save_dir else file.parent

    losses_dir = save_dir / "losses"
    components_dir = losses_dir / "components"
    metrics_dir = save_dir / "metrics_history"

    losses_dir.mkdir(parents=True, exist_ok=True)
    components_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not file.is_file():
        print(f"[SSD/utility] Advertencia: No se encontró {file}")
        return

    try:
        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"[SSD/utility] Error leyendo CSV: {e}")
        return

    epochs = df["epoch"].values

    if "val_mAP_0.5_0.95" in df.columns:
        df = df.rename(columns={"val_mAP_0.5_0.95": "val_mAP_0.95"})

    # -------------------------------------------------------------------------
    # 1. Gráficos de Pérdidas (Losses)
    # -------------------------------------------------------------------------

    # 1.1 Total Loss (Train vs Val)
    if "train_loss_total" in df.columns and "val_loss_total" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

        y_train = df["train_loss_total"].values
        y_val = df["val_loss_total"].values

        ax.plot(epochs, y_train, color=COLOR_TRAIN, alpha=0.3, linewidth=1)
        ax.plot(epochs, smooth(y_train), color=COLOR_TRAIN, linewidth=2, label="Train mean loss")

        ax.plot(epochs, y_val, color=COLOR_VAL, alpha=0.3, linewidth=1)
        ax.plot(epochs, smooth(y_val), color=COLOR_VAL, linewidth=2, label="Val mean loss")

        ax.set_title(f"Loss curves | {model_name}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean loss (box, obj, cls)")  # Etiqueta genérica estilo YOLO
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.savefig(losses_dir / "total_loss.png", dpi=300)
        plt.close(fig)
        print(f"[SSD/utility] Guardado: {losses_dir / 'total_loss.png'}")

    # 1.2 Train Components (Loc vs Conf)
    if "train_loss_loc" in df.columns and "train_loss_conf" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

        y_loc = df["train_loss_loc"].values
        y_conf = df["train_loss_conf"].values

        ax.plot(epochs, smooth(y_loc), color=COLOR_TRAIN, linewidth=2, label="train loc (box)")
        ax.plot(epochs, smooth(y_conf), color=COLOR_VAL, linewidth=2, label="train conf (cls)")

        ax.set_title(f"Train loss components | {model_name}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.savefig(components_dir / "train_loss_components.png", dpi=300)
        plt.close(fig)
        print(f"[SSD/utility] Guardado: {components_dir / 'train_loss_components.png'}")

    # 1.3 Val Components (Loc vs Conf)
    if "val_loss_loc" in df.columns and "val_loss_conf" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

        y_loc = df["val_loss_loc"].values
        y_conf = df["val_loss_conf"].values

        ax.plot(epochs, smooth(y_loc), color=COLOR_TRAIN, linewidth=2, label="val loc (box)")
        ax.plot(epochs, smooth(y_conf), color=COLOR_VAL, linewidth=2, label="val conf (cls)")

        ax.set_title(f"Val loss components | {model_name}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.savefig(components_dir / "val_loss_components.png", dpi=300)
        plt.close(fig)
        print(f"[SSD/utility] Guardado: {components_dir / 'val_loss_components.png'}")

    # -------------------------------------------------------------------------
    # 2. Gráficos de Métricas Históricas (Metrics History)
    # -------------------------------------------------------------------------

    metric_cols = ["val_mAP_0.5", "val_mAP_0.95", "val_P", "val_R", "val_F1"]
    if all(c in df.columns for c in metric_cols):

        # 2.1 mAP Curves
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        ax.plot(epochs, smooth(df["val_mAP_0.5"].values), color=COLOR_MAP50, linewidth=2, label="mAP@0.5")
        ax.plot(epochs, smooth(df["val_mAP_0.95"].values), color=COLOR_MAP95, linewidth=2, label="mAP@0.5:0.95")

        ax.set_title(f"mAP curves | {model_name}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.savefig(metrics_dir / "map_curves.png", dpi=300)
        plt.close(fig)
        print(f"[SSD/utility] Guardado: {metrics_dir / 'map_curves.png'}")

        # 2.2 Precision, Recall, F1 (Separados para claridad o juntos)
        # Aquí los graficamos juntos pero con colores consistentes
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        ax.plot(epochs, smooth(df["val_P"].values), color=COLOR_TRAIN, linewidth=2, label="Precision")
        ax.plot(epochs, smooth(df["val_R"].values), color=COLOR_VAL, linewidth=2, label="Recall")
        ax.plot(epochs, smooth(df["val_F1"].values), color='tab:green', linewidth=2,
                label="F1")  # F1 en verde para distinguir

        ax.set_title(f"Precision, Recall, F1 | {model_name}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.savefig(metrics_dir / "prf1_curves.png", dpi=300)
        plt.close(fig)
        print(f"[SSD/utility] Guardado: {metrics_dir / 'prf1_curves.png'}")

    else:
        print("[SSD/utility] Nota: No se encontraron columnas de métricas históricas completas.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="results.csv", help="Ruta al archivo results.csv")
    parser.add_argument("--dir", type=str, default="", help="Directorio de salida (opcional)")
    parser.add_argument("--name", type=str, default="SSD300", help="Nombre del modelo para títulos")
    args = parser.parse_args()

    plot_results(args.file, args.dir, args.name)