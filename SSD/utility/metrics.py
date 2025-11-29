# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/metrics.py
# Descripción: Utilidades para visualización de historial de entrenamiento.
#              Procesa results.csv y genera curvas de Loss/LR estilo YOLO.
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


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Suaviza la curva y usando una ventana de convolución (Box filter)."""
    if len(y) == 0:
        return y
    nf = round(len(y) * f * 2) // 2 + 1  # número impar
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def plot_results(file: str | Path = "results.csv", save_dir: str | Path = "") -> None:
    """
    Lee results.csv y genera gráficos de evolución de entrenamiento.
    Genera:
      1. results.png: Grid con Loc, Conf, Total Loss (Train/Val) y LR.
      2. val_loss_components.png: Comparativa de componentes de Val Loss.
    """
    file = Path(file)
    save_dir = Path(save_dir) if save_dir else file.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if not file.is_file():
        print(f"[SSD/utility] Advertencia: No se encontró {file}")
        return

    # Leer CSV usando pandas para robustez
    try:
        df = pd.read_csv(file)
        # Limpiar espacios en nombres de columnas
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"[SSD/utility] Error leyendo CSV: {e}")
        return

    # Verificar columnas esperadas
    required_cols = ["epoch", "train_loss_loc", "train_loss_conf", "val_loss_loc", "val_loss_conf"]
    if not all(c in df.columns for c in required_cols):
        print(f"[SSD/utility] El CSV no tiene las columnas esperadas: {required_cols}")
        print(f"Columnas encontradas: {df.columns.tolist()}")
        return

    epochs = df["epoch"].values

    # -------------------------------------------------------------------------
    # Gráfico 1: Grid de Resultados (Train vs Val + LR)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    ax = ax.ravel()

    # Definición de métricas a plotear
    # (Título, Columna Train, Columna Val)
    metrics_map = [
        ("Localization Loss", "train_loss_loc", "val_loss_loc"),
        ("Confidence Loss", "train_loss_conf", "val_loss_conf"),
        ("Total Loss", "train_loss_total", "val_loss_total"),
        ("Learning Rate", "lr", None),
    ]

    for i, (title, train_col, val_col) in enumerate(metrics_map):
        if train_col not in df.columns:
            continue

        y_train = df[train_col].values
        # Plot Train (Original tenue + Suavizado fuerte)
        ax[i].plot(epochs, y_train, color="tab:blue", alpha=0.3, linewidth=1)
        ax[i].plot(epochs, smooth(y_train), color="tab:blue", linewidth=2, label="Train")

        if val_col and val_col in df.columns:
            y_val = df[val_col].values
            # Plot Val (Original tenue + Suavizado fuerte)
            ax[i].plot(epochs, y_val, color="tab:orange", alpha=0.3, linewidth=1)
            ax[i].plot(epochs, smooth(y_val), color="tab:orange", linewidth=2, label="Val")

        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch")
        ax[i].grid(True, linestyle="--", alpha=0.5)

        if i == 3:  # LR scale logarítmica a veces es útil, pero lineal está bien si hay steps
            ax[i].set_ylabel("LR")
        else:
            ax[i].set_ylabel("Loss")
            ax[i].legend()

    fig.savefig(save_dir / "results.png", dpi=300)
    plt.close(fig)
    print(f"[SSD/utility] Gráfico guardado: {save_dir / 'results.png'}")

    # -------------------------------------------------------------------------
    # Gráfico 2: Componentes de Val Loss (Estilo input_file_15.png)
    # -------------------------------------------------------------------------
    fig_comp, ax_comp = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)

    # SSD tiene Loc (Box) y Conf (Cls + Obj implícito)
    # Mapeamos a nombres estilo YOLO para consistencia visual
    val_loc = df["val_loss_loc"].values
    val_conf = df["val_loss_conf"].values

    # Plot Loc
    ax_comp.plot(epochs, smooth(val_loc), color="tab:blue", linewidth=2, label="val loc (box)")
    # Plot Conf
    ax_comp.plot(epochs, smooth(val_conf), color="tab:orange", linewidth=2, label="val conf (cls)")

    # Si existiera Total, lo podríamos poner, pero suele ensuciar la escala si es la suma
    # ax_comp.plot(epochs, smooth(df["val_loss_total"].values), color="tab:green", linewidth=2, label="val total")

    ax_comp.set_title("Val loss components | SSD300", fontsize=14)
    ax_comp.set_xlabel("Epoch", fontsize=12)
    ax_comp.set_ylabel("Loss", fontsize=12)
    ax_comp.legend(fontsize=12)
    ax_comp.grid(True, linestyle="--", alpha=0.5)

    fig_comp.savefig(save_dir / "val_loss_components.png", dpi=300)
    plt.close(fig_comp)
    print(f"[SSD/utility] Gráfico guardado: {save_dir / 'val_loss_components.png'}")


if __name__ == "__main__":
    # Bloque de prueba manual
    # Se puede ejecutar: python SSD/utility/metrics.py --file SSD/metrics/.../results.csv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="results.csv", help="Ruta al archivo results.csv")
    parser.add_argument("--dir", type=str, default="", help="Directorio de salida (opcional)")
    args = parser.parse_args()

    plot_results(args.file, args.dir)