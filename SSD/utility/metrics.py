# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/metrics.py
# Descripción: Utilidades para visualización de historial de entrenamiento.
#              Procesa results.csv y genera curvas de Loss/LR/mAP estilo YOLO.
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
      3. historical_metrics.png: Curvas de mAP, P, R, F1 vs Epoch.
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
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"[SSD/utility] Error leyendo CSV: {e}")
        return

    epochs = df["epoch"].values

    # Columnas de métricas de detección
    metric_cols = ["val_mAP_0.5", "val_mAP_0.95", "val_P", "val_R", "val_F1"]
    has_metrics = all(c in df.columns for c in metric_cols)

    # Renombrar mAP_0.5_0.95 si existe
    if "val_mAP_0.5_0.95" in df.columns:
        df = df.rename(columns={"val_mAP_0.5_0.95": "val_mAP_0.95"})

    # -------------------------------------------------------------------------
    # Gráfico 1: Grid de Resultados (Loss + LR)
    # -------------------------------------------------------------------------

    # Verificar columnas de Loss
    loss_cols = ["train_loss_loc", "train_loss_conf", "val_loss_loc", "val_loss_conf", "train_loss_total",
                 "val_loss_total", "lr"]
    if all(c in df.columns for c in loss_cols):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
        ax = ax.ravel()

        metrics_map = [
            ("Localization Loss", "train_loss_loc", "val_loss_loc"),
            ("Confidence Loss", "train_loss_conf", "val_loss_conf"),
            ("Total Loss", "train_loss_total", "val_loss_total"),
            ("Learning Rate", "lr", None),
        ]

        for i, (title, train_col, val_col) in enumerate(metrics_map):
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

            if i == 3:
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

        val_loc = df["val_loss_loc"].values
        val_conf = df["val_loss_conf"].values

        ax_comp.plot(epochs, smooth(val_loc), color="tab:blue", linewidth=2, label="val loc (box)")
        ax_comp.plot(epochs, smooth(val_conf), color="tab:orange", linewidth=2, label="val conf (cls)")

        ax_comp.set_title("Val loss components | SSD300", fontsize=14)
        ax_comp.set_xlabel("Epoch", fontsize=12)
        ax_comp.set_ylabel("Loss", fontsize=12)
        ax_comp.legend(fontsize=12)
        ax_comp.grid(True, linestyle="--", alpha=0.5)

        fig_comp.savefig(save_dir / "val_loss_components.png", dpi=300)
        plt.close(fig_comp)
        print(f"[SSD/utility] Gráfico guardado: {save_dir / 'val_loss_components.png'}")
    else:
        print("[SSD/utility] Advertencia: Columnas de Loss incompletas. Saltando gráficos de Loss.")

    # -------------------------------------------------------------------------
    # Gráfico 3: Métricas Históricas (mAP, P, R, F1 vs Epoch)
    # -------------------------------------------------------------------------
    if has_metrics:
        fig_hist, ax_hist = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
        ax_hist = ax_hist.ravel()

        # 3.1 mAP Curves (Estilo input_file_20.png)
        ax_hist[0].plot(epochs, smooth(df["val_mAP_0.5"].values), linewidth=2, label="mAP@0.5")
        ax_hist[0].plot(epochs, smooth(df["val_mAP_0.95"].values), linewidth=2, label="mAP@0.5:0.95")
        ax_hist[0].set_title("mAP curves | SSD300", fontsize=14)
        ax_hist[0].set_xlabel("Epoch")
        ax_hist[0].set_ylabel("mAP")
        ax_hist[0].set_ylim(0, 1.05)
        ax_hist[0].legend()
        ax_hist[0].grid(True, linestyle="--", alpha=0.5)

        # 3.2 Precision (P)
        ax_hist[1].plot(epochs, smooth(df["val_P"].values), linewidth=2, color='tab:green')
        ax_hist[1].set_title("Precision (P) | SSD300")
        ax_hist[1].set_xlabel("Epoch")
        ax_hist[1].set_ylabel("Precision")
        ax_hist[1].set_ylim(0, 1.05)
        ax_hist[1].grid(True, linestyle="--", alpha=0.5)

        # 3.3 Recall (R)
        ax_hist[2].plot(epochs, smooth(df["val_R"].values), linewidth=2, color='tab:red')
        ax_hist[2].set_title("Recall (R) | SSD300")
        ax_hist[2].set_xlabel("Epoch")
        ax_hist[2].set_ylabel("Recall")
        ax_hist[2].set_ylim(0, 1.05)
        ax_hist[2].grid(True, linestyle="--", alpha=0.5)

        # 3.4 F1 Score
        ax_hist[3].plot(epochs, smooth(df["val_F1"].values), linewidth=2, color='tab:purple')
        ax_hist[3].set_title("F1 Score | SSD300")
        ax_hist[3].set_xlabel("Epoch")
        ax_hist[3].set_ylabel("F1")
        ax_hist[3].set_ylim(0, 1.05)
        ax_hist[3].grid(True, linestyle="--", alpha=0.5)

        fig_hist.savefig(save_dir / "historical_metrics.png", dpi=300)
        plt.close(fig_hist)
        print(f"[SSD/utility] Gráfico guardado: {save_dir / 'historical_metrics.png'}")
    else:
        print("[SSD/utility] Advertencia: Columnas de métricas de detección incompletas. Saltando gráficos históricos.")


if __name__ == "__main__":
    # Bloque de prueba manual
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="results.csv", help="Ruta al archivo results.csv")
    parser.add_argument("--dir", type=str, default="", help="Directorio de salida (opcional)")
    args = parser.parse_args()

    plot_results(args.file, args.dir)