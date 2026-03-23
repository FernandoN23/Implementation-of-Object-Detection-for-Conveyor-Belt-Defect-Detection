# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/metrics.py
# Descripción: Herramienta CLI para visualización y comparación
#              de métricas de entrenamiento DETR.
#              Procesa logs nativos (log.txt) y genera curvas de
#              pérdida, error de clase y mAP (COCO).
# ==============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración de estilo
try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    sns.set_palette("tab10")
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

# ---------------------------------------------------------------------------
# Definición de Rutas y Constantes
# ---------------------------------------------------------------------------

FILE = Path(__file__).resolve()
UTILITY_ROOT = FILE.parent
DETR_ROOT = UTILITY_ROOT.parent
RUNS_ROOT = DETR_ROOT / "runs"
METRICS_ROOT = DETR_ROOT / "metrics"

# Parámetros aproximados de DETR (ResNet Backbones) en Millones
DETR_PARAMS_M = {
    "r50": 41.3,
    "r50_dc5": 41.3,
    "r101": 60.1,
    "r101_dc5": 60.1,
}

VARIANT_COLORS = {
    "r50": "#1f77b4",  # Azul
    "r50_dc5": "#2ca02c",  # Verde
    "r101": "#d62728",  # Rojo
    "r101_dc5": "#9467bd",  # Púrpura
}

COLOR_TRAIN = "#1f77b4"
COLOR_VAL = "#ff7f0e"


# ---------------------------------------------------------------------------
# Configuración (Dataclass)
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    task_model: str = "detect"
    variant: str = "r50"
    train_run: str = ""
    merge_mode: bool = False
    variants_to_compare: List[str] = None  # type: ignore

    @property
    def train_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / self.variant / "train" / self.train_run

    @property
    def final_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / "global_comparison"


# ---------------------------------------------------------------------------
# Lógica de Carga de Datos (Parser JSON Lines)
# ---------------------------------------------------------------------------

def load_detr_log(path: Path) -> pd.DataFrame:
    """Lee log.txt y lo convierte en un DataFrame estructurado."""
    if not path.is_file():
        return pd.DataFrame()

    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Mapeo de métricas COCO (índices estándar de pycocotools)
    # [0]: mAP@0.5:0.95, [1]: mAP@0.5
    if 'test_coco_eval_bbox' in df.columns:
        df['val_map_50_95'] = df['test_coco_eval_bbox'].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
        df['val_map_50'] = df['test_coco_eval_bbox'].apply(lambda x: x[1] if isinstance(x, list) else np.nan)

    return df


def smooth_signal(scalars: List[float], weight: float = 0.6) -> List[float]:
    if not scalars: return []
    series = pd.Series(scalars).interpolate(limit_direction='both')
    clean_scalars = series.tolist()
    last = clean_scalars[0]
    smoothed = []
    for point in clean_scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ---------------------------------------------------------------------------
# Lógica de Ploteo
# ---------------------------------------------------------------------------

def plot_train_val_curve(df: pd.DataFrame, train_col: str, val_col: str, title: str, ylabel: str, out_path: Path):
    if df.empty: return
    plt.figure(figsize=(10, 6))

    for col, label, color in [(train_col, "Train", COLOR_TRAIN), (val_col, "Validation", COLOR_VAL)]:
        if col in df.columns:
            raw = df[col].values
            smoothed = smooth_signal(raw.tolist())
            plt.plot(df['epoch'], raw, color=color, alpha=0.2)
            plt.plot(df['epoch'], smoothed, label=label, color=color, linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_comparative(data_map: Dict[str, pd.DataFrame], col_name: str, title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(10, 6))
    for var, df in data_map.items():
        if col_name in df.columns:
            color = VARIANT_COLORS.get(var, "#808080")
            smoothed = smooth_signal(df[col_name].tolist(), weight=0.8)
            plt.plot(df['epoch'], smoothed, label=var.upper(), color=color, linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Modos de Ejecución
# ---------------------------------------------------------------------------

def run_single_mode(cfg: MetricsConfig):
    print(f"\n--- Procesando métricas DETR: {cfg.variant} / {cfg.train_run} ---")
    run_dir = RUNS_ROOT / cfg.variant / "train" / cfg.train_run
    log_path = run_dir / "log.txt"

    df = load_detr_log(log_path)
    if df.empty:
        print(f"[Error] No se encontró log válido en {log_path}")
        return

    out_dir = cfg.train_metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "losses").mkdir(exist_ok=True)

    # 1. Gráficos de Pérdida
    plot_train_val_curve(df, 'train_loss', 'test_loss', 'Total Loss', 'Loss', out_dir / "losses/loss_total.png")
    plot_train_val_curve(df, 'train_loss_ce', 'test_loss_ce', 'Classification Loss (CE)', 'Loss',
                         out_dir / "losses/loss_ce.png")
    plot_train_val_curve(df, 'train_loss_bbox', 'test_loss_bbox', 'BBox Loss (L1)', 'Loss',
                         out_dir / "losses/loss_bbox.png")

    # 2. Gráficos de Performance
    plot_train_val_curve(df, 'train_class_error', 'test_class_error', 'Classification Error', 'Error %',
                         out_dir / "class_error.png")

    if 'val_map_50' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['val_map_50'], label="mAP@0.5", linewidth=2)
        plt.plot(df['epoch'], df['val_map_50_95'], label="mAP@0.5:0.95", linewidth=2)
        plt.title("mAP Evolution (COCO)")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.legend()
        plt.savefig(out_dir / "map_curves.png", dpi=200)
        plt.close()

    print(f"✓ Reporte generado en: {out_dir}")


def run_comparison_mode(cfg: MetricsConfig):
    print("\n--- Iniciando Comparación Global DETR ---")
    variants = cfg.variants_to_compare or list(DETR_PARAMS_M.keys())
    data_map = {}

    for var in variants:
        base = RUNS_ROOT / var / "train"
        if not base.exists(): continue
        # Tomar el run más reciente de esa variante
        runs = sorted([d for d in base.iterdir() if d.is_dir()], key=os.path.getmtime)
        if not runs: continue
        df = load_detr_log(runs[-1] / "log.txt")
        if not df.empty:
            data_map[var] = df
            print(f"[Merge] Incluida variante '{var}': {runs[-1].name}")

    if not data_map: return

    out = cfg.final_metrics_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_comparative(data_map, 'test_loss', 'Comparative: Validation Loss', 'Loss', out / "compare_loss.png")
    plot_comparative(data_map, 'val_map_50_95', 'Comparative: mAP@0.5:0.95', 'mAP', out / "compare_map.png")

    # Trade-off Plot
    plt.figure(figsize=(9, 6))
    for var, df in data_map.items():
        best_map = df['val_map_50_95'].max()
        params = DETR_PARAMS_M.get(var, 0)
        plt.scatter(params, best_map, s=200, label=var.upper(), color=VARIANT_COLORS.get(var))
        plt.annotate(var.upper(), (params, best_map), xytext=(5, 5), textcoords='offset points', fontweight='bold')

    plt.xlabel("Parámetros (Millones)")
    plt.ylabel("Best mAP@0.5:0.95")
    plt.title("Trade-off: Performance vs Complejidad (DETR)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out / "tradeoff_detr.png", dpi=200)
    plt.close()

    print(f"✓ Comparación global finalizada en: {out}")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="r50")
    parser.add_argument("--run", default="")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    cfg = MetricsConfig(variant=args.variant, train_run=args.run, merge_mode=args.merge)

    if cfg.merge_mode:
        run_comparison_mode(cfg)
    else:
        if not cfg.train_run:
            base = RUNS_ROOT / cfg.variant / "train"
            if base.exists():
                options = sorted([d.name for d in base.iterdir() if d.is_dir()])
                if options:
                    print(f"\nRuns disponibles para {cfg.variant}:")
                    for i, o in enumerate(options): print(f" [{i}] {o}")
                    idx = int(input("Seleccione índice: "))
                    cfg.train_run = options[idx]
        run_single_mode(cfg)


if __name__ == "__main__":
    main()