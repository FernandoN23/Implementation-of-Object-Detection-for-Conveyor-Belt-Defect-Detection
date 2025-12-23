# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/metrics.py
# Descripción: Herramienta CLI para visualización y comparación
#              de métricas de entrenamiento SSD.
#              Genera curvas de pérdida detalladas (Loc, Conf, Total)
#              y métricas de validación (P, R, F1, mAP).
# ==============================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración de estilo para Matplotlib
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
SSD_ROOT = FILE.parents[1]  # SSD/
METRICS_ROOT = SSD_ROOT / "metrics"

# Parámetros aproximados de SSD (VGG16 Backbone) en Millones
SSD_PARAMS_M = {
    "ssd300": 26.3,
    "ssd512": 27.1,
}

# Colores distintivos para las variantes (Modo Comparativo)
VARIANT_COLORS = {
    "ssd300": "#1f77b4",  # Azul
    "ssd512": "#d62728",  # Rojo
}

# Colores para Train/Val (Modo Individual)
COLOR_TRAIN = "#1f77b4"  # Azul
COLOR_VAL = "#ff7f0e"  # Naranja


# ---------------------------------------------------------------------------
# Configuración (Dataclass)
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    task_model: str = "detect"
    variant: str = "ssd300"
    train_run: str = ""
    merge_mode: bool = False
    variants_to_compare: List[str] = None  # type: ignore

    @property
    def train_metrics_dir(self) -> Path:
        # Estructura: SSD/metrics/detect/{variant}/train/{run_name}
        return METRICS_ROOT / self.task_model / self.variant / "train" / self.train_run

    @property
    def final_metrics_dir(self) -> Path:
        # Salida para modo comparativo (Merge)
        return METRICS_ROOT / self.task_model / "global_comparison"


# ---------------------------------------------------------------------------
# Utilidades Generales
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_runs(root: Path) -> List[str]:
    if not root.exists(): return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def interactive_select(prompt: str, options: List[str]) -> str:
    if not options: raise RuntimeError(f"No hay opciones disponibles para: {prompt}")
    print(f"\n{prompt}")
    for i, name in enumerate(options): print(f"  [{i}] {name}")
    while True:
        idx = input("Seleccione índice: ").strip()
        if idx.isdigit() and 0 <= int(idx) < len(options): return options[int(idx)]
        print("Índice inválido.")


def smooth_signal(scalars: List[float], weight: float = 0.6) -> List[float]:
    """
    Aplica suavizado exponencial robusto a NaNs.
    Usa interpolación de Pandas para rellenar huecos antes de suavizar.
    """
    series = pd.Series(scalars)
    series = series.interpolate(limit_direction='both')

    if series.isnull().all():
        return scalars

    clean_scalars = series.tolist()
    last = clean_scalars[0]
    smoothed = []
    for point in clean_scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ---------------------------------------------------------------------------
# Carga de Datos
# ---------------------------------------------------------------------------

def load_results_csv(path: Path) -> pd.DataFrame:
    """Carga results.csv de SSD y asegura tipos numéricos."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró results.csv en {path}")

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        cols_to_numeric = [c for c in df.columns if c not in ['epoch', 'iteration']]
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        print(f"[Error] Leyendo CSV {path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Lógica de Ploteo (Individual)
# ---------------------------------------------------------------------------

def plot_train_val_curve(
        df: pd.DataFrame,
        train_col: str,
        val_col: str,
        title: str,
        ylabel: str,
        out_path: Path,
        smooth_factor: float = 0.6
) -> None:
    """
    Genera un gráfico combinado de Train vs Val para una métrica específica.
    Útil para Loss Total, Loss Loc y Loss Conf.
    """
    if df.empty: return

    # Verificar existencia de columnas
    has_train = train_col in df.columns
    has_val = val_col in df.columns

    if not has_train and not has_val:
        return

    plt.figure(figsize=(10, 6))
    epochs = df["epoch"]

    # Plot Train
    if has_train:
        raw_train = df[train_col].values.astype(float)
        smooth_train = smooth_signal(raw_train.tolist(), weight=smooth_factor)
        plt.plot(epochs, raw_train, color=COLOR_TRAIN, alpha=0.2, linewidth=1)
        plt.plot(epochs, smooth_train, label="Train", color=COLOR_TRAIN, alpha=1.0, linewidth=2.5)

    # Plot Val
    if has_val:
        raw_val = df[val_col].values.astype(float)
        smooth_val = smooth_signal(raw_val.tolist(), weight=smooth_factor)
        plt.plot(epochs, raw_val, color=COLOR_VAL, alpha=0.2, linewidth=1)
        plt.plot(epochs, smooth_val, label="Validation", color=COLOR_VAL, alpha=1.0, linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Lógica de Ploteo (Comparativo)
# ---------------------------------------------------------------------------

def plot_comparative_metric(
        data_map: Dict[str, pd.DataFrame],
        metric_col_keyword: str,
        title: str,
        ylabel: str,
        out_path: Path,
        smooth_factor: float = 0.6
) -> None:
    """Genera gráfico comparativo superponiendo variantes."""
    plt.figure(figsize=(10, 6))

    has_data = False
    for var, df in data_map.items():
        if df.empty: continue

        # Buscar columna exacta o que contenga la keyword
        if metric_col_keyword in df.columns:
            col = metric_col_keyword
        else:
            col = next((c for c in df.columns if metric_col_keyword in c), None)

        if not col: continue

        df_clean = df.dropna(subset=['epoch', col])
        if df_clean.empty: continue

        has_data = True
        epochs = df_clean["epoch"]
        values = df_clean[col].values.astype(float)

        # Color según variante o default si es modo single
        color = VARIANT_COLORS.get(var, "#2ca02c")
        if len(data_map) == 1: color = "#2ca02c"

        # 1. Plot datos crudos
        plt.plot(epochs, values, color=color, alpha=0.2, linewidth=1)

        # 2. Plot datos suavizados
        smoothed = smooth_signal(values.tolist(), weight=smooth_factor)
        plt.plot(epochs, smoothed, label=f"{var.upper()}", color=color, alpha=1.0, linewidth=2.5)

    if not has_data:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_variant_tradeoff(data_map: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Gráfico de dispersión: mAP vs Parámetros."""
    variants, maps, params = [], [], []

    for var, df in data_map.items():
        if df.empty: continue
        col = next((c for c in df.columns if "mAP_0.5_0.95" in c), None)
        if not col: continue

        best_map = df[col].max()
        if pd.isna(best_map) or best_map == 0: continue

        variants.append(var)
        maps.append(best_map)
        params.append(SSD_PARAMS_M.get(var, 0))

    if not variants: return

    plt.figure(figsize=(9, 6))
    colors = [VARIANT_COLORS.get(v, "gray") for v in variants]
    plt.scatter(params, maps, c=colors, s=150, zorder=3, edgecolors='black')

    if len(params) > 1:
        sorted_indices = np.argsort(params)
        sorted_params = np.array(params)[sorted_indices]
        sorted_maps = np.array(maps)[sorted_indices]
        plt.plot(sorted_params, sorted_maps, linestyle='--', color='gray', alpha=0.5, zorder=1)

    for i, txt in enumerate(variants):
        plt.annotate(f"  {txt.upper()}", (params[i], maps[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=11, fontweight='bold')

    plt.xlabel("Parámetros (Millones)")
    plt.ylabel("Best mAP@0.5:0.95")
    plt.title("Trade-off: Performance vs Complejidad (SSD)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Modos de Ejecución
# ---------------------------------------------------------------------------

def discover_best_runs(task: str, variants: List[str]) -> Dict[str, Path]:
    found_runs = {}
    for var in variants:
        base_path = METRICS_ROOT / task / var / "train"
        if not base_path.exists(): continue
        runs = [p for p in base_path.iterdir() if p.is_dir()]
        if not runs: continue
        latest_run = max(runs, key=lambda p: p.stat().st_mtime)
        csv_path = latest_run / "results.csv"
        if csv_path.is_file():
            found_runs[var] = csv_path
            print(f"[Merge] Variante '{var}': {latest_run.name}")
    return found_runs


def run_comparison_mode(cfg: MetricsConfig) -> None:
    print("\n=== Iniciando Modo Comparativo SSD (Merge) ===")

    variants = cfg.variants_to_compare or ["ssd300", "ssd512"]
    runs_map = discover_best_runs(cfg.task_model, variants)

    if not runs_map:
        print("[Error] No se encontraron runs válidos para comparar.")
        return

    data_map = {var: load_results_csv(path) for var, path in runs_map.items()}

    # Estructura de carpetas de salida (Global)
    global_out = cfg.final_metrics_dir
    losses_out = global_out / "losses"
    metrics_out = global_out / "metrics"

    _ensure_dir(global_out)
    _ensure_dir(losses_out)
    _ensure_dir(metrics_out)

    print(f"Generando gráficos comparativos en: {global_out}")

    # --- 1. Gráficos de Pérdidas (Comparativo) ---
    # Ahora incluimos componentes para ver cuál variante optimiza mejor qué cosa
    loss_types = {
        "Total Loss (Val)": "val_loss_total",
        "Localization Loss (Val)": "val_loss_loc",
        "Confidence Loss (Val)": "val_loss_conf"
    }

    for title, keyword in loss_types.items():
        plot_comparative_metric(
            data_map, keyword,
            title, "Loss",
            losses_out / f"compare_{keyword}.png",
            smooth_factor=0.8
        )

    # --- 2. Gráficos de Métricas ---
    metric_types = {
        "Precision": "val_P",
        "Recall": "val_R",
        "F1-Score": "val_F1",
        "mAP@0.5": "val_mAP_0.5",
        "mAP@0.5:0.95": "val_mAP_0.5_0.95"
    }

    for title, keyword in metric_types.items():
        safe_name = keyword.replace(":", "_")
        plot_comparative_metric(
            data_map, keyword,
            title, title,
            metrics_out / f"compare_{safe_name}.png",
            smooth_factor=0.6
        )

    # --- 3. Trade-off ---
    plot_variant_tradeoff(data_map, global_out / "tradeoff_performance_size.png")

    # --- 4. Resumen JSON ---
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": list(data_map.keys()),
        "best_metrics": {}
    }

    for var, df in data_map.items():
        if df.empty: continue
        map_col = next((c for c in df.columns if "mAP_0.5_0.95" in c), None)
        f1_col = next((c for c in df.columns if "F1" in c), None)

        summary["best_metrics"][var] = {
            "map50_95": float(df[map_col].max()) if map_col else 0,
            "f1": float(df[f1_col].max()) if f1_col else 0,
            "params_M": SSD_PARAMS_M.get(var, 0)
        }

    with open(global_out / "global_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Comparación Finalizada ===")


def run_single_mode(cfg: MetricsConfig) -> None:
    print(f"\n=== Iniciando Modo Individual ({cfg.variant}) ===")

    if not cfg.train_run:
        base_path = METRICS_ROOT / cfg.task_model / cfg.variant / "train"
        if not base_path.exists():
            print(f"[Error] No existe directorio: {base_path}")
            return
        options = list_runs(base_path)
        if not options:
            print(f"[Error] No se encontraron runs en {base_path}")
            return
        cfg.train_run = interactive_select("Seleccione entrenamiento:", options)

    csv_path = cfg.train_metrics_dir / "results.csv"
    df = load_results_csv(csv_path)

    if df.empty:
        print("[Error] DataFrame vacío o archivo no encontrado.")
        return

    # Directorio de salida: La misma carpeta del run
    out_dir = cfg.train_metrics_dir
    losses_dir = out_dir / "losses"
    _ensure_dir(losses_dir)

    # Usamos la lógica comparativa pero con un solo elemento en el mapa para métricas
    data_map = {cfg.variant: df}

    # ---------------------------------------------------------
    # 1. Gráficos de Pérdidas (Train vs Val) en subcarpeta 'losses'
    # ---------------------------------------------------------

    # Total Loss
    plot_train_val_curve(
        df, "train_loss_total", "val_loss_total",
        "Total Loss (Train vs Val)", "Loss",
        losses_dir / "loss_total_combined.png"
    )

    # Localization Loss
    plot_train_val_curve(
        df, "train_loss_loc", "val_loss_loc",
        "Localization Loss (Train vs Val)", "Loss",
        losses_dir / "loss_loc_combined.png"
    )

    # Confidence Loss
    plot_train_val_curve(
        df, "train_loss_conf", "val_loss_conf",
        "Confidence Loss (Train vs Val)", "Loss",
        losses_dir / "loss_conf_combined.png"
    )

    # Gráfico "Resumen" en la raíz (Total Loss)
    plot_train_val_curve(
        df, "train_loss_total", "val_loss_total",
        "Total Loss", "Loss",
        out_dir / "loss_combined.png"
    )

    # ---------------------------------------------------------
    # 2. Gráficos de Métricas en raíz
    # ---------------------------------------------------------
    plot_comparative_metric(data_map, "val_mAP_0.5", "mAP@0.5", "mAP", out_dir / "map_05.png")
    plot_comparative_metric(data_map, "val_mAP_0.5_0.95", "mAP@0.5:0.95", "mAP", out_dir / "map_05_95.png")
    plot_comparative_metric(data_map, "val_F1", "F1 Score", "F1", out_dir / "f1_score.png")
    plot_comparative_metric(data_map, "val_P", "Precision", "Precision", out_dir / "precision.png")
    plot_comparative_metric(data_map, "val_R", "Recall", "Recall", out_dir / "recall.png")

    print(f"Métricas individuales generadas en: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--task-model", default="detect", help="Tipo de tarea (detect).")
    parser.add_argument("--variant", default="ssd300", help="Variante para modo individual (ssd300, ssd512).")
    parser.add_argument("--train-run", default="", help="Nombre específico del run (opcional).")
    parser.add_argument("--merge", action="store_true", help="Activar modo comparativo entre variantes.")
    parser.add_argument("--variants-to-compare", nargs="+", default=["ssd300", "ssd512"],
                        help="Lista de variantes a comparar.")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = MetricsConfig(
        task_model=args.task_model,
        variant=args.variant,
        train_run=args.train_run,
        merge_mode=args.merge,
        variants_to_compare=args.variants_to_compare
    )

    if cfg.merge_mode:
        run_comparison_mode(cfg)
    else:
        run_single_mode(cfg)


if __name__ == "__main__":
    main()