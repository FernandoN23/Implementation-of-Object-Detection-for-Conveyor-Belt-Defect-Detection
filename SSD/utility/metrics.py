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
#              Incluye modo individual y modo comparativo (merge)
#              interactivo con paleta de colores dinámica.
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

# GFLOPs aproximados de SSD (VGG16 Backbone)
SSD_GFLOPS = {
    "ssd300": 31.4,
    "ssd512": 90.8,
}

# Paleta estándar para asignar colores únicos por experimento (run)
STANDARD_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

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
        return pd.DataFrame()

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
# Clase MergeManager: Gestión Interactiva de Comparativas
# ---------------------------------------------------------------------------

class MergeManager:
    def __init__(self, task_model: str = "detect"):
        self.task_model = task_model
        self.selected_runs: Dict[str, Tuple[str, pd.DataFrame]] = {}  # run_name -> (variant, df)
        self.run_colors: Dict[str, str] = {}  # run_name -> color hex
        self.comparison_name = ""
        self.base_output_dir = METRICS_ROOT / self.task_model / "global_comparison"

    def get_available_variants(self) -> List[str]:
        task_dir = METRICS_ROOT / self.task_model
        if not task_dir.exists(): return []
        return sorted([d.name for d in task_dir.iterdir() if d.is_dir() and d.name != "global_comparison"])

    def get_available_runs(self, variant: str) -> List[str]:
        train_path = METRICS_ROOT / self.task_model / variant / "train"
        if not train_path.exists(): return []
        return sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    def interactive_selection(self):
        print(f"\n[metrics.py] --- Configuración de Comparativa Global (SSD) ---")

        while True:
            variants = self.get_available_variants()
            if not variants:
                print(f"[metrics.py] ERROR: No se detectaron métricas en metrics/{self.task_model}/")
                return False

            print(f"\n[metrics.py] Variantes disponibles:")
            for i, v in enumerate(variants): print(f"  [{i}] {v}")

            v_idx = input("[metrics.py] Seleccione índice de variante (o 'q' para finalizar selección): ").strip()
            if v_idx.lower() == 'q': break

            if not v_idx.isdigit() or int(v_idx) >= len(variants):
                print("[metrics.py] Selección inválida.")
                continue

            variant = variants[int(v_idx)]
            runs = self.get_available_runs(variant)

            if not runs:
                print(f"[metrics.py] No hay entrenamientos registrados para {variant}.")
                continue

            print(f"\n[metrics.py] Experimentos (runs) para {variant}:")
            for i, r in enumerate(runs): print(f"  [{i}] {r}")

            r_idx = input(f"[metrics.py] Seleccione índice de experimento para {variant}: ").strip()
            if not r_idx.isdigit() or int(r_idx) >= len(runs):
                print("[metrics.py] Selección inválida.")
                continue

            run_name = runs[int(r_idx)]
            csv_path = METRICS_ROOT / self.task_model / variant / "train" / run_name / "results.csv"

            if csv_path.exists():
                df = load_results_csv(csv_path)
                if not df.empty:
                    unique_key = f"{run_name} ({variant})"
                    self.selected_runs[unique_key] = (variant, df)
                    print(f"[metrics.py] ✓ Añadido: {unique_key}")
                else:
                    print(f"[metrics.py] ERROR: El archivo results.csv está vacío en {run_name}")
            else:
                print(f"[metrics.py] ERROR: No se encontró results.csv en {run_name}")

            cont = input("\n[metrics.py] ¿Desea añadir otro experimento a la comparación? (s/n): ").strip().lower()
            if cont != 's': break

        if not self.selected_runs:
            print("[metrics.py] No se seleccionó ningún modelo. Abortando.")
            return False

        print(f"\n[metrics.py] --- RESUMEN DE COMPARACIÓN ---")
        print(f"Se generarán gráficos comparativos para {len(self.selected_runs)} modelos:")
        for key in self.selected_runs.keys():
            print(f"  • {key}")

        confirm = input("\n[metrics.py] ¿Confirmar inicio de procesamiento? (s/n): ").strip().lower()
        if confirm == 's':
            print(f"\n[metrics.py] Configuración de salida:")
            folder_name = input("[metrics.py] Ingrese nombre para la carpeta de comparación: ").strip()
            self.comparison_name = folder_name if folder_name else "unnamed_comparison"
            return True

        return False

    def run_comparison(self):
        output_dir = self.base_output_dir / self.comparison_name
        output_dir.mkdir(parents=True, exist_ok=True)

        losses_out = output_dir / "losses"
        metrics_out = output_dir / "metrics"
        losses_out.mkdir(exist_ok=True)
        metrics_out.mkdir(exist_ok=True)

        # Asignar colores únicos a cada experimento seleccionado
        for i, label in enumerate(self.selected_runs.keys()):
            self.run_colors[label] = STANDARD_PALETTE[i % len(STANDARD_PALETTE)]

        print(f"[metrics.py] Generando gráficos en {output_dir}...")

        # Columnas específicas de SSD
        loss_types = {
            "Total Loss (Val)": "val_loss_total",
            "Localization Loss (Val)": "val_loss_loc",
            "Confidence Loss (Val)": "val_loss_conf"
        }
        for title, keyword in loss_types.items():
            self._plot_comparative(keyword, f"Comparativa {title}", "Loss", losses_out / f"compare_{keyword}.png", 0.8)

        metric_types = {
            "Precision": "val_P",
            "Recall": "val_R",
            "F1-Score": "val_F1",
            "mAP@0.5": "val_mAP_0.5",
            "mAP@0.5:0.95": "val_mAP_0.5_0.95"
        }
        for title, keyword in metric_types.items():
            safe_name = keyword.replace(":", "_")
            self._plot_comparative(keyword, f"Comparativa {title}", title, metrics_out / f"compare_{safe_name}.png",
                                   0.6)

        self._plot_tradeoffs(output_dir)
        self._save_summary_json(output_dir)

        print(f"[metrics.py] === Comparación Finalizada con Éxito en: {self.comparison_name} ===")

    def _plot_comparative(self, metric_col: str, title: str, ylabel: str, out_path: Path, smooth_factor: float):
        plt.figure(figsize=(10, 6))
        has_data = False

        for label, (variant, df) in self.selected_runs.items():
            if metric_col not in df.columns: continue

            df_clean = df.dropna(subset=['epoch', metric_col])
            if df_clean.empty: continue

            has_data = True
            color = self.run_colors[label]
            epochs = df_clean["epoch"]
            values = df_clean[metric_col].values.astype(float)

            plt.plot(epochs, values, color=color, alpha=0.15, linewidth=1)
            smoothed = smooth_signal(values.tolist(), weight=smooth_factor)
            plt.plot(epochs, smoothed, label=label, color=color, alpha=1.0, linewidth=2.5)

        if not has_data:
            plt.close()
            return

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def _plot_tradeoffs(self, output_dir: Path):
        """Genera gráficos de trade-off para Parámetros y GFLOPs."""
        data_points = []
        for label, (variant, df) in self.selected_runs.items():
            if "val_mAP_0.5_0.95" not in df.columns: continue
            best_map = df["val_mAP_0.5_0.95"].max()
            if pd.isna(best_map) or best_map == 0: continue

            data_points.append({
                'label': label,
                'variant': variant,
                'params': SSD_PARAMS_M.get(variant, 0),
                'gflops': SSD_GFLOPS.get(variant, 0),
                'map': best_map,
                'color': self.run_colors[label]
            })

        if not data_points: return

        # --- Gráfico 1: Performance vs Parámetros ---
        data_points.sort(key=lambda x: x['params'])
        self._render_scatter_plot(
            data_points, 'params', "Parámetros (Millones)", "Best mAP@0.5:0.95",
            "Trade-off: Performance vs Parámetros (SSD)",
            output_dir / "tradeoff_performance_params.png"
        )

        # --- Gráfico 2: Performance vs GFLOPs ---
        data_points.sort(key=lambda x: x['gflops'])
        self._render_scatter_plot(
            data_points, 'gflops', "GFLOPs", "Best mAP@0.5:0.95",
            "Trade-off: Performance vs Costo Computacional (GFLOPs) (SSD)",
            output_dir / "tradeoff_performance_gflops.png"
        )

    def _render_scatter_plot(self, data_points, x_key, xlabel, ylabel, title, out_path):
        """Función auxiliar para renderizar los gráficos de dispersión."""
        x_vals = [d[x_key] for d in data_points]
        y_vals = [d['map'] for d in data_points]
        colors = [d['color'] for d in data_points]
        labels = [d['label'] for d in data_points]

        plt.figure(figsize=(10, 6))

        # Dibujar línea de tendencia (solo si hay más de un punto)
        if len(x_vals) > 1:
            plt.plot(x_vals, y_vals, linestyle='--', color='#7f8c8d', alpha=0.6, zorder=1, linewidth=1.5)

        # Dibujar puntos
        for i in range(len(x_vals)):
            plt.scatter(x_vals[i], y_vals[i], color=colors[i], s=200, zorder=3, edgecolors='black', linewidth=1.2)
            plt.annotate(f"  {labels[i]}", (x_vals[i], y_vals[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9, fontweight='bold')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.margins(0.15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def _save_summary_json(self, output_dir: Path):
        summary = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
        for label, (variant, df) in self.selected_runs.items():
            summary["models"][label] = {
                "variant": variant,
                "best_mAP_0.5": float(df["val_mAP_0.5"].max()) if "val_mAP_0.5" in df.columns else 0,
                "best_mAP_0.5_0.95": float(df["val_mAP_0.5_0.95"].max()) if "val_mAP_0.5_0.95" in df.columns else 0,
                "best_F1": float(df["val_F1"].max()) if "val_F1" in df.columns else 0,
                "params_M": SSD_PARAMS_M.get(variant, 0),
                "gflops": SSD_GFLOPS.get(variant, 0)
            }
        with open(output_dir / "global_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


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

    has_train = train_col in df.columns
    has_val = val_col in df.columns

    if not has_train and not has_val:
        return

    plt.figure(figsize=(10, 6))
    epochs = df["epoch"]

    if has_train:
        raw_train = df[train_col].values.astype(float)
        smooth_train = smooth_signal(raw_train.tolist(), weight=smooth_factor)
        plt.plot(epochs, raw_train, color=COLOR_TRAIN, alpha=0.2, linewidth=1)
        plt.plot(epochs, smooth_train, label="Train", color=COLOR_TRAIN, alpha=1.0, linewidth=2.5)

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


def plot_single_metric(
        df: pd.DataFrame,
        metric_col: str,
        title: str,
        ylabel: str,
        out_path: Path,
        smooth_factor: float = 0.6
) -> None:
    """Genera un gráfico para una métrica individual (ej. mAP, F1)."""
    if df.empty or metric_col not in df.columns: return

    df_clean = df.dropna(subset=['epoch', metric_col])
    if df_clean.empty: return

    plt.figure(figsize=(10, 6))
    epochs = df_clean["epoch"]
    values = df_clean[metric_col].values.astype(float)

    plt.plot(epochs, values, color=COLOR_VAL, alpha=0.2, linewidth=1)
    smoothed = smooth_signal(values.tolist(), weight=smooth_factor)
    plt.plot(epochs, smoothed, label="Validation", color=COLOR_VAL, alpha=1.0, linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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

    # ---------------------------------------------------------
    # 1. Gráficos de Pérdidas (Train vs Val) en subcarpeta 'losses'
    # ---------------------------------------------------------
    plot_train_val_curve(df, "train_loss_total", "val_loss_total", "Total Loss (Train vs Val)", "Loss",
                         losses_dir / "loss_total_combined.png")
    plot_train_val_curve(df, "train_loss_loc", "val_loss_loc", "Localization Loss (Train vs Val)", "Loss",
                         losses_dir / "loss_loc_combined.png")
    plot_train_val_curve(df, "train_loss_conf", "val_loss_conf", "Confidence Loss (Train vs Val)", "Loss",
                         losses_dir / "loss_conf_combined.png")

    # Gráfico "Resumen" en la raíz (Total Loss)
    plot_train_val_curve(df, "train_loss_total", "val_loss_total", "Total Loss", "Loss", out_dir / "loss_combined.png")

    # ---------------------------------------------------------
    # 2. Gráficos de Métricas en raíz
    # ---------------------------------------------------------
    plot_single_metric(df, "val_mAP_0.5", "mAP@0.5", "mAP", out_dir / "map_05.png")
    plot_single_metric(df, "val_mAP_0.5_0.95", "mAP@0.5:0.95", "mAP", out_dir / "map_05_95.png")
    plot_single_metric(df, "val_F1", "F1 Score", "F1", out_dir / "f1_score.png")
    plot_single_metric(df, "val_P", "Precision", "Precision", out_dir / "precision.png")
    plot_single_metric(df, "val_R", "Recall", "Recall", out_dir / "recall.png")

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
        manager = MergeManager(task_model=cfg.task_model)
        if manager.interactive_selection():
            manager.run_comparison()
    else:
        run_single_mode(cfg)


if __name__ == "__main__":
    main()