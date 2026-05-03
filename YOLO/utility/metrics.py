# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/utility/metrics.py
# Descripción: Utilidades para estandarizar métricas de
#              entrenamiento y validación YOLO.
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
import yaml

# Configuración de estilo para Matplotlib (Estilo académico/limpio)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

# Raíces de proyecto
FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parents[1]
PROJECT_ROOT = YOLO_ROOT.parent
METRICS_ROOT = YOLO_ROOT / "metrics"
RUNS_ROOT = YOLO_ROOT / "runs"
CONFIGS_ROOT = YOLO_ROOT / "configs"
YOLOV5_ROOT = YOLO_ROOT / "yolov5"

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

# Parámetros aproximados de YOLOv5 (Millones de parámetros)
YOLOV5_PARAMS_M = {
    "n": 1.9, "s": 7.2, "m": 21.2, "l": 46.5, "x": 86.7,
}

# Paleta estándar para asignar colores únicos por experimento (run)
STANDARD_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    task_model: str = "detect"
    variant: str = "s"
    train_run: str = ""
    val_run: str = ""
    experiment_id: Optional[str] = None
    dataset_cfg: Path = CONFIGS_ROOT / "dataset.yaml"
    weights: Optional[str] = None
    imgsz: int = 640
    batch_size: int = 4
    device: str = ""
    compute_iou: bool = False
    merge_mode: bool = False
    variants_to_compare: List[str] = None  # type: ignore

    def final_experiment_id(self) -> str:
        if self.experiment_id:
            return self.experiment_id
        return self.train_run or f"{self.task_model}_{self.variant}_experiment"

    @property
    def train_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / self.variant / "train" / self.train_run

    @property
    def val_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / self.variant / "val" / self.val_run

    @property
    def final_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / self.variant / "final_metrics" / self.final_experiment_id()

    @property
    def train_runs_root(self) -> Path:
        return RUNS_ROOT / self.task_model / self.variant / "train"


# ---------------------------------------------------------------------------
# Utilidades Generales
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_runs(root: Path) -> List[str]:
    if not root.exists(): return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def interactive_select(prompt: str, options: List[str]) -> str:
    if not options: raise RuntimeError(f"No hay opciones para {prompt}")
    print(prompt)
    for i, name in enumerate(options): print(f"  [{i}] {name}")
    while True:
        idx = input("Seleccione índice: ").strip()
        if idx.isdigit() and 0 <= int(idx) < len(options): return options[int(idx)]
        print("Índice inválido.")


def smooth_signal(scalars: List[float], weight: float = 0.6) -> List[float]:
    """Aplica suavizado exponencial (tipo Tensorboard) para limpiar curvas ruidosas."""
    series = pd.Series(scalars).interpolate(limit_direction='both')
    if series.isnull().all(): return scalars
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
    """Carga results.csv, limpia columnas y calcula F1-Score."""
    if not path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Calcular F1 Score si existen P y R
    p_col = next((c for c in df.columns if "precision" in c), None)
    r_col = next((c for c in df.columns if "recall" in c), None)

    if p_col and r_col:
        P = df[p_col]
        R = df[r_col]
        # F1 = 2 * (P * R) / (P + R), evitando división por cero
        df["metrics/F1"] = 2 * (P * R) / (P + R + 1e-16)

    return df


def load_hyp(cfg: MetricsConfig) -> Dict:
    hyp_path = cfg.train_runs_root / cfg.train_run / "hyp.yaml"
    if hyp_path.is_file():
        with open(hyp_path, "r") as f: return yaml.safe_load(f) or {}
    return {}


def load_dataset_stats(dataset_cfg: Path) -> Dict:
    if not dataset_cfg.exists(): return {}
    with open(dataset_cfg, "r") as f:
        data = yaml.safe_load(f)
    stats = {"config": str(dataset_cfg)}
    if "nc" in data: stats["num_classes"] = int(data["nc"])
    return stats


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
        print(f"\n[metrics.py] --- Configuración de Comparativa Global (YOLO) ---")

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
            for i, r in enumerate(runs): print(f"[{i}] {r}")

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

        # Columnas específicas de YOLOv5
        loss_types = {
            "Box Loss (Val)": "val/box_loss",
            "Objectness Loss (Val)": "val/obj_loss",
            "Classification Loss (Val)": "val/cls_loss"
        }
        for title, keyword in loss_types.items():
            self._plot_comparative(keyword, f"Comparativa {title}", "Loss",
                                   losses_out / f"compare_{keyword.replace('/', '_')}.png", 0.7)

        metric_types = {
            "Precision": "metrics/precision",
            "Recall": "metrics/recall",
            "F1-Score": "metrics/F1",
            "mAP@0.5": "metrics/mAP_0.5",
            "mAP@0.5:0.95": "metrics/mAP_0.5:0.95"
        }
        for title, keyword in metric_types.items():
            safe_name = keyword.replace("metrics/", "").replace(":", "_")
            self._plot_comparative(keyword, f"Comparativa {title}", title, metrics_out / f"compare_{safe_name}.png",
                                   0.5)

        self._plot_tradeoff(output_dir)
        self._save_summary_json(output_dir)

        print(f"[metrics.py] === Comparación Finalizada con Éxito en: {self.comparison_name} ===")

    def _plot_comparative(self, metric_col: str, title: str, ylabel: str, out_path: Path, smooth_factor: float):
        plt.figure(figsize=(10, 6))
        has_data = False

        for label, (variant, df) in self.selected_runs.items():
            # Buscar columna que contenga la keyword
            col = next((c for c in df.columns if metric_col in c), None)
            if not col: continue

            df_clean = df.dropna(subset=['epoch' if 'epoch' in df.columns else df.columns[0], col])
            if df_clean.empty: continue

            has_data = True
            color = self.run_colors[label]
            epochs = df_clean["epoch"] if "epoch" in df_clean.columns else df_clean.index
            values = df_clean[col].values.astype(float)

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

    def _plot_tradeoff(self, output_dir: Path):
        data_points = []
        for label, (variant, df) in self.selected_runs.items():
            col = next((c for c in df.columns if "mAP_0.5:0.95" in c), None)
            if not col: continue
            best_map = df[col].max()
            if pd.isna(best_map) or best_map == 0: continue

            data_points.append({
                'label': label,
                'variant': variant,
                'params': YOLOV5_PARAMS_M.get(variant, 0),
                'map': best_map,
                'color': self.run_colors[label]
            })

        if not data_points: return

        # Ordenar por parámetros para la línea continua
        data_points.sort(key=lambda x: x['params'])

        params = [d['params'] for d in data_points]
        maps = [d['map'] for d in data_points]
        colors = [d['color'] for d in data_points]
        labels = [d['label'] for d in data_points]

        plt.figure(figsize=(10, 6))

        # Dibujar línea de tendencia (solo si hay más de un punto)
        if len(params) > 1:
            plt.plot(params, maps, linestyle='--', color='#7f8c8d', alpha=0.6, zorder=1, linewidth=1.5)

        # Dibujar puntos
        for i in range(len(params)):
            plt.scatter(params[i], maps[i], color=colors[i], s=200, zorder=3, edgecolors='black', linewidth=1.2)
            plt.annotate(f"  {labels[i]}", (params[i], maps[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9, fontweight='bold')

        plt.xlabel("Parámetros (Millones)")
        plt.ylabel("Best mAP@0.5:0.95")
        plt.title("Trade-off: Performance vs Complejidad")

        plt.margins(0.15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_performance_size.png", dpi=200)
        plt.close()

    def _save_summary_json(self, output_dir: Path):
        summary = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
        for label, (variant, df) in self.selected_runs.items():
            map50_col = next((c for c in df.columns if "mAP_0.5" in c and "0.95" not in c), None)
            map95_col = next((c for c in df.columns if "mAP_0.5:0.95" in c), None)
            f1_col = next((c for c in df.columns if "F1" in c), None)

            summary["models"][label] = {
                "variant": variant,
                "best_mAP_0.5": float(df[map50_col].max()) if map50_col else 0,
                "best_mAP_0.5_0.95": float(df[map95_col].max()) if map95_col else 0,
                "best_F1": float(df[f1_col].max()) if f1_col else 0,
                "params_M": YOLOV5_PARAMS_M.get(variant, 0)
            }
        with open(output_dir / "global_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Plotting Individual (Legacy)
# ---------------------------------------------------------------------------

def compute_mean_loss(df: pd.DataFrame, split: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in df.columns if split in c and "loss" in c]
    if not cols: raise KeyError(f"No loss cols for {split}")
    losses = df[cols].to_numpy(dtype=float)
    return df.index.to_numpy(), losses.mean(axis=1)


def plot_loss_curves(cfg: MetricsConfig, hyp: Dict, ds: Dict, df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    try:
        epochs, loss_t = compute_mean_loss(df, "train")
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_t, label="Train")
        try:
            epochs_v, loss_v = compute_mean_loss(df, "val")
            plt.plot(epochs_v, loss_v, label="Val")
        except KeyError:
            pass
        plt.title(f"Loss Curves - {cfg.variant}")
        plt.legend()
        plt.savefig(out_dir / "loss_curves.png")
        plt.close()
    except KeyError:
        pass


def plot_map_curves(cfg: MetricsConfig, hyp: Dict, ds: Dict, df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    map50 = next((c for c in df.columns if "mAP_0.5" in c and "0.95" not in c), None)
    map95 = next((c for c in df.columns if "mAP_0.5:0.95" in c), None)
    if map50 and map95:
        plt.figure(figsize=(8, 5))
        plt.plot(df[map50], label="mAP@0.5")
        plt.plot(df[map95], label="mAP@0.5:0.95")
        plt.title(f"mAP Curves - {cfg.variant}")
        plt.legend()
        plt.savefig(out_dir / "map_curves.png")
        plt.close()


# ---------------------------------------------------------------------------
# CLI y Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Individual
    parser.add_argument("--task-model", default="detect", help="Tipo de modelo.")
    parser.add_argument("--variant", default="s", help="Variante (n, s, m, l, x).")
    parser.add_argument("--train-run", default="", help="Run entrenamiento.")
    parser.add_argument("--val-run", default="", help="Run validación.")
    parser.add_argument("--experiment-id", default="", help="ID salida.")
    parser.add_argument("--dataset-cfg", default=str(CONFIGS_ROOT / "dataset.yaml"))
    parser.add_argument("--weights", default="")
    parser.add_argument("--compute-iou", action="store_true")

    # Merge
    parser.add_argument("--merge", action="store_true", help="Activar modo comparativo interactivo.")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = MetricsConfig(
        task_model=args.task_model, variant=args.variant,
        train_run=args.train_run, val_run=args.val_run,
        experiment_id=args.experiment_id, dataset_cfg=Path(args.dataset_cfg),
        weights=args.weights, compute_iou=args.compute_iou,
        merge_mode=args.merge
    )

    if cfg.merge_mode:
        manager = MergeManager(task_model=cfg.task_model)
        if manager.interactive_selection():
            manager.run_comparison()
        return

    # Lógica Individual (Legacy)
    if not cfg.train_run:
        options = list_runs(METRICS_ROOT / cfg.task_model / cfg.variant / "train")
        if not options:
            print(f"[Error] No hay entrenamientos en {METRICS_ROOT / cfg.task_model / cfg.variant / 'train'}")
            return
        cfg.train_run = interactive_select("Seleccione entrenamiento:", options)

    try:
        df = load_results_csv(cfg.train_metrics_dir / "results.csv")
        if df.empty: raise FileNotFoundError
    except FileNotFoundError:
        print(f"[Error] No results.csv en {cfg.train_metrics_dir}")
        return

    hyp = load_hyp(cfg)
    ds = load_dataset_stats(cfg.dataset_cfg)
    out_dir = cfg.final_metrics_dir
    _ensure_dir(out_dir)

    plot_loss_curves(cfg, hyp, ds, df, out_dir / "losses")
    plot_map_curves(cfg, hyp, ds, df, out_dir)

    summary = {"variant": cfg.variant, "train_run": cfg.train_run}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Métricas individuales en: {out_dir}")


if __name__ == "__main__":
    main()