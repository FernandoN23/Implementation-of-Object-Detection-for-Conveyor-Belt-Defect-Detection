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
#              con suavizado de curvas y cálculo de F1.
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
plt.style.use('seaborn-v0_8-whitegrid')
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

# Colores distintivos para las variantes
VARIANT_COLORS = {
    "n": "#1f77b4",  # Azul
    "s": "#ff7f0e",  # Naranja
    "m": "#2ca02c",  # Verde
    "l": "#d62728",  # Rojo
    "x": "#9467bd"  # Morado
}


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
    last = scalars[0]
    smoothed = []
    for point in scalars:
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
        raise FileNotFoundError(f"No se encontró results.csv en {path}")

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
# Plotting Individual (Legacy)
# ---------------------------------------------------------------------------
# (Se mantienen versiones simplificadas para no romper compatibilidad)

def compute_mean_loss(df: pd.DataFrame, split: str) -> Tuple[np.ndarray, np.ndarray]:
    # Lógica simplificada de búsqueda de columnas
    cols = [c for c in df.columns if split in c and "loss" in c]
    if not cols: raise KeyError(f"No loss cols for {split}")
    losses = df[cols].to_numpy(dtype=float)
    return df.index.to_numpy(), losses.mean(axis=1)


def plot_loss_curves(cfg: MetricsConfig, hyp: Dict, ds: Dict, df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
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
# Lógica de Comparación (Merge Mode) - MEJORADA
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


def plot_comparative_metric(
        data_map: Dict[str, pd.DataFrame],
        metric_col_keyword: str,
        title: str,
        ylabel: str,
        out_path: Path,
        smooth_factor: float = 0.6
) -> None:
    """
    Genera gráfico comparativo con suavizado.
    Dibuja la línea cruda (transparente) y la suavizada (sólida).
    """
    plt.figure(figsize=(10, 6))

    has_data = False
    for var, df in data_map.items():
        # Buscar columna que contenga la keyword (ej: "val/obj_loss")
        col = next((c for c in df.columns if metric_col_keyword in c), None)
        if not col: continue

        has_data = True
        epochs = df["epoch"] if "epoch" in df.columns else df.index
        values = df[col].values.astype(float)

        color = VARIANT_COLORS.get(var, "gray")

        # 1. Plot datos crudos (muy transparentes)
        plt.plot(epochs, values, color=color, alpha=0.2, linewidth=1)

        # 2. Plot datos suavizados (sólidos)
        smoothed = smooth_signal(values, weight=smooth_factor)
        plt.plot(epochs, smoothed, label=f"YOLOv5-{var}", color=color, alpha=1.0, linewidth=2.5)

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
    variants, maps, params = [], [], []

    for var, df in data_map.items():
        col = next((c for c in df.columns if "mAP_0.5:0.95" in c), None)
        if not col: continue
        best_map = df[col].max()
        variants.append(var)
        maps.append(best_map)
        params.append(YOLOV5_PARAMS_M.get(var, 0))

    if not variants: return

    plt.figure(figsize=(9, 6))
    # Scatter con colores por variante
    colors = [VARIANT_COLORS.get(v, "gray") for v in variants]
    plt.scatter(params, maps, c=colors, s=150, zorder=3, edgecolors='black')

    # Línea conectora
    sorted_indices = np.argsort(params)
    sorted_params = np.array(params)[sorted_indices]
    sorted_maps = np.array(maps)[sorted_indices]
    plt.plot(sorted_params, sorted_maps, linestyle='--', color='gray', alpha=0.5, zorder=1)

    for i, txt in enumerate(variants):
        plt.annotate(f"  {txt.upper()}", (params[i], maps[i]), fontsize=11, fontweight='bold')

    plt.xlabel("Parámetros (Millones)")
    plt.ylabel("Best mAP@0.5:0.95")
    plt.title("Trade-off: Performance vs Complejidad")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_comparison_mode(cfg: MetricsConfig) -> None:
    print("\n=== Iniciando Modo Comparativo (Merge) ===")

    variants = cfg.variants_to_compare or ["n", "s", "m", "l", "x"]
    runs_map = discover_best_runs(cfg.task_model, variants)

    if not runs_map:
        print("[Error] No se encontraron runs válidos.")
        return

    data_map = {var: load_results_csv(path) for var, path in runs_map.items()}

    # Estructura de carpetas
    global_out = METRICS_ROOT / cfg.task_model / "global_comparison"
    losses_out = global_out / "losses"
    metrics_out = global_out / "metrics"

    _ensure_dir(global_out)
    _ensure_dir(losses_out)
    _ensure_dir(metrics_out)

    print(f"Generando gráficos en: {global_out}")

    # --- 1. Gráficos de Pérdidas (Losses) ---
    # Buscamos patrones comunes en las columnas de YOLOv5
    loss_types = {
        "Box Loss": "val/box_loss",
        "Objectness Loss": "val/obj_loss",
        "Classification Loss": "val/cls_loss"
    }

    for title, keyword in loss_types.items():
        plot_comparative_metric(
            data_map, keyword,
            f"Comparativa {title} (Validation)", "Loss",
            losses_out / f"compare_{keyword.replace('/', '_')}.png",
            smooth_factor=0.7  # Mayor suavizado para losses
        )

    # --- 2. Gráficos de Métricas ---
    metric_types = {
        "Precision": "metrics/precision",
        "Recall": "metrics/recall",
        "F1-Score": "metrics/F1",
        "mAP@0.5": "metrics/mAP_0.5",
        "mAP@0.5:0.95": "metrics/mAP_0.5:0.95"
    }

    for title, keyword in metric_types.items():
        # Ajuste de nombre de archivo para evitar caracteres raros
        safe_name = keyword.replace("metrics/", "").replace(":", "_")
        plot_comparative_metric(
            data_map, keyword,
            f"Comparativa {title}", title,
            metrics_out / f"compare_{safe_name}.png",
            smooth_factor=0.5  # Menor suavizado para métricas
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
        # Obtener el mejor valor de cada métrica (max)
        summary["best_metrics"][var] = {
            "map50_95": float(df.filter(like="mAP_0.5:0.95").max().max()) if not df.filter(
                like="mAP_0.5:0.95").empty else 0,
            "f1": float(df.filter(like="F1").max().max()) if not df.filter(like="F1").empty else 0,
            "params_M": YOLOV5_PARAMS_M.get(var, 0)
        }

    with open(global_out / "global_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Comparación Finalizada ===")


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
    parser.add_argument("--merge", action="store_true", help="Activar modo comparativo.")
    parser.add_argument("--variants-to-compare", nargs="+", default=["n", "s", "m", "l", "x"])

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = MetricsConfig(
        task_model=args.task_model, variant=args.variant,
        train_run=args.train_run, val_run=args.val_run,
        experiment_id=args.experiment_id, dataset_cfg=Path(args.dataset_cfg),
        weights=args.weights, compute_iou=args.compute_iou,
        merge_mode=args.merge, variants_to_compare=args.variants_to_compare
    )

    if cfg.merge_mode:
        run_comparison_mode(cfg)
        return

    # Lógica Individual (Legacy)
    if not cfg.train_run:
        options = list_runs(METRICS_ROOT / cfg.task_model / cfg.variant / "train")
        cfg.train_run = interactive_select("Seleccione entrenamiento:", options)

    try:
        df = load_results_csv(cfg.train_metrics_dir / "results.csv")
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