# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/metrics.py
# Descripción: Motor gráfico para reportes de validación y
#              herramienta CLI para comparación global de
#              variantes DETR (r50, r101, dc5).
# ==============================================================

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# --- CONFIGURACIÓN DE ESTILO Y CONSTANTES ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.facecolor': '#f0f0f0',
    'grid.color': 'white',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

FILE = Path(__file__).resolve()
UTILITY_ROOT = FILE.parent
DETR_ROOT = UTILITY_ROOT.parent
METRICS_ROOT = DETR_ROOT / "metrics"

# Parámetros aproximados de DETR (Millones)
DETR_PARAMS_M = {
    "r50": 41.3,
    "r50_dc5": 41.3,
    "r101": 60.1,
    "r101_dc5": 60.1,
}

# Paleta de colores estricta para comparativas
VARIANT_COLORS = {
    "r50": "#1f77b4",  # Azul
    "r50_dc5": "#2ca02c",  # Verde
    "r101": "#d62728",  # Rojo
    "r101_dc5": "#9467bd",  # Púrpura
}


@dataclass
class MetricsConfig:
    task_model: str = "detect"
    variant: str = "r50"
    train_run: str = ""
    merge_mode: bool = False
    variants_to_compare: List[str] = None  # type: ignore

    @property
    def final_metrics_dir(self) -> Path:
        return METRICS_ROOT / self.task_model / "global_comparison"


# ---------------------------------------------------------------------------
# Utilidades Generales y Suavizado
# ---------------------------------------------------------------------------

def smooth_signal(scalars: List[float], weight: float = 0.6) -> List[float]:
    """Aplica suavizado exponencial robusto a NaNs."""
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


def load_results_csv(path: Path) -> pd.DataFrame:
    """Carga results.csv y asegura tipos numéricos."""
    if not path.is_file(): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Calcular F1 si no existe explícitamente
    if 'metrics/mAP_0.5' in df.columns and 'metrics/recall' in df.columns:
        p = df['metrics/mAP_0.5']
        r = df['metrics/recall']
        df['metrics/F1'] = 2 * (p * r) / (p + r + 1e-16)
    return df


# ---------------------------------------------------------------------------
# Lógica de Comparación Global (Merge Mode)
# ---------------------------------------------------------------------------

def discover_best_runs(task: str, variants: List[str]) -> Dict[str, Path]:
    """Busca el run más reciente para cada variante solicitada."""
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


def plot_comparative_metric(data_map: Dict[str, pd.DataFrame], metric_col: str, title: str, ylabel: str, out_path: Path,
                            smooth_factor: float = 0.6):
    """Genera gráfico comparativo superponiendo variantes con colores fijos."""
    plt.figure(figsize=(10, 6))
    has_data = False

    for var, df in data_map.items():
        if metric_col not in df.columns: continue
        df_clean = df.dropna(subset=['epoch', metric_col])
        if df_clean.empty: continue

        has_data = True
        epochs = df_clean["epoch"]
        values = df_clean[metric_col].values.astype(float)
        color = VARIANT_COLORS.get(var, "gray")

        # Plot crudo (transparente) y suavizado (sólido)
        plt.plot(epochs, values, color=color, alpha=0.2, linewidth=1)
        smoothed = smooth_signal(values.tolist(), weight=smooth_factor)
        plt.plot(epochs, smoothed, label=f"DETR-{var.upper()}", color=color, alpha=1.0, linewidth=2.5)

    if not has_data:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_variant_tradeoff(data_map: Dict[str, pd.DataFrame], out_path: Path):
    """Gráfico de dispersión: mAP vs Parámetros."""
    variants, maps, params = [], [], []
    for var, df in data_map.items():
        if 'metrics/mAP_0.5:0.95' not in df.columns: continue
        best_map = df['metrics/mAP_0.5:0.95'].max()
        if pd.isna(best_map) or best_map == 0: continue
        variants.append(var)
        maps.append(best_map)
        params.append(DETR_PARAMS_M.get(var, 0))

    if not variants: return

    plt.figure(figsize=(9, 6))
    colors = [VARIANT_COLORS.get(v, "gray") for v in variants]
    plt.scatter(params, maps, c=colors, s=150, zorder=3, edgecolors='black')

    # Línea conectora
    if len(params) > 1:
        sorted_indices = np.argsort(params)
        plt.plot(np.array(params)[sorted_indices], np.array(maps)[sorted_indices], linestyle='--', color='gray',
                 alpha=0.5, zorder=1)

    for i, txt in enumerate(variants):
        plt.annotate(f"  {txt.upper()}", (params[i], maps[i]), xytext=(5, 5), textcoords='offset points', fontsize=11,
                     fontweight='bold')

    plt.xlabel("Parámetros (Millones)")
    plt.ylabel("Best mAP@0.5:0.95")
    plt.title("Trade-off: Performance vs Complejidad (DETR)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_comparison_mode(cfg: MetricsConfig):
    print("\n=== Iniciando Modo Comparativo DETR (Merge) ===")
    variants = cfg.variants_to_compare or list(DETR_PARAMS_M.keys())
    runs_map = discover_best_runs(cfg.task_model, variants)

    if not runs_map:
        print("[Error] No se encontraron runs válidos para comparar.")
        return

    data_map = {var: load_results_csv(path) for var, path in runs_map.items()}
    global_out = cfg.final_metrics_dir
    losses_out = global_out / "losses"
    metrics_out = global_out / "metrics"

    global_out.mkdir(parents=True, exist_ok=True)
    losses_out.mkdir(exist_ok=True)
    metrics_out.mkdir(exist_ok=True)

    # 1. Gráficos de Pérdidas
    loss_types = {"Total Loss (Val)": "val/loss", "Classification Loss (Val)": "val/loss_ce",
                  "BBox Loss (Val)": "val/loss_bbox"}
    for title, keyword in loss_types.items():
        plot_comparative_metric(data_map, keyword, title, "Loss",
                                losses_out / f"compare_{keyword.replace('/', '_')}.png", smooth_factor=0.7)

    # 2. Gráficos de Métricas
    metric_types = {"Precision (mAP@0.5)": "metrics/mAP_0.5", "mAP@0.5:0.95": "metrics/mAP_0.5:0.95",
                    "Recall": "metrics/recall", "F1-Score": "metrics/F1"}
    for title, keyword in metric_types.items():
        safe_name = keyword.replace("metrics/", "").replace(":", "_")
        plot_comparative_metric(data_map, keyword, f"Comparativa {title}", title,
                                metrics_out / f"compare_{safe_name}.png", smooth_factor=0.5)

    # 3. Trade-off
    plot_variant_tradeoff(data_map, global_out / "tradeoff_performance_size.png")

    # 4. Resumen JSON
    summary = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": list(data_map.keys()), "best_metrics": {}}
    for var, df in data_map.items():
        summary["best_metrics"][var] = {
            "map50_95": float(df['metrics/mAP_0.5:0.95'].max()) if 'metrics/mAP_0.5:0.95' in df.columns else 0,
            "f1": float(df['metrics/F1'].max()) if 'metrics/F1' in df.columns else 0,
            "params_M": DETR_PARAMS_M.get(var, 0)
        }
    with open(global_out / "global_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== Comparación Finalizada en {global_out} ===")


# ---------------------------------------------------------------------------
# Lógica de Validación (Reporte Completo)
# ---------------------------------------------------------------------------

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def plot_validation_report(preds, gts, class_names, save_dir, iou_threshold=0.5):
    nc = len(class_names)
    conf_levels = np.linspace(0, 1, 100)
    curve_data = {c: {'p': [], 'r': [], 'f1': []} for c in range(nc)}
    all_ious = []
    confusion_matrix = np.zeros((nc + 1, nc + 1))

    for p, g in zip(preds, gts):
        p_boxes, p_scores, p_labels = p['boxes'], p['scores'], p['labels']
        g_boxes, g_labels = g['boxes'], g['labels']
        matched_gts = [False] * len(g_labels)

        for i in range(len(p_labels)):
            if p_scores[i] < 0.25: continue
            best_iou, best_gt_idx = 0, -1
            for j in range(len(g_labels)):
                iou = calculate_iou(p_boxes[i], g_boxes[j])
                if iou > best_iou: best_iou, best_gt_idx = iou, j
            if best_iou > iou_threshold:
                confusion_matrix[p_labels[i], g_labels[best_gt_idx]] += 1
                matched_gts[best_gt_idx] = True
                all_ious.append(best_iou)
            else:
                confusion_matrix[p_labels[i], nc] += 1
        for j, matched in enumerate(matched_gts):
            if not matched: confusion_matrix[nc, g_labels[j]] += 1

    for c in range(nc):
        for conf in conf_levels:
            tp, fp, fn = 0, 0, 0
            for p, g in zip(preds, gts):
                p_mask = (p['labels'] == c) & (p['scores'] >= conf)
                g_mask = (g['labels'] == c)
                curr_p_boxes, curr_g_boxes = p['boxes'][p_mask], g['boxes'][g_mask]
                matched = [False] * len(curr_g_boxes)
                for pb in curr_p_boxes:
                    found = False
                    for idx, gb in enumerate(curr_g_boxes):
                        if not matched[idx] and calculate_iou(pb, gb) > iou_threshold:
                            tp += 1;
                            matched[idx] = True;
                            found = True;
                            break
                    if not found: fp += 1
                fn += len(curr_g_boxes) - sum(matched)
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            curve_data[c]['p'].append(prec)
            curve_data[c]['r'].append(rec)
            curve_data[c]['f1'].append(2 * prec * rec / (prec + rec + 1e-6))

    # F1 Curve
    plt.figure(figsize=(10, 7))
    f1_all = []
    for c in range(nc):
        plt.plot(conf_levels, curve_data[c]['f1'], label=class_names[c], linewidth=1)
        f1_all.append(curve_data[c]['f1'])
    mean_f1 = np.mean(f1_all, axis=0)
    best_idx = np.argmax(mean_f1)
    plt.plot(conf_levels, mean_f1, label=f'all classes {mean_f1[best_idx]:.2f} at {conf_levels[best_idx]:.3f}',
             color='blue', linewidth=3)
    plt.title('F1-Confidence Curve');
    plt.xlabel('Confidence');
    plt.ylabel('F1');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout();
    plt.savefig(save_dir / "F1_curve.png", dpi=200);
    plt.close()

    # PR Curve
    plt.figure(figsize=(10, 7))
    for c in range(nc): plt.plot(curve_data[c]['r'], curve_data[c]['p'], label=class_names[c], linewidth=1)
    plt.title('Precision-Recall Curve');
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout();
    plt.savefig(save_dir / "PR_curve.png", dpi=200);
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(12, 9))
    cm_norm = confusion_matrix / (confusion_matrix.sum(axis=0) + 1e-6)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names + ['background'],
                yticklabels=class_names + ['background'])
    plt.title('Confusion Matrix');
    plt.xlabel('True');
    plt.ylabel('Predicted')
    plt.tight_layout();
    plt.savefig(save_dir / "confusion_matrix.png", dpi=200);
    plt.close()

    # IoU Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=20, color='cornflowerblue', edgecolor='black')
    plt.title('IoU Distribution');
    plt.xlabel('IoU');
    plt.ylabel('Frequency')
    plt.tight_layout();
    plt.savefig(save_dir / "iou_distribution.png", dpi=200);
    plt.close()

    return {'F1': float(mean_f1[best_idx]),
            'mAP_0.5': float(np.mean([np.trapz(curve_data[c]['p'], curve_data[c]['r']) for c in range(nc)]))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-model", default="detect")
    parser.add_argument("--variant", default="r50")
    parser.add_argument("--train-run", default="")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--variants-to-compare", nargs="+", default=["r50", "r50_dc5", "r101", "r101_dc5"])
    args = parser.parse_args()

    cfg = MetricsConfig(task_model=args.task_model, variant=args.variant, train_run=args.train_run,
                        merge_mode=args.merge, variants_to_compare=args.variants_to_compare)
    if cfg.merge_mode: run_comparison_mode(cfg)


if __name__ == "__main__":
    main()