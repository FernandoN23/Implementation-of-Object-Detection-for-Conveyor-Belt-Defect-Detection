# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/utility/metrics.py
# Descripción: Motor gráfico para reportes de validación y
#              herramienta CLI interactiva para comparación
#              global de experimentos (Modo Merge).
#              *Actualizado para soportar métricas CDN de DINO*
#              *Actualizado con métricas estándar de Trade-off
#               (GFLOPs/Params) alineadas con YOLO*
#              *CORREGIDO: Curvas P y R ahora muestran clases
#               individuales alineadas al estándar YOLO*
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
from typing import Dict, List, Optional, Tuple

# --- CONFIGURACIÓN DE ESTILO Y CONSTANTES ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

FILE = Path(__file__).resolve()
UTILITY_ROOT = FILE.parent
DINO_ROOT = UTILITY_ROOT.parent
METRICS_ROOT = DINO_ROOT / "metrics" / "detect"

# Parámetros extraídos de la literatura oficial de DINO (en Millones)
DINO_PARAMS_M = {
    "r50_4scale": 47.0,
    "r50_5scale": 47.0,
    "swin_l": 218.0,
}

# GFLOPs extraídos de la literatura oficial de DINO y estimaciones técnicas
DINO_GFLOPS = {
    "r50_4scale": 279.0,  # Dato oficial tabla DINO
    "r50_5scale": 860.0,  # Dato oficial tabla DINO
    "swin_l": 1040.0,  # Estimado (Swin-L base + 4-scale attention)
}

# Paleta estándar para asignar colores únicos por experimento (run)
STANDARD_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


# ---------------------------------------------------------------------------
# Utilidades de Procesamiento
# ---------------------------------------------------------------------------

def smooth_signal(scalars: List[float], weight: float = 0.6) -> List[float]:
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
    if not path.is_file(): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Clase MergeManager: Gestión Interactiva de Comparativas
# ---------------------------------------------------------------------------

class MergeManager:
    def __init__(self):
        self.selected_runs: Dict[str, Tuple[str, pd.DataFrame]] = {}  # run_name -> (variant, df)
        self.run_colors: Dict[str, str] = {}  # run_name -> color hex
        self.comparison_name = ""
        self.base_output_dir = METRICS_ROOT / "global_comparison"

    def get_available_variants(self) -> List[str]:
        if not METRICS_ROOT.exists(): return []
        return sorted([d.name for d in METRICS_ROOT.iterdir() if d.is_dir() and d.name != "global_comparison"])

    def get_available_runs(self, variant: str) -> List[str]:
        train_path = METRICS_ROOT / variant / "train"
        if not train_path.exists(): return []
        return sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    def interactive_selection(self):
        print(f"\n[metrics.py] --- Configuración de Comparativa Global ---")

        while True:
            variants = self.get_available_variants()
            if not variants:
                print("[metrics.py] ERROR: No se detectaron métricas en metrics/detect/")
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
            csv_path = METRICS_ROOT / variant / "train" / run_name / "results.csv"

            if csv_path.exists():
                df = load_results_csv(csv_path)
                unique_key = f"{run_name} ({variant})"
                self.selected_runs[unique_key] = (variant, df)
                print(f"[metrics.py] ✓ Añadido: {unique_key}")
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

        loss_types = {"Total Loss (Val)": "val/loss", "Classification Loss (Val)": "val/loss_ce",
                      "BBox Loss (Val)": "val/loss_bbox"}
        for title, keyword in loss_types.items():
            self._plot_comparative(keyword, title, "Loss", losses_out / f"compare_{keyword.replace('/', '_')}.png", 0.7)

        metric_types = {"Precision": "metrics/precision", "Recall": "metrics/recall", "F1-Score": "metrics/F1",
                        "mAP@0.5": "metrics/mAP_0.5", "mAP@0.5:0.95": "metrics/mAP_0.5:0.95"}
        for title, keyword in metric_types.items():
            safe_name = keyword.replace("metrics/", "").replace(":", "_")
            self._plot_comparative(keyword, f"Comparativa {title}", title, metrics_out / f"compare_{safe_name}.png",
                                   0.5)

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

        plt.xlabel("Epoch");
        plt.ylabel(ylabel);
        plt.title(title)
        plt.legend(frameon=True, framealpha=0.9);
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout();
        plt.savefig(out_path, dpi=200);
        plt.close()

    def _plot_tradeoffs(self, output_dir: Path):
        """Genera gráficos de trade-off para Parámetros y GFLOPs."""
        data_points = []
        for label, (variant, df) in self.selected_runs.items():
            if 'metrics/mAP_0.5:0.95' not in df.columns: continue
            best_map = df['metrics/mAP_0.5:0.95'].max()
            if pd.isna(best_map) or best_map == 0: continue

            data_points.append({
                'label': label,
                'variant': variant,
                'params': DINO_PARAMS_M.get(variant, 0),
                'gflops': DINO_GFLOPS.get(variant, 0),
                'map': best_map,
                'color': self.run_colors[label]
            })

        if not data_points: return

        # --- Gráfico 1: Performance vs Parámetros ---
        data_points.sort(key=lambda x: x['params'])
        self._render_scatter_plot(
            data_points, 'params', "Parámetros (Millones)", "Best mAP@0.5:0.95",
            "Trade-off: Performance vs Parámetros (DINO)",
            output_dir / "tradeoff_performance_params.png"
        )

        # --- Gráfico 2: Performance vs GFLOPs ---
        data_points.sort(key=lambda x: x['gflops'])
        self._render_scatter_plot(
            data_points, 'gflops', "GFLOPs", "Best mAP@0.5:0.95",
            "Trade-off: Performance vs Costo Computacional (GFLOPs) (DINO)",
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
                "best_mAP_0.5": float(df['metrics/mAP_0.5'].max()) if 'metrics/mAP_0.5' in df.columns else 0,
                "best_mAP_0.5_0.95": float(
                    df['metrics/mAP_0.5:0.95'].max()) if 'metrics/mAP_0.5:0.95' in df.columns else 0,
                "best_F1": float(df['metrics/F1'].max()) if 'metrics/F1' in df.columns else 0,
                "params_M": DINO_PARAMS_M.get(variant, 0),
                "gflops": DINO_GFLOPS.get(variant, 0)
            }
        with open(output_dir / "global_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Lógica de Validación (Reporte Individual)
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
            prec = tp / (tp + fp + 1e-6);
            rec = tp / (tp + fn + 1e-6)
            curve_data[c]['p'].append(prec);
            curve_data[c]['r'].append(rec);
            curve_data[c]['f1'].append(2 * prec * rec / (prec + rec + 1e-6))

    # --- Generación de Gráficos ---

    # 1. F1-Confidence Curve
    plt.figure(figsize=(10, 7))
    f1_all = []
    for c in range(nc):
        plt.plot(conf_levels, curve_data[c]['f1'], label=class_names[c], linewidth=1)
        f1_all.append(curve_data[c]['f1'])
    mean_f1 = np.mean(f1_all, axis=0);
    best_idx = np.argmax(mean_f1)
    plt.plot(conf_levels, mean_f1, label=f'all classes {mean_f1[best_idx]:.2f} at {conf_levels[best_idx]:.3f}',
             color='blue', linewidth=3)
    plt.title('F1-Confidence Curve');
    plt.xlabel('Confidence');
    plt.ylabel('F1');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.tight_layout();
    plt.savefig(save_dir / "F1_curve.png", dpi=200, bbox_inches='tight');
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(10, 7))
    for c in range(nc): plt.plot(curve_data[c]['r'], curve_data[c]['p'], label=class_names[c], linewidth=1)
    plt.title('Precision-Recall Curve');
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.tight_layout();
    plt.savefig(save_dir / "PR_curve.png", dpi=200, bbox_inches='tight');
    plt.close()

    # 3. Precision-Confidence Curve [CORREGIDO]
    plt.figure(figsize=(10, 7))
    p_all = []
    for c in range(nc):
        plt.plot(conf_levels, curve_data[c]['p'], label=class_names[c], linewidth=1)
        p_all.append(curve_data[c]['p'])
    plt.plot(conf_levels, np.mean(p_all, axis=0), color='blue', linewidth=3, label='all classes')
    plt.title('Precision-Confidence Curve');
    plt.xlabel('Confidence');
    plt.ylabel('Precision');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.tight_layout();
    plt.savefig(save_dir / "P_curve.png", dpi=200, bbox_inches='tight');
    plt.close()

    # 4. Recall-Confidence Curve [CORREGIDO]
    plt.figure(figsize=(10, 7))
    r_all = []
    for c in range(nc):
        plt.plot(conf_levels, curve_data[c]['r'], label=class_names[c], linewidth=1)
        r_all.append(curve_data[c]['r'])
    plt.plot(conf_levels, np.mean(r_all, axis=0), color='blue', linewidth=3, label='all classes')
    plt.title('Recall-Confidence Curve');
    plt.xlabel('Confidence');
    plt.ylabel('Recall');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.tight_layout();
    plt.savefig(save_dir / "R_curve.png", dpi=200, bbox_inches='tight');
    plt.close()

    # 5. Confusion Matrix
    plt.figure(figsize=(12, 9))
    sns.heatmap(confusion_matrix / (confusion_matrix.sum(axis=0) + 1e-6), annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names + ['background'], yticklabels=class_names + ['background'])
    plt.title('Confusion Matrix');
    plt.xlabel('True');
    plt.ylabel('Predicted');
    plt.tight_layout();
    plt.savefig(save_dir / "confusion_matrix.png", dpi=200);
    plt.close()

    # 6. IoU Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=20, color='cornflowerblue', edgecolor='black')
    plt.title('IoU Distribution');
    plt.xlabel('IoU');
    plt.ylabel('Frequency');
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.tight_layout();
    plt.savefig(save_dir / "iou_distribution.png", dpi=200);
    plt.close()

    return {'F1': float(mean_f1[best_idx]), 'precision': float(np.mean(p_all, axis=0)[best_idx]),
            'recall': float(np.mean(r_all, axis=0)[best_idx])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="Activar modo comparativo interactivo.")
    args = parser.parse_args()

    if args.merge:
        manager = MergeManager()
        if manager.interactive_selection():
            manager.run_comparison()
    else:
        print("[metrics.py] Use --merge para iniciar la comparativa interactiva.")


if __name__ == "__main__":
    main()