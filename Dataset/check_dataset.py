# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: Dataset/check_dataset.py
# Chequea consistencia del dataset en formato YOLO (imágenes y labels) a partir
# de configs/dataset.yaml o ruta provista. ACEPTA explícitamente imágenes sin
# clases (negativas), ya sea con label vacío o sin archivo de label, y las
# reporta como tales sin marcarlas como error (salvo que se use --strict).
# ==============================================================

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Intentamos importar librerías de ploteo de forma segura
try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LBL_EXT = ".txt"


def load_dataset_yaml(ds_yaml: Path) -> Dict:
    """Carga y normaliza el YAML del dataset."""
    try:
        with open(ds_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"CRITICAL: No se encontró el archivo YAML en: {ds_yaml}")
        sys.exit(1)

    keys_map = {"val": "valid", "test": "test"}
    for k_old, k_new in keys_map.items():
        if k_old in data and k_new not in data:
            data[k_new] = data[k_old]
    return data


def resolve_path(path_str: str, parent_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (parent_dir / p).resolve()


def images_in(path: Path) -> List[Path]:
    if not path.exists():
        print(f"WARNING: El directorio de imágenes no existe: {path}")
        return []
    files = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def label_path_for(img_path: Path) -> Path:
    """Deduce la ruta del label (soporta estructura flat y standard)."""
    parts = list(img_path.parts)
    try:
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            lbl = Path(*parts).with_suffix(LBL_EXT)
            if lbl.exists(): return lbl
    except ValueError:
        pass

    if img_path.parent.name == "images":
        lbl = img_path.parent.parent / "labels" / img_path.with_suffix(LBL_EXT).name
        if lbl.exists(): return lbl

    return img_path.with_suffix(LBL_EXT)


def parse_label_line(line: str) -> Tuple[int, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Formato incorrecto (esperado: class x y w h): '{line.strip()}'")
    c = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    return c, x, y, w, h


def check_split(name: str, images_dir: Path, nc: int, allow_empty: bool, allow_missing: bool) -> Dict:
    imgs = images_in(images_dir)
    total = len(imgs)

    # Contenedores de errores
    issues = {
        "missing_labels": [], "empty_labels": [], "bad_lines": [], "bad_ranges": [], "class_oob": [],
    }
    negatives = {"negatives_missing": [], "negatives_empty": []}

    # Métricas
    affected_images_set = set()  # Para contar imágenes únicas con problemas
    total_failures_count = 0
    class_counts = Counter()

    print(f"   > Analizando split '{name}' ({total} imágenes)...")

    for img in imgs:
        lbl = label_path_for(img)
        img_has_error = False

        # 1. Chequeo de existencia
        if not lbl.exists():
            if allow_missing:
                negatives["negatives_missing"].append(str(img))
                # No es error técnico si se permiten, no suma a failures
            else:
                issues["missing_labels"].append(str(img))
                img_has_error = True
                total_failures_count += 1

            if img_has_error: affected_images_set.add(img)
            continue  # Si no existe, no podemos leer líneas

        # 2. Lectura
        try:
            content = lbl.read_text(encoding="utf-8").strip()
        except Exception as e:
            issues["bad_lines"].append(f"{lbl} :: error I/O: {e}")
            affected_images_set.add(img)
            total_failures_count += 1
            continue

        if not content:
            if allow_empty:
                negatives["negatives_empty"].append(str(img))
            else:
                issues["empty_labels"].append(str(lbl))
                affected_images_set.add(img)
                total_failures_count += 1
            continue

        # 3. Validación de contenido
        for i, line in enumerate(content.splitlines()):
            line_error = False
            try:
                c, x, y, w, h = parse_label_line(line)

                # Clase fuera de rango
                if c < 0 or c >= nc:
                    issues["class_oob"].append(f"{lbl} (L{i + 1}): Class {c} not in [0, {nc - 1}]")
                    line_error = True
                else:
                    # Conteo de clases válidas
                    class_counts[c] += 1

                # Coordenadas fuera de rango
                for v, k in zip((x, y, w, h), "xywh"):
                    if not (0.0 <= v <= 1.0):
                        issues["bad_ranges"].append(f"{lbl} (L{i + 1}): {k}={v} not in [0,1]")
                        line_error = True
                        break  # Reportamos una vez por línea para no saturar

            except Exception as e:
                issues["bad_lines"].append(f"{lbl} (L{i + 1}): {e}")
                line_error = True

            if line_error:
                img_has_error = True
                total_failures_count += 1

        if img_has_error:
            affected_images_set.add(img)

    ok = (total > 0 or (total == 0 and allow_missing)) and all(len(v) == 0 for k, v in issues.items())

    return {
        "split": name,
        "images_dir": str(images_dir),
        "total_images": total,
        "affected_images_count": len(affected_images_set),
        "total_failures": total_failures_count,
        "class_counts": class_counts,
        "ok": ok,
        **issues, **negatives,
    }


def print_summary_table(results: List[Dict], names: List[str]):
    """Imprime una tabla resumen formateada a la consola."""
    # Ajuste dinámico de ancho de columnas para clases
    col_width = max(8, max([len(n) for n in names] + [0]) + 2) if names else 8
    class_headers = "".join([f" | {name[:col_width - 2].center(col_width - 2)}" for name in names])
    table_width = 85 + len(class_headers)

    print("\n" + "=" * table_width)
    print(f" RESUMEN EJECUTIVO DEL DATASET")
    print("=" * table_width)
    # Header
    header = f"{'PARTICIÓN':<10} | {'IMÁGENES':<10} | {'AFECTADAS':<10} | {'FALLAS TOT.':<12}{class_headers} | {'ESTADO':<10}"
    print(header)
    print("-" * table_width)

    total_imgs = 0
    total_affected = 0
    total_fails = 0
    total_class_counts = Counter()
    global_status = True

    for r in results:
        status_str = "✅ OK" if r['ok'] else "❌ FAIL"

        # Formateo de columnas de clases
        c_counts = r['class_counts']
        class_cols = "".join([f" | {str(c_counts.get(i, 0)).center(col_width - 2)}" for i in range(len(names))])

        row = f"{r['split'].upper():<10} | {r['total_images']:<10} | {r['affected_images_count']:<10} | {r['total_failures']:<12}{class_cols} | {status_str:<10}"
        print(row)

        total_imgs += r['total_images']
        total_affected += r['affected_images_count']
        total_fails += r['total_failures']
        total_class_counts += c_counts
        if not r['ok']: global_status = False

    print("-" * table_width)
    final_status = "LISTO" if global_status else "REVISAR"

    # Fila de totales de clases
    total_class_cols = "".join(
        [f" | {str(total_class_counts.get(i, 0)).center(col_width - 2)}" for i in range(len(names))])

    print(
        f"{'TOTAL':<10} | {total_imgs:<10} | {total_affected:<10} | {total_fails:<12}{total_class_cols} | {final_status:<10}")
    print("=" * table_width + "\n")


def ensure_info_dir(ds_yaml: Path) -> Path:
    """Asegura que exista la carpeta Dataset/info para guardar gráficos."""
    info_dir = ds_yaml.parent / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    return info_dir


def plot_class_distribution(results: List[Dict], names: List[str], output_dir: Path):
    """Genera un gráfico de barras agrupadas con la distribución de clases."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n[INFO] matplotlib no está instalado. Omitiendo generación de gráfico.")
        return

    splits = [r['split'].upper() for r in results]
    n_splits = len(splits)
    n_classes = len(names)

    x = np.arange(n_classes)
    width = 0.8 / n_splits

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.8), 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, r in enumerate(results):
        counts = [r['class_counts'].get(idx, 0) for idx in range(n_classes)]
        pos = x + (i * width) - (n_splits * width / 2) + (width / 2)

        rects = ax.bar(pos, counts, width, label=r['split'].upper(), color=colors[i % len(colors)], alpha=0.85,
                       edgecolor='black', linewidth=0.5)

        if n_classes < 20:
            # Corrección solicitada: Números horizontales para mejor lectura
            ax.bar_label(rects, padding=3, fontsize=9, rotation=0)

    ax.set_ylabel('Frecuencia (Cantidad de etiquetas)', fontsize=10, fontweight='bold')
    ax.set_title('Distribución de Clases por Partición del Dataset', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend(title="Partición")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    filename = output_dir / "class_distribution.png"
    try:
        plt.savefig(filename, dpi=300)
        print(f"📊 Gráfico de Clases guardado en: {filename}")
    except Exception as e:
        print(f"ERROR al guardar gráfico de clases: {e}")
    finally:
        plt.close()


def plot_split_composition(results: List[Dict], output_dir: Path):
    """
    Genera un gráfico de barras apiladas mostrando la relación Señal/Ruido
    (Imágenes con Objetos vs Background) para cada partición.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    splits = [r['split'].upper() for r in results]

    # Preparación de datos
    total_imgs = np.array([r['total_images'] for r in results])
    # Background (Negativos) = Missing labels + Empty labels
    negatives = np.array([len(r['negatives_missing']) + len(r['negatives_empty']) for r in results])
    # Objetos (Positivos) = Total - Negativos
    positives = total_imgs - negatives

    # Configuración del plot
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.55  # Un poco más angostas para estética académica

    # Colores corporativos
    color_pos = '#2b83ba'  # Azul corporativo
    color_neg = '#fdae61'  # Naranja suave

    # Barras apiladas
    p1 = ax.bar(splits, positives, width, label='Con Objetos', color=color_pos, edgecolor='black', alpha=0.9)
    p2 = ax.bar(splits, negatives, width, bottom=positives, label='Background (Negativos)', color=color_neg,
                edgecolor='black', alpha=0.9, hatch='//')

    # Umbral dinámico para decidir si el texto va dentro o fuera (5% de la altura máxima)
    y_limit = max(total_imgs) * 1.2  # Damos 20% de aire arriba
    ax.set_ylim(0, y_limit)
    threshold = y_limit * 0.05

    # Aplicar etiquetas inteligentes
    for i, (rect_pos, rect_neg) in enumerate(zip(p1, p2)):
        # 1. Capa Positivos (Azul)
        h_pos = rect_pos.get_height()
        if h_pos > 0:
            pct_pos = h_pos / total_imgs[i] * 100
            # Si el segmento es muy pequeño, la etiqueta va afuera.
            # Casi siempre esta es la capa grande, así que suele ir adentro.
            if h_pos > threshold:
                ax.text(rect_pos.get_x() + rect_pos.get_width() / 2., rect_pos.get_y() + h_pos / 2.,
                        f'{int(h_pos)}\n({pct_pos:.1f}%)', ha='center', va='center', color='white', fontweight='bold',
                        fontsize=9)
            else:
                # Si es muy chico, lo ponemos arriba de su propia barra (pero esto podría chocar con la naranja)
                # En composición típica, si POS es muy chico, casi todo es NEG.
                ax.text(rect_pos.get_x() + rect_pos.get_width() / 2., h_pos / 2.,
                        f'{int(h_pos)}', ha='center', va='center', color='white', fontsize=8)

        # 2. Capa Negativos (Naranja/Background)
        h_neg = rect_neg.get_height()
        if h_neg > 0:
            pct_neg = h_neg / total_imgs[i] * 100

            # Lógica crítica: Si es muy pequeño (típico), poner etiqueta FLOTANTE arriba de la barra total
            if h_neg > threshold:
                # Cabe adentro
                ax.text(rect_neg.get_x() + rect_neg.get_width() / 2., rect_neg.get_y() + h_neg / 2.,
                        f'{int(h_neg)}\n({pct_neg:.1f}%)', ha='center', va='center', color='black', fontweight='bold',
                        fontsize=9)
            else:
                # NO cabe adentro -> Ponerlo flotando encima de la barra completa
                # El offset vertical ayuda a separarlo de la barra
                ax.text(rect_neg.get_x() + rect_neg.get_width() / 2., rect_neg.get_y() + h_neg + (y_limit * 0.01),
                        f'{int(h_neg)} ({pct_neg:.1f}%)', ha='center', va='bottom', color='black', fontsize=9,
                        fontweight='bold')

        # 3. Etiqueta de TOTAL general
        # La ponemos bastante más arriba para evitar colisión con etiquetas flotantes
        ax.text(i, total_imgs[i] + (y_limit * 0.08), f'Total: {total_imgs[i]}',
                ha='center', va='bottom', fontweight='bold', color='#333333')

    ax.set_ylabel('Cantidad de Imágenes', fontsize=10, fontweight='bold')
    ax.set_title('Composición del Dataset: Objetos vs. Background', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    filename = output_dir / "split_composition.png"
    try:
        plt.savefig(filename, dpi=300)
        print(f"📊 Gráfico de Composición guardado en: {filename}")
    except Exception as e:
        print(f"ERROR al guardar gráfico de composición: {e}")
    finally:
        plt.close()


def pretty_print_report(results: List[Dict], ds_yaml: Path, nc: int, names: List[str], generate_plots: bool) -> int:
    # 1. Imprimir Tabla Resumen
    print_summary_table(results, names)

    # 2. Generar Gráficos en Dataset/info
    if generate_plots:
        info_dir = ensure_info_dir(ds_yaml)
        plot_class_distribution(results, names, info_dir)
        plot_split_composition(results, info_dir)

    # 3. Imprimir Detalles si hay errores
    exit_code = 0
    any_issue = False

    print("\n DETALLE DE INCONSISTENCIAS:")
    for r in results:
        if not r['ok']:
            any_issue = True
            print(f"\n📂 [{r['split'].upper()}]")
            issue_keys = ["missing_labels", "empty_labels", "bad_lines", "bad_ranges", "class_oob"]
            for k in issue_keys:
                if r[k]:
                    exit_code = 1
                    print(f"   ⚠  {k}: {len(r[k])} incidencias.")
                    for s in r[k][:5]:  # Muestra los primeros 5 errores
                        print(f"      - {s}")
                    if len(r[k]) > 5:
                        print(f"      - ... (+{len(r[k]) - 5} restantes)")

    if not any_issue:
        print("   Ninguna inconsistencia técnica detectada.")

    # 4. Reporte de Negativos (Informativo)
    print("\n INFORME DE NEGATIVOS (Permitidos):")
    has_negatives = False
    for r in results:
        total_neg = len(r['negatives_missing']) + len(r['negatives_empty'])
        if total_neg > 0:
            has_negatives = True
            print(f"   - {r['split'].upper()}: {total_neg} imágenes negativas (Background only).")
    if not has_negatives:
        print("   - No se detectaron imágenes negativas.")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Validación técnica de Dataset YOLO con reporte detallado.")
    default_yaml = Path(__file__).parent / "data.yaml"

    parser.add_argument("--dataset-yaml", type=str, default=str(default_yaml),
                        help=f"Ruta al archivo .yaml (Default: {default_yaml})")
    parser.add_argument("--strict", action="store_true",
                        help="Modo estricto: Prohíbe imágenes sin etiqueta (negativos).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Desactiva la generación de gráficos.")

    args = parser.parse_args()
    ds_yaml = Path(args.dataset_yaml).resolve()

    if not ds_yaml.exists():
        print(f"Error: No se encuentra el archivo de configuración: {ds_yaml}")
        sys.exit(1)

    print(f"⚙️  Iniciando auditoría de dataset en: {ds_yaml}")
    data = load_dataset_yaml(ds_yaml)

    nc = int(data.get("nc", 0))
    names = data.get("names", [])

    allow_empty = not args.strict
    allow_missing = not args.strict

    results = []
    splits = ["train", "valid", "test"]

    for split_key in splits:
        if split_key in data:
            images_dir = resolve_path(data[split_key], ds_yaml.parent)
            results.append(check_split(split_key, images_dir, nc, allow_empty, allow_missing))

    if not results:
        print("Error: No se encontraron particiones válidas en el YAML.")
        sys.exit(1)

    sys.exit(pretty_print_report(results, ds_yaml, nc, names, generate_plots=not args.no_plot))


if __name__ == "__main__":
    main()