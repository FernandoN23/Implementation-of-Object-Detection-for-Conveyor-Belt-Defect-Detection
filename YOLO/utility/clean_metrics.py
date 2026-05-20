# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLO/utility/clean_metrics.py
# Descripción: Limpia métricas del modelo YOLOv5 adaptado para correas
#              transportadoras, eliminando el contenido interno de:
#              YOLO/metrics/detect/{variante}/{carpeta}
#              donde {carpeta} ∈ {final_metrics, train, val}.
#              Se preservan siempre los directorios base. Consola
#              interactiva con confirmación (s/n).
# =============================================================

from pathlib import Path
import shutil
from typing import Optional

# Variantes soportadas del modelo YOLOv5 (n, s, m, l, x)
VARIANTS = ["n", "s", "m", "l", "x"]
METRIC_FOLDERS = ["final_metrics", "train", "val"]


# --- Utilidades de path/proyecto ---

def find_project_root(start: Optional[Path] = None) -> Path:
    """Intenta localizar la raíz del proyecto YOLO.

    Heurística: asciende desde este archivo hasta encontrar un
    directorio que contenga al menos:
      - configs/
      - models/

    En el repositorio actual, esto corresponde a la carpeta YOLO/.
    Si no se encuentra, se devuelve el directorio de trabajo actual.
    """
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()


# --- Entrada interactiva ---

def select_variant() -> str:
    """Solicita al usuario la variante del modelo cuyas métricas desea limpiar."""
    while True:
        v = input("Seleccione variante [n/s/m/l/x]: ").strip().lower()
        if v in VARIANTS:
            return v
        print("Entrada inválida. Intente nuevamente.")


def select_metrics_folder() -> str:
    """Selecciona la carpeta de métricas a limpiar para la variante dada."""
    opts = "/".join(METRIC_FOLDERS)
    while True:
        f = input(f"Seleccione carpeta de métricas [{opts}]: ").strip().lower()
        if f in METRIC_FOLDERS:
            return f
        print("Entrada inválida. Intente nuevamente.")


def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"


# --- Operaciones sobre métricas ---

def collect_metric_files(root: Path, variant: str, folder: str) -> tuple[Path, list[Path]]:
    """Determina la carpeta objetivo y los archivos internos a eliminar.

    Estructura objetivo:
        root/YOLO/metrics/detect/{variant}/{folder}

    Se preserva siempre el directorio {folder}, eliminando solo su contenido
    (archivos y subdirectorios).
    """
    metrics_root = root / "YOLO" /"metrics" / "detect" / variant / folder
    files: list[Path] = []

    if metrics_root.exists() and metrics_root.is_dir():
        for child in metrics_root.iterdir():
            files.append(child)

    return metrics_root, files


def delete_child(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink(missing_ok=True)
        print(f"✓ Eliminado: {path}")
    except Exception as e:
        print(f"✗ Error al eliminar {path}: {e}")


# --- Main ---

def main() -> None:
    # 1) Localizar raíz del proyecto (carpeta YOLO/)
    root = find_project_root()

    # 2) Elegir variante y carpeta de métricas
    variant = select_variant()
    folder = select_metrics_folder()

    # 3) Recolectar objetivos
    metrics_root, children = collect_metric_files(root, variant, folder)

    if not metrics_root.exists() or not metrics_root.is_dir():
        print(f"La carpeta de métricas no existe: {metrics_root}")
        return

    if not children:
        print(f"No se encontraron elementos dentro de: {metrics_root}")
        return

    print("Se eliminará el contenido interno de la siguiente carpeta, preservando el directorio base:")
    print(f" - Carpeta: {metrics_root}")
    print("Elementos a eliminar:")
    for ch in children[:20]:
        print(f"   · {ch}")
    if len(children) > 20:
        print(f"   ... (+{len(children) - 20} elementos más)")

    if not confirm("¿Confirma eliminación del contenido interno?"):
        print("Operación cancelada.")
        return

    # 4) Ejecutar eliminación de contenido interno
    for ch in children:
        delete_child(ch)


if __name__ == "__main__":
    main()
