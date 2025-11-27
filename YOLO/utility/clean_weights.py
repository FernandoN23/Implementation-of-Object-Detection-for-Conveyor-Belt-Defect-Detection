# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_weights.py
# Descripción: Limpia pesos del modelo YOLOv5 adaptado para correas
#              transportadoras, eliminando:
#              (i) el checkpoint principal yolov5{variante}.pt en
#                  YOLO/weights/ y
#              (ii) todos los archivos dentro de YOLO/weights/detect/{variante}/train.
#              Se preservan los directorios base. Consola interactiva
#              con confirmación (s/n).
# =============================================================

from pathlib import Path

# Variantes soportadas del modelo YOLOv5 (n, s, m, l, x)
VARIANTS = ["n", "s", "m", "l", "x"]


# --- Utilidades de path/proyecto ---

def find_project_root(start: Path | None = None) -> Path:
    """Intenta localizar la raíz del proyecto YOLO.

    Heurística: asciende desde este archivo hasta encontrar un
    directorio que contenga al menos:
      - configs/
      - models/

    En tu repositorio actual, esto corresponde a la carpeta YOLO/.
    Si no se encuentra, se devuelve el directorio de trabajo actual.
    """
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()


# --- Entrada interactiva ---

def select_variant() -> str:
    """Solicita al usuario la variante del modelo a limpiar."""
    while True:
        v = input("Seleccione variante [n/s/m/l/x]: ").strip().lower()
        if v in VARIANTS:
            return v
        print("Entrada inválida. Intente nuevamente.")


def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"


# --- Descubrimiento de objetivos a eliminar ---

def collect_targets(root: Path, variant: str) -> tuple[Path | None, list[Path]]:
    """Determina los elementos a borrar para una variante dada.

    - Modelo principal:
        YOLO/weights/yolov5{variant}.pt
    - Pesos de entrenamiento asociados:
        YOLO/weights/detect/{variant}/train/* (solo archivos).
    """
    # Raíces relevantes
    weights_root = root / "YOLO" / "weights"

    # 1) Checkpoint principal
    main_ckpt = weights_root / f"yolov5{variant}.pt"
    if not main_ckpt.exists():
        main_ckpt = None

    # 2) Carpeta de pesos de entrenamiento por variante
    train_dir = weights_root / "detect" / variant / "train"
    train_files: list[Path] = []
    if train_dir.exists() and train_dir.is_dir():
        for child in train_dir.iterdir():
            if child.is_file():
                train_files.append(child)

    return main_ckpt, train_files


# --- Eliminación ---

def delete_file(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
        print(f"✓ Eliminado archivo: {p}")
    except Exception as e:
        print(f"✗ Error al eliminar archivo {p}: {e}")


def main() -> None:
    # 1) Localizar raíz del proyecto (carpeta YOLO/)
    root = find_project_root()

    # 2) Elegir variante
    variant = select_variant()

    # 3) Recolectar objetivos
    main_ckpt, train_files = collect_targets(root, variant)

    if main_ckpt is None and not train_files:
        print("No se encontraron pesos para la variante seleccionada.")
        return

    print("Se eliminarán los siguientes pesos:")
    if main_ckpt is not None:
        print(f" - Checkpoint principal: {main_ckpt}")
    for f in train_files:
        print(f" - Peso en train/: {f}")

    if not confirm("¿Confirma eliminación de estos archivos?"):
        print("Operación cancelada.")
        return

    # 4) Ejecutar eliminación
    if main_ckpt is not None:
        delete_file(main_ckpt)
    for f in train_files:
        delete_file(f)


if __name__ == "__main__":
    main()
