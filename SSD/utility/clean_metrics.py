# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/clean_metrics.py
# Limpia carpetas de métricas en metrics/ filtrando por:
#   1. Variante (detectada dinámicamente en metrics/detect/)
#   2. Fase [train/valid] (No existe 'test' en métricas)
#
# ACCIÓN: Elimina la CARPETA COMPLETA seleccionada.
#         (Ej: metrics/detect/ssd300/train)
# ==============================================================

import shutil
from pathlib import Path
from typing import List

# --- Configuración ---
PHASES = ["train", "valid"]


# --- Utilidades de path/proyecto ---

def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "ssd").exists():
            return parent
    return Path.cwd()


def get_available_variants(metrics_root: Path) -> List[str]:
    """
    Escanea metrics/detect/ y lista todas las carpetas que encuentre.
    Asume que cualquier carpeta ahí es una variante (ssd300, ssd512, etc).
    """
    found = set()
    detect_path = metrics_root / "detect"

    if detect_path.exists():
        for p in detect_path.iterdir():
            if p.is_dir():
                # Guardamos el nombre real de la carpeta
                found.add(p.name)

    return sorted(list(found))


# --- Entrada interactiva (Estilo HUD) ---

def select_variant(available_variants: List[str]) -> str:
    if not available_variants:
        return "none"

    options_str = "/".join(available_variants)
    # Si hay más de una variante, ofrecemos 'all'
    prompt_opts = f"{options_str}/all" if len(available_variants) > 1 else options_str

    while True:
        v = input(f"Seleccione variante [{prompt_opts}]: ").strip()  # Case sensitive por si acaso, o lower

        # Manejo de 'all'
        if v.lower() == "all" and len(available_variants) > 1:
            return "all"

        # Buscamos coincidencia exacta o insensible a mayúsculas
        for av in available_variants:
            if v.lower() == av.lower():
                return av

        print("Entrada inválida. Intente nuevamente.")


def select_phase() -> str:
    while True:
        p = input("Seleccione fase [train/valid] (o 'all'): ").strip().lower()
        if p in PHASES or p == "all":
            return p
        print("Entrada inválida. Intente nuevamente.")


def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"


# --- Lógica de Selección de Carpetas ---

def list_directories_to_delete(metrics_root: Path, variant_sel: str, phase_sel: str, available_variants: List[str]) -> \
List[Path]:
    """
    Construye las rutas: metrics/detect/{variant}/{phase}
    """
    candidates = []
    detect_root = metrics_root / "detect"

    if not detect_root.exists():
        return []

    # 1. Determinar variantes
    target_variants = available_variants if variant_sel == "all" else [variant_sel]

    for var_name in target_variants:
        var_path = detect_root / var_name

        # 2. Determinar fases
        target_phases = PHASES if phase_sel == "all" else [phase_sel]

        for phase in target_phases:
            phase_path = var_path / phase

            if phase_path.exists():
                candidates.append(phase_path)

    return candidates


# --- Eliminación ---

def remove_directories(paths: List[Path]):
    for p in paths:
        try:
            shutil.rmtree(p)
            print(f"✓ Carpeta eliminada: {p.name}")
        except Exception as e:
            print(f"✗ Error al eliminar {p.name}: {e}")


# --- Main ---

def main():
    root = find_project_root()
    metrics_root = root / "metrics"

    print(f"Raíz detectada: {root}")

    # 1. Escaneo
    available_vars = get_available_variants(metrics_root)

    if not available_vars:
        print("No se encontraron carpetas de variantes en metrics/detect/.")
        return

    # 2. Interacción HUD
    variant = select_variant(available_vars)
    phase = select_phase()

    # 3. Búsqueda
    candidates = list_directories_to_delete(metrics_root, variant, phase, available_vars)

    if not candidates:
        print("\nNo se encontraron carpetas que coincidan con los criterios.")
        return

    print("\nSe eliminarán las siguientes CARPETAS COMPLETAS en metrics/:")
    for c in candidates:
        rel = c.relative_to(root) if c.is_relative_to(root) else c
        print(f" - {rel}")

    # 4. Ejecución
    print("\n[ADVERTENCIA] Se borrarán todos los archivos dentro de estas carpetas.")
    if confirm("¿Confirma eliminación?"):
        remove_directories(candidates)

        # Limpieza de variantes vacías
        print("\n--- Verificación de residuos ---")
        detect_root = metrics_root / "detect"
        if detect_root.exists():
            for var_dir in detect_root.iterdir():
                if var_dir.is_dir() and not any(var_dir.iterdir()):
                    if confirm(f"La carpeta raíz '{var_dir.name}' quedó vacía. ¿Eliminar?"):
                        var_dir.rmdir()
                        print(f"✓ Eliminada: {var_dir.name}")
    else:
        print("Operación cancelada.")


if __name__ == "__main__":
    main()