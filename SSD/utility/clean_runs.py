# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/clean_runs.py
# Limpia carpetas de logs/runs en runs/ filtrando por:
#   1. Variante (detectada dinámicamente en runs/detect/)
#   2. Fase [train/valid/test]
#
# ACCIÓN: Elimina la CARPETA DE FASE COMPLETA.
#         (Ej: runs/detect/ssd300/train -> Borra todos los experimentos dentro)
# ==============================================================

import shutil
from pathlib import Path
from typing import List

# --- Configuración ---
# En runs sí es común tener 'test' además de train/valid
PHASES = ["train", "valid", "test"]


# --- Utilidades de path/proyecto ---

def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "ssd").exists():
            return parent
    return Path.cwd()


def get_available_variants(runs_root: Path) -> List[str]:
    """
    Escanea runs/detect/ y lista todas las carpetas que encuentre.
    (ej: ssd300, ssd512)
    """
    found = set()
    detect_path = runs_root / "detect"

    if detect_path.exists():
        for p in detect_path.iterdir():
            if p.is_dir():
                found.add(p.name)

    return sorted(list(found))


# --- Entrada interactiva (Estilo HUD) ---

def select_variant(available_variants: List[str]) -> str:
    if not available_variants:
        return "none"

    options_str = "/".join(available_variants)
    prompt_opts = f"{options_str}/all" if len(available_variants) > 1 else options_str

    while True:
        v = input(f"Seleccione variante [{prompt_opts}]: ").strip()

        if v.lower() == "all" and len(available_variants) > 1:
            return "all"

        for av in available_variants:
            if v.lower() == av.lower():
                return av

        print("Entrada inválida. Intente nuevamente.")


def select_phase() -> str:
    while True:
        p = input("Seleccione fase [train/valid/test] (o 'all'): ").strip().lower()
        if p in PHASES or p == "all":
            return p
        print("Entrada inválida. Intente nuevamente.")


def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"


# --- Lógica de Selección de Carpetas ---

def list_directories_to_delete(runs_root: Path, variant_sel: str, phase_sel: str, available_variants: List[str]) -> \
List[Path]:
    """
    Construye las rutas: runs/detect/{variant}/{phase}
    """
    candidates = []
    detect_root = runs_root / "detect"

    if not detect_root.exists():
        return []

    # 1. Variantes
    target_variants = available_variants if variant_sel == "all" else [variant_sel]

    for var_name in target_variants:
        var_path = detect_root / var_name

        # 2. Fases
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
    runs_root = root / "runs"

    print(f"Raíz detectada: {root}")

    # 1. Escaneo
    available_vars = get_available_variants(runs_root)

    if not available_vars:
        print("No se encontraron carpetas de variantes en runs/detect/.")
        return

    # 2. Interacción HUD
    variant = select_variant(available_vars)
    phase = select_phase()

    # 3. Búsqueda
    candidates = list_directories_to_delete(runs_root, variant, phase, available_vars)

    if not candidates:
        print("\nNo se encontraron carpetas que coincidan con los criterios.")
        return

    print("\nSe eliminarán las siguientes CARPETAS COMPLETAS en runs/:")
    for c in candidates:
        rel = c.relative_to(root) if c.is_relative_to(root) else c
        print(f" - {rel}")

    # 4. Ejecución
    print("\n[ADVERTENCIA] Se borrarán todos los experimentos (logs, yamls) dentro de estas fases.")
    if confirm("¿Confirma eliminación?"):
        remove_directories(candidates)

        # Limpieza de variantes vacías
        print("\n--- Verificación de residuos ---")
        detect_root = runs_root / "detect"
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