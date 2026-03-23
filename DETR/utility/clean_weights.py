# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/clean_weights.py
# Descripción: Limpia checkpoints en weights/ filtrando por:
#              Variante (r50, r101, dc5), Fase y Escenario.
# ==============================================================

import re
import shutil
from pathlib import Path
from typing import Iterable, List

# Constantes base de DETR
DEFAULT_VARIANTS = ["r50", "r50_dc5", "r101", "r101_dc5"]
PHASES = ["train", "valid", "test"]
SCENARIOS = ["tests", "final", "all"]

FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parents[1]
WEIGHTS_ROOT = DETR_ROOT / "weights"


def get_available_variants():
    """Escanea weights/ para detectar variantes de DETR."""
    found = set()
    if WEIGHTS_ROOT.exists():
        # Busca patrones tipo r50, r101, dc5
        regex = re.compile(r"(r\d+|dc5)", re.IGNORECASE)
        for p in WEIGHTS_ROOT.rglob("*"):
            if p.is_dir():
                match = regex.search(p.name)
                if match:
                    found.add(match.group(1).lower())
    return sorted(list(found)) if found else DEFAULT_VARIANTS


def select_option(prompt, options):
    options_str = "/".join(options)
    while True:
        v = input(f"{prompt} [{options_str}]: ").strip().lower()
        if v in options or v == "all":
            return v
        print("Entrada inválida.")


def main():
    print(f"\n--- Mantenimiento de Pesos DETR ---")
    available_vars = get_available_variants()

    variant = select_option("Seleccione variante", available_vars)
    phase = select_option("Seleccione fase", PHASES + ["all"])
    which = select_option("¿Qué pesos eliminar?", ["best", "last", "all"])

    candidates = []
    for p in WEIGHTS_ROOT.rglob("*.pth"):
        path_str = str(p).lower()
        # Filtro de variante
        if variant != "all" and variant not in path_str:
            continue
        # Filtro de fase
        if phase != "all" and phase not in path_str:
            continue
        # Filtro de archivo específico
        if which != "all" and f"{which}.pth" not in path_str:
            continue
        candidates.append(p)

    if not candidates:
        print("No se encontraron archivos para eliminar.")
        return

    print(f"\nSe eliminarán {len(candidates)} archivos:")
    for c in candidates[:10]:
        print(f" - {c.relative_to(DETR_ROOT)}")

    if input("\n¿Confirma eliminación? (s/n): ").lower() == "s":
        for p in candidates:
            p.unlink()
            print(f"✓ Eliminado: {p.name}")

        # Limpieza de carpetas vacías
        for d in sorted([p for p in WEIGHTS_ROOT.rglob("*") if p.is_dir()], reverse=True):
            try:
                d.rmdir()
                print(f"✓ Carpeta vacía eliminada: {d.name}")
            except OSError:
                pass
    else:
        print("Operación cancelada.")


if __name__ == "__main__":
    main()