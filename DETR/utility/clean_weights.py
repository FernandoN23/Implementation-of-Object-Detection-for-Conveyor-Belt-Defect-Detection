# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/clean_weights.py
# Descripción: Limpia checkpoints consolidados en weights/ filtrando
#              por variante. Protege estrictamente weights/base/.
# ==============================================================

import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parents[1]
WEIGHTS_ROOT = DETR_ROOT / "weights"


def get_available_variants():
    """Escanea weights/ para detectar variantes, ignorando 'base'."""
    found = set()
    if WEIGHTS_ROOT.exists():
        for p in WEIGHTS_ROOT.iterdir():
            if p.is_dir() and p.name != "base":
                found.add(p.name.lower())
    return sorted(list(found))


def select_option(prompt, options):
    options_str = " / ".join(options)
    while True:
        v = input(f"{prompt} [{options_str}]: ").strip().lower()
        if v in options or v == "all":
            return v
        print("Entrada inválida.")


def main():
    print(f"\n[clean_weights.py] --- Mantenimiento de Pesos Consolidados DETR ---")

    if not WEIGHTS_ROOT.exists():
        print("[clean_weights.py] No existe la carpeta weights/.")
        return

    available_vars = get_available_variants()

    if not available_vars:
        print("[clean_weights.py] No se encontraron pesos consolidados (fuera de 'base/').")
        return

    print("NOTA: La carpeta 'weights/base/' (pesos pre-entrenados) está protegida y no será alterada.")
    variant = select_option("Seleccione variante a limpiar", available_vars + ["all"])

    candidates = []
    for p in WEIGHTS_ROOT.rglob("*.pt"):
        # Protección estricta: Ignorar cualquier cosa dentro de 'base'
        if "base" in p.parts:
            continue

        path_str = str(p).lower()

        # Filtro de variante
        if variant != "all" and f"weights\\{variant}" not in path_str and f"weights/{variant}" not in path_str:
            continue

        candidates.append(p)

    if not candidates:
        print("[clean_weights.py] No se encontraron archivos para eliminar con esos criterios.")
        return

    print(f"\nSe eliminarán {len(candidates)} archivos de pesos consolidados:")
    for c in candidates:
        print(f" - {c.relative_to(DETR_ROOT)}")

    if input("\n¿Confirma eliminación? (s/n): ").lower() == "s":
        for p in candidates:
            p.unlink()
            print(f"[clean_weights.py] ✓ Eliminado: {p.name}")

        # Limpieza de carpetas de variantes vacías (ej. weights/r50 si quedó vacía)
        for d in sorted([p for p in WEIGHTS_ROOT.iterdir() if p.is_dir() and p.name != "base"], reverse=True):
            try:
                d.rmdir()
                print(f"[clean_weights.py] ✓ Carpeta vacía eliminada: {d.name}")
            except OSError:
                pass  # La carpeta no está vacía
    else:
        print("[clean_weights.py] Operación cancelada.")


if __name__ == "__main__":
    main()