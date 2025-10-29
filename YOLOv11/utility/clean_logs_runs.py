# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_logs_runs.py
# Limpia carpetas de logs/ y runs/ filtrando por variante (n/s/m/l/xl) y fase (train/valid/test). Consola interactiva con confirmación (s/n).
#==============================================================

import shutil
from pathlib import Path

VARIANTS = ["n", "s", "m", "l", "xl"]
PHASES = ["train", "valid", "test"]

def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()

def select_variant() -> str:
    while True:
        v = input("Seleccione variante [n/s/m/l/xl]: ").strip().lower()
        if v in VARIANTS:
            return v
        print("Entrada inválida. Intente nuevamente.")

def select_phase() -> str:
    while True:
        p = input("Seleccione fase [train/valid/test] (o 'all'): ").strip().lower()
        if p in PHASES or p == "all":
            return p
        print("Entrada inválida. Intente nuevamente.")

def select_target() -> str:
    while True:
        t = input("¿Qué limpiar? [logs/runs/ambos]: ").strip().lower()
        if t in {"logs", "runs", "ambos"}:
            return t
        print("Entrada inválida. Intente nuevamente.")

def list_candidates(base: Path, variant: str, phase: str):
    if not base.exists():
        return []
    items = []
    for p in base.rglob("*"):
        if p.is_dir():
            name = str(p).lower().replace("\\", "/")
            cond_v = variant in name
            cond_p = (phase in name) if phase in PHASES else True
            if cond_v and cond_p:
                items.append(p)
    return sorted(set(items))

def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"

def remove_paths(paths):
    for p in paths:
        try:
            shutil.rmtree(p)
            print(f"✓ Eliminado: {p}")
        except Exception as e:
            print(f"✗ Error al eliminar {p}: {e}")

def main():
    root = find_project_root()
    variant = select_variant()
    phase = select_phase()
    target = select_target()

    bases = []
    if target in {"logs", "ambos"}:
        bases.append(root / "logs")
    if target in {"runs", "ambos"}:
        bases.append(root / "runs")

    all_candidates = []
    for b in bases:
        cands = list_candidates(b, variant, phase)
        if cands:
            print(f"\nEn {b}:")
            for c in cands[:20]:
                print(f" - {c}")
            if len(cands) > 20:
                print(f" ... (+{len(cands)-20} directorios más)")
        all_candidates.extend(cands)

    if not all_candidates:
        print("\nNo se encontraron directorios que coincidan con los criterios.")
        return

    if confirm("\n¿Confirma eliminación de TODOS los directorios listados?"):
        remove_paths(all_candidates)
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    main()
