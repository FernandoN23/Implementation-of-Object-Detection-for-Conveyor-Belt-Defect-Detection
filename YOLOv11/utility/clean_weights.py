# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_weights.py
# Limpia pesos/chekpoints en weights/ filtrando por variante (n/s/m/l/xl) y fase (train/valid/test). Permite elegir 'best', 'last' o 'all'. Consola interactiva con confirmación (s/n).
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

def select_which() -> str:
    while True:
        w = input("¿Qué pesos desea eliminar? [best/last/all]: ").strip().lower()
        if w in {"best", "last", "all"}:
            return w
        print("Entrada inválida. Intente nuevamente.")

def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"

def list_candidates(weights_root: Path, variant: str, phase: str, which: str):
    if not weights_root.exists():
        return []
    items = []
    patterns = []
    if which in {"best", "last"}:
        patterns = [f"{which}.pt"]
    # Gather files and directories
    for p in weights_root.rglob("*"):
        name = str(p).lower().replace("\\", "/")
        cond_v = variant in name
        cond_p = (phase in name) if phase in PHASES else True
        if not (cond_v and cond_p):
            continue
        if which == "all":
            items.append(p)
        else:
            if p.is_file() and any(name.endswith(ptn) for ptn in patterns):
                items.append(p)
    # Deduplicate and sort: delete files first, then dirs
    files = sorted({p for p in items if p.is_file()})
    dirs = sorted({p for p in items if p.is_dir()})
    return files + dirs

def remove_paths(paths):
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            print(f"✓ Eliminado: {p}")
        except Exception as e:
            print(f"✗ Error al eliminar {p}: {e}")

def main():
    root = find_project_root()
    variant = select_variant()
    phase = select_phase()
    which = select_which()
    weights_root = root / "weights"

    candidates = list_candidates(weights_root, variant, phase, which)
    if not candidates:
        print("No se encontraron pesos/checkpoints que coincidan con los criterios.")
        return

    print("Se eliminarán los siguientes elementos en weights/:")
    for c in candidates[:20]:
        print(f" - {c}")
    if len(candidates) > 20:
        print(f" ... (+{len(candidates)-20} más)")

    if confirm("¿Confirma eliminación?"):
        remove_paths(candidates)
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    main()
