# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_logs_runs.py
# Limpia carpetas de logs/ y runs/ filtrando por variante (n/s/m/l/xl) y fase (train/valid/test).
# \- Revisión: precisión de filtrado de variante (evitar falsos positivos por coincidencia parcial
#    como 'n' en 'runs' o 'train'). Coincidencia por tokens y patrones tipo 'yolo11n'.
# Consola interactiva con confirmación (s/n).
#==============================================================

import re
import shutil
from pathlib import Path
from typing import Iterable, List

VARIANTS = ["n", "s", "m", "l", "xl"]
PHASES = ["train", "valid", "test"]
PHASE_ALIASES = {
    "train": {"train"},
    "valid": {"valid", "val"},
    "test": {"test"},
}

# --- Utilidades de path/proyecto ---

def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()

# --- Entrada interactiva ---

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

# --- Coincidencia robusta por tokens ---
TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
YOLO_VAR_RE = re.compile(r"^yolo\d{0,3}(n|s|m|l|xl)$")


def tokenize_path_components(path: Path) -> List[str]:
    tokens: List[str] = []
    for part in str(path).lower().replace("\\", "/").split("/"):
        if not part:
            continue
        # tokens por separadores ( - _ . = etc.)
        for tok in TOKEN_SPLIT.split(part):
            if tok:
                tokens.append(tok)
        # además, conservar el segmento completo si es patrón yoloXXv
        # (para soportar 'yolo11n' como un único token)
        if YOLO_VAR_RE.match(part):
            tokens.append(part)
    return tokens


def match_variant(tokens: Iterable[str], variant: str) -> bool:
    tokens = list(tokens)
    # 1) Coincidencia exacta del token (evita 'n' dentro de 'train' o 'runs')
    if variant in tokens:
        return True
    # 2) Patrones de estilo 'yolo{ver}{var}' p.ej. 'yolo11n'
    if any(t == f"yolo{ver}{variant}" for t in tokens for ver in ["", "5", "8", "9", "10", "11", "12", "v5", "v8"]):
        return True
    if any(re.fullmatch(rf"yolo\d{{1,3}}{variant}", t) for t in tokens):
        return True
    # 3) pares tipo ['variant', 'n'] ya quedan cubiertos por (1), porque 'n' es token propio
    return False


def match_phase(tokens: Iterable[str], phase: str) -> bool:
    if phase == "all":
        return True
    aliases = PHASE_ALIASES.get(phase, {phase})
    return any(t in aliases for t in tokens)

# --- Descubrimiento de candidatos ---

def list_candidates(base: Path, variant: str, phase: str):
    if not base.exists():
        return []
    items = []
    for p in base.rglob("*"):
        if not p.is_dir():
            continue
        toks = tokenize_path_components(p)
        if match_variant(toks, variant) and match_phase(toks, phase):
            items.append(p)
    # eliminar duplicados, ordenar
    return sorted(set(items))

# --- Confirmación/Eliminación ---

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

# --- Main ---

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
