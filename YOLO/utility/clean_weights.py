# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_weights.py
# Limpia pesos/checkpoints en weights/ filtrando por variante (n/s/m/l/xl),
# fase (train/valid/test) **y sub-escenario de entrenamiento** (tests/final/all).
# Permite elegir 'best', 'last' o 'all'. Consola interactiva con confirmación (s/n).
# - Revisión: precisión de filtrado de variante/fase (evitar falsos positivos por substrings
#   como 'n' en 'runs'/'train'). Coincidencia por tokens + patrones tipo 'yolo11n'.
# - Nuevo: selector de **escenario** ('tests' vs 'final' vs 'all') para evitar borrar artefactos
#   de producción al limpiar resultados de pruebas y viceversa.
# - Mejora: en **tests** normalmente no se generan .pt; el script ahora limpia **el contenido**
#   del directorio `tests/` (archivos .json, etc.) manteniendo la carpeta, salvo que se elija
#   `which=all`, donde se elimina por completo.
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
# Escenarios bajo 'train' (según estructura observada en la imagen: weights/<var>/train/{final,tests})
SCENARIOS = ["tests", "final", "all"]
SCENARIO_ALIASES = {
    "tests": {"tests"},
    "final": {"final"},
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

def select_scenario(phase: str) -> str:
    """Cuando phase == 'train', permite elegir 'tests'/'final'/'all'. Para otras fases devuelve 'all'."""
    if phase != "train":
        return "all"
    while True:
        s = input("Seleccione escenario bajo 'train' [tests/final] (o 'all'): ").strip().lower()
        if s in SCENARIOS:
            return s
        print("Entrada inválida. Intente nuevamente.")

def select_which() -> str:
    while True:
        w = input("¿Qué pesos desea eliminar? [best/last/all]: ").strip().lower()
        if w in {"best", "last", "all"}:
            return w
        print("Entrada inválida. Intente nuevamente.")

# --- Confirmación ---

def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"

# --- Coincidencia robusta por tokens ---
TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
YOLO_VAR_RE = re.compile(r"^yolo\d{0,3}(n|s|m|l|xl)$")


def tokenize_path_components(path: Path) -> List[str]:
    tokens: List[str] = []
    for part in str(path).lower().replace("\\", "/").split("/"):
        if not part:
            continue
        # tokens por separadores ( - _ . etc.)
        for tok in TOKEN_SPLIT.split(part):
            if tok:
                tokens.append(tok)
        # conservar el segmento completo si cumple patrón 'yoloXXn'
        if YOLO_VAR_RE.match(part):
            tokens.append(part)
    return tokens


def match_variant(tokens: Iterable[str], variant: str) -> bool:
    tokens = list(tokens)
    # 1) token exacto (evita 'n' dentro de 'train'/'runs')
    if variant in tokens:
        return True
    # 2) patrones estilo 'yolo{ver}{variant}' (p.ej., 'yolo11n')
    if any(re.fullmatch(rf"yolo\d{{0,3}}{variant}", t) for t in tokens):
        return True
    return False


def match_phase(tokens: Iterable[str], phase: str) -> bool:
    if phase == "all":
        return True
    aliases = PHASE_ALIASES.get(phase, {phase})
    return any(t in aliases for t in tokens)


def match_scenario(tokens: Iterable[str], phase: str, scenario: str) -> bool:
    """Restringe a 'tests'/'final' sólo cuando phase == 'train'."""
    if phase != "train" or scenario == "all":
        return True
    aliases = SCENARIO_ALIASES.get(scenario, {scenario})
    return any(t in aliases for t in tokens)

# --- Descubrimiento de candidatos ---

def list_candidates(weights_root: Path, variant: str, phase: str, scenario: str, which: str):
    if not weights_root.exists():
        return []
    items = []
    for p in weights_root.rglob("*"):
        toks = tokenize_path_components(p)
        if not (match_variant(toks, variant) and match_phase(toks, phase) and match_scenario(toks, phase, scenario)):
            continue

        # --- Caso especial: escenario 'tests' ---
        # No se esperan .pt; limpiamos **archivos** dentro del directorio tests, manteniendo la carpeta.
        if phase == "train" and scenario == "tests":
            if p.is_file():
                items.append(p)  # incluir cualquier archivo (.json, .yaml, .csv, .png, etc.)
            # no añadimos directorios aquí para mantener la estructura
            continue

        # --- Resto de escenarios ---
        if which == "all":
            items.append(p)  # incluir archivos y directorios (se eliminarán por completo)
        else:
            # Para 'best'/'last' solo borrar archivos EXACTOS 'best.pt'/'last.pt'
            if p.is_file() and p.name.lower() == f"{which}.pt":
                items.append(p)
    # Deduplicar y ordenar: primero archivos, luego directorios
    files = sorted({q for q in items if q.is_file()})
    dirs = sorted({q for q in items if q.is_dir()})
    return files + dirs

# --- Eliminación ---

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

# --- Main ---

def main():
    root = find_project_root()
    variant = select_variant()
    phase = select_phase()
    scenario = select_scenario(phase)
    which = select_which()
    weights_root = root / "weights"

    # Nota UX: si estamos en 'tests' y el usuario eligió 'best'/'last', aclaramos la acción.
    if phase == "train" and scenario == "tests" and which in {"best", "last"}:
        print("[Aviso] Escenario 'tests': no se esperan checkpoints '.pt'. Se limpiará el **contenido** de 'tests/'.")

    candidates = list_candidates(weights_root, variant, phase, scenario, which)
    if not candidates:
        print("No se encontraron elementos que coincidan con los criterios dados.")
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
