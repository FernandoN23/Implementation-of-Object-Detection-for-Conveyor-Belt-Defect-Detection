# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: SSD/utility/clean_weights.py
# Limpia pesos/checkpoints en weights/ filtrando por:
#   1. Variante (detectada dinámicamente, ej: [ssd300/ssd512])
#   2. Fase [train/valid/test]
#   3. Sub-escenario [tests/final/all] (Solo para train)
# Permite elegir 'best', 'last' o 'all'.
#
# CORRECCIÓN:
# - Solo selecciona ARCHIVOS para eliminar. Nunca selecciona carpetas
#   en la fase principal para evitar conflictos de "File not found".
# - Las carpetas vacías se pueden limpiar en el paso opcional final.
# ==============================================================

import re
import shutil
from pathlib import Path
from typing import Iterable, List, Set

# Constantes base
DEFAULT_VARIANTS = ["ssd300", "ssd512"]
PHASES = ["train", "valid", "test"]
PHASE_ALIASES = {
    "train": {"train"},
    "valid": {"valid", "val"},
    "test": {"test"},
}
SCENARIOS = ["tests", "final", "all"]
SCENARIO_ALIASES = {
    "tests": {"tests"},
    "final": {"final"},
}


# --- Utilidades de path/proyecto ---

def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "ssd").exists():
            return parent
    return Path.cwd()


def get_available_variants(weights_root: Path) -> List[str]:
    """Escanea weights/ para ver qué variantes existen realmente."""
    found = set()
    if weights_root.exists():
        regex = re.compile(r"(ssd\d{3})", re.IGNORECASE)
        for p in weights_root.rglob("*"):
            if p.is_dir():
                match = regex.search(p.name)
                if match:
                    found.add(match.group(1).lower())
    return sorted(list(found)) if found else DEFAULT_VARIANTS


# --- Entrada interactiva (Estilo HUD YOLO) ---

def select_variant(available_variants: List[str]) -> str:
    options_str = "/".join(available_variants)
    while True:
        v = input(f"Seleccione variante [{options_str}]: ").strip().lower()
        if v in available_variants:
            return v
        print("Entrada inválida. Intente nuevamente.")


def select_phase() -> str:
    while True:
        p = input("Seleccione fase [train/valid/test] (o 'all'): ").strip().lower()
        if p in PHASES or p == "all":
            return p
        print("Entrada inválida. Intente nuevamente.")


def select_scenario(phase: str) -> str:
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


def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"


# --- Coincidencia robusta por tokens ---

TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
SSD_VAR_RE = re.compile(r"^ssd\d{3}.*$")


def tokenize_path_components(path: Path) -> List[str]:
    tokens: List[str] = []
    for part in str(path).lower().replace("\\", "/").split("/"):
        if not part: continue
        for tok in TOKEN_SPLIT.split(part):
            if tok: tokens.append(tok)
        if SSD_VAR_RE.match(part):
            tokens.append(part)
    return tokens


def match_variant(tokens: Iterable[str], variant: str) -> bool:
    tokens = list(tokens)
    if variant in tokens: return True
    if any(variant in t for t in tokens if SSD_VAR_RE.match(t)): return True
    return False


def match_phase(tokens: Iterable[str], phase: str) -> bool:
    if phase == "all": return True
    aliases = PHASE_ALIASES.get(phase, {phase})
    return any(t in aliases for t in tokens)


def match_scenario(tokens: Iterable[str], phase: str, scenario: str) -> bool:
    if phase != "train" or scenario == "all": return True
    aliases = SCENARIO_ALIASES.get(scenario, {scenario})
    return any(t in aliases for t in tokens)


# --- Descubrimiento de candidatos ---

def list_candidates(weights_root: Path, variant: str, phase: str, scenario: str, which: str):
    if not weights_root.exists():
        return []
    items = []

    # Recorremos recursivamente
    for p in weights_root.rglob("*"):
        # IMPORTANTE: Ignoramos directorios en la selección principal
        # para evitar conflictos al borrar. Solo seleccionamos archivos.
        if not p.is_file():
            continue

        toks = tokenize_path_components(p)

        # Filtros principales
        if not (match_variant(toks, variant) and match_phase(toks, phase) and match_scenario(toks, phase, scenario)):
            continue

        # --- Lógica de Selección ---
        if which == "all":
            # Si es 'all', borramos cualquier archivo dentro de la ruta coincidente
            # (pesos, logs, json, etc.)
            items.append(p)
        else:
            # Si es 'best' o 'last', solo borramos el .pth exacto
            if p.name.lower() == f"{which}.pth":
                items.append(p)

    # Ordenar alfabéticamente
    return sorted(items)


# --- Eliminación ---

def remove_paths(paths):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
            print(f"✓ Eliminado: {p.name}")
        except Exception as e:
            print(f"✗ Error al eliminar {p.name}: {e}")


# --- Main ---

def main():
    root = find_project_root()
    weights_root = root / "weights"

    print(f"Raíz detectada: {root}")

    # 1. Escaneo de variantes
    available_vars = get_available_variants(weights_root)

    # 2. Interacción HUD
    variant = select_variant(available_vars)
    phase = select_phase()
    scenario = select_scenario(phase)
    which = select_which()

    # Aviso UX
    if phase == "train" and scenario == "tests" and which in {"best", "last"}:
        print("\n[Aviso] Escenario 'tests': Se limpiará el contenido de la carpeta, ignorando filtro 'best/last'.")

    # 3. Búsqueda
    candidates = list_candidates(weights_root, variant, phase, scenario, which)

    if not candidates:
        print("\nNo se encontraron archivos que coincidan con los criterios.")
        return

    print("\nSe eliminarán los siguientes ARCHIVOS en weights/ (las carpetas se mantendrán):")
    for c in candidates[:20]:
        rel = c.relative_to(root) if c.is_relative_to(root) else c
        print(f" - {rel}")
    if len(candidates) > 20:
        print(f" ... (+{len(candidates) - 20} más)")

    # 4. Ejecución
    if confirm("\n¿Confirma eliminación?"):
        remove_paths(candidates)

        # 5. Limpieza opcional de carpetas vacías (Residuales)
        print("\n--- Limpieza de Directorios Vacíos ---")
        if input("¿Desea eliminar las carpetas que quedaron vacías? (s/n): ").lower() == "s":
            # Recorremos de abajo hacia arriba (reverse) para borrar subcarpetas primero
            # Convertimos a lista para poder invertir el orden
            all_dirs = sorted([p for p in weights_root.rglob("*") if p.is_dir()], reverse=True)

            count = 0
            for p in all_dirs:
                try:
                    # rmdir solo borra si está vacío
                    p.rmdir()
                    print(f"✓ Carpeta vacía eliminada: {p.relative_to(root)}")
                    count += 1
                except OSError:
                    # La carpeta no está vacía, la ignoramos silenciosamente
                    pass

            if count == 0:
                print("No se encontraron carpetas vacías para eliminar.")
            else:
                print(f"Total carpetas eliminadas: {count}")

    else:
        print("Operación cancelada.")


if __name__ == "__main__":
    main()