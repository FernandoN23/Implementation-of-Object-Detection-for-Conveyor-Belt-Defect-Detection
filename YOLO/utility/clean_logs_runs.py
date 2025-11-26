# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_logs_runs.py
# Limpia carpetas de logs/ y runs/ filtrando por variante (n/s/m/l/xl) y fase (train/valid/test).
# - Revisión: precisión de filtrado de variante (evitar falsos positivos por coincidencia parcial
#   como 'n' en 'runs' o 'train'). Coincidencia por tokens y patrones tipo 'yolo11n'.
# - Selector de **escenario** cuando phase == 'train' → {tests, final, all}.
#   En 'tests' o 'final' se **vacía el contenido** de dichas carpetas (archivos y subcarpetas)
#   preservando el directorio base; en 'all' se eliminan directorios completos (comportamiento previo).
# - Mejora: compresión de acciones y orden de borrado para evitar falsos errores (WinError 3) al
#   intentar borrar subrutas ya eliminadas por el borrado de un ancestro. Además, tolerancia a rutas
#   inexistentes: se informa como "Ya no existe (ok)" en lugar de error.
# Consola interactiva con confirmación (s/n).
#==============================================================

import re
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

VARIANTS = ["n", "s", "m", "l", "xl"]
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

def select_scenario(phase: str) -> str:
    if phase != "train":
        return "all"
    while True:
        s = input("Seleccione escenario bajo 'train' [tests/final] (o 'all'): ").strip().lower()
        if s in SCENARIOS:
            return s
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
        # conservar segmento completo si es patrón tipo 'yolo11n'
        if YOLO_VAR_RE.match(part):
            tokens.append(part)
        # añadir el segmento completo para detectar alias exactos de escenario
        tokens.append(part)
    return tokens


def match_variant(tokens: Iterable[str], variant: str) -> bool:
    tokens = list(tokens)
    # 1) token exacto
    if variant in tokens:
        return True
    # 2) patrones 'yolo{ver}{variant}' p.ej. 'yolo11n'
    if any(re.fullmatch(rf"yolo\d{{0,3}}{variant}", t) for t in tokens):
        return True
    return False


def match_phase(tokens: Iterable[str], phase: str) -> bool:
    if phase == "all":
        return True
    aliases = PHASE_ALIASES.get(phase, {phase})
    return any(t in aliases for t in tokens)


def match_scenario(tokens: Iterable[str], phase: str, scenario: str) -> bool:
    if phase != "train" or scenario == "all":
        return True
    aliases = SCENARIO_ALIASES.get(scenario, {scenario})
    return any(t in aliases for t in tokens)

# --- Helpers de selección/compresión ---

def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def scenario_root_only(path: Path, phase: str, scenario: str) -> bool:
    """Si filtramos 'tests'/'final', quedarse solo con el directorio raíz del escenario.
    Ej.: .../train/tests/<run>/ -> acepta solo .../train/tests"""
    if phase == "train" and scenario in {"tests", "final"}:
        return path.name.lower() in SCENARIO_ALIASES.get(scenario, {scenario})
    return True

# --- Descubrimiento de candidatos ---

def list_candidates(base: Path, variant: str, phase: str, scenario: str) -> List[Path]:
    if not base.exists():
        return []
    items: List[Path] = []
    for p in base.rglob("*"):
        if not p.is_dir():
            continue
        toks = tokenize_path_components(p)
        if match_variant(toks, variant) and match_phase(toks, phase) and match_scenario(toks, phase, scenario):
            if scenario_root_only(p, phase, scenario):
                items.append(p)
    return sorted(set(items))

# --- Confirmación/Eliminación ---

def confirm(prompt: str) -> bool:
    resp = input(f"{prompt} (s/n): ").strip().lower()
    return resp == "s"


def has_contents(d: Path) -> bool:
    try:
        next(d.iterdir())
        return True
    except StopIteration:
        return False
    except Exception:
        return True  # si no podemos listar, asumimos que tiene algo


def clear_dir_contents(d: Path) -> None:
    if not d.exists():
        return
    # Elimina todo el contenido inmediato (archivos y subdirectorios) preservando d
    for child in list(d.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception as e:
            print(f"✗ Error al limpiar {child}: {e}")


def compress_actions(actions: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    """Quita hijos cuando un ancestro ya será eliminado; prioriza 'delete' sobre 'clear'.
    Ordena para ejecutar primero 'clear', y luego 'delete' del más profundo al más superficial."""
    # Normalizar: preferir 'delete' si hay duplicados del mismo path
    by_path = {}
    for p, mode in actions:
        p = p.resolve()
        prev = by_path.get(p)
        if prev is None or (prev == "clear" and mode == "delete"):
            by_path[p] = mode
    pairs = [(p, m) for p, m in by_path.items()]
    # Remover descendientes si algún ancestro se borrará ('delete')
    delete_roots = [p for p, m in pairs if m == "delete"]
    kept: List[Tuple[Path, str]] = []
    for p, m in pairs:
        if any(is_subpath(p, root) and p != root for root in delete_roots):
            continue  # un ancestro se eliminará; no acciones redundantes
        kept.append((p, m))
    # Orden: primero 'clear' (profundidad ascendente), luego 'delete' (profundidad descendente)
    clears = sorted([(p, m) for p, m in kept if m == "clear"], key=lambda t: len(t[0].parts))
    deletes = sorted([(p, m) for p, m in kept if m == "delete"], key=lambda t: len(t[0].parts), reverse=True)
    return clears + deletes


def remove_paths(paths: List[Tuple[Path, str]]):
    for p, mode in paths:
        try:
            if mode == "delete":
                if not p.exists():
                    print(f"✓ Ya no existe (ok): {p}")
                    continue
                shutil.rmtree(p, ignore_errors=False)
                print(f"✓ Eliminado: {p}")
            elif mode == "clear":
                if not p.exists():
                    print(f"✓ Ya no existe (ok): {p}")
                    continue
                if has_contents(p):
                    clear_dir_contents(p)
                    print(f"✓ Contenido limpiado: {p}")
                else:
                    print(f"✓ Vacío (sin cambios): {p}")
        except FileNotFoundError:
            print(f"✓ Ya no existe (ok): {p}")
        except Exception as e:
            print(f"✗ Error en {p}: {e}")

# --- Main ---

def main():
    root = find_project_root()
    variant = select_variant()
    phase = select_phase()
    scenario = select_scenario(phase)
    target = select_target()

    bases = []
    if target in {"logs", "ambos"}:
        bases.append(root / "logs")
    if target in {"runs", "ambos"}:
        bases.append(root / "runs")

    raw_actions: List[Tuple[Path, str]] = []  # (path, mode)

    for b in bases:
        cands = list_candidates(b, variant, phase, scenario)
        if cands:
            print(f"\nEn {b}:")
            for c in cands[:20]:
                print(f" - {c}")
            if len(cands) > 20:
                print(f" ... (+{len(cands)-20} directorios más)")
        # Clasificación por modo de acción
        for c in cands:
            if phase == "train" and scenario in {"tests", "final"}:
                raw_actions.append((c, "clear"))  # limpiar contenido, preservar carpeta
            else:
                raw_actions.append((c, "delete"))  # eliminar carpeta completa

    if not raw_actions:
        print("\nNo se encontraron directorios que coincidan con los criterios.")
        return

    actions = compress_actions(raw_actions)

    # Resumen de acciones
    n_clear = sum(1 for _, m in actions if m == "clear")
    n_delete = sum(1 for _, m in actions if m == "delete")
    print("\nResumen de acciones (después de compresión):")
    if n_clear:
        print(f" - Limpiar contenido (preservar carpeta): {n_clear}")
    if n_delete:
        print(f" - Eliminar carpeta completa: {n_delete}")

    if confirm("\n¿Proceder con las acciones listadas?"):
        remove_paths(actions)
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    main()
