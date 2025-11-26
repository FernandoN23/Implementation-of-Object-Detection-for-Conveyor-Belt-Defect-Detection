# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: clean_metrics.py
# Limpia métricas almacenadas en metrics/ filtrando por variante (n/s/m/l/xl) y fase
# (train/valid/test/test_metrics). Consola interactiva con confirmación (s/n).
# - Revisión: precisión de filtrado de variante/fase para evitar falsos positivos por
#   coincidencias parciales (p.ej., 'n' dentro de 'runs' o 'train'). Se usa corte por
#   tokens y patrones tipo 'yolo11n'.
# - Nuevo: soporte para la fase adicional 'test_metrics' bajo YOLOv11/metrics/.
# - Mejora: compresión de rutas y borrado robusto para evitar errores al intentar
#   eliminar subdirectorios/archivos ya borrados por el borrado de un ancestro.
#   Rutas inexistentes se reportan como "Ya no existe (ok)" en lugar de error.
#==============================================================

import re
import shutil
from pathlib import Path
from typing import Iterable, List

VARIANTS = ["n", "s", "m", "l", "xl"]
PHASES = ["train", "valid", "test", "test_metrics"]
PHASE_ALIASES = {
    "train": {"train"},
    "valid": {"valid", "val"},
    "test": {"test"},
    # Soporta nombres con guion/bajo y abreviatura
    "test_metrics": {"test_metrics", "test-metrics", "testmetrics", "tm"},
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
        p = input("Seleccione fase [train/valid/test/test_metrics] (o 'all'): ").strip().lower()
        if p in PHASES or p == "all":
            return p
        print("Entrada inválida. Intente nuevamente.")


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
        # 1) tokens por separadores ( - _ . = etc.)
        for tok in TOKEN_SPLIT.split(part):
            if tok:
                tokens.append(tok)
        # 2) conservar el segmento completo para detectar compuestos ('test_metrics', 'test-metrics', 'yolo11n')
        tokens.append(part)
        if YOLO_VAR_RE.match(part):
            tokens.append(part)
    return tokens


def match_variant(tokens: Iterable[str], variant: str) -> bool:
    tokens = list(tokens)
    # 1) token exacto
    if variant in tokens:
        return True
    # 2) patrones estilo 'yolo{ver}{variant}' (yolo11n, yolo8s, etc.)
    if any(re.fullmatch(rf"yolo\d{{0,3}}{variant}", t) for t in tokens):
        return True
    return False


def match_phase(tokens: Iterable[str], phase: str) -> bool:
    if phase == "all":
        return True
    aliases = PHASE_ALIASES.get(phase, {phase})
    return any(t in aliases for t in tokens)

# --- Helpers de selección / compresión ---

def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def compress_dirs(dirs: List[Path]) -> List[Path]:
    """Elimina subdirectorios si algún ancestro ya será borrado.
    Devuelve solo raíces mínimas, ordenadas de más profundo a más superficial
    para minimizar conflictos de borrado.
    """
    unique = sorted({d.resolve() for d in dirs}, key=lambda p: len(p.parts))
    roots: List[Path] = []
    for d in unique:
        if any(is_subpath(d, r) for r in roots):
            continue
        roots.append(d)
    # borrar primero los más profundos
    return sorted(roots, key=lambda p: len(p.parts), reverse=True)

# --- Descubrimiento de candidatos ---

def list_candidates(metrics_root: Path, variant: str, phase: str) -> List[Path]:
    if not metrics_root.exists():
        return []
    items: List[Path] = []
    for p in metrics_root.rglob("*"):
        if not p.is_dir():
            continue
        toks = tokenize_path_components(p)
        if match_variant(toks, variant) and match_phase(toks, phase):
            items.append(p)
    return compress_dirs(items)

# --- Eliminación ---

def remove_paths(paths: List[Path]):
    for p in paths:
        try:
            if not p.exists():
                print(f"✓ Ya no existe (ok): {p}")
                continue
            shutil.rmtree(p, ignore_errors=False)
            print(f"✓ Eliminado: {p}")
        except FileNotFoundError:
            print(f"✓ Ya no existe (ok): {p}")
        except Exception as e:
            print(f"✗ Error al eliminar {p}: {e}")

# --- Main ---

def main():
    root = find_project_root()
    variant = select_variant()
    phase = select_phase()
    metrics_root = root / "metrics"

    candidates = list_candidates(metrics_root, variant, phase)
    if not candidates:
        print("No se encontraron métricas para los criterios dados.")
        return

    print("Se eliminarán los siguientes directorios de métricas (ya comprimidos por raíz):")
    for c in candidates[:20]:
        print(f" - {c}")
    if len(candidates) > 20:
        print(f" ... (+{len(candidates)-20} directorios más)")

    if confirm("¿Confirma eliminación?"):
        remove_paths(candidates)
    else:
        print("Operación cancelada.")


if __name__ == "__main__":
    main()
