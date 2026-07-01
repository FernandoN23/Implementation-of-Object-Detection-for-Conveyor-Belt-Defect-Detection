# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/utility/clean_runs.py
# Descripción: Elimina carpetas de experimentos en runs/ filtrando
#              por variante de modelo y fase lógica.
#              Ignora la carpeta runs/data/ (datasets descargados).
# ==============================================================

import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
DINO_ROOT = FILE.parents[1]
RUNS_ROOT = DINO_ROOT / "runs"


def main():
    print(f"\n[clean_runs.py] --- Limpieza de Runs DINO ---")
    if not RUNS_ROOT.exists():
        print("[clean_runs.py] No existe la carpeta runs/.")
        return

    # Detectar variantes, ignorando la carpeta 'data'
    variants = sorted([d.name for d in RUNS_ROOT.iterdir() if d.is_dir() and d.name != "data"])

    if not variants:
        print("[clean_runs.py] No hay variantes registradas en runs/.")
        return

    v_sel = input(f"Seleccione variante [{' / '.join(variants)} / all]: ").strip().lower()

    if v_sel != "all" and v_sel not in variants:
        print("[clean_runs.py] Variante no válida.")
        return

    phase = input("Seleccione fase[train / valid / test / all]: ").strip().lower()

    targets =[]
    for v in variants:
        if v_sel != "all" and v != v_sel:
            continue

        v_path = RUNS_ROOT / v
        if not v_path.exists(): continue

        for p_dir in v_path.iterdir():
            if p_dir.is_dir():
                if phase == "all" or p_dir.name == phase:
                    # Añadimos la carpeta de la fase completa (ej. runs/r50_4scale/train)
                    targets.append(p_dir)

    if not targets:
        print("[clean_runs.py] No se encontraron carpetas que coincidan con los criterios.")
        return

    print("\nSe eliminarán las siguientes carpetas COMPLETAS:")
    for t in targets:
        print(f" - {t.relative_to(DINO_ROOT)}")

    if input("\n¿Está seguro? Se perderán todos los logs, pesos y gráficos de estos runs. (s/n): ").lower() == "s":
        for t in targets:
            shutil.rmtree(t)
            print(f"[clean_runs.py] ✓ Eliminada: {t.relative_to(DINO_ROOT)}")

        # Limpiar carpetas de variantes vacías
        for v in variants:
            v_path = RUNS_ROOT / v
            if v_path.exists() and not any(v_path.iterdir()):
                v_path.rmdir()
    else:
        print("[clean_runs.py] Operación cancelada.")


if __name__ == "__main__":
    main()