# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/clean_runs.py
# Descripción: Elimina carpetas de experimentos en runs/ filtrando
#              por variante de modelo y fase lógica.
# ==============================================================

import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parents[1]
RUNS_ROOT = DETR_ROOT / "runs"

def main():
    print(f"\n--- Limpieza de Runs DETR ---")
    if not RUNS_ROOT.exists():
        print("No existe la carpeta runs/.")
        return

    variants = sorted([d.name for d in RUNS_ROOT.iterdir() if d.is_dir()])
    if not variants:
        print("No hay variantes registradas en runs/.")
        return

    v_sel = input(f"Seleccione variante [{'/'.join(variants)}/all]: ").strip().lower()
    phase = input("Seleccione fase [train/valid/test/all]: ").strip().lower()

    targets = []
    for v in variants:
        if v_sel != "all" and v != v_sel:
            continue
        v_path = RUNS_ROOT / v
        for p_dir in v_path.iterdir():
            if p_dir.is_dir():
                if phase == "all" or p_dir.name == phase:
                    targets.append(p_dir)

    if not targets:
        print("No se encontraron carpetas que coincidan.")
        return

    print("\nSe eliminarán las siguientes carpetas COMPLETAS:")
    for t in targets:
        print(f" - {t.relative_to(DETR_ROOT)}")

    if input("\n¿Está seguro? Se perderán todos los logs y gráficos. (s/n): ").lower() == "s":
        for t in targets:
            shutil.rmtree(t)
            print(f"✓ Eliminada: {t.name}")
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    main()