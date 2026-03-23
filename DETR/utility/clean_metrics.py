# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/clean_metrics.py
# Descripción: Limpia carpetas de métricas procesadas en metrics/.
# ==============================================================

import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parents[1]
METRICS_ROOT = DETR_ROOT / "metrics"


def main():
    print(f"\n--- Limpieza de Métricas DETR ---")
    if not METRICS_ROOT.exists():
        print("No existe la carpeta metrics/.")
        return

    # Estructura: metrics/<task>/<variant>/<phase>/<run>
    # Simplificamos para buscar variantes directamente
    variants = set()
    for p in METRICS_ROOT.rglob("*"):
        if p.is_dir() and p.name in ["r50", "r101", "dc5"]:
            variants.add(p.name)

    v_list = sorted(list(variants))
    if not v_list:
        print("No se detectaron carpetas de variantes estándar.")
        v_list = ["all"]

    v_sel = input(f"Seleccione variante [{'/'.join(v_list)}/all]: ").strip().lower()

    if v_sel == "all":
        confirm = input("¿Desea vaciar TODA la carpeta metrics/? (s/n): ").lower()
        if confirm == "s":
            for item in METRICS_ROOT.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print("✓ Carpeta metrics/ vaciada.")
    else:
        # Buscar carpetas que contengan el nombre de la variante
        for d in METRICS_ROOT.rglob(v_sel):
            if d.is_dir():
                shutil.rmtree(d)
                print(f"✓ Eliminada carpeta de métricas: {d.name}")


if __name__ == "__main__":
    main()