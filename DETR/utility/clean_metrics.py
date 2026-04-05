# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/utility/clean_metrics.py
# Descripción: Limpia carpetas de métricas procesadas en metrics/.
#              Adaptado a la estructura metrics/detect/<variante>.
# ==============================================================

import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
DETR_ROOT = FILE.parents[1]
METRICS_ROOT = DETR_ROOT / "metrics" / "detect"


def main():
    print(f"\n[clean_metrics.py] --- Limpieza de Métricas DETR ---")
    if not METRICS_ROOT.exists():
        print("[clean_metrics.py] No existe la carpeta metrics/detect/.")
        return

    # Detectar variantes disponibles en metrics/detect/
    variants = sorted([d.name for d in METRICS_ROOT.iterdir() if d.is_dir() and d.name != "global_comparison"])

    if not variants:
        print("[clean_metrics.py] No se detectaron carpetas de variantes estándar.")
        variants = ["all"]

    v_sel = input(f"Seleccione variante a eliminar [{' / '.join(variants)} / all]: ").strip().lower()

    if v_sel == "all":
        confirm = input("¿Desea vaciar TODA la carpeta metrics/detect/? (s/n): ").lower()
        if confirm == "s":
            for item in METRICS_ROOT.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print("[clean_metrics.py] ✓ Carpeta metrics/detect/ vaciada.")
        else:
            print("[clean_metrics.py] Operación cancelada.")
    elif v_sel in variants:
        target_dir = METRICS_ROOT / v_sel
        confirm = input(f"¿Desea eliminar todas las métricas de la variante '{v_sel}'? (s/n): ").lower()
        if confirm == "s":
            shutil.rmtree(target_dir)
            print(f"[clean_metrics.py] ✓ Eliminada carpeta de métricas: {target_dir.relative_to(DETR_ROOT)}")
        else:
            print("[clean_metrics.py] Operación cancelada.")
    else:
        print("[clean_metrics.py] Variante no válida.")


if __name__ == "__main__":
    main()