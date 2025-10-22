"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: clean_metrics.py
Limpieza interactiva de métricas en YOLOv11.
Estructura soportada:
  YOLOv11/metrics/<variant>/<train|valid|test>/
-------------------------------------------------------------
"""

import shutil
from pathlib import Path

VARIANTS = ["n", "s", "m", "l", "xl"]
PHASES = ["train", "valid", "test"]

def confirm(prompt: str) -> bool:
    """Pide confirmación al usuario antes de eliminar."""
    return input(f"{prompt} [y/N]: ").lower() in ("y", "yes")

def choose_variant():
    """Selecciona la variante por letra."""
    print("\n📦 Variantes disponibles: " + ", ".join(v.upper() for v in VARIANTS))
    variant = input("👉 Escribe la letra de la variante: ").lower()
    if variant not in VARIANTS:
        print("⚠️ Variante inválida.")
        return None
    return variant

def choose_phase():
    """Selecciona la fase (train, valid o test)."""
    print("\n📂 Fases disponibles: train / valid / test")
    phase = input("👉 Escribe la fase: ").lower()
    if phase not in PHASES:
        print("⚠️ Fase inválida.")
        return None
    return phase

def clean_metrics():
    """Limpieza principal de métricas por variante y fase."""
    base_dir = Path(__file__).resolve().parents[1] / "metrics"
    variant = choose_variant()
    if not variant:
        return
    phase = choose_phase()
    if not phase:
        return

    target_dir = base_dir / variant / phase
    print(f"\n🧩 Carpeta objetivo: {target_dir}")

    if not target_dir.exists():
        print("⚠️ La carpeta seleccionada no existe.")
        return

    contents = list(target_dir.iterdir())
    if not contents:
        print("ℹ️ No se encontraron métricas para eliminar.")
        return

    print(f"🔍 Se encontraron {len(contents)} elementos (archivos o carpetas).")
    if not confirm(f"¿Eliminar TODO el contenido de metrics/{variant}/{phase}?"):
        print("❌ Operación cancelada.")
        return

    for item in contents:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f"🗑️ Eliminado: {item.name}")
        except Exception as e:
            print(f"⚠️ Error eliminando {item.name}: {e}")

    print("✅ Limpieza de métricas completada con éxito.")


if __name__ == "__main__":
    clean_metrics()
