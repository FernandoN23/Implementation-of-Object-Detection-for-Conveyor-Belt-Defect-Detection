"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: clean_weights.py
Limpieza interactiva de weights/pesos en YOLOv11.
Estructura soportada:
  YOLOv11/weights/<variant>/<train|valid|test>/
-------------------------------------------------------------
"""

import os
from pathlib import Path
import shutil

VARIANTS = ["n", "s", "m", "l", "xl"]
PHASES = ["train", "valid", "test"]

def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").lower() in ("y", "yes")

def choose_variant():
    print("\n📦 Selecciona la variante a limpiar:")
    for i, v in enumerate(VARIANTS, 1):
        print(f"  {i}) {v.upper()}")
    choice = input("👉 Variante (número): ")
    try:
        return VARIANTS[int(choice) - 1]
    except (ValueError, IndexError):
        print("⚠️ Selección inválida.")
        return None

def choose_phase():
    print("\n📂 Selecciona el tipo de datos:")
    for i, p in enumerate(PHASES, 1):
        print(f"  {i}) {p}")
    choice = input("👉 Tipo (número): ")
    try:
        return PHASES[int(choice) - 1]
    except (ValueError, IndexError):
        print("⚠️ Selección inválida.")
        return None

def clean_weights():
    base_dir = Path(__file__).resolve().parents[1] / "weights"
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

    files = list(target_dir.glob("*.pt"))
    if not files:
        print("ℹ️ No se encontraron archivos .pt en esta carpeta.")
        return

    print(f"🔍 Se encontraron {len(files)} archivos de checkpoint.")
    if not confirm("¿Deseas eliminar TODOS los archivos de esta carpeta?"):
        print("❌ Operación cancelada.")
        return

    for f in files:
        try:
            os.remove(f)
            print(f"🗑️ Eliminado: {f.name}")
        except Exception as e:
            print(f"⚠️ Error eliminando {f.name}: {e}")

    print("✅ Limpieza completada con éxito.")


if __name__ == "__main__":
    clean_weights()
