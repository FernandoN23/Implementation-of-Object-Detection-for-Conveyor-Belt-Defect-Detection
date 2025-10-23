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
  YOLOv11/weights/<variant>/train/
-------------------------------------------------------------
"""

import os
from pathlib import Path

VARIANTS = ["n", "s", "m", "l", "xl"]

def confirm(prompt: str) -> bool:
    """Pide confirmación con formato (s/n)."""
    return input(f"{prompt} (s/n): ").strip().lower() == "s"

def choose_variant():
    """Selecciona la variante a limpiar."""
    print("\n📦 Variantes disponibles: " + ", ".join(v.upper() for v in VARIANTS))
    variant = input("👉 Escribe la letra de la variante a limpiar: ").lower()
    if variant not in VARIANTS:
        print("⚠️ Variante inválida.")
        return None
    return variant

def clean_weights():
    """Elimina los pesos de entrenamiento de una variante YOLOv11."""
    base_dir = Path(__file__).resolve().parents[1] / "weights"
    variant = choose_variant()
    if not variant:
        return

    target_dir = base_dir / variant / "train"
    print(f"\n🧩 Carpeta objetivo: {target_dir}")

    if not target_dir.exists():
        print("⚠️ La carpeta seleccionada no existe.")
        return

    files = list(target_dir.glob("*.pt"))
    if not files:
        print("ℹ️ No se encontraron archivos .pt en esta carpeta.")
        return

    print(f"🔍 Se encontraron {len(files)} archivos de checkpoint en la carpeta de entrenamiento.")
    if not confirm(f"¿Deseas eliminar TODOS los pesos de entrenamiento de la variante '{variant.upper()}'?"):
        print("❌ Operación cancelada.")
        return

    for f in files:
        try:
            os.remove(f)
            print(f"🗑️ Eliminado: {f.name}")
        except Exception as e:
            print(f"⚠️ Error eliminando {f.name}: {e}")

    print(f"✅ Limpieza completada para la variante '{variant.upper()}' (train).")

if __name__ == "__main__":
    clean_weights()
