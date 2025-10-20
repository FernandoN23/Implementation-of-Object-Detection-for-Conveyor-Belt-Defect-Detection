"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: clean_logs_runs.py
Limpieza interactiva de logs y runs en YOLOv11.
Estructura soportada:
  YOLOv11/{logs|runs}/<variant>/<train|valid|test>/
-------------------------------------------------------------
"""

import shutil
from pathlib import Path

VARIANTS = ["n", "s", "m", "l", "xl"]
PHASES = ["train", "valid", "test"]
TARGETS = ["logs", "runs"]

def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").lower() in ("y", "yes")

def choose_variant():
    print("\n📦 Selecciona la variante:")
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

def choose_target():
    print("\n🧭 Selecciona el destino:")
    for i, t in enumerate(TARGETS, 1):
        print(f"  {i}) {t}")
    choice = input("👉 Destino (número o 3 para ambos): ")
    if choice == "3":
        return TARGETS
    try:
        return [TARGETS[int(choice) - 1]]
    except (ValueError, IndexError):
        print("⚠️ Selección inválida.")
        return []

def clean_logs_runs():
    base_dir = Path(__file__).resolve().parents[1]
    variant = choose_variant()
    if not variant:
        return
    phase = choose_phase()
    if not phase:
        return
    targets = choose_target()
    if not targets:
        return

    for target in targets:
        target_dir = base_dir / target / variant / phase
        print(f"\n🧩 Carpeta objetivo: {target_dir}")

        if not target_dir.exists():
            print("⚠️ No existe esta ruta, se omite.")
            continue

        contents = list(target_dir.iterdir())
        if not contents:
            print("ℹ️ Carpeta vacía, nada que eliminar.")
            continue

        print(f"🔍 Se encontraron {len(contents)} elementos en {target_dir}")
        if not confirm(f"¿Eliminar TODO el contenido de {target}/{variant}/{phase}?"):
            print("❌ Operación cancelada para esta carpeta.")
            continue

        try:
            for item in contents:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print(f"✅ Limpieza completada en {target}/{variant}/{phase}")
        except Exception as e:
            print(f"⚠️ Error eliminando contenido: {e}")


if __name__ == "__main__":
    clean_logs_runs()
