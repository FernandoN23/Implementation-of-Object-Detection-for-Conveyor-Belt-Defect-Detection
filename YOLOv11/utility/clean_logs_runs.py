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
    """Pide confirmación de usuario antes de eliminar."""
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def choose_variant():
    print("\n📦 Variantes disponibles: " + ", ".join(v.upper() for v in VARIANTS))
    variant = input("👉 Escribe la letra de la variante: ").strip().lower()
    if variant not in VARIANTS:
        print("⚠️ Variante inválida.")
        return None
    return variant


def choose_phase():
    print("\n📂 Fases disponibles: train / valid / test")
    phase = input("👉 Escribe la fase: ").strip().lower()
    if phase not in PHASES:
        print("⚠️ Fase inválida.")
        return None
    return phase


def choose_target():
    print("\n🧭 Destinos disponibles: logs / runs / both")
    choice = input("👉 Escribe destino: ").strip().lower()
    if choice == "both":
        return TARGETS
    elif choice in TARGETS:
        return [choice]
    else:
        print("⚠️ Destino inválido.")
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

        # 🔥 Eliminación segura
        for item in contents:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"⚠️ Error al eliminar {item}: {e}")
        print(f"✅ Contenido eliminado correctamente en {target}/{variant}/{phase}.")


if __name__ == "__main__":
    print("=== LIMPIEZA DE LOGS Y RUNS - YOLOv11 ===")
    clean_logs_runs()
    print("\n🏁 Proceso finalizado.\n")
