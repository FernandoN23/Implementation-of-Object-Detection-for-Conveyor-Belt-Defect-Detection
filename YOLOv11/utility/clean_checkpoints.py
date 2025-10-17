"""
===============================================================
  Trabajo de Memoria de Título
  Memorista: Fernando Navarrete
  Modelo actual: YOLOv11
  Código actual: clean_checkpoints.py
===============================================================
Descripción:
Script automático para limpiar la carpeta de checkpoints del
modelo YOLOv11. Permite eliminar todos los checkpoints o
conservar los N más recientes según configuración del usuario.

Uso:
    python clean_checkpoints.py --mode all
    python clean_checkpoints.py --keep 3
===============================================================
"""

import os
import argparse
from pathlib import Path


def clean_checkpoints(mode: str = "keep", keep: int = 3):
    """
    Limpia la carpeta YOLOv11/checkpoints según el modo elegido.

    Args:
        mode (str): 'all' para eliminar todo, 'keep' para conservar los N más recientes.
        keep (int): Número de checkpoints a conservar (si mode='keep').
    """
    base_dir = Path(__file__).resolve().parents[1]
    checkpoint_dir = base_dir / "checkpoints"

    print(f"[i] Buscando checkpoints en: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        print(f"[!] Carpeta no encontrada: {checkpoint_dir}")
        return

    files = sorted(
        [f for f in checkpoint_dir.glob("*") if f.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not files:
        print("[i] No se encontraron archivos de checkpoint.")
        return

    if mode == "all":
        to_delete = files
        print(f"[!] Modo ALL: Se eliminarán {len(to_delete)} archivos.")
    else:
        if len(files) <= keep:
            print(f"[i] Solo hay {len(files)} archivos, no se eliminará nada.")
            return
        to_delete = files[keep:]
        print(f"[i] Se conservarán los {keep} más recientes y se eliminarán {len(to_delete)} restantes.")

    for f in to_delete:
        try:
            os.remove(f)
            print(f"[-] Eliminado: {f.name}")
        except Exception as e:
            print(f"[x] Error eliminando {f.name}: {e}")

    print(f"[✔] Limpieza completada ({mode.upper()}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpia checkpoints del modelo YOLOv11.")
    parser.add_argument("--mode", type=str, choices=["all", "keep"], default="keep",
                        help="Modo de limpieza: 'all' elimina todo, 'keep' conserva los más recientes.")
    parser.add_argument("--keep", type=int, default=3, help="Número de checkpoints a conservar (modo keep).")

    args = parser.parse_args()
    clean_checkpoints(mode=args.mode, keep=args.keep)
