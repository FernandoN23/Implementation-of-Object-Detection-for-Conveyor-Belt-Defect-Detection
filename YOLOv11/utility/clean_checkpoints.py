"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: clean_checkpoints.py
Herramienta de limpieza de checkpoints de YOLOv11.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Función principal: clean_checkpoints()
#   - Elimina archivos de checkpoints (.pt) según el modo elegido.
#   - Modo 'all': borra todos los archivos.
#   - Modo 'keep': conserva los N más recientes.
#
# Características:
#   • Soporta limpieza por variante (n, s, m, l, xl)
#   • Ordena automáticamente por fecha de modificación
#   • Evita errores si la carpeta no existe
#
# Uso típico:
#   python clean_checkpoints.py --mode keep --keep 3 --variant s
#
# Conexión:
#   Utilizado fuera del flujo de entrenamiento para mantener
#   la estructura de YOLOv11/checkpoints/ organizada.
# -------------------------------------------------------------


import os
import argparse
from pathlib import Path


def clean_checkpoints(mode: str = "keep", keep: int = 3, variant: str | None = None):
    """
    Limpia la carpeta YOLOv11/checkpoints según el modo elegido.
    Args:
        mode (str): 'all' para eliminar todo, 'keep' para conservar los N más recientes.
        keep (int): Número de checkpoints a conservar (si mode='keep').
        variant (str): Variante de modelo ('n', 's', 'm', 'l', 'xl'). Si None, limpia todas.
    """
    base_dir = Path(__file__).resolve().parents[1]
    checkpoint_dir = base_dir / "checkpoints"

    # Si se especifica una variante, actuar sobre su subcarpeta
    if variant:
        checkpoint_dir = checkpoint_dir / variant.lower()

    print(f"[i] Buscando checkpoints en: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        print(f"[!] Carpeta no encontrada: {checkpoint_dir}")
        return

    files = sorted(
        [f for f in checkpoint_dir.glob("*.pt") if f.is_file()],
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

    print(f"[✔] Limpieza completada ({mode.upper()}) en variante: {variant or 'todas'}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpia checkpoints del modelo YOLOv11.")
    parser.add_argument("--mode", type=str, choices=["all", "keep"], default="keep",
                        help="Modo de limpieza: 'all' elimina todo, 'keep' conserva los más recientes.")
    parser.add_argument("--keep", type=int, default=3, help="Número de checkpoints a conservar (modo keep).")
    parser.add_argument("--variant", type=str, default=None,
                        help="Variante del modelo YOLOv11 a limpiar (n, s, m, l, xl). Si no se especifica, limpia todas.")

    args = parser.parse_args()
    clean_checkpoints(mode=args.mode, keep=args.keep, variant=args.variant)
