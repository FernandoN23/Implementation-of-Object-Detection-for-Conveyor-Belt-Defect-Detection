#!/usr/bin/env python3
"""
===============================================================
  Trabajo de Memoria de Título
  Memorista: Fernando Navarrete
  Modelo actual: YOLOv11
  Código actual: clean_logs_runs.py
===============================================================
Descripción:
Script automático para limpiar las carpetas de logs y runs del
modelo YOLOv11. Permite eliminar registros globales o específicos
de una variante del modelo (n, s, m, l, xl).

Características:
 - Puede operar sobre logs, runs o ambos.
 - Soporta selección de variante (--variant n|s|m|l|xl).
 - Permite conservar las N carpetas más recientes (--keep-last).
 - Admite limpieza por antigüedad (--older-than DAYS).
 - Incluye modo de simulación (--dry-run) y confirmación opcional (--yes).

Uso:
    python clean_logs_runs.py --target both
    python clean_logs_runs.py --target runs --variant s --keep-last 2
    python clean_logs_runs.py --target logs --variant l --older-than 7
===============================================================
"""
import argparse
import shutil
import sys
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cleaner")


def confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N]: ").lower() in ("y", "yes")
    except KeyboardInterrupt:
        return False


def is_safe_base(base: Path) -> bool:
    return any(part.lower() == "yolov11" for part in base.parts)


def list_candidates(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)


def remove_path(path: Path, dry_run: bool = True):
    if not path.exists():
        return
    if dry_run:
        logger.info(f"[DRY-RUN] delete: {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
        logger.info(f"Removed directory: {path}")
    else:
        path.unlink()
        logger.info(f"Removed file: {path}")


def safe_iter_delete(folder: Path, keep_last: int = 0, older_than_days: int | None = None, dry_run: bool = True):
    candidates = list_candidates(folder)
    total = len(candidates)
    logger.info(f"Found {total} items in {folder}")
    cutoff_ts = None
    if older_than_days is not None:
        cutoff_ts = time.time() - older_than_days * 86400

    for idx, item in enumerate(candidates):
        name = item.name
        if name == ".gitkeep":
            continue
        if keep_last and idx < keep_last:
            continue
        if cutoff_ts is not None and item.stat().st_mtime > cutoff_ts:
            continue
        remove_path(item, dry_run=dry_run)


def parse_args():
    p = argparse.ArgumentParser(description="Limpia logs y runs de YOLOv11, con soporte por variante.")
    p.add_argument("--target", choices=("logs", "runs", "both"), default="both",
                   help="Carpeta a limpiar.")
    p.add_argument("--base-path", default="..",
                   help="Ruta base del proyecto YOLOv11 (default: ..).")
    p.add_argument("--variant", type=str, default=None,
                   help="Variante del modelo YOLOv11 (n, s, m, l, xl). Si se omite, limpia todo.")
    p.add_argument("--keep-last", type=int, default=0,
                   help="Conservar las N carpetas más recientes.")
    p.add_argument("--older-than", type=int, default=None,
                   help="Borrar solo carpetas/archivos más antiguos que N días.")
    p.add_argument("--dry-run", action="store_true", help="No borra nada, solo muestra el plan.")
    p.add_argument("--yes", action="store_true", help="No pedir confirmación interactiva.")
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    base = Path(args.base_path)
    if not base.is_absolute():
        base = (script_dir / base).resolve()

    if not is_safe_base(base):
        logger.error("Ruta base no válida: no contiene 'YOLOv11' en su ruta. Abortando.")
        sys.exit(1)

    variant = args.variant.lower() if args.variant else None
    logger.info(f"🧩 Variante seleccionada: {variant or 'todas'}")

    targets = []
    if args.target in ("logs", "both"):
        t = base / "logs"
        if variant:
            t = t / variant
        targets.append(t)
    if args.target in ("runs", "both"):
        t = base / "runs"
        if variant:
            t = t / variant
        targets.append(t)

    logger.info("Plan de limpieza:")
    for t in targets:
        logger.info(f" - {t} (keep_last={args.keep_last}, older_than={args.older_than}, dry_run={args.dry_run})")

    if not args.yes and not confirm("¿Continuar con la limpieza según el plan anterior?"):
        logger.info("Operación cancelada por el usuario.")
        sys.exit(0)

    for t in targets:
        if not t.exists():
            logger.warning(f"Target inexistente: {t} (omitido)")
            continue
        safe_iter_delete(folder=t, keep_last=args.keep_last, older_than_days=args.older_than, dry_run=args.dry_run)

    logger.info("🧹 Limpieza finalizada correctamente.")


if __name__ == "__main__":
    main()
