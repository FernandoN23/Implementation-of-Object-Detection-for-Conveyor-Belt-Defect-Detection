#!/usr/bin/env python3
"""
clean_logs_runs.py

Utility para limpiar carpetas de logs y runs en el proyecto YOLOv11.

Ubicación recomendada:
YOLOv11/utility/clean_logs_runs.py

Características:
 - Operar sobre YOLOv11/logs y YOLOv11/runs por defecto.
 - Opciones: target (logs|runs|both), --path (ruta base del proyecto), --keep-last N, --older-than DAYS, --dry-run, --yes.
 - Protecciones: verifica que la ruta base contenga 'YOLOv11' antes de ejecutar borrados.
 - Conserva archivos nombrados .gitkeep por defecto.
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
    # seguridad básica: la ruta debe contener "YOLOv11" en algún componente
    return any(part.lower() == "yolov11" for part in base.parts)


def list_candidates(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)


def remove_path(path: Path, dry_run: bool = True):
    if not path.exists():
        return
    if path.is_dir():
        # remove directory contents (and dir itself)
        if dry_run:
            logger.info(f"[DRY-RUN] rmdir: {path}")
        else:
            shutil.rmtree(path)
            logger.info(f"Removed directory: {path}")
    else:
        if dry_run:
            logger.info(f"[DRY-RUN] unlink: {path}")
        else:
            path.unlink()
            logger.info(f"Removed file: {path}")


def safe_iter_delete(folder: Path, keep_last: int = 0, older_than_days: int | None = None, dry_run: bool = True):
    """
    Borra elementos dentro de folder aplicando reglas:
     - conserva .gitkeep
     - preserva keep_last elementos más recientes (por mtime)
     - si older_than_days se especifica, borra solo si mtime más viejo que ese umbral
    """
    candidates = list_candidates(folder)
    total = len(candidates)
    logger.info(f"Found {total} items in {folder}")
    cutoff_ts = None
    if older_than_days is not None:
        cutoff_ts = time.time() - older_than_days * 86400

    for idx, item in enumerate(candidates):
        name = item.name
        # skip .gitkeep
        if name == ".gitkeep":
            logger.debug(f"Skipping .gitkeep: {item}")
            continue

        # preserve last N
        if keep_last and idx < keep_last:
            logger.info(f"Keeping (recent #{idx+1}): {item}")
            continue

        # check older_than
        if cutoff_ts is not None:
            mtime = item.stat().st_mtime
            if mtime > cutoff_ts:
                logger.info(f"Keeping (newer than threshold): {item}")
                continue

        # finally remove
        remove_path(item, dry_run=dry_run)


def parse_args():
    p = argparse.ArgumentParser(description="Limpiar YOLOv11/logs y YOLOv11/runs")
    p.add_argument(
        "--target",
        choices=("logs", "runs", "both"),
        default="both",
        help="Carpeta a atacar (default: both)",
    )
    p.add_argument(
        "--base-path",
        default="..",
        help="Ruta base del proyecto YOLOv11 desde la carpeta utility (default: ..). Puede ser absoluta.",
    )
    p.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="Conservar las N carpetas/archivos más recientes en cada target (ordenadas por mtime).",
    )
    p.add_argument(
        "--older-than",
        type=int,
        default=None,
        help="Borrar solo archivos/carpetas más antiguos que N días. (int)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No borra nada, solo muestra lo que se haría.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="No pedir confirmación interactiva (usar con precaución).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve base path relative to this script location if relative
    script_dir = Path(__file__).resolve().parent
    base = Path(args.base_path)
    if not base.is_absolute():
        base = (script_dir / base).resolve()

    logger.info(f"Base path resolved to: {base}")

    if not is_safe_base(base):
        logger.error("Ruta base no válida: no contiene 'YOLOv11' en su ruta. Abortando por seguridad.")
        sys.exit(1)

    targets = []
    if args.target in ("logs", "both"):
        targets.append(base / "logs")
    if args.target in ("runs", "both"):
        targets.append(base / "runs")

    # show plan
    logger.info("Plan de limpieza:")
    for t in targets:
        logger.info(f" - Target: {t}  (keep_last={args.keep_last}, older_than={args.older_than}, dry_run={args.dry_run})")

    if not args.yes:
        if not confirm("¿Continuar con la limpieza según el plan anterior?"):
            logger.info("Operación cancelada por el usuario.")
            sys.exit(0)

    for t in targets:
        if not t.exists():
            logger.warning(f"Target inexistente: {t} (se omite)")
            continue
        safe_iter_delete(folder=t, keep_last=args.keep_last, older_than_days=args.older_than, dry_run=args.dry_run)

    logger.info("Proceso finalizado.")


if __name__ == "__main__":
    main()
