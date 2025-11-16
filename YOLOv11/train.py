# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/train.py
# Descripción: Script principal de entrenamiento. Orquesta bootstrap ROCm,
#  construcción de modelo/datos, y delega el loop de entrenamiento a
#  engine/Trainer.py. Integra AMP/EMA, callbacks, validación interna (val_int)
#  y HUD de consola. Incluye modo --test para prueba de ensamblado y warmup.
#==============================================================

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

# ---------------------------------------------------------------------------
# 0) Configuración global de warnings (antes de todo)
# ---------------------------------------------------------------------------

# Centraliza filtros y formateo de warnings (pin_memory, MIOpen, LR sched)
from YOLOv11.engine.warnings import configure_warnings

configure_warnings()


# ---------------------------------------------------------------------------
# 1) Bootstrap ROCm/MIOpen DEBE ocurrir antes de importar torch
# ---------------------------------------------------------------------------


def _bootstrap_before_torch(args: argparse.Namespace) -> None:
    """Configura ROCm/MIOpen antes de importar torch.

    Usa sólo parámetros válidos de `MIOpenConfig`.
    Permite overrides vía variables de entorno (MIOPEN_*),
    y aplica `strict_before_torch=True` por seguridad.
    """
    try:
        from YOLOv11.engine.bootstrap_miopen import bootstrap, MIOpenConfig
    except Exception as e:
        print(f"[bootstrap_miopen] Advertencia: no se pudo importar bootstrap_miopen: {e}")
        return

    # Lee posibles overrides desde el entorno. Si no existen, usa defaults.
    find_mode = os.environ.get("MIOPEN_FIND_MODE", "FAST")
    user_db_path = os.environ.get("MIOPEN_USER_DB_PATH", None)
    disable_cache_env = os.environ.get("MIOPEN_DISABLE_CACHE", "1").strip().lower() in {"1", "true", "yes"}
    log_level_env = os.environ.get("MIOPEN_LOG_LEVEL", "0")

    try:
        log_level = int(log_level_env)
    except Exception:
        log_level = 0

    miopen_cfg = MIOpenConfig(
        find_mode=find_mode,
        user_db_path=user_db_path,
        disable_cache=disable_cache_env,
        log_level=log_level,
        extra_env={},
        strict_before_torch=True,
        verbose=1,
    )
    try:
        bootstrap(miopen_cfg)
    except Exception as e:
        print(f"[bootstrap_miopen] Advertencia: fallo en bootstrap MIOpen: {e}")


# ---------------------------------------------------------------------------
# 2) Imports del engine y dependencias (luego de bootstrap)
# ---------------------------------------------------------------------------


def _lazy_import_engine() -> Dict[str, Any]:
    """Importa módulos del engine una vez realizado el bootstrap.

    Nota: no importar `engine.CLI` aquí; el parser se usa sólo en `__main__`.
    """
    from YOLOv11.engine import (
        amp as engine_amp,
        optim as engine_optim,
        ema as engine_ema,
        callbacks as engine_callbacks,
        validator as engine_validator,
        warmup_sanity as engine_warmup,
        utils as ut,
        bn2gn_patch as b2g,
        hud as engine_hud,
    )
    # Utilidades solicitadas: DataLoader, Losses, Logger y Weights desde utility/
    from YOLOv11.utility import data_loader as util_data
    from YOLOv11.utility.losses import YOLOLoss
    from YOLOv11.utility.logger import ExperimentLogger
    from YOLOv11.utility.weights import WeightsManager

    import torch  # ahora sí, después de bootstrap

    return dict(
        amp=engine_amp,
        optim=engine_optim,
        ema=engine_ema,
        callbacks=engine_callbacks,
        validator=engine_validator,  # usado para validación interna (val_int)
        warmup=engine_warmup,
        utils=ut,
        b2g=b2g,
        hud=engine_hud,
        torch=torch,
        WeightsManager=WeightsManager,
        util_data=util_data,
        YOLOLoss=YOLOLoss,
        ExperimentLogger=ExperimentLogger,
    )


# ---------------------------------------------------------------------------
# 3) Utilidad local: construcción de modelo y datos
# ---------------------------------------------------------------------------

from YOLOv11.engine.Trainer import Trainer, DotDict


def _build_model_and_data(cfg: DotDict, engine: Dict[str, Any]):
    """Construye modelo y dataloader de entrenamiento usando utilidades del proyecto.

    - Delegación total a `utility/data_loader.py` (API `build_train_bundle`).
    - Normalización de `names` a lista y telemetría de dataset.
    - **Sin** crear partición de validación externa.
    """
    ut = engine["utils"]

    # 1) Modelo
    model = ut.build_model(cfg.model, variant=cfg.variant)

    # 2) DataLoader + names + resumen desde utility/data_loader.py
    project_root = Path(__file__).resolve().parent  # raíz que contiene configs/ y models/
    udata = engine["util_data"]

    train_loader, names_list, info = udata.build_train_bundle(
        project_root=project_root,
        split="train",
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        workers=cfg.workers,
        augment=True,
    )

    # 3) Telemetría de datos (el prefijo [data_loader] se emite desde train.py)
    if cfg.get("dl_info", False):
        names_fmt = "[" + ", ".join(str(n) for n in names_list) + "]"
        msg = (
            f"[data_loader] split={info.split}\n"
            f"[data_loader] images={info.images_dir}\n"
            f"[data_loader] labels={info.labels_dir}\n"
            f"[data_loader] count={info.count} nc={info.nc}  names={names_fmt}\n"
            f"[data_loader] imgsz={info.imgsz}  workers={info.workers}  "
            f"pin_memory={info.pin_memory}  persistent={info.persistent_workers}"
        )
        print(msg, flush=True)

    return model, train_loader, names_list


# ---------------------------------------------------------------------------
# 4) main
# ---------------------------------------------------------------------------


def main(cli: argparse.Namespace) -> None:
    # 1) Bootstrap ROCm/MIOpen antes de cualquier import de torch
    _bootstrap_before_torch(cli)

    # 2) Cargar engine (torch + submódulos)
    engine = _lazy_import_engine()
    ut = engine["utils"]

    # 3) Autodetección AMP
    amp_mode = cli.amp
    if amp_mode == "auto":
        amp_mode = ut.auto_amp_mode()  # bf16 si disponible, luego fp16, si no off

    # 4) HUD por defecto: ON si stdout es TTY (si CLI no lo resolvió)
    if getattr(cli, "hud", None) is None:
        cli.hud = sys.stdout.isatty()

    # 5) Nombre por timestamp si no se especifica
    name = cli.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 6) Construcción de configuración normalizada (DotDict)
    cfg = DotDict(
        data=cli.data,
        model=cli.model,
        variant=cli.variant,
        epochs=cli.epochs,
        batch=cli.batch,
        imgsz=cli.imgsz,
        workers=cli.workers,
        device=cli.device,
        seed=cli.seed,
        lr=cli.lr,
        wd=cli.wd,
        clip_norm=cli.clip_norm,
        clip_mode=cli.clip_mode,
        warmup=cli.warmup,
        warmup_epochs=int(cli.warmup_epochs) if getattr(cli, "warmup_epochs", None) is not None else 0,
        bn2gn=cli.bn2gn,
        amp=amp_mode,
        ema=bool(cli.ema),
        compile=bool(cli.compile),
        hud=bool(cli.hud),
        resume=cli.resume,
        project=cli.project or "runs/train",
        name=name,
        exist_ok=bool(getattr(cli, "exist_ok", False)),
        time_limit=float(getattr(cli, "time_limit", 0.0)),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        test=bool(cli.test),
        dl_info=bool(getattr(cli, "dl_info", False)),
        # val_int
        val_int_interval=int(getattr(cli, "val_int_interval", 5)),
        val_int_max_batches=int(getattr(cli, "val_int_max_batches", 1)),
        val_int_use_train_subset=bool(getattr(cli, "val_int_use_train_subset", False)),
        val_int_conf=float(getattr(cli, "val_int_conf", 0.25)),
        val_int_split=str(getattr(cli, "val_int_split", "val")),
        val_int_pivots=bool(getattr(cli, "val_int_pivots", True)),
        val_int_tb=bool(getattr(cli, "val_int_tb", True)),
        val_int_tb_nrow=int(getattr(cli, "val_int_tb_nrow", 3)),
        val_int_tb_conf=float(getattr(cli, "val_int_tb_conf", 0.25)),
        val_int_tb_topk=int(getattr(cli, "val_int_tb_topk", 5)),
        dataset_base=getattr(cli, "dataset_base", None),
    )

    # 7) Construcción de modelo y dataloader (sin validación externa)
    model, train_loader, names = _build_model_and_data(cfg, engine)

    # 8) Instanciar trainer del engine
    trainer = Trainer(model, train_loader, names, cfg, engine)

    # 9) Ejecutar (el banner se imprime dentro de Trainer.fit())
    trainer.fit()


if __name__ == "__main__":
    # Parseo modular en dos etapas (presets + YAML) sin dependencias a torch
    from YOLOv11.engine.CLI import parse_args_two_stage

    args = parse_args_two_stage(sys.argv[1:])
    main(args)
