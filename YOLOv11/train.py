# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/train.py
# Descripción: Script principal de entrenamiento. Orquesta bootstrap ROCm,
#  construcción de modelo/datos, loop de entrenamiento con AMP/EMA,
#  callbacks/overlays, validación y HUD de consola. Incluye modo --test
#  para prueba de ensamblado y warmup configurable.
#==============================================================

from __future__ import annotations
import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Any, Dict, Tuple

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
        print(f"[bootstrap] Advertencia: no se pudo importar bootstrap_miopen: {e}")
        return

    # Lee posibles overrides desde el entorno. Si no existen, usa defaults.
    find_mode = os.environ.get("MIOPEN_FIND_MODE", "FAST")
    user_db_path = os.environ.get("MIOPEN_USER_DB_PATH", None)
    disable_cache_env = os.environ.get("MIOPEN_DISABLE_CACHE", "0").strip().lower() in {"1","true","yes"}
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
        print(f"[bootstrap] Advertencia: fallo en bootstrap MIOpen: {e}")

# ---------------------------------------------------------------------------
# 2) Imports del engine y dependencias (luego de bootstrap)
# ---------------------------------------------------------------------------

def _lazy_import_engine():
    """Importa módulos del engine una vez realizado el bootstrap."""
    from YOLOv11.engine import (
        amp as engine_amp,
        optim as engine_optim,
        ema as engine_ema,
        callbacks as engine_callbacks,
        validator as engine_validator,
        warmup_sanity as engine_warmup,
        utils as ut,
        bn2gn_patch as b2g,
        overlays as _engine_overlays,  # usado por callbacks
        hud as engine_hud,
    )
    import torch  # ahora sí
    return dict(
        amp=engine_amp,
        optim=engine_optim,
        ema=engine_ema,
        callbacks=engine_callbacks,
        validator=engine_validator,
        warmup=engine_warmup,
        utils=ut,
        b2g=b2g,
        hud=engine_hud,
        torch=torch,
    )


# ---------------------------------------------------------------------------
# 3) Utilidades locales
# ---------------------------------------------------------------------------

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _print_banner(cfg: DotDict, engine: Dict[str, Any]) -> None:
    ut = engine["utils"]
    device_info = ut.device_info()
    mode = "TEST" if cfg.test else ("WARMUP" if cfg.warmup != "off" else "TRAIN")
    print(
        "\n[YOLOv11] "
        f"MODE={mode}  VARIANT={cfg.variant}  BN2GN={cfg.bn2gn}  "
        f"AMP={cfg.amp}  EMA={'ON' if cfg.ema else 'OFF'}  HUD={'ON' if cfg.hud else 'OFF'}\n"
        f"Device={device_info}  Batch={cfg.batch}  ImgSz={cfg.imgsz}  Epochs={cfg.epochs}\n"
        f"Project={cfg.project}  SaveDir={cfg.save_dir}\n"
        f"Warmup={cfg.warmup}  Overlays every {cfg.overlays_interval} epochs\n"
    )


def _build_model_and_data(cfg: DotDict, engine: Dict[str, Any]):
    """Construye modelo y dataloaders usando utilidades del proyecto.
    Requiere que YOLOv11/engine/utils.py provea funciones compatibles:
      - build_model(model_yaml, variant) -> nn.Module
      - build_dataloaders(data_yaml, imgsz, batch, workers) -> (train_loader, val_loader, names)
    """
    ut = engine["utils"]
    model = ut.build_model(cfg.model, variant=cfg.variant)
    train_loader, val_loader, names = ut.build_dataloaders(
        cfg.data, imgsz=cfg.imgsz, batch=cfg.batch, workers=cfg.workers
    )
    return model, train_loader, val_loader, names


def _fitness(metrics: Dict[str, float]) -> float:
    # Fitness clásica: 0.1*mAP50 + 0.9*mAP50-95 si existen; tolerante a faltantes
    m50 = float(metrics.get("map50", 0.0))
    m5095 = float(metrics.get("map", metrics.get("map50-95", 0.0)))
    return 0.1 * m50 + 0.9 * m5095


# ---------------------------------------------------------------------------
# 4) Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_loader, val_loader, names, cfg: DotDict, engine: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.names = names
        self.cfg = cfg
        self.engine = engine

        ut = engine["utils"]
        self.device = ut.select_device(cfg.device)
        ut.seed_everything(cfg.seed)
        self.save_dir = ut.setup_save_dir(cfg.project, cfg.name, exist_ok=cfg.exist_ok)
        self.cfg.save_dir = self.save_dir  # para banner

        # compile opcional
        self.model.to(self.device)
        self.model = ut.maybe_compile(self.model, cfg.compile)

        # BN→GN
        engine["b2g"].apply_bn2gn_patch(self.model, policy=cfg.bn2gn, verbose=1)

        # AMP, Optim, Scheduler, Accumulate
        iters_per_epoch = len(self.train_loader)
        AmpConfig = engine["amp"].AmpConfig
        mode = str(self.cfg.amp).lower()
        if mode == "off":
            amp_cfg = AmpConfig(enabled=False)
        elif mode in ("bf16", "fp16"):
            amp_cfg = AmpConfig(enabled=True, dtype=mode)
        else:
            # 'auto' ya fue resuelto antes con utils.auto_amp_mode(); por compatibilidad, asuma fp16
            amp_cfg = AmpConfig(enabled=True, dtype="fp16")
        self.ampmgr = engine["amp"].AmpManager(amp_cfg)
        optcfg = dict(
            epochs=cfg.epochs,
            base_lr=cfg.lr,
            weight_decay=cfg.wd,
            warmup_type="linear",
            cosine=True,
            clip_norm=cfg.clip_norm,
            clip_mode=cfg.clip_mode,
            iters_per_epoch=iters_per_epoch,
        )
        self.optimizer, self.scheduler, self.accumulate = engine["optim"].build_optimizer_and_scheduler(
            self.model,
            engine["optim"].OptimConfig(**optcfg),
            batch_per_gpu=cfg.batch,
            world_size=cfg.world_size,
        )

        # EMA
        self.ema = engine["ema"].ModelEMA(self.model, cfg=DotDict(enabled=bool(cfg.ema))) if cfg.ema else None

        # Callbacks
        self.cb = engine["callbacks"].build_default_callbacks(self.save_dir, cfg=DotDict(overlays_interval=cfg.overlays_interval))

        # HUD
        self.hud = engine["hud"].HUD(engine["hud"].HUDConfig(enable=cfg.hud)) if cfg.hud else None

        # Checkpoints
        self.ckpt = ut.CheckpointManager(self.save_dir)
        self.start_epoch = 0
        self.best_fitness = -1e9

        # Límite de tiempo (0 = ilimitado)
        self.timer = ut.timed_stop(cfg.time_limit)

    # -----------------------------
    def fit(self) -> None:
        self._print_mode_banner()

        # Modo --test: prueba de ensamblado rápida y salida
        if self.cfg.test:
            self._assembly_test()
            print("[TEST] Assembly test passed ✔")
            return

        engine = self.engine
        ut = engine["utils"]

        self.cb.on_train_start(trainer=self)
        iters_per_epoch = len(self.train_loader)

        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.model.train()
            if self.hud:
                self.hud.on_epoch_start(epoch, self.cfg.epochs, iters_per_epoch)

            t_iter = time.perf_counter()
            for i, batch in enumerate(self.train_loader, start=1):
                with self.ampmgr.autocast():
                    loss, items = self.model(batch)  # contrato: (Tensor, dict)

                do_step = (i % self.accumulate) == 0
                engine["amp"].safe_backward_step(
                    loss, self.optimizer, self.ampmgr,
                    clip_fn=lambda: engine["optim"].clip_gradients(self.model, self.cfg.clip_norm, self.cfg.clip_mode),
                    zero_grad=do_step, set_to_none=True,
                )
                if do_step:
                    self.scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)

                dt_ms = (time.perf_counter() - t_iter) * 1000.0
                if self.hud:
                    lr = float(self.optimizer.param_groups[0]['lr'])
                    self.hud.update(epoch, i, iters_per_epoch, lr, float(loss.item()), items, dt_ms)
                self.cb.on_train_batch_end(self, epoch * iters_per_epoch + i, float(loss.item()), items)
                t_iter = time.perf_counter()

                if ut.SIGNALS.stop or self.timer.expired():
                    break

            if self.hud:
                self.hud.on_epoch_end()

            # Validación
            model_eval = self.ema.ema if self.ema else self.model
            val_metrics = self.engine["validator"].validate(
                model_eval, self.val_loader, names=self.names,
                save_dir=self.save_dir, phase="val", slot="epoch", step_tag=f"epoch_{epoch:03d}"
            )

            fit = _fitness(val_metrics)
            best = fit > self.best_fitness
            self.best_fitness = max(self.best_fitness, fit)
            path, _ = self.ckpt.save(epoch, self.model, self.optimizer, self.scheduler, self.ema, val_metrics, best=best, best_fitness=self.best_fitness)
            self.cb.on_model_save(self, path, best)

            self.cb.on_fit_epoch_end(self, epoch, train_stats={}, val_stats=val_metrics)
            if self.hud:
                self.hud.update_epoch(epoch, **val_metrics)

            if ut.SIGNALS.stop or self.timer.expired():
                break

        if self.hud:
            self.hud.close()

    # -----------------------------
    def _assembly_test(self) -> None:
        engine = self.engine
        ut = engine["utils"]
        iters_per_epoch = len(self.train_loader)

        # sanity: dummy + un minibatch real con backward/step
        if self.cfg.warmup in ("sanity", "fast", "full"):
            engine["warmup"].warmup_sanity(self.model, device=self.device, cfg=DotDict(mode=self.cfg.warmup))

        self.model.train()
        for i, batch in enumerate(self.train_loader, start=1):
            with self.ampmgr.autocast():
                loss, items = self.model(batch)
            engine["amp"].safe_backward_step(
                loss, self.optimizer, self.ampmgr,
                clip_fn=lambda: engine["optim"].clip_gradients(self.model, self.cfg.clip_norm, self.cfg.clip_mode),
                zero_grad=True, set_to_none=True,
            )
            if self.ema:
                self.ema.update(self.model)
            break  # solo 1 minibatch

        self.model.eval()
        with engine["torch"].inference_mode():
            _ = self.engine["validator"].validate(
                self.model, self.val_loader, names=self.names,
                save_dir=self.save_dir, phase="val", slot="test", step_tag="assembly_test"
            )

    # -----------------------------
    def _print_mode_banner(self):
        _print_banner(self.cfg, self.engine)


# ---------------------------------------------------------------------------
# 5) Argumentos CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("YOLOv11 Trainer")
    # rutas / datos / modelo
    p.add_argument('--data', type=str, required=True, help='dataset.yaml')
    p.add_argument('--model', type=str, required=True, help='yolo11.yaml')
    p.add_argument('--parser', type=str, required = True, help='parser.yaml')

    # variantes y params clave
    p.add_argument('--variant', type=str, default='s', choices=['n','s','m','l','x'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--workers', type=int, default=8)

    # sistema/semilla/device
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)

    # optim
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--clip-norm', type=float, default=0.0)
    p.add_argument('--clip-mode', type=str, default='norm', choices=['norm','value'])

    # mitigaciones / precisión / ema / compile
    p.add_argument('--warmup', type=str, default='sanity', choices=['off','sanity','fast','full'])
    p.add_argument('--bn2gn', type=str, default='on_error', choices=['off','on','on_error'])
    p.add_argument('--amp', type=str, default='auto', choices=['auto','off','fp16','bf16'])
    p.add_argument('--ema', action='store_true', default=True)
    p.add_argument('--no-ema', dest='ema', action='store_false')
    p.add_argument('--compile', action='store_true', default=False)

    # overlays / hud / resume / tiempo / outputs
    p.add_argument('--overlays-interval', type=int, default=5)
    p.add_argument('--hud', action='store_true', default=None)
    p.add_argument('--no-hud', dest='hud', action='store_false')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--project', type=str, default='runs/train')
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--exist-ok', action='store_true', default=False)
    p.add_argument('--time-limit', type=float, default=0.0)

    # modo prueba
    p.add_argument('--test', action='store_true', help='Ejecuta prueba de ensamblado y sale')

    return p


# ---------------------------------------------------------------------------
# 6) main
# ---------------------------------------------------------------------------

def main(cli: argparse.Namespace) -> None:
    _bootstrap_before_torch(cli)
    engine = _lazy_import_engine()
    ut = engine["utils"]

    # autodetección AMP
    amp_mode = cli.amp
    if amp_mode == 'auto':
        amp_mode = ut.auto_amp_mode()  # bf16 si disponible, luego fp16, si no off

    # HUD por defecto: ON si stdout es TTY
    if cli.hud is None:
        cli.hud = sys.stdout.isatty()

    # nombre por timestamp si no se especifica
    name = cli.name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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
        bn2gn=cli.bn2gn,
        amp=amp_mode,
        ema=bool(cli.ema),
        compile=bool(cli.compile),
        overlays_interval=cli.overlays_interval,
        hud=bool(cli.hud),
        resume=cli.resume,
        project=cli.project,
        name=name,
        exist_ok=cli.exist_ok,
        time_limit=cli.time_limit,
        world_size=int(os.environ.get('WORLD_SIZE', '1')),
        test=bool(cli.test),
    )

    # Construcción de modelo y dataloaders
    model, train_loader, val_loader, names = _build_model_and_data(cfg, engine)

    # Instanciar trainer
    trainer = Trainer(model, train_loader, val_loader, names, cfg, engine)

    # Banner de configuración efectiva
    _print_banner(cfg, engine)

    # Ejecutar
    trainer.fit()


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
