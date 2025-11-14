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
#  callbacks, validación interna (val_int) y HUD de consola. Incluye modo --test
#  para prueba de ensamblado y warmup configurable.
#==============================================================

from __future__ import annotations
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

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
    disable_cache_env = os.environ.get("MIOPEN_DISABLE_CACHE", "1").strip().lower() in {"1","true","yes"}
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
    # Utilidades solicitadas: DataLoader, Losses y Logger desde utility/
    from YOLOv11.utility import data_loader as util_data
    from YOLOv11.utility.losses import YOLOLoss
    from YOLOv11.utility.logger import ExperimentLogger

    from YOLOv11.utility.weights import WeightsManager
    import torch  # ahora sí
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
    warm_ep = f" (epochs={cfg.warmup_epochs})" if int(cfg.get("warmup_epochs", 0)) > 0 else ""
    print(
        "\n[YOLOv11] "
        f"MODE={mode}  VARIANT={cfg.variant}  BN2GN={cfg.bn2gn}  "
        f"AMP={cfg.amp}  EMA={'ON' if cfg.ema else 'OFF'}  HUD={'ON' if cfg.hud else 'OFF'}\n"
        f"Device={device_info}  Batch={cfg.batch}  ImgSz={cfg.imgsz}  Epochs={cfg.epochs}\n"
        f"Project={cfg.project}  SaveDir={cfg.save_dir}\n"
        f"Warmup={cfg.warmup}{warm_ep}  ValInt every {cfg.val_int_interval} epochs\n"
    )


# ---------------------------------------------------------------------------
# 3.1) Construcción de modelo y datos (delegando en data_loader.py)
# ---------------------------------------------------------------------------


def _build_model_and_data(cfg: DotDict, engine: Dict[str, Any]):
    """Construye modelo y dataloader de entrenamiento usando utilidades del proyecto.

    Cambios:
      - Delegación total al `utility/data_loader.py` (API `build_train_bundle`).
      - Normalización de `names` a lista y telemetría de dataset impresa aquí.
      - **Sin** crear partición de validación externa.
    """
    ut = engine["utils"]

    # 1) Modelo (igual que antes)
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


def _fitness(metrics: Dict[str, float]) -> float:
    # Fitness clásica: 0.1*mAP50 + 0.9*mAP50-95 si existen; tolerante a faltantes
    m50 = float(metrics.get("map50", 0.0))
    m5095 = float(metrics.get("map", metrics.get("map50-95", 0.0)))
    return 0.1 * m50 + 0.9 * m5095


# ---------------------------------------------------------------------------
# 4) Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_loader, names, cfg: DotDict, engine: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.names = names
        self.cfg = cfg
        self.engine = engine

        ut = engine["utils"]
        self.device = ut.select_device(cfg.device)
        ut.seed_everything(cfg.seed)

        # === Logger de experimento (utility/logger.py) ===
        self.logger = engine["ExperimentLogger"](
            variant=cfg.variant,
            phase="train",
            is_test=cfg.test,
            run_name=cfg.name,
            reset_final=not cfg.exist_ok,
        )

        # Directorio oficial de guardado: slot de runs definido por el logger
        self.save_dir = Path(self.logger.runs_dir)
        self.cfg.save_dir = str(self.save_dir)  # para banner y downstream
        # Usamos `project` como etiqueta de raíz de runs (p.ej. YOLOv11/runs)
        try:
            self.cfg.project = str(self.save_dir.parents[3])
        except Exception:
            self.cfg.project = str(self.save_dir.parent)

        # Snapshot de config y resumen del modelo
        try:
            cfg_dict = dict(self.cfg)
            self.logger.save_config_json(cfg_dict)
        except Exception:
            pass
        try:
            n_params = sum(p.numel() for p in model.parameters())
            extra = {
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "amp": cfg.amp,
                "device": str(self.device),
                "params": int(n_params),
                "variant": cfg.variant,
            }
            self.logger.save_model_summary(self.model, extra=extra)
        except Exception:
            pass

        # BN→GN antes de fijar device/compile para evitar capas nuevas en CPU
        engine["b2g"].apply_bn2gn_patch(self.model, policy=cfg.bn2gn, verbose=1)

        # Mover modelo (ya parcheado) al dispositivo destino
        self.model.to(self.device)

        # compile opcional
        self.model = ut.maybe_compile(self.model, cfg.compile)

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

        # Delegar completamente en engine.optim para obtener configuraciones desde parser
        self.optimizer, self.scheduler, self.accumulate, self._optim_cfg = engine["optim"].build_optim_from_parser(
            self.model,
            None,  # el optimizador resolverá internamente el parser/config
            iters_per_epoch=iters_per_epoch,
            batch_per_gpu=cfg.batch,
            world_size=cfg.world_size,
        )

        # EMA
        self.ema = engine["ema"].ModelEMA(self.model, cfg=engine["ema"].EMAConfig()) if cfg.ema else None

        # Callbacks (sin overlays)
        self.cb = engine["callbacks"].build_default_callbacks(self.save_dir, cfg=DotDict(val_int_interval=cfg.val_int_interval))

        # HUD
        self.hud = engine["hud"].HUD(engine["hud"].HUDConfig(enable=cfg.hud)) if cfg.hud else None

        # Weights Manager (reemplazo de CheckpointManager)
        self.wm = engine["WeightsManager"](
            project_root=Path(__file__).resolve().parent,
            variant=cfg.variant,
            phase="train",
            run_name=cfg.name,
            is_test=cfg.test,
            reset_final=False,
        )
        self.start_epoch = 0
        self.best_fitness = -1e9

        # === Criterio de pérdida (utility/losses.YOLOLoss) ===
        try:
            nc = len(self.names) if hasattr(self.names, "__len__") else int(self.names)
        except Exception:
            nc = 80
        self.criterion = engine["YOLOLoss"](nc=nc).to(self.device)

        # Reanudación si corresponde
        if cfg.resume:
            prefer = str(cfg.resume).lower()
            info = None
            if prefer in ("last", "best"):
                info = self.wm.try_resume(self.model, optimizer=self.optimizer, scheduler=self.scheduler, prefer=prefer)
            else:
                try:
                    # Interpretar cfg.resume como ruta explícita
                    ckpt = self.wm.load(Path(prefer))
                    if "state_dict" in ckpt and ckpt["state_dict"] is not None:
                        self.model.load_state_dict(ckpt["state_dict"], strict=False)
                    if self.optimizer is not None and ckpt.get("optimizer") is not None:
                        self.optimizer.load_state_dict(ckpt["optimizer"])
                    if self.scheduler is not None and ckpt.get("scheduler") is not None:
                        self.scheduler.load_state_dict(ckpt["scheduler"])
                    info = {"resumed": True, "start_epoch": int(ckpt.get("epoch", 0)) + 1, "ckpt_path": Path(prefer)}
                except Exception:
                    info = {"resumed": False, "start_epoch": 0, "ckpt_path": None}
            self.start_epoch = int((info or {}).get("start_epoch", 0))

        # Límite de tiempo (0 = ilimitado)
        self.timer = ut.timed_stop(cfg.time_limit)

        # Adaptador de loader a dict en dispositivo (delegado al módulo utility.data_loader)
        self.train_loader_adapt = engine["util_data"].as_dict_loader(self.train_loader, self.device)

    # -----------------------------
    def fit(self) -> None:
        self._print_mode_banner()

        # Warm-up previo a TRAIN si se solicita por CLI (no interrumpe flujo normal)
        if not self.cfg.test and int(self.cfg.get("warmup_epochs", 0)) > 0 and self.hud:
            self._run_warmup_hud(loops=int(self.cfg.warmup_epochs))

        # Modo --test: prueba de ensamblado rápida y salida
        if self.cfg.test:
            self._assembly_test()
            print("[TEST] Assembly test passed ✔")
            try:
                self.logger.close()
            except Exception:
                pass
            return

        engine = self.engine
        ut = engine["utils"]

        self.cb.on_train_start(trainer=self)
        iters_per_epoch = len(self.train_loader)

        # Preparar nombres como lista para validator (si vienen como dict)
        if isinstance(self.names, dict):
            names_list: List[str] = [self.names[i] for i in sorted(self.names.keys())]
        else:
            names_list = list(self.names) if hasattr(self.names, "__iter__") else []

        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.model.train()
            if self.hud:
                self.hud.on_epoch_start(epoch, self.cfg.epochs, iters_per_epoch)

            # Acumuladores de métricas de entrenamiento por época
            sum_loss = 0.0
            count = 0
            scalars_sum: Dict[str, float] = {}

            t_iter = time.perf_counter()
            for i, batch in enumerate(self.train_loader_adapt, start=1):
                with self.ampmgr.autocast():
                    # forward del core + pérdida YOLOLoss
                    x = batch["img"]
                    core = getattr(self.model, "core", self.model)
                    preds = core(x)
                    loss, scalars = self.criterion(preds, batch["targets"])
                    items = {"loss": float(loss.detach()), **{k: float(v) for k, v in scalars.items()}}

                # acumular métricas
                sum_loss += float(loss.item())
                count += 1
                for k, v in scalars.items():
                    scalars_sum[k] = scalars_sum.get(k, 0.0) + float(v)

                do_step = (i % self.accumulate) == 0
                self.engine["amp"].safe_backward_step(
                    loss, self.optimizer, self.ampmgr,
                    clip_fn=lambda: self.engine["optim"].clip_gradients(self.model, self.cfg.clip_norm, self.cfg.clip_mode),
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

            # Métricas promedio de train por época
            train_metrics: Dict[str, float] = {"loss": (sum_loss / max(1, count))}
            for k, v in scalars_sum.items():
                train_metrics[k] = v / max(1, count)

            # ===== Validación interna (val_int) por intervalo =====
            val_metrics: Dict[str, float] = {}
            run_val_int = (epoch % max(1, int(self.cfg.val_int_interval)) == 0) or (epoch == 0)
            if run_val_int:
                model_eval = self.ema.ema if self.ema is not None else self.model
                try:
                    val_metrics = self.engine["validator"].validate_interna(
                        model_eval,
                        loader=self.train_loader_adapt if self.cfg.val_int_use_train_subset else None,
                        names=names_list,
                        save_dir=str(self.save_dir),
                        conf_thres=float(self.cfg.val_int_conf),
                        iou_thres=0.60,
                        device=str(self.device),
                        # internos
                        epoch=int(epoch),
                        max_batches=int(self.cfg.val_int_max_batches),
                        split=str(self.cfg.val_int_split),
                        use_pivots=bool(self.cfg.val_int_pivots),
                        # TB
                        tb_enable=bool(self.cfg.val_int_tb),
                        tb_variant=str(self.cfg.variant),
                        tb_run_name=str(self.cfg.name),
                        tb_nrow=int(self.cfg.val_int_tb_nrow),
                        tb_conf_thr=float(self.cfg.val_int_tb_conf),
                        tb_topk=int(self.cfg.val_int_tb_topk),
                        dataset_base=self.cfg.dataset_base,
                        # slots
                        phase="val",
                        slot="epoch",
                        run_name=str(self.cfg.name),
                        step_tag=f"epoch_{epoch:03d}",
                        verbose=1,
                    )
                except Exception as e:
                    print(f"[val_int] Advertencia: validación interna falló en época {epoch}: {e}")
                    val_metrics = {}

            # Fitness y guardado de pesos
            fitness = _fitness(val_metrics) if val_metrics else -1e9
            is_best = fitness > self.best_fitness
            if is_best:
                self.best_fitness = fitness

            path = self.wm.save_epoch(
                self.model,
                epoch,
                score=(fitness if fitness > -1e8 else 0.0),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                extra={"imgsz": self.cfg.imgsz, "batch": self.cfg.batch},
                save_full_model=False,
            )
            self.cb.on_model_save(self, path, best=is_best)

            # Logging de época (train + val_int si existió)
            try:
                self.logger.log_epoch(epoch, train_metrics, split="train")
                if val_metrics:
                    self.logger.log_epoch(epoch, val_metrics, split="val_int")
            except Exception:
                pass

            # Callback de fin de época (usar métricas disponibles)
            self.cb.on_fit_epoch_end(self, epoch, train_stats=train_metrics, val_stats=(val_metrics or {}))
            if self.hud:
                self.hud.update_epoch(epoch)

            if ut.SIGNALS.stop or self.timer.expired():
                break

        if self.hud:
            self.hud.close()
        try:
            self.logger.close()
        except Exception:
            pass

    # -----------------------------
    def _run_warmup_hud(self, loops: int = 1) -> None:
        """Ejecuta warm-up sintético con HUD durante `loops` épocas virtuales.
        Usa **exclusivamente** la API de warmup del HUD (on_warmup_*),
        separada del HUD de TRAIN, para evitar encabezados como `[TRAIN] 2/1`.
        """
        engine = self.engine
        torch_mod = engine["torch"]
        core = getattr(self.model, "core", self.model)

        # Inferir stride del modelo (fallback 32)
        stride = None
        for obj in (core, getattr(core, "model", None)):
            if obj is None:
                continue
            for attr in ("stride", "strides", "max_stride"):
                if hasattr(obj, attr):
                    stride = getattr(obj, attr)
                    break
            if stride is not None:
                break
        try:
            if isinstance(stride, (list, tuple)):
                stride = int(max(stride))
            elif isinstance(stride, dict):
                stride = int(max(int(v) for v in stride.values()))
            elif hasattr(stride, "max"):
                try:
                    stride = int(getattr(stride, "max")().item())
                except Exception:
                    stride = int(stride)
            elif isinstance(stride, (int, float)):
                stride = int(stride)
            else:
                stride = None
        except Exception:
            stride = None
        if not stride or stride <= 0:
            stride = 32

        # Determinar dtype por AMP (para HUD)
        amp_mode = str(self.cfg.amp).lower()
        if amp_mode == "bf16":
            dtype = "bf16"
        elif amp_mode == "fp16":
            dtype = "fp16"
        else:
            dtype = "fp32"

        # Iteraciones por modo warmup
        iters = 2 if self.cfg.warmup == "sanity" else (5 if self.cfg.warmup == "fast" else 10)
        bs = int(self.cfg.batch)
        imgsz = int(self.cfg.imgsz)
        channels = 3
        x = torch_mod.randn(bs, channels, imgsz, imgsz, device=self.device, dtype=torch_mod.float32)

        # Contexto MIOpen para HUD
        find_mode = os.environ.get("MIOPEN_FIND_MODE", None)
        cache_env = os.environ.get("MIOPEN_DISABLE_CACHE", None)
        cache_disabled = None
        if cache_env is not None:
            try:
                cache_disabled = cache_env.strip().lower() in {"1", "true", "yes"}
            except Exception:
                cache_disabled = None

        # Cabecera de warmup
        print("[WARMUP] Comenzando warmup...", flush=True)
        if self.hud:
            self.hud.on_warmup_start(
                total_iters=iters,
                dtype=dtype,
                compile=bool(self.cfg.compile),
                stride=int(stride),
                bn2gn=str(self.cfg.bn2gn),
                amp=(amp_mode != "off"),
                find_mode=find_mode,
                cache_disabled=cache_disabled,
            )

        # Épocas sintéticas de warmup
        for ep in range(int(max(1, loops))):
            print(f"[WARMUP] Epoch {ep+1}/{int(max(1, loops))}", flush=True)
            for i in range(1, iters + 1):
                t0 = time.perf_counter()
                with self.ampmgr.autocast():
                    _ = core(x)
                if torch_mod.cuda.is_available():
                    try:
                        torch_mod.cuda.synchronize()
                    except Exception:
                        pass
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if self.hud:
                    self.hud.update_warmup(i, iters, dt_ms)

        # Resumen y cierre de warmup
        if self.hud:
            self.hud.on_warmup_end()
        print("[WARMUP] Finalizado.", flush=True)

    # -----------------------------
    def _assembly_test(self) -> None:
        # Ejecuta warm-up con HUD si se solicitó por modo o por `warmup_epochs`
        if self.cfg.warmup in ("sanity", "fast", "full") or int(self.cfg.get("warmup_epochs", 0)) > 0:
            self._run_warmup_hud(loops=int(max(1, int(self.cfg.get("warmup_epochs", 0)))))

        # sanity: un minibatch real con backward/step
        self.model.train()
        for i, batch in enumerate(self.train_loader_adapt, start=1):
            with self.ampmgr.autocast():
                core = getattr(self.model, "core", self.model)
                preds = core(batch["img"])  # forward
                loss, _ = self.criterion(preds, batch["targets"])  # pérdida
            self.engine["amp"].safe_backward_step(
                loss, self.optimizer, self.ampmgr,
                clip_fn=lambda: self.engine["optim"].clip_gradients(self.model, self.cfg.clip_norm, self.cfg.clip_mode),
                zero_grad=True, set_to_none=True,
            )
            if self.ema:
                self.ema.update(self.model)
            break  # solo 1 minibatch

    # -----------------------------
    def _print_mode_banner(self):
        _print_banner(self.cfg, self.engine)


# ---------------------------------------------------------------------------
# 5) main
# ---------------------------------------------------------------------------


def main(cli: argparse.Namespace) -> None:
    _bootstrap_before_torch(cli)
    engine = _lazy_import_engine()
    ut = engine["utils"]

    # autodetección AMP
    amp_mode = cli.amp
    if amp_mode == 'auto':
        amp_mode = ut.auto_amp_mode()  # bf16 si disponible, luego fp16, si no off

    # HUD por defecto: ON si stdout es TTY (si CLI no lo resolvió)
    if getattr(cli, 'hud', None) is None:
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
        warmup_epochs=int(cli.warmup_epochs) if getattr(cli, 'warmup_epochs', None) is not None else 0,
        bn2gn=cli.bn2gn,
        amp=amp_mode,
        ema=bool(cli.ema),
        compile=bool(cli.compile),
        hud=bool(cli.hud),
        resume=cli.resume,
        project=cli.project or 'runs/train',
        name=name,
        exist_ok=bool(getattr(cli, 'exist_ok', False)),
        time_limit=float(getattr(cli, 'time_limit', 0.0)),
        world_size=int(os.environ.get('WORLD_SIZE', '1')),
        test=bool(cli.test),
        dl_info=bool(getattr(cli, 'dl_info', False)),
        # val_int
        val_int_interval=int(getattr(cli, 'val_int_interval', 5)),
        val_int_max_batches=int(getattr(cli, 'val_int_max_batches', 1)),
        val_int_use_train_subset=bool(getattr(cli, 'val_int_use_train_subset', False)),
        val_int_conf=float(getattr(cli, 'val_int_conf', 0.25)),
        val_int_split=str(getattr(cli, 'val_int_split', 'val')),
        val_int_pivots=bool(getattr(cli, 'val_int_pivots', True)),
        val_int_tb=bool(getattr(cli, 'val_int_tb', True)),
        val_int_tb_nrow=int(getattr(cli, 'val_int_tb_nrow', 3)),
        val_int_tb_conf=float(getattr(cli, 'val_int_tb_conf', 0.25)),
        val_int_tb_topk=int(getattr(cli, 'val_int_tb_topk', 5)),
        dataset_base=getattr(cli, 'dataset_base', None),
    )

    # Construcción de modelo y dataloader (sin validación externa)
    model, train_loader, names = _build_model_and_data(cfg, engine)

    # Instanciar trainer
    trainer = Trainer(model, train_loader, names, cfg, engine)

    # Ejecutar (el banner se imprime dentro de Trainer.fit())
    trainer.fit()


if __name__ == "__main__":
    # Parseo modular en dos etapas (presets + YAML) sin dependencias a torch
    from YOLOv11.engine.CLI import parse_args_two_stage
    args = parse_args_two_stage(sys.argv[1:])
    main(args)
