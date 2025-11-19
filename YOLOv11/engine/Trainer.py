# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/Trainer.py
# Descripción: Implementación de la clase Trainer para el pipeline
#              de entrenamiento YOLOv11. Orquesta el loop de train,
#              AMP/EMA, callbacks, validación interna (val_int), HUD
#              de consola y guardado de pesos vía WeightsManager.
#==============================================================

from __future__ import annotations

import os
import time
import copy
from pathlib import Path
from typing import Any, Dict, List


__all__ = ["DotDict", "Trainer", "_fitness", "_print_banner"]


# -------------------------------
# Utilitarios locales
# -------------------------------


class DotDict(dict):
    """Diccionario con acceso por atributos (cfg.x en vez de cfg["x"])."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _print_banner(cfg: DotDict, engine: Dict[str, Any]) -> None:
    """Imprime banner inicial de modo/variant/config en consola.

    Casos contemplados para MODE:
    - "TEST" o "TEST+WARMUP" cuando se ejecuta en modo ensamblado.
    - "WARMUP+TRAIN" cuando hay warmup previo y luego entrenamiento real.
    - "TRAIN" cuando solo hay entrenamiento sin warmup explícito.
    """
    ut = engine["utils"]
    device_info = ut.device_info()

    has_warmup_cfg = str(cfg.warmup) != "off"
    has_warmup_epochs = int(cfg.get("warmup_epochs", 0)) > 0
    has_warmup = has_warmup_cfg or has_warmup_epochs

    if cfg.test:
        mode = "TEST+WARMUP" if has_warmup else "TEST"
    else:
        mode = "WARMUP+TRAIN" if has_warmup else "TRAIN"

    warm_ep = f" (epochs={cfg.warmup_epochs})" if has_warmup_epochs else ""
    print(
        "[YOLOv11] "
        f" MODE={mode}  VARIANT={cfg.variant}  BN2GN={cfg.bn2gn} \n"
        f"├─ AMP={cfg.amp}  EMA={'ON' if cfg.ema else 'OFF'}  HUD={'ON' if cfg.hud else 'OFF'} \n"
        f"├─ Device: {device_info} \n"
        f"├─ Batch={cfg.batch}  Imgsize={cfg.imgsz}  Epochs={cfg.epochs} \n"
        f"├─ Project: ...{cfg.project[26:]} \n"
        f"├─ SaveDir: ...{cfg.save_dir[109:]} \n"
        f"└─ Warmup: {cfg.warmup}{warm_ep}  ValInt every {cfg.val_int_interval} epochs"
    )


def _fitness(metrics: Dict[str, float]) -> float:
    """Criterio de fitness clásico: 0.1*mAP50 + 0.9*mAP50-95.

    Tolera ausencia de claves, usando 0.0 por defecto.
    """
    m50 = float(metrics.get("map50", 0.0))
    m5095 = float(metrics.get("map", metrics.get("map50-95", 0.0)))
    return 0.1 * m50 + 0.9 * m5095


# -------------------------------
# Trainer
# -------------------------------


class Trainer:
    """Clase principal de entrenamiento para YOLOv11.

    Parámetros
    -----------
    model : torch.nn.Module
        Modelo ya construido (core envuelto si aplica).
    train_loader : iterable
        DataLoader de entrenamiento (formato adaptado por utility.data_loader).
    names : list or dict
        Nombres de clases; se normaliza internamente a lista para el validador.
    cfg : DotDict
        Configuración normalizada desde CLI/archivos YAML.
    engine : dict
        Mapa de submódulos del engine (amp, optim, ema, callbacks, validator,
        warmup, utils, bn2gn_patch, hud, torch, WeightsManager, util_data,
        YOLOLoss, ExperimentLogger).
    """

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
            # 'auto' ya fue resuelto antes con utils.auto_amp_mode(); por compatibilidad, asumimos fp16
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
        self.cb = engine["callbacks"].build_default_callbacks(
            self.save_dir, cfg=DotDict(val_int_interval=cfg.val_int_interval)
        )

        # HUD
        self.hud = engine["hud"].HUD(engine["hud"].HUDConfig(enable=cfg.hud)) if cfg.hud else None

        # Weights Manager (reemplazo de CheckpointManager)
        self.wm = engine["WeightsManager"](
            project_root=Path(__file__).resolve().parents[1],  # YOLOv11/
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
                info = self.wm.try_resume(
                    self.model, optimizer=self.optimizer, scheduler=self.scheduler, prefer=prefer
                )
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
                    info = {
                        "resumed": True,
                        "start_epoch": int(ckpt.get("epoch", 0)) + 1,
                        "ckpt_path": Path(prefer),
                    }
                except Exception:
                    info = {"resumed": False, "start_epoch": 0, "ckpt_path": None}
            self.start_epoch = int((info or {}).get("start_epoch", 0))

        # Límite de tiempo (0 = ilimitado)
        self.timer = ut.timed_stop(cfg.time_limit)

        # Adaptador de loader a dict en dispositivo (delegado al módulo utility.data_loader).
        # IMPORTANTE: no guardamos aquí el generador adaptado para evitar agotarlo;
        # se recrea en cada uso mediante `_iter_train_loader()`.

    # -----------------------------
    # Helpers internos
    # -----------------------------

    def _iter_train_loader(self):
        """Devuelve un generador fresco adaptado (dict en device) para cada uso.

        Esto evita que el generador se agote tras la primera época y garantiza
        que tanto el loop de entrenamiento como la validación interna (val_int)
        vean batches consistentes.
        """

        util_data = self.engine["util_data"]
        return util_data.as_dict_loader(self.train_loader, self.device)

    # -----------------------------
    # Loop principal de entrenamiento
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
            # Cierre explícito de HUD en modo --test para liberar hilos/recursos
            if self.hud and hasattr(self.hud, "close"):
                try:
                    self.hud.close()
                except Exception:
                    pass
            try:
                self.logger.close()
            except Exception:
                pass
            return

        engine = self.engine
        ut = engine["utils"]

        self.cb.on_train_start(trainer=self)
        iters_per_epoch = len(self.train_loader)
        print("[TRAIN] >>> Inicio entrenamiento (~2-5 min)", flush=True)

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
            # Generador fresco por época para evitar agotamiento
            train_loader_adapt = self._iter_train_loader()
            for i, batch in enumerate(train_loader_adapt, start=1):
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
                    loss,
                    self.optimizer,
                    self.ampmgr,
                    clip_fn=lambda: self.engine["optim"].clip_gradients(
                        self.model, self.cfg.clip_norm, self.cfg.clip_mode
                    ),
                    zero_grad=do_step,
                    set_to_none=True,
                )
                if do_step:
                    # Orden lógico: backward/step (en safe_backward_step) -> scheduler -> EMA
                    self.scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)

                dt_ms = (time.perf_counter() - t_iter) * 1000.0
                if self.hud:
                    lr = float(self.optimizer.param_groups[0]["lr"])
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
                # Preparar modelo de evaluación según política EMA
                if self.ema is not None:
                    try:
                        model_eval = copy.deepcopy(self.model)
                        model_eval.to(self.device)
                        model_eval.eval()
                        self.ema.copy_to(model_eval)
                    except Exception as e:
                        print(
                            f"[EMA] Advertencia: fallo al preparar modelo EMA para validación interna: {e}"
                        )
                        model_eval = self.model
                else:
                    model_eval = self.model

                # Para val_int usamos siempre un adaptador fresco del train_loader; el tamaño efectivo
                # lo controla `max_batches` cuando se activa `val_int_use_train_subset`.
                max_batches = (
                    int(self.cfg.val_int_max_batches)
                    if self.cfg.val_int_use_train_subset
                    else 0
                )

                val_loader_adapt = self._iter_train_loader()

                try:
                    val_metrics = self.engine["validator"].validate_interna(
                        model_eval,
                        loader=val_loader_adapt,
                        names=names_list,
                        save_dir=str(self.save_dir),
                        conf_thres=float(self.cfg.val_int_conf),
                        iou_thres=0.60,
                        device=str(self.device),
                        # internos
                        epoch=int(epoch),
                        max_batches=max_batches,
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
            self.cb.on_model_save(self, path, is_best=is_best)

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
                # Preparado para métricas explícitas en el futuro
                self.hud.update_epoch(epoch, train_metrics=train_metrics, val_metrics=val_metrics or None)

            if ut.SIGNALS.stop or self.timer.expired():
                break

        print("[TRAIN] <<< Fin entrenamiento (~2-5 min)", flush=True)
        if self.hud and hasattr(self.hud, "close"):
            try:
                self.hud.close()
            except Exception:
                pass
        try:
            self.logger.close()
        except Exception:
            pass

    # -----------------------------
    # Warmup sintético con HUD (delegado a engine.warmup)
    # -----------------------------

    def _run_warmup_hud(self, loops: int = 1) -> None:
        """Ejecuta warm-up sintético delegando el trabajo a `engine.warmup`.

        - Usa `engine.warmup.build_warmup_config_from_train` para derivar la
          configuración desde `self.cfg`.
        - Ejecuta `warmup_sanity` una vez y reconstruye la barra/estadísticas
          en el HUD a partir de los tiempos medidos.

        Nota: el parámetro `loops` se interpreta como número de "épocas
        virtuales" para el encabezado, pero el trabajo pesado lo realiza
        una sola invocación de warmup (suficiente para inicializar kernels
        MIOpen/HIP y validar el forward).
        """

        if self.hud is None:
            return

        warm_mod = self.engine["warmup"]
        core = getattr(self.model, "core", self.model)

        # Configuración de iteraciones según política CLI de warmup
        amp_mode = str(self.cfg.amp).lower()
        if amp_mode == "bf16":
            dtype = "bf16"
        elif amp_mode == "fp16":
            dtype = "fp16"
        else:
            dtype = "fp32"

        # Iteraciones por warmup (sanity/fast/full) – coherente con versiones previas
        if self.cfg.warmup == "sanity":
            base_iters = 2
        elif self.cfg.warmup == "fast":
            base_iters = 5
        elif self.cfg.warmup == "full":
            base_iters = 10
        else:
            base_iters = 2

        total_iters = max(2, base_iters)

        warm_cfg = warm_mod.build_warmup_config_from_train(
            self.cfg,
            device=str(self.device),
            iters=total_iters,
            verbose=1,
        )

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
        print("[WARMUP] >>> Inicio warmup (~2-5 min)", flush=True)
        self.hud.on_warmup_start(
            total_iters=total_iters,
            dtype=dtype,
            compile=bool(self.cfg.compile),
            stride=int(warm_cfg.stride),
            bn2gn=str(self.cfg.bn2gn),
            amp=(amp_mode != "off"),
            find_mode=find_mode,
            cache_disabled=cache_disabled,
        )

        # Ejecutar warmup real (sin HUD interno) y luego reconstruir barra con los tiempos
        summary = warm_mod.warmup_sanity(core, device=str(self.device), cfg=warm_cfg)
        times = summary.get("timings_ms", {}).get("all", [])
        if not times:
            times = [0.0] * total_iters

        # Simular progreso de HUD usando los tiempos medidos
        for i, dt_ms in enumerate(times, start=1):
            self.hud.update_warmup(i, total_iters, float(dt_ms))

        # Resumen y cierre de warmup
        self.hud.on_warmup_end()
        print("[WARMUP] <<< Fin warmup (~2-5 min)", flush=True)

    # -----------------------------
    # Prueba de ensamblado (modo --test)
    # -----------------------------

    def _assembly_test(self) -> None:
        # Ejecuta warm-up con HUD si se solicitó por modo o por `warmup_epochs`
        if self.cfg.warmup in ("sanity", "fast", "full") or int(self.cfg.get("warmup_epochs", 0)) > 0:
            self._run_warmup_hud(loops=int(max(1, int(self.cfg.get("warmup_epochs", 0)))))

        # sanity: un minibatch real con backward/step
        self.model.train()
        # Usar siempre un adaptador fresco del train_loader
        for i, batch in enumerate(self._iter_train_loader(), start=1):
            with self.ampmgr.autocast():
                core = getattr(self.model, "core", self.model)
                preds = core(batch["img"])  # forward
                loss, _ = self.criterion(preds, batch["targets"])  # pérdida
            self.engine["amp"].safe_backward_step(
                loss,
                self.optimizer,
                self.ampmgr,
                clip_fn=lambda: self.engine["optim"].clip_gradients(
                    self.model, self.cfg.clip_norm, self.cfg.clip_mode
                ),
                zero_grad=True,
                set_to_none=True,
            )
            if self.ema:
                self.ema.update(self.model)
            break  # solo 1 minibatch

    # -----------------------------
    # Utilitario: banner de modo
    # -----------------------------

    def _print_mode_banner(self) -> None:
        _print_banner(self.cfg, self.engine)
