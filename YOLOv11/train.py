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
import json
import argparse
from datetime import datetime
from typing import Any, Dict, Tuple, Iterable, Iterator, List
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
    print(
        "\n[YOLOv11] "
        f"MODE={mode}  VARIANT={cfg.variant}  BN2GN={cfg.bn2gn}  "
        f"AMP={cfg.amp}  EMA={'ON' if cfg.ema else 'OFF'}  HUD={'ON' if cfg.hud else 'OFF'}\n"
        f"Device={device_info}  Batch={cfg.batch}  ImgSz={cfg.imgsz}  Epochs={cfg.epochs}\n"
        f"Project={cfg.project}  SaveDir={cfg.save_dir}\n"
        f"Warmup={cfg.warmup}  ValInt every {cfg.val_int_interval} epochs\n"
    )


class _DictLoaderAdapter:
    """Adaptador minimal para que entrenamiento reciba batches en dict.
    Envuelve un DataLoader que entrega (imgs, targets, meta) y produce
    {"img": imgs, "targets": targets, "meta": meta}.
    """
    def __init__(self, base_loader: Iterable, device) -> None:
        self.base = base_loader
        self._len = None
        try:
            self._len = len(base_loader)  # tipo DataLoader tiene __len__
        except Exception:
            self._len = None
        self.device = device

    def __len__(self) -> int:
        if self._len is None:
            raise TypeError("Base loader no expone __len__")
        return int(self._len)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for b in self.base:
            if isinstance(b, tuple) and len(b) == 3:
                imgs, targets, meta = b
                yield {"img": imgs.to(self.device), "targets": targets.to(self.device), "meta": meta}
            elif isinstance(b, dict):
                d = dict(b)
                if "img" in d and hasattr(d["img"], "to"):
                    d["img"] = d["img"].to(self.device)
                if "targets" in d and hasattr(d["targets"], "to"):
                    d["targets"] = d["targets"].to(self.device)
                yield d
            else:
                yield b


def _build_model_and_data(cfg: DotDict, engine: Dict[str, Any]):
    """Construye modelo y dataloader de entrenamiento usando utilidades del proyecto.

    Cambio solicitado:
      - Adoptar `utility/data_loader.py` para los datos reales de *train*.
      - Mantener `engine.utils.build_model` para el ensamblado del modelo.
      - **Excluir** la partición de validación (no se crea `val_loader`).
    """
    ut = engine["utils"]

    # 1) Modelo (igual que antes)
    model = ut.build_model(cfg.model, variant=cfg.variant)

    # 2) DataLoader de entrenamiento desde utility/data_loader.py
    project_root = Path(__file__).resolve().parent  # raíz que contiene configs/ y models/
    udata = engine["util_data"]

    train_loader = udata.build_yolo_dataloader(
        split="train", batch=cfg.batch, imgsz=cfg.imgsz, workers=cfg.workers, project_root=project_root
    )

    # 3) Nombres/clases desde dataset.yaml
    ds_yaml = udata.load_dataset_yaml(project_root)
    names = ds_yaml.get("names", {})
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}

    # 4) (Opcional) Mostrar info del dataset si se activa el flag (solo train)
    if cfg.get("dl_info", False):
        try:
            nc = len(names) if hasattr(names, "__len__") else int(names)
        except Exception:
            nc = 0
        try:
            n_train = len(getattr(train_loader, "dataset", []))
        except Exception:
            n_train = "?"
        train_path = str(Path(ds_yaml.get("train", "")).resolve())
        print(f"Dataset cargado / partición: train / N° imágenes = {n_train} / nc = {nc} / ruta: {train_path}")

    return model, train_loader, names


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
        self.save_dir = ut.setup_save_dir(cfg.project, cfg.name, exist_ok=cfg.exist_ok)
        self.cfg.save_dir = self.save_dir  # para banner

        # === Logger de experimento (utility/logger.py) ===
        self.logger = engine["ExperimentLogger"](
            variant=cfg.variant,
            phase="train",
            is_test=cfg.test,
            run_name=cfg.name,
            reset_final=not cfg.exist_ok,
        )
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

        # Adaptador de loader a dict en dispositivo
        self.train_loader_adapt = _DictLoaderAdapter(self.train_loader, self.device)

    # -----------------------------
    def fit(self) -> None:
        self._print_mode_banner()

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
                    val_metrics = engine["validator"].validate_interna(
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
    def _assembly_test(self) -> None:
        engine = self.engine
        ut = engine["utils"]

        # Warm-up robusto con configuración completa
        if self.cfg.warmup in ("sanity", "fast", "full"):
            # Inferir stride desde el modelo o usar fallback 32
            torch_mod = engine["torch"]
            core = getattr(self.model, "core", self.model)
            stride = None
            # Buscar atributos típicos de stride
            for obj in (core, getattr(core, "model", None)):
                if obj is None:
                    continue
                for attr in ("stride", "strides", "max_stride"):
                    if hasattr(obj, attr):
                        stride = getattr(obj, attr)
                        break
                if stride is not None:
                    break
            # Normalizar a entero
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

            # Mapear dtype según AMP
            amp_mode = str(self.cfg.amp).lower()
            if amp_mode == "bf16":
                dtype = "bf16"
            elif amp_mode == "fp16":
                dtype = "fp16"
            else:
                dtype = "fp32"

            # Iteraciones por modo
            iters = 2 if self.cfg.warmup == "sanity" else (5 if self.cfg.warmup == "fast" else 10)

            WarmupConfig = engine["warmup"].WarmupConfig
            warm_cfg = WarmupConfig(
                imgsz=int(self.cfg.imgsz),
                bs=int(self.cfg.batch),
                nc=(len(self.names) if hasattr(self.names, "__len__") else int(self.names)),
                amp=(amp_mode != "off"),
                device=str(self.device),
                channels=3,
                stride=int(stride),
                iters=int(iters),
                compile=bool(self.cfg.compile),
                dtype=dtype,
                verbose=1,
            )
            engine["warmup"].warmup_sanity(self.model, device=self.device, cfg=warm_cfg)

        # sanity: un minibatch real con backward/step
        self.model.train()
        for i, batch in enumerate(self.train_loader_adapt, start=1):
            with self.ampmgr.autocast():
                core = getattr(self.model, "core", self.model)
                preds = core(batch["img"])  # forward
                loss, _ = self.criterion(preds, batch["targets"])  # pérdida
            engine["amp"].safe_backward_step(
                loss, self.optimizer, self.ampmgr,
                clip_fn=lambda: engine["optim"].clip_gradients(self.model, self.cfg.clip_norm, self.cfg.clip_mode),
                zero_grad=True, set_to_none=True,
            )
            if self.ema:
                self.ema.update(self.model)
            break  # solo 1 minibatch

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

    # HUD / resume / tiempo / outputs
    p.add_argument('--hud', action='store_true', default=None)
    p.add_argument('--no-hud', dest='hud', action='store_false')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--project', type=str, default='runs/train')
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--exist-ok', action='store_true', default=False)
    p.add_argument('--time-limit', type=float, default=0.0)

    # info de data_loader (flag solicitado)
    p.add_argument('--dl-info', action='store_true', help='Muestra info del dataset por partición al cargar')

    # === Validación interna (val_int) ===
    p.add_argument('--val-int-interval', type=int, default=5, help='Intervalo de épocas para validación interna')
    p.add_argument('--val-int-max-batches', type=int, default=1, help='Nº de batches (subset) para métrica rápida')
    p.add_argument('--val-int-use-train-subset', action='store_true', default=False,
                   help='Usa subset del train_loader para métricas internas (si no, sólo pivotes/TB)')
    p.add_argument('--val-int-conf', type=float, default=0.25, help='Confianza mínima para val_int/TB')
    p.add_argument('--val-int-split', type=str, default='val', choices=['train','val'],
                   help='Split de pivotes para TB (visualización)')
    p.add_argument('--val-int-pivots', action='store_true', default=True,
                   help='Dibujar también imágenes pivote en TB (GT o GT+Pred)')
    p.add_argument('--no-val-int-pivots', dest='val_int_pivots', action='store_false')

    # TensorBoard para val_int
    p.add_argument('--val-int-tb', action='store_true', default=True, help='Habilita TB en validación interna')
    p.add_argument('--no-val-int-tb', dest='val_int_tb', action='store_false')
    p.add_argument('--val-int-tb-nrow', type=int, default=3)
    p.add_argument('--val-int-tb-conf', type=float, default=0.25)
    p.add_argument('--val-int-tb-topk', type=int, default=5)
    p.add_argument('--dataset-base', type=str, default=None, help='Ruta base del dataset (para pivotes TB)')

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
        hud=bool(cli.hud),
        resume=cli.resume,
        project=cli.project,
        name=name,
        exist_ok=cli.exist_ok,
        time_limit=cli.time_limit,
        world_size=int(os.environ.get('WORLD_SIZE', '1')),
        test=bool(cli.test),
        dl_info=bool(cli.dl_info),
        # val_int
        val_int_interval=int(cli.val_int_interval),
        val_int_max_batches=int(cli.val_int_max_batches),
        val_int_use_train_subset=bool(cli.val_int_use_train_subset),
        val_int_conf=float(cli.val_int_conf),
        val_int_split=str(cli.val_int_split),
        val_int_pivots=bool(cli.val_int_pivots),
        val_int_tb=bool(cli.val_int_tb),
        val_int_tb_nrow=int(cli.val_int_tb_nrow),
        val_int_tb_conf=float(cli.val_int_tb_conf),
        val_int_tb_topk=int(cli.val_int_tb_topk),
        dataset_base=cli.dataset_base,
    )

    # Construcción de modelo y dataloader (sin validación externa)
    model, train_loader, names = _build_model_and_data(cfg, engine)

    # Instanciar trainer
    trainer = Trainer(model, train_loader, names, cfg, engine)

    # Ejecutar (el banner se imprime dentro de Trainer.fit())
    trainer.fit()


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
