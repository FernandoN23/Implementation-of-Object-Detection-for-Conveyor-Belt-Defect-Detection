# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/callbacks.py
# Descripción: Sistema de callbacks estilo Ultralytics para enganchar
#              eventos del ciclo de entrenamiento/validación sin
#              contaminar el bucle principal. Incluye un gestor
#              (CallbackManager), interfaz base y callbacks ejemplo
#              (TensorBoard/Overlays/Checkpoints). CSV eliminado.
#==============================================================

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

__all__ = [
    "Callback",
    "CallbackManager",
    "CallbackConfig",
    "TensorBoardCallback",
    "OverlaysCallback",
    "CheckpointCallback",
]


# -------------------------------
# Interfaz base
# -------------------------------

class Callback(Protocol):
    """Interfaz de callback: cada método es opcional.

    Los nombres de métodos corresponden a eventos del ciclo de entrenamiento.
    """

    # Entrenamiento
    def on_train_start(self, trainer: Any) -> None: ...
    def on_train_epoch_start(self, trainer: Any, epoch: int) -> None: ...
    def on_train_batch_start(self, trainer: Any, step: int, batch: Any) -> None: ...
    def on_train_batch_end(self, trainer: Any, step: int, loss: float, items: Dict[str, float]) -> None: ...
    def on_train_epoch_end(self, trainer: Any, epoch: int, train_stats: Dict[str, float]) -> None: ...
    def on_train_end(self, trainer: Any) -> None: ...  # ← cierre ordenado

    # Validación
    def on_val_start(self, trainer: Any, epoch: int) -> None: ...
    def on_val_end(self, trainer: Any, epoch: int, val_stats: Dict[str, float]) -> None: ...

    # Fit (época completa)
    def on_fit_epoch_end(self, trainer: Any, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None: ...

    # Guardado de modelo
    def on_model_save(self, trainer: Any, path: str, is_best: bool) -> None: ...

    # Excepciones
    def on_exception(self, trainer: Any, exc: BaseException) -> None: ...


# -------------------------------
# Gestor de callbacks
# -------------------------------

@dataclass
class CallbackConfig:
    enable_tb: bool = True
    enable_overlays: bool = True
    enable_ckpt: bool = True
    overlays_interval: int = 10
    overlays_samples: int = 8
    overlays_tb: bool = True                 # enviar overlays a TensorBoard
    overlays_train_fallback: bool = True     # usar train_loader si no hay val_loader


class CallbackManager:
    """Orquesta y despacha eventos a la lista de callbacks registrados."""

    def __init__(self, save_dir: str, cfg: Optional[CallbackConfig] = None) -> None:
        self.save_dir = Path(save_dir)
        self.cfg = cfg or CallbackConfig()
        self.callbacks: List[Callback] = []

    # Registro
    def add(self, cb: Callback) -> None:
        self.callbacks.append(cb)

    def extend(self, cbs: Iterable[Callback]) -> None:
        for cb in cbs:
            self.add(cb)

    def load_from_paths(self, import_paths: List[str]) -> None:
        """Carga callbacks por rutas de import (p. ej., "pkg.mod:Class")."""
        for spec in import_paths:
            mod_name, _, cls_name = spec.partition(":")
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            self.add(cls())

    # Despacho seguro
    def _safe_call(self, name: str, *args, **kwargs) -> None:
        for cb in list(self.callbacks):
            fn = getattr(cb, name, None)
            if fn is None:
                continue
            try:
                fn(*args, **kwargs)
            except Exception as e:  # no romper entrenamiento por callback
                print(f"[callbacks] Excepción en {cb.__class__.__name__}.{name}: {e}")

    # Eventos (API pública)
    def on_train_start(self, trainer: Any) -> None:
        self._safe_call("on_train_start", trainer)

    def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._safe_call("on_train_epoch_start", trainer, epoch)

    def on_train_batch_start(self, trainer: Any, step: int, batch: Any) -> None:
        self._safe_call("on_train_batch_start", trainer, step, batch)

    def on_train_batch_end(self, trainer: Any, step: int, loss: float, items: Dict[str, float]) -> None:
        self._safe_call("on_train_batch_end", trainer, step, loss, items)

    def on_train_epoch_end(self, trainer: Any, epoch: int, train_stats: Dict[str, float]) -> None:
        self._safe_call("on_train_epoch_end", trainer, epoch, train_stats)

    def on_train_end(self, trainer: Any) -> None:
        self._safe_call("on_train_end", trainer)

    def on_val_start(self, trainer: Any, epoch: int) -> None:
        self._safe_call("on_val_start", trainer, epoch)

    def on_val_end(self, trainer: Any, epoch: int, val_stats: Dict[str, float]) -> None:
        self._safe_call("on_val_end", trainer, epoch, val_stats)

    def on_fit_epoch_end(self, trainer: Any, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None:
        self._safe_call("on_fit_epoch_end", trainer, epoch, train_stats, val_stats)

    def on_model_save(self, trainer: Any, path: str, is_best: bool) -> None:
        self._safe_call("on_model_save", trainer, path, is_best)

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        self._safe_call("on_exception", trainer, exc)


# -------------------------------
# Callbacks incluidos
# -------------------------------

class TensorBoardCallback:
    """Emite scalars a TensorBoard si tensorboard está disponible y habilitado."""

    def __init__(self, save_dir: str, enabled: bool = True) -> None:
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tb = None

    def on_train_start(self, trainer: Any) -> None:
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.tb = SummaryWriter(log_dir=str(self.save_dir / "tb"))
        except Exception as e:
            print(f"[callbacks] TensorBoard no disponible: {e}")
            self.enabled = False

    def on_fit_epoch_end(self, trainer: Any, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None:
        if not self.enabled or self.tb is None:
            return
        for k, v in (train_stats or {}).items():
            self.tb.add_scalar(f"train/{k}", float(v), epoch)
        for k, v in (val_stats or {}).items():
            self.tb.add_scalar(f"val/{k}", float(v), epoch)
        self.tb.flush()

    def on_train_end(self, trainer: Any) -> None:
        if self.tb:
            try:
                self.tb.flush(); self.tb.close()
            except Exception:
                pass


class OverlaysCallback:
    """Genera overlays pred–GT cada N épocas usando engine.overlays.

    Si no existe `trainer.val_loader`, usa `trainer.train_loader` (fallback)
    para una validación interna con el mismo set de entrenamiento.
    Opcionalmente publica las imágenes en TensorBoard.
    """

    def __init__(self, save_dir: str, interval: int = 10, samples: int = 8, to_tensorboard: bool = True) -> None:
        from .overlays import OverlaysManager, OverlayConfig  # import local
        self.save_dir = Path(save_dir) / "overlays"
        self.mgr = OverlaysManager(OverlayConfig(save_dir=str(self.save_dir), interval=interval, num_samples=samples))
        self.to_tb = to_tensorboard
        self.tb = None  # SummaryWriter opcional

    def on_train_start(self, trainer: Any) -> None:
        if not self.to_tb:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            tb_dir = Path(getattr(trainer, "save_dir", self.save_dir.parent)) / "tb"
            self.tb = SummaryWriter(log_dir=str(tb_dir))
        except Exception as e:
            print(f"[callbacks] TensorBoard (overlays) no disponible: {e}")
            self.to_tb = False

    def on_fit_epoch_end(self, trainer: Any, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None:
        # Seleccionar loader: validación real o fallback a train
        loader = getattr(trainer, "val_loader", None)
        if loader is None:
            loader = getattr(trainer, "train_loader", None)
        if loader is None:
            return

        names = getattr(trainer, "names", None)
        try:
            out = self.mgr.run(epoch, getattr(trainer, "model_eval", trainer.model), loader, names=names)
            if out:
                print(f"[callbacks] overlays → {out}")
                if self.to_tb and self.tb is not None:
                    # Cargar y publicar algunas imágenes en TB (mejor esfuerzo)
                    try:
                        from PIL import Image  # type: ignore
                        import numpy as np
                        paths = out if isinstance(out, (list, tuple)) else [out]
                        for i, p in enumerate(paths):
                            if i >= 4:  # publicar pocas por época
                                break
                            img = Image.open(p).convert("RGB")
                            arr = np.asarray(img)
                            # HWC→CHW
                            chw = arr.transpose(2, 0, 1)
                            self.tb.add_image(f"overlays/epoch_{epoch}/sample_{i}", chw, epoch)
                        self.tb.flush()
                    except Exception as e:
                        print(f"[callbacks] overlays→TB error: {e}")
        except Exception as e:
            print(f"[callbacks] overlays error: {e}")

    def on_train_end(self, trainer: Any) -> None:
        if self.tb:
            try:
                self.tb.flush(); self.tb.close()
            except Exception:
                pass


class CheckpointCallback:
    """Notificación simple al guardar checkpoints (best/last) para integraciones externas."""

    def __init__(self) -> None:
        pass

    def on_model_save(self, trainer: Any, path: str, is_best: bool) -> None:
        tag = "best" if is_best else "last"
        print(f"[callbacks] checkpoint saved ({tag}): {path}")


# -------------------------------
# Helper de ensamblado por defecto
# -------------------------------

def build_default_callbacks(save_dir: str, cfg: Optional[CallbackConfig] = None) -> CallbackManager:
    cfg = cfg or CallbackConfig()
    mgr = CallbackManager(save_dir, cfg)

    if cfg.enable_tb:
        mgr.add(TensorBoardCallback(save_dir, enabled=True))
    if cfg.enable_overlays:
        mgr.add(OverlaysCallback(save_dir, interval=cfg.overlays_interval, samples=cfg.overlays_samples, to_tensorboard=cfg.overlays_tb))
    if cfg.enable_ckpt:
        mgr.add(CheckpointCallback())

    return mgr
