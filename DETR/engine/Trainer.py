# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Trainer.py
# Descripción: Orquestador de entrenamiento con auto-descarga de
#              pesos oficiales, parches ROCm y mitigación de ruido.
# ==============================================================

import os
import sys
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn

# --- INTEGRACIÓN DE SUBMÓDULO DETR ---
FILE = Path(__file__).resolve()
ENGINE_ROOT = FILE.parent
DETR_ROOT = ENGINE_ROOT.parent
DETR_SUBMODULE = DETR_ROOT / "detr"

if str(DETR_SUBMODULE) not in sys.path:
    sys.path.append(str(DETR_SUBMODULE))

try:
    from models import build_model
    from util.misc import save_on_master
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig
    from utility.data_loader import build_dataloader
    from engine.Validator import Validator
    from engine.bootstrap_miopen import MuteStderr  # [NUEVO]
except ImportError as e:
    print(f"[Trainer] ERROR: Fallo al importar componentes esenciales: {e}")
    sys.exit(1)

# Mapeo de pesos oficiales de Facebook Research (Transformer + Heads)
DETR_URLS = {
    "r50": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
    "r50_dc5": "https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth",
    "r101": "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",
    "r101_dc5": "https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth"
}


@dataclass
class TrainerConfig:
    variant: str = "r50"
    run_name: str = "exp"
    phase: str = "train"
    epochs: int = 300
    batch_size: int = 2
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    lr_drop: int = 200
    clip_max_norm: float = 0.1
    device: str = "cuda"
    pretrain_weights: str = ""  # Ruta local en weights/base/
    nc: int = 5
    model_args: Any = None
    bn2gn_policy: str = "on"


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)

        # Desactivar benchmark para evitar spam de MIOpen con tamaños variables
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Rutas canónicas
        self.save_dir = DETR_ROOT / "runs" / self.cfg.variant / self.cfg.phase / self.cfg.run_name
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # 1. Preparar pesos y modelo
        self.model, self.criterion, self.postprocessors = self._setup_model()

        # 2. Preparar optimización
        self.optimizer, self.lr_scheduler = self._setup_optimizer()

        # 3. Preparar validador
        self.validator = Validator(self.model, self.criterion, self.postprocessors, self.device)
        self.best_map = 0.0

    def _maybe_download_weights(self):
        """Descarga automática de pesos si no existen localmente (estilo YOLO)."""
        w_path = Path(self.cfg.pretrain_weights)
        if not w_path.exists():
            variant = self.cfg.variant
            if variant in DETR_URLS:
                print(f"[Trainer] Pesos base no encontrados en: {w_path}")
                print(f"[Trainer] Descargando pesos oficiales de Facebook para variante '{variant}'...")
                w_path.parent.mkdir(parents=True, exist_ok=True)

                # Descarga vía torch.hub
                torch.hub.download_url_to_file(DETR_URLS[variant], str(w_path))
                print(f"✓ Descarga completada: {w_path.name}")
            else:
                print(f"[Trainer] AVISO: No hay URL de descarga definida para '{variant}'.")

    def _setup_model(self):
        """Instancia, descarga, carga y adapta el modelo."""
        self._maybe_download_weights()

        model, criterion, postprocessors = build_model(self.cfg.model_args)

        # Carga de pesos base (COCO)
        w_path = Path(self.cfg.pretrain_weights)
        if w_path.exists():
            print(f"[Trainer] Cargando pesos pre-entrenados: {w_path}")
            checkpoint = torch.load(w_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)

        # Adaptación dinámica: 91 clases (COCO) -> N clases (Proyecto)
        hidden_dim = model.transformer.d_model
        print(f"[Trainer] Adaptando cabezal para {self.cfg.nc} clases + fondo.")
        model.class_embed = nn.Linear(hidden_dim, self.cfg.nc + 1)

        # Adaptación del criterio de pérdida
        criterion.num_classes = self.cfg.nc
        empty_weight = torch.ones(self.cfg.nc + 1)
        empty_weight[-1] = self.cfg.model_args.eos_coef
        criterion.register_buffer('empty_weight', empty_weight)

        # Parche BN2GN para estabilidad en ROCm
        if self.cfg.bn2gn_policy != "off":
            replace_bn_with_gn(model, BN2GNConfig(policy=self.cfg.bn2gn_policy))

        model.to(self.device)
        criterion.to(self.device)
        return model, criterion, postprocessors

    def _setup_optimizer(self):
        """Configuración AdamW con LR reducido para el Backbone."""
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.cfg.lr_backbone}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.cfg.lr_drop)
        return optimizer, scheduler

    def fit(self):
        """Ciclo principal de entrenamiento y validación."""
        print(f"\n--- Iniciando Entrenamiento DETR: {self.cfg.run_name} ---")

        train_loader = build_dataloader("train", self.cfg.batch_size)
        val_loader = build_dataloader("valid", self.cfg.batch_size)

        start_time = time.time()
        for epoch in range(self.cfg.epochs):
            # 1. Entrenamiento
            train_stats = self._train_one_epoch(train_loader, epoch)
            self.lr_scheduler.step()

            # 2. Validación
            val_stats = self.validator.validate(val_loader, self.save_dir)

            # 3. Logs JSON Lines (para utility/metrics.py)
            log_stats = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_loss_ce": train_stats["loss_ce"],
                "train_loss_bbox": train_stats["loss_bbox"],
                "train_loss_giou": train_stats["loss_giou"],
                "train_class_error": train_stats["class_error"],
                **{f"test_{k}": v for k, v in val_stats.items()}
            }
            with open(self.save_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 4. Checkpoints (last y best)
            self._save_checkpoints(epoch, val_stats)

        print(f"\n[Trainer] Finalizado en {(time.time() - start_time) / 60:.2f} min.")

    def _train_one_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()
        stats = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "class_error": 0.0}

        # [NUEVO] Silenciar ruido de C++ durante el bucle de batches
        with MuteStderr():
            for i, (samples, targets) in enumerate(loader):
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(samples)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                if not math.isfinite(losses.item()):
                    # Si hay error crítico, el print saldrá fuera del context manager
                    # o podemos imprimirlo forzadamente.
                    continue

                self.optimizer.zero_grad()
                losses.backward()

                if self.cfg.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)

                self.optimizer.step()

                stats["loss"] += losses.item()
                stats["loss_ce"] += loss_dict["loss_ce"].item()
                stats["loss_bbox"] += loss_dict["loss_bbox"].item()
                stats["loss_giou"] += loss_dict["loss_giou"].item()

                if "class_error" in loss_dict:
                    stats["class_error"] += loss_dict["class_error"].item()

        # Imprimir progreso fuera del MuteStderr para que sea visible
        print(f"Epoch [{epoch}] completada. Loss promedio: {stats['loss'] / len(loader):.4f}")

        num_batches = len(loader)
        return {k: v / num_batches for k, v in stats.items()}

    def _save_checkpoints(self, epoch, val_stats):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'cfg': self.cfg
        }
        save_on_master(checkpoint, self.weights_dir / "last.pt")

        current_map = val_stats.get("coco_eval_bbox", [0, 0])[1]
        if current_map > self.best_map:
            self.best_map = current_map
            save_on_master(checkpoint, self.weights_dir / "best.pt")
            print(f"  --> Nuevo Mejor mAP@0.5: {current_map:.4f} (Guardado en best.pt)")