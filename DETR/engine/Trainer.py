# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Trainer.py
# Descripción: Motor de entrenamiento para DETR. Gestiona la
#              optimización AdamW, el ciclo de épocas y la
#              generación de logs compatibles con metrics.py.
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
from torch.utils.data import DataLoader

# --- CONFIGURACIÓN DE RUTAS ---
FILE = Path(__file__).resolve()
ENGINE_ROOT = FILE.parent
DETR_ROOT = ENGINE_ROOT.parent
DETR_SUBMODULE = DETR_ROOT / "detr"

if str(DETR_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(DETR_SUBMODULE))

try:
    from models import build_model
    from util.misc import save_on_master, reduce_dict
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig
    from utility.data_loader import build_dataloader
    from engine.Validator import Validator
except ImportError as e:
    print(f"[Trainer] ERROR: Fallo en imports de dependencias: {e}")
    sys.exit(1)


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
    pretrain_weights: str = ""
    nc: int = 5
    model_args: Any = None
    bn2gn_policy: str = "on"


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)

        # Rutas de salida
        self.save_dir = DETR_ROOT / "runs" / self.cfg.variant / self.cfg.phase / self.cfg.run_name
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar componentes
        self.model, self.criterion, self.postprocessors = self._setup_model()
        self.optimizer, self.lr_scheduler = self._setup_optimizer()

        # Inicializar Validador
        self.validator = Validator(self.model, self.criterion, self.postprocessors, self.device)

        # Mejor mAP registrado
        self.best_map = 0.0

    def _setup_model(self):
        """Instancia, parcha y adapta el modelo DETR."""
        model, criterion, postprocessors = build_model(self.cfg.model_args)

        # 1. Cargar pesos base (Fine-tuning)
        if self.cfg.pretrain_weights and os.path.exists(self.cfg.pretrain_weights):
            print(f"[Trainer] Cargando pesos: {self.cfg.pretrain_weights}")
            checkpoint = torch.load(self.cfg.pretrain_weights, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)

        # 2. Adaptar última capa para N clases + fondo
        hidden_dim = model.transformer.d_model
        model.class_embed = nn.Linear(hidden_dim, self.cfg.nc + 1)

        # 3. Adaptar criterio de pérdida
        criterion.num_classes = self.cfg.nc
        empty_weight = torch.ones(self.cfg.nc + 1)
        empty_weight[-1] = self.cfg.model_args.eos_coef
        criterion.register_buffer('empty_weight', empty_weight)

        # 4. Aplicar Parche BN2GN para ROCm
        if self.cfg.bn2gn_policy != "off":
            replace_bn_with_gn(model, BN2GNConfig(policy=self.cfg.bn2gn_policy))

        model.to(self.device)
        criterion.to(self.device)
        return model, criterion, postprocessors

    def _setup_optimizer(self):
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.cfg.lr_backbone}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.cfg.lr_drop)
        return optimizer, scheduler

    def fit(self):
        """Bucle principal de entrenamiento y validación."""
        print(f"\n--- Iniciando Entrenamiento DETR: {self.cfg.run_name} ---")

        # Cargar Datos
        train_loader = build_dataloader("train", self.cfg.batch_size)
        val_loader = build_dataloader("valid", self.cfg.batch_size)

        start_time = time.time()
        for epoch in range(self.cfg.epochs):
            # 1. Fase de Entrenamiento
            train_stats = self._train_one_epoch(train_loader, epoch)
            self.lr_scheduler.step()

            # 2. Fase de Validación
            val_stats = self.validator.validate(val_loader, self.save_dir)

            # 3. Registro de Logs (formato JSON Lines para metrics.py)
            log_stats = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_loss_ce": train_stats["loss_ce"],
                "train_loss_bbox": train_stats["loss_bbox"],
                "train_class_error": train_stats["class_error"],
                **{f"test_{k}": v for k, v in val_stats.items()}
            }

            with open(self.save_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 4. Guardar Checkpoints
            self._save_checkpoints(epoch, val_stats)

        print(f"\n[Trainer] Entrenamiento finalizado en {(time.time() - start_time) / 60:.2f} min.")

    def _train_one_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()

        total_loss = 0.0
        stats = {"loss": 0, "loss_ce": 0, "loss_bbox": 0, "class_error": 0}

        for i, (samples, targets) in enumerate(loader):
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = self.model(samples)
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if not math.isfinite(losses.item()):
                print(f"ERROR: Pérdida infinita detectada en batch {i}. Abortando.")
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()

            # Recorte de gradiente: Imprescindible en Transformers para estabilidad
            if self.cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)

            self.optimizer.step()

            # Acumular estadísticas reducidas
            total_loss += losses.item()
            stats["loss_ce"] += loss_dict["loss_labels"].item()
            stats["loss_bbox"] += loss_dict["loss_boxes"].item()

            if i % 20 == 0:
                print(f"Epoch [{epoch}] Batch [{i}/{len(loader)}] Loss: {losses.item():.4f}")

        # Promediar
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

        # Guardar último
        save_on_master(checkpoint, self.weights_dir / "last.pt")

        # Guardar mejor según mAP@0.5 (índice 1 de COCO stats)
        current_map = val_stats.get("coco_eval_bbox", [0, 0])[1]
        if current_map > self.best_map:
            self.best_map = current_map
            save_on_master(checkpoint, self.weights_dir / "best.pt")
            print(f"  --> Nuevo mejor modelo guardado (mAP@0.5: {current_map:.4f})")