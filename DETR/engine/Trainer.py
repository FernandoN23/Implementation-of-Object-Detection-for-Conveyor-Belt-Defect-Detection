# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Trainer.py
# Descripción: Orquestador de entrenamiento. Gestiona la carga de
#              pesos, parches BN2GN y el ciclo de optimización AdamW.
# ==============================================================

import os
import sys
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any

# --- SOLUCIÓN DE REFERENCIAS DINÁMICAS ---
FILE = Path(__file__).resolve()
ENGINE_ROOT = FILE.parent
DETR_ROOT = ENGINE_ROOT.parent
DETR_SUBMODULE = DETR_ROOT / "detr"

# Insertar el submódulo al principio del path para que el IDE y Python lo encuentren
if str(DETR_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(DETR_SUBMODULE))

import torch
import torch.nn as nn

# Ahora podemos importar desde el submódulo 'detr' con total seguridad
try:
    from models import build_model
    from util.misc import save_on_master, reduce_dict
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig
except ImportError as e:
    print(f"[Trainer] ERROR: No se pudieron cargar los módulos de DETR: {e}")
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

        # Estructura de directorios canónica
        self.save_dir = DETR_ROOT / "runs" / self.cfg.variant / self.cfg.phase / self.cfg.run_name
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar componentes
        self.model, self.criterion = self._setup_model()
        self.optimizer, self.lr_scheduler = self._setup_optimizer()

    def _setup_model(self):
        """Construye, parcha y adapta el modelo DETR."""
        print(f"[Trainer] Construyendo modelo variante: {self.cfg.variant}")
        model, criterion, _ = build_model(self.cfg.model_args)

        # 1. Cargar pesos base (COCO)
        if self.cfg.pretrain_weights and os.path.exists(self.cfg.pretrain_weights):
            print(f"[Trainer] Cargando pesos base desde: {self.cfg.pretrain_weights}")
            checkpoint = torch.load(self.cfg.pretrain_weights, map_location='cpu')
            # strict=False permite cargar aunque falte el cabezal de salida
            model.load_state_dict(checkpoint['model'], strict=False)

        # 2. Adaptar dinámicamente el cabezal de clasificación (5 clases + fondo)
        hidden_dim = model.transformer.d_model
        model.class_embed = nn.Linear(hidden_dim, self.cfg.nc + 1)

        # 3. Adaptar el criterio de pérdida
        criterion.num_classes = self.cfg.nc
        empty_weight = torch.ones(self.cfg.nc + 1)
        empty_weight[-1] = self.cfg.model_args.eos_coef
        criterion.register_buffer('empty_weight', empty_weight)

        # 4. Aplicar Parche BN2GN (Vital para ROCm)
        if self.cfg.bn2gn_policy != "off":
            replace_bn_with_gn(model, BN2GNConfig(policy=self.cfg.bn2gn_policy))

        model.to(self.device)
        criterion.to(self.device)
        return model, criterion

    def _setup_optimizer(self):
        """Configuración de AdamW con LR diferenciado para el backbone."""
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.cfg.lr_backbone}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.cfg.lr_drop)
        return optimizer, scheduler

    def fit(self, train_loader, val_loader=None):
        """Bucle principal de entrenamiento."""
        print(f"[Trainer] Iniciando entrenamiento de {self.cfg.run_name}...")
        start_time = time.time()

        for epoch in range(self.cfg.epochs):
            self._train_one_epoch(train_loader, epoch)
            self.lr_scheduler.step()

            # Guardado del checkpoint 'last.pt'
            save_on_master({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': self.cfg
            }, self.weights_dir / "last.pt")

        print(f"[Trainer] Finalizado en {(time.time() - start_time) / 60:.2f} min.")

    def _train_one_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()

        for i, (samples, targets) in enumerate(loader):
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = self.model(samples)
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            self.optimizer.zero_grad()
            losses.backward()

            # Recorte de gradiente: Imprescindible en Transformers
            if self.cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)

            self.optimizer.step()