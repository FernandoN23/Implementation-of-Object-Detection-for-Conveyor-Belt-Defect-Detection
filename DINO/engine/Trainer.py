# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/engine/Trainer.py
# Descripción: Orquestador de entrenamiento para DINO.
#              Optimizado con backend 'Agg', limpieza de caché,
#              AMP, ModelEma y Contrastive DeNoising (CDN).
# ==============================================================

import os
import sys
import time
import json
import math
import csv
import yaml
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Union, List

import torch
import torch.nn as nn

# --- INTEGRACIÓN DE SUBMÓDULO DINO ---
FILE = Path(__file__).resolve()
ENGINE_ROOT = FILE.parent
DINO_ROOT = ENGINE_ROOT.parent
DINO_SUBMODULE = DINO_ROOT / "dino"

if str(DINO_SUBMODULE) not in sys.path:
    sys.path.append(str(DINO_SUBMODULE))

try:
    from models import build_model
    from util.misc import save_on_master
    from util.get_param_dicts import get_param_dict
    from util.utils import ModelEma
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig
    from utility.data_loader import build_dataloader
    from engine.Validator import Validator
    from engine.bootstrap_miopen import MuteStderr
except ImportError as e:
    print(f"[Trainer] ERROR: Fallo al importar componentes esenciales de DINO: {e}")
    sys.exit(1)

# URLs oficiales de DINO para inicialización
DINO_URLS = {
    "r50_4scale": "https://github.com/IDEA-Research/DINO/releases/download/v0.1.0/checkpoint0011_4scale.pth",
    "r50_5scale": "https://github.com/IDEA-Research/DINO/releases/download/v0.1.0/checkpoint0011_5scale.pth",
    "swin_l_4scale": "https://github.com/IDEA-Research/DINO/releases/download/v0.1.0/checkpoint0029_4scale_swin.pth"
}


class Dict2Obj:
    """Convierte un diccionario en un objeto para que DINO pueda acceder a args.atributo de forma segura."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        # Retorna None si DINO pide un argumento que no definimos en el YAML (evita crashes)
        return None


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if exist_ok and path.exists():
        return path
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class TrainerConfig:
    variant: str = "r50_4scale"
    run_name: str = "exp"
    phase: str = "train"
    epochs: int = 12
    batch_size: int = 2
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    lr_drop: Union[int, List[int]] = 11
    lr_gamma: float = 0.1
    clip_max_norm: float = 0.1
    device: str = "cuda"
    pretrain_weights: str = ""
    nc: int = 5
    class_names: List[str] = None  # type: ignore
    model_args: Any = None
    bn2gn_policy: str = "on"
    exist_ok: bool = False
    metrics_root: Path = DINO_ROOT / "metrics"
    resume: Union[bool, str] = False
    start_epoch: int = 0
    use_coco128: bool = False
    empty_cache_freq: int = 1
    use_amp: bool = True
    ema_decay: float = 0.9997


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        base_subdir = Path(self.cfg.variant) / self.cfg.phase / self.cfg.run_name
        if self.cfg.resume: self.cfg.exist_ok = True

        self.save_dir = increment_path(DINO_ROOT / "runs" / base_subdir, exist_ok=self.cfg.exist_ok)
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        rel_path = self.save_dir.relative_to(DINO_ROOT / "runs")
        self.metrics_dir = self.cfg.metrics_root / "detect" / rel_path
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        (self.metrics_dir / "losses").mkdir(parents=True, exist_ok=True)

        self.csv_path = self.metrics_dir / "results.csv"

        self._check_and_save_hyp()

        self.model, self.criterion, self.postprocessors = self._setup_model()
        self.ema_m = ModelEma(self.model, self.cfg.ema_decay)

        self.optimizer, self.lr_scheduler = self._setup_optimizer()
        self.validator = Validator(self.model, self.criterion, self.postprocessors, self.device)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.use_amp)

        self.best_map = 0.0
        self.best_metrics = {"mAP_0.5": 0.0, "mAP_0.5:0.95": 0.0, "precision": 0.0, "recall": 0.0, "F1": 0.0}

        self._resume_training()

    def _check_and_save_hyp(self):
        hyp_path = self.save_dir / "hyp.yaml"
        drop_epochs = [self.cfg.lr_drop] if isinstance(self.cfg.lr_drop, int) else self.cfg.lr_drop

        new_hyp = {
            "experiment": {"variant": self.cfg.variant, "run_name": self.cfg.run_name, "save_dir": str(self.save_dir),
                           "device": self.cfg.device, "use_coco128": self.cfg.use_coco128},
            "training": {"epochs": self.cfg.epochs, "batch_size": self.cfg.batch_size, "lr": self.cfg.lr,
                         "lr_backbone": self.cfg.lr_backbone, "lr_drop_epochs": drop_epochs,
                         "lr_gamma": self.cfg.lr_gamma, "weight_decay": self.cfg.weight_decay,
                         "clip_max_norm": self.cfg.clip_max_norm, "ema_decay": self.cfg.ema_decay},
            "architecture": vars(self.cfg.model_args) if self.cfg.model_args else {},
            "hardware_policy": {"bn2gn_policy": self.cfg.bn2gn_policy, "empty_cache_freq": self.cfg.empty_cache_freq,
                                "use_amp": self.cfg.use_amp}
        }

        if self.cfg.resume and hyp_path.exists():
            with open(hyp_path, "r", encoding="utf-8") as f:
                old_hyp = yaml.safe_load(f)
            if old_hyp.get("training") != new_hyp["training"] or old_hyp.get("architecture") != new_hyp["architecture"]:
                print("[Trainer] Cambio detectado en la configuración de hiperparámetros.")
            else:
                print("[Trainer] Hiperparámetros cargados desde train.yaml sin modificaciones.")
        elif self.cfg.resume:
            print("[Trainer] Cambio detectado en la configuración de hiperparámetros.")

        with open(hyp_path, "w", encoding="utf-8") as f:
            yaml.dump(new_hyp, f, default_flow_style=False, sort_keys=False)

    def _maybe_download_weights(self):
        w_path = Path(self.cfg.pretrain_weights)
        if not w_path.exists() and self.cfg.pretrain_weights != "":
            variant = self.cfg.variant
            if variant in DINO_URLS:
                print(f"[Trainer] Descargando pesos oficiales para '{variant}'...")
                w_path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(DINO_URLS[variant], str(w_path))

    def _setup_model(self):
        self._maybe_download_weights()

        # Convertir el diccionario de argumentos a un objeto para DINO
        dino_args = Dict2Obj(vars(self.cfg.model_args))
        dino_args.device = self.cfg.device
        dino_args.num_classes = self.cfg.nc

        model, criterion, postprocessors = build_model(dino_args)

        if not self.cfg.resume:
            w_path = Path(self.cfg.pretrain_weights)
            if w_path.exists():
                print(f"[Trainer] Cargando pesos pre-entrenados desde {w_path}")
                checkpoint = torch.load(w_path, map_location='cpu', weights_only=False)
                # Limpiar state_dict para evitar conflictos con el número de clases
                state_dict = checkpoint['model']
                state_dict = {k: v for k, v in state_dict.items() if 'class_embed' not in k and 'label_enc' not in k}
                model.load_state_dict(state_dict, strict=False)

        if self.cfg.bn2gn_policy != "off":
            replace_bn_with_gn(model, BN2GNConfig(policy=self.cfg.bn2gn_policy))

        model.to(self.device)
        criterion.to(self.device)
        return model, criterion, postprocessors

    def _setup_optimizer(self):
        # Usar el gestor de parámetros oficial de DINO
        dino_args = Dict2Obj({
            'param_dict_type': 'default',
            'lr_backbone': self.cfg.lr_backbone,
            'lr': self.cfg.lr,
            'weight_decay': self.cfg.weight_decay
        })
        param_dicts = get_param_dict(dino_args, self.model)

        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        milestones = [self.cfg.lr_drop] if isinstance(self.cfg.lr_drop, int) else self.cfg.lr_drop
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=self.cfg.lr_gamma)
        return optimizer, scheduler

    def _resume_training(self):
        if not self.cfg.resume: return
        resume_path = Path(self.cfg.resume) if isinstance(self.cfg.resume, str) else self.weights_dir / "last.pt"
        if resume_path and resume_path.exists():
            print(f"[Trainer] Reanudando entrenamiento desde: {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            if 'ema_model' in checkpoint:
                self.ema_m.module.load_state_dict(checkpoint['ema_model'])

            if 'scaler' in checkpoint and self.cfg.use_amp:
                self.scaler.load_state_dict(checkpoint['scaler'])

            self.cfg.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.param_groups[0]['lr'] = self.cfg.lr
            self.optimizer.param_groups[0]['initial_lr'] = self.cfg.lr
            self.optimizer.param_groups[1]['lr'] = self.cfg.lr_backbone
            self.optimizer.param_groups[1]['initial_lr'] = self.cfg.lr_backbone
            milestones = [self.cfg.lr_drop] if isinstance(self.cfg.lr_drop, int) else self.cfg.lr_drop
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                     gamma=self.cfg.lr_gamma)
            for _ in range(self.cfg.start_epoch): self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"[Trainer] Estado restaurado. Continuando desde la época {self.cfg.start_epoch}. LR actual: {current_lr:.2e}")
        else:
            print(f"[Trainer] ADVERTENCIA: No se encontró archivo para reanudar. Iniciando desde cero.")
            self.cfg.start_epoch = 0

    def _log_to_csv(self, epoch, train_stats, val_stats):
        header = ['epoch', 'train/loss', 'train/loss_ce', 'train/loss_bbox', 'train/loss_giou',
                  'train/loss_ce_dn', 'train/loss_bbox_dn', 'train/loss_giou_dn',
                  'val/loss', 'val/loss_ce', 'val/loss_bbox', 'val/loss_giou',
                  'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'metrics/F1']

        row = [epoch, train_stats.get('loss', 0), train_stats.get('loss_ce', 0), train_stats.get('loss_bbox', 0),
               train_stats.get('loss_giou', 0),
               train_stats.get('loss_ce_dn', 0), train_stats.get('loss_bbox_dn', 0), train_stats.get('loss_giou_dn', 0),
               val_stats.get('loss', 0), val_stats.get('loss_ce', 0), val_stats.get('loss_bbox', 0),
               val_stats.get('loss_giou', 0),
               val_stats.get('precision', 0.0), val_stats.get('recall', 0.0), val_stats.get('mAP_0.5', 0.0),
               val_stats.get('mAP_0.5:0.95', 0.0), val_stats.get('F1', 0.0)]

        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(header)
            writer.writerow(row)

    def _plot_live_results(self):
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            plt.style.use(
                'seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')

            # Total Loss
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train/loss'], label='Train', linewidth=2.5, color='#1f77b4')
            plt.plot(df['epoch'], df['val/loss'], label='Validation', linewidth=2.5, color='#ff7f0e')
            plt.title('Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.metrics_dir / "loss_combined.png", dpi=200)
            plt.close('all')

            # Standard Losses
            for key in ['loss_ce', 'loss_bbox', 'loss_giou']:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df[f'train/{key}'], label='Train', linewidth=2.5, color='#1f77b4')
                plt.plot(df['epoch'], df[f'val/{key}'], label='Validation', linewidth=2.5, color='#ff7f0e')
                plt.title(key.replace('_', ' ').upper())
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(self.metrics_dir / f"losses/{key}_combined.png", dpi=200)
                plt.close('all')

            # DeNoising Losses (Solo Train)
            for key in ['loss_ce_dn', 'loss_bbox_dn', 'loss_giou_dn']:
                if f'train/{key}' in df.columns and not df[f'train/{key}'].isnull().all():
                    plt.figure(figsize=(10, 6))
                    plt.plot(df['epoch'], df[f'train/{key}'], label='Train (DeNoising)', linewidth=2.5, color='#2ca02c')
                    plt.title(key.replace('_', ' ').upper())
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(self.metrics_dir / f"losses/{key}_train.png", dpi=200)
                    plt.close('all')

        except Exception as e:
            print(f"[Trainer] Error en live plotting: {e}")

    def _plot_final_metrics(self):
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            plt.style.use(
                'seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
            metrics = [('metrics/mAP_0.5', 'map_05.png', 'mAP @ 50%', 'green'),
                       ('metrics/mAP_0.5:0.95', 'map_05_95.png', 'mAP @ 50-95%', 'red'),
                       ('metrics/precision', 'precision.png', 'Precision', 'blue'),
                       ('metrics/recall', 'recall.png', 'Recall', 'orange'),
                       ('metrics/F1', 'f1_score.png', 'F1-Score', 'purple')]
            for col, fname, title, color in metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df[col], color=color, linewidth=2.5)
                plt.title(title)
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(self.metrics_dir / fname, dpi=200)
                plt.close('all')
        except Exception as e:
            print(f"[Trainer] Error en final plotting: {e}")

    def _consolidate_final_weights(self):
        best_local = self.weights_dir / "best.pt"
        if best_local.exists():
            global_weights_dir = DINO_ROOT / "weights" / self.cfg.variant
            global_weights_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_local, global_weights_dir / f"{self.cfg.run_name}_best.pt")
            print(f"[Trainer] Pesos finales consolidados en: {global_weights_dir.relative_to(DINO_ROOT)}")

    def fit(self):
        print(f"\n[Trainer] --- Iniciando Entrenamiento DINO: {self.save_dir.name} ---")
        if self.cfg.use_amp:
            print(f"[Trainer] Automatic Mixed Precision (AMP) activado.")

        train_loader = build_dataloader("train", self.cfg.batch_size, use_coco128=self.cfg.use_coco128,
                                        class_names=self.cfg.class_names)
        val_loader = build_dataloader("valid", self.cfg.batch_size, use_coco128=self.cfg.use_coco128,
                                      class_names=self.cfg.class_names)
        start_time = time.time()
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            old_lr = self.optimizer.param_groups[0]['lr']

            train_stats = self._train_one_epoch(train_loader, epoch)
            self.lr_scheduler.step()

            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < (old_lr - 1e-8):
                print(f"[Trainer] Info: Learning Rate Drop. Época: {epoch}. Valor: {new_lr:.2e}")

            # Validar usando el modelo EMA para mayor estabilidad
            val_stats = self.validator.validate(val_loader, self.save_dir, model_override=self.ema_m.module)

            self._log_to_csv(epoch, train_stats, val_stats)
            self._plot_live_results()
            self._save_checkpoints(epoch, val_stats)

            if self.cfg.empty_cache_freq > 0 and (epoch + 1) % self.cfg.empty_cache_freq == 0:
                torch.cuda.empty_cache()

        self._plot_final_metrics()
        with open(self.metrics_dir / "metrics.yaml", "w") as f:
            yaml.dump(self.best_metrics, f)
        self._consolidate_final_weights()
        print(f"\n[Trainer] Finalizado en {(time.time() - start_time) / 60:.2f} min.")

    def _train_one_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()

        stats = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0,
                 "loss_ce_dn": 0.0, "loss_bbox_dn": 0.0, "loss_giou_dn": 0.0, "class_error": 0.0}

        for i, (samples, targets) in enumerate(loader):
            with MuteStderr():
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()

                # Forward Pass con AMP y Targets (Requerido para Contrastive DeNoising)
                with torch.autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                    outputs = self.model(samples, targets)
                    loss_dict = self.criterion(outputs, targets)
                    weight_dict = self.criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                if not math.isfinite(losses.item()): continue

                # Backward Pass y Step con GradScaler
                self.scaler.scale(losses).backward()

                if self.cfg.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_max_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Actualizar pesos EMA
                self.ema_m.update(self.model)

                # Registrar estadísticas
                stats["loss"] += losses.item()
                for k in ["loss_ce", "loss_bbox", "loss_giou", "loss_ce_dn", "loss_bbox_dn", "loss_giou_dn"]:
                    if k in loss_dict: stats[k] += loss_dict[k].item()
                if "class_error" in loss_dict: stats["class_error"] += loss_dict["class_error"].item()

            if i % 10 == 0:
                print(f"[Trainer] Epoch [{epoch}] Batch[{i}/{len(loader)}] - Loss: {losses.item():.4f}", flush=True)

        return {k: v / len(loader) for k, v in stats.items()}

    def _save_checkpoints(self, epoch, val_stats):
        checkpoint = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_m.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.cfg.use_amp else None,
            'epoch': epoch,
            'cfg': self.cfg
        }
        save_on_master(checkpoint, self.weights_dir / "last.pt")
        current_map = val_stats.get("mAP_0.5", 0.0)
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_metrics = {"mAP_0.5": float(val_stats.get("mAP_0.5", 0.0)),
                                 "mAP_0.5:0.95": float(val_stats.get("mAP_0.5:0.95", 0.0)),
                                 "precision": float(val_stats.get("precision", 0.0)),
                                 "recall": float(val_stats.get("recall", 0.0)), "F1": float(val_stats.get("F1", 0.0))}
            save_on_master(checkpoint, self.weights_dir / "best.pt")
            print(f"[Trainer] --> Nuevo Mejor mAP@0.5: {current_map:.4f}")