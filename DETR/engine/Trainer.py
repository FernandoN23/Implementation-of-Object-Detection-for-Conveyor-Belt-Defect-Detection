# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Trainer.py
# Descripción: Orquestador de entrenamiento para DETR. Gestiona el
#              ciclo de vida del experimento, incluyendo auto-incremento
#              de carpetas, registro de métricas detalladas en CSV y
#              generación de gráficas de rendimiento (Loss/mAP) en vivo.
# ==============================================================

import os
import sys
import time
import json
import math
import csv
import yaml
import matplotlib.pyplot as plt
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
    from engine.bootstrap_miopen import MuteStderr
except ImportError as e:
    print(f"[Trainer] ERROR: Fallo al importar componentes esenciales: {e}")
    sys.exit(1)

DETR_URLS = {
    "r50": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
    "r50_dc5": "https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth",
    "r101": "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",
    "r101_dc5": "https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth"
}


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """Incrementa el path si ya existe, ej: exp -> exp2, exp3, etc."""
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
    exist_ok: bool = False
    metrics_root: Path = DETR_ROOT / "metrics"


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # 1. Gestión de Rutas con Auto-incremento
        base_subdir = Path(self.cfg.variant) / self.cfg.phase / self.cfg.run_name
        self.save_dir = increment_path(DETR_ROOT / "runs" / base_subdir, exist_ok=self.cfg.exist_ok)
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Ruta de métricas espejo de runs
        rel_path = self.save_dir.relative_to(DETR_ROOT / "runs")
        self.metrics_dir = self.cfg.metrics_root / "detect" / rel_path
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        (self.metrics_dir / "losses").mkdir(parents=True, exist_ok=True)

        self.csv_path = self.metrics_dir / "results.csv"

        # 2. Guardar Hiperparámetros (hyp.yaml)
        self._save_hyp_yaml()

        # 3. Inicializar componentes
        self.model, self.criterion, self.postprocessors = self._setup_model()
        self.optimizer, self.lr_scheduler = self._setup_optimizer()
        self.validator = Validator(self.model, self.criterion, self.postprocessors, self.device)

        # Seguimiento de mejores métricas
        self.best_map = 0.0
        self.best_metrics = {"mAP_0.5": 0.0, "mAP_0.5:0.95": 0.0, "recall": 0.0, "F1": 0.0}

    def _save_hyp_yaml(self):
        """Consolida y guarda la configuración completa en hyp.yaml."""
        hyp = {
            "experiment": {"variant": self.cfg.variant, "run_name": self.cfg.run_name, "save_dir": str(self.save_dir),
                           "device": self.cfg.device},
            "training": {"epochs": self.cfg.epochs, "batch_size": self.cfg.batch_size, "lr": self.cfg.lr,
                         "lr_backbone": self.cfg.lr_backbone},
            "architecture": vars(self.cfg.model_args) if self.cfg.model_args else {},
            "hardware_policy": {"bn2gn_policy": self.cfg.bn2gn_policy}
        }
        with open(self.save_dir / "hyp.yaml", "w", encoding="utf-8") as f:
            yaml.dump(hyp, f, default_flow_style=False, sort_keys=False)

    def _maybe_download_weights(self):
        """Descarga automática de pesos si no existen localmente."""
        w_path = Path(self.cfg.pretrain_weights)
        if not w_path.exists():
            variant = self.cfg.variant
            if variant in DETR_URLS:
                print(f"[Trainer] Descargando pesos oficiales para '{variant}'...")
                w_path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(DETR_URLS[variant], str(w_path))

    def _setup_model(self):
        self._maybe_download_weights()
        model, criterion, postprocessors = build_model(self.cfg.model_args)
        w_path = Path(self.cfg.pretrain_weights)
        if w_path.exists():
            checkpoint = torch.load(w_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)

        hidden_dim = model.transformer.d_model
        model.class_embed = nn.Linear(hidden_dim, self.cfg.nc + 1)
        criterion.num_classes = self.cfg.nc
        empty_weight = torch.ones(self.cfg.nc + 1)
        empty_weight[-1] = self.cfg.model_args.eos_coef
        criterion.register_buffer('empty_weight', empty_weight)

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

    def _log_to_csv(self, epoch, train_stats, val_stats):
        """Escribe métricas detalladas en results.csv."""
        header = [
            'epoch', 'train/loss', 'train/loss_ce', 'train/loss_bbox', 'train/loss_giou',
            'val/loss', 'val/loss_ce', 'val/loss_bbox', 'val/loss_giou',
            'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'metrics/recall'
        ]
        row = [
            epoch,
            train_stats['loss'], train_stats['loss_ce'], train_stats['loss_bbox'], train_stats['loss_giou'],
            val_stats['loss'], val_stats['loss_ce'], val_stats['loss_bbox'], val_stats['loss_giou'],
            val_stats['mAP_0.5'], val_stats['mAP_0.5:0.95'], val_stats['recall']
        ]
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(header)
            writer.writerow(row)

    def _plot_live_results(self):
        """Genera gráficas comparativas Train vs Val actualizadas en cada época."""
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            # Loss Total
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train/loss'], label='Train', linewidth=2)
            plt.plot(df['epoch'], df['val/loss'], label='Validation', linewidth=2)
            plt.title('Total Loss Evolution');
            plt.xlabel('Epoch');
            plt.ylabel('Loss');
            plt.legend();
            plt.grid(True, alpha=0.3)
            plt.savefig(self.metrics_dir / "loss_combined.png", dpi=200);
            plt.close()
            # Losses Específicas
            for key in ['loss_ce', 'loss_bbox', 'loss_giou']:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df[f'train/{key}'], label='Train', linewidth=2)
                plt.plot(df['epoch'], df[f'val/{key}'], label='Validation', linewidth=2)
                plt.title(key.replace('_', ' ').upper());
                plt.xlabel('Epoch');
                plt.ylabel('Loss');
                plt.legend();
                plt.grid(True, alpha=0.3)
                plt.savefig(self.metrics_dir / f"losses/{key}_combined.png", dpi=200);
                plt.close()
        except Exception as e:
            print(f"[Trainer] Error en live plotting: {e}")

    def _plot_final_metrics(self):
        """Genera gráficas individuales de precisión al finalizar el entrenamiento."""
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            metrics = [('metrics/mAP_0.5', 'map_05.png', 'mAP @ 50%'),
                       ('metrics/mAP_0.5:0.95', 'map_05_95.png', 'mAP @ 50-95%'),
                       ('metrics/recall', 'recall.png', 'Average Recall')]
            for col, fname, title in metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df[col], color='blue', linewidth=2.5)
                plt.title(title);
                plt.xlabel('Epoch');
                plt.ylabel('Value');
                plt.grid(True, alpha=0.3)
                plt.savefig(self.metrics_dir / fname, dpi=200);
                plt.close()
            # F1 Score
            plt.figure(figsize=(10, 6))
            f1 = 2 * (df['metrics/mAP_0.5'] * df['metrics/recall']) / (
                        df['metrics/mAP_0.5'] + df['metrics/recall'] + 1e-16)
            plt.plot(df['epoch'], f1, color='purple', linewidth=2.5)
            plt.title('F1-Score Evolution');
            plt.xlabel('Epoch');
            plt.ylabel('F1');
            plt.grid(True, alpha=0.3)
            plt.savefig(self.metrics_dir / "f1_score.png", dpi=200);
            plt.close()
        except Exception as e:
            print(f"[Trainer] Error en final plotting: {e}")

    def fit(self):
        print(f"\n--- Iniciando Entrenamiento DETR: {self.save_dir.name} ---")
        train_loader = build_dataloader("train", self.cfg.batch_size)
        val_loader = build_dataloader("valid", self.cfg.batch_size)

        start_time = time.time()
        for epoch in range(self.cfg.epochs):
            train_stats = self._train_one_epoch(train_loader, epoch)
            self.lr_scheduler.step()
            val_stats = self.validator.validate(val_loader, self.save_dir)

            # 1. Log JSON (Nativo DETR)
            log_stats = {
                "epoch": epoch, "train_loss": train_stats["loss"],
                **{f"test_{k}": v for k, v in val_stats.items()}
            }
            with open(self.save_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 2. Estandarización de Métricas (CSV y Live PNG)
            self._log_to_csv(epoch, train_stats, val_stats)
            self._plot_live_results()

            self._save_checkpoints(epoch, val_stats)

        # 3. Generación de métricas finales
        self._plot_final_metrics()
        with open(self.metrics_dir / "metrics.yaml", "w") as f:
            yaml.dump(self.best_metrics, f)

        print(f"\n[Trainer] Finalizado en {(time.time() - start_time) / 60:.2f} min.")

    def _train_one_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()
        stats = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "class_error": 0.0}
        print_freq = 10

        for i, (samples, targets) in enumerate(loader):
            with MuteStderr():
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(samples)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                if not math.isfinite(losses.item()): continue

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

            if i % print_freq == 0 or i == len(loader) - 1:
                print(f"Epoch [{epoch}] Batch [{i}/{len(loader)}] - Loss: {losses.item():.4f}", flush=True)

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
        current_map = val_stats.get("mAP_0.5", 0.0)
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_metrics = {
                "mAP_0.5": float(val_stats["mAP_0.5"]),
                "mAP_0.5:0.95": float(val_stats["mAP_0.5:0.95"]),
                "recall": float(val_stats["recall"]),
                "F1": float(2 * (val_stats["mAP_0.5"] * val_stats["recall"]) / (
                            val_stats["mAP_0.5"] + val_stats["recall"] + 1e-16))
            }
            save_on_master(checkpoint, self.weights_dir / "best.pt")
            print(f"  --> Nuevo Mejor mAP@0.5: {current_map:.4f}")