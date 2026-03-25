# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Validator.py
# Descripción: Motor de validación para DETR. Realiza inferencia
#              sobre el conjunto de validación y estandariza las
#              métricas de COCO (mAP, Recall) para su posterior
#              registro y visualización.
# ==============================================================

import torch
import sys
from pathlib import Path

# --- INTEGRACIÓN DE SUBMÓDULO DETR ---
FILE = Path(__file__).resolve()
ENGINE_ROOT = FILE.parent
DETR_ROOT = ENGINE_ROOT.parent
DETR_SUBMODULE = DETR_ROOT / "detr"

if str(DETR_SUBMODULE) not in sys.path:
    sys.path.append(str(DETR_SUBMODULE))

try:
    from datasets.coco_eval import CocoEvaluator
    from engine.bootstrap_miopen import MuteStderr
except ImportError as e:
    print(f"[Validator] ERROR: No se pudo importar dependencias: {e}")


class Validator:
    def __init__(self, model, criterion, postprocessors, device):
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.device = device

    @torch.no_grad()
    def validate(self, loader, output_dir):
        """Ejecuta inferencia y retorna diccionario de métricas estandarizadas."""
        self.model.eval()
        self.criterion.eval()

        base_ds = loader.dataset
        evaluator = CocoEvaluator(base_ds.coco, iou_types=['bbox'])

        stats = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "class_error": 0.0}

        print("[Validator] Iniciando validación...", flush=True)

        with MuteStderr():
            for samples, targets in loader:
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(samples)

                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                stats["loss"] += losses.item()
                stats["loss_ce"] += loss_dict["loss_ce"].item()
                stats["loss_bbox"] += loss_dict["loss_bbox"].item()
                stats["loss_giou"] += loss_dict["loss_giou"].item()
                if "class_error" in loss_dict:
                    stats["class_error"] += loss_dict["class_error"].item()

                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = self.postprocessors['bbox'](outputs, orig_target_sizes)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                evaluator.update(res)

        num_batches = len(loader)
        final_stats = {k: v / num_batches for k, v in stats.items()}

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        # Mapeo de índices COCO a nombres legibles para el CSV
        if 'bbox' in evaluator.coco_eval:
            coco_stats = evaluator.coco_eval['bbox'].stats.tolist()
            final_stats["mAP_0.5:0.95"] = coco_stats[0]
            final_stats["mAP_0.5"] = coco_stats[1]
            final_stats["recall"] = coco_stats[8]  # AR@100

        return final_stats