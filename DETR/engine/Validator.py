# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DETR/engine/Validator.py
# Descripción: Motor de validación para DETR. Evalúa el modelo
#              sobre el split de validación y calcula métricas COCO.
# ==============================================================

import torch
import sys
from pathlib import Path

# Inyectar submódulo si no está
FILE = Path(__file__).resolve()
DETR_SUBMODULE = FILE.parents[1] / "detr"
if str(DETR_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(DETR_SUBMODULE))

try:
    from datasets.coco_eval import CocoEvaluator
    from utility.data_loader import build_dataloader  # Para obtener el dataset base
except ImportError as e:
    print(f"[Validator] ERROR: {e}")


class Validator:
    def __init__(self, model, criterion, postprocessors, device):
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.device = device

    @torch.no_grad()
    def validate(self, loader, output_dir):
        """Ejecuta inferencia y retorna diccionario de métricas."""
        self.model.eval()
        self.criterion.eval()

        # Para el evaluador de COCO necesitamos el dataset original (con metadatos)
        # Accedemos a través del loader -> dataset
        base_ds = loader.dataset

        # DETR usa CocoEvaluator para agrupar predicciones
        # iou_types = ('bbox',)
        evaluator = CocoEvaluator(base_ds, iou_types=['bbox'])

        stats = {"loss": 0, "loss_ce": 0, "loss_bbox": 0, "class_error": 0}

        print("[Validator] Evaluando...")
        for samples, targets in loader:
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = self.model(samples)

            # 1. Calcular pérdidas de validación
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            stats["loss"] += losses.item()
            stats["loss_ce"] += loss_dict["loss_labels"].item()
            stats["loss_bbox"] += loss_dict["loss_boxes"].item()

            # 2. Post-procesar para formato COCO (boxes xyxy absolutas)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = self.postprocessors['bbox'](outputs, orig_target_sizes)

            # 3. Alimentar evaluador
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            evaluator.update(res)

        # Promediar pérdidas
        num_batches = len(loader)
        final_stats = {k: v / num_batches for k, v in stats.items()}

        # Sincronizar y resumir métricas COCO
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        # Extraer el array de stats de COCO ([0]: mAP@.5:.95, [1]: mAP@.5, etc)
        if 'bbox' in evaluator.coco_eval:
            final_stats["coco_eval_bbox"] = evaluator.coco_eval['bbox'].stats.tolist()

        return final_stats