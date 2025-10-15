"""
metrics.py
---------------------------------
Cálculo de métricas para detección de objetos:
mAP, mAP50-95, IoU y FPS.
"""

import torch
import numpy as np
import time

def bbox_iou(box1, box2, eps=1e-6):
    """Calcula el IoU entre dos cajas (x1, y1, x2, y2)."""
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter_area / (box1_area + box2_area - inter_area + eps)


def calculate_map(preds, targets, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Cálculo simplificado de mAP."""
    aps = []
    for t in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for pred, gt in zip(preds, targets):
            iou = bbox_iou(pred, gt)
            if iou > t:
                tp += 1
            else:
                fp += 1
        ap = tp / (tp + fp + 1e-6)
        aps.append(ap)
    return np.mean(aps)


def measure_fps(model, sample_input, device="cpu", runs=20):
    """Evalúa FPS promedio del modelo."""
    model.eval()
    sample_input = sample_input.to(device)
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(sample_input)
    total = time.time() - start
    return runs / total
