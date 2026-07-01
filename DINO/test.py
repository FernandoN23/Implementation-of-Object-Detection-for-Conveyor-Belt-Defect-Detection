# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: DINO/test.py
# Descripción: Punto de entrada principal para pruebas de inferencia
#              sobre modelos DINO entrenados.
#              *CORREGIDO: lr_backbone > 0 para evitar crash en build_backbone*
#              *CORREGIDO: Dict2Obj unificado con soporte 'in'*
#              *CORREGIDO: Lógica de Resize (800x1333) inyectada
#               para igualar el preprocesamiento de entrenamiento.*
#              *CORREGIDO: Sincronización de MIOpen DB.*
#              *CORREGIDO: Uso de make_coco_transforms oficial para
#               garantizar preprocesamiento idéntico a valid.py.*
#              *CORREGIDO: Importación diferida estricta para evitar
#               inicialización prematura de PyTorch (MIOpen).*
#              *NUEVO: Auto-detección de arquitectura desde el checkpoint
#               para evitar recortes destructivos en pesos fine-tuned.*
# ==============================================================

from __future__ import annotations
import argparse
import os
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

FILE = Path(__file__).resolve()
DINO_ROOT = FILE.parent
PROJECT_ROOT = DINO_ROOT.parent
CONFIGS_ROOT = DINO_ROOT / "configs"

if str(DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DINO_ROOT))

DINO_SUBMODULE = DINO_ROOT / "dino"
if DINO_SUBMODULE.is_dir() and str(DINO_SUBMODULE) not in sys.path:
    sys.path.insert(1, str(DINO_SUBMODULE))

DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"

try:
    from engine.bootstrap_miopen import MIOpenConfig, bootstrap, MuteStderr
except Exception as e:
    print(f"[test.py] ERROR: No se pudo importar bootstrap_miopen: {e}")
    sys.exit(1)

DINO_DEFAULTS = {
    'query_dim': 4,
    'unic_layers': 0,
    'decoder_layer_noise': False,
    'dln_xy_noise': 0.2,
    'dln_hw_noise': 0.2,
    'add_channel_attention': False,
    'add_pos_value': False,
    'random_refpoints_xy': False,
    'two_stage_type': 'standard',
    'two_stage_pat_embed': 0,
    'two_stage_add_query_num': 0,
    'two_stage_learn_wh': False,
    'two_stage_keep_all_tokens': False,
    'dec_layer_number': None,
    'decoder_sa_type': 'sa',
    'decoder_module_seq': ['sa', 'ca', 'ffn'],
    'embed_init_tgt': True,
    'use_detached_boxes_dec_out': False,
    'transformer_activation': 'relu',
    'num_patterns': 0,
    'dec_pred_class_embed_share': True,
    'dec_pred_bbox_embed_share': True,
    'two_stage_bbox_embed_share': False,
    'two_stage_class_embed_share': False,
    'use_deformable_box_attn': False,
    'box_attn_type': 'roi_align',
    'match_unstable_error': True,
    'fix_refpoints_hw': -1,
    'use_dn': True,
    'dn_number': 100,
    'dn_box_noise_scale': 0.4,
    'dn_label_noise_ratio': 0.5,
    'matcher_type': 'HungarianMatcher',
    'num_select': 300,
    'nms_iou_threshold': -1,
    'interm_loss_coef': 1.0,
    'no_interm_box_loss': False,
    'pe_temperatureH': 20,
    'pe_temperatureW': 20,
    'backbone_freeze_keywords': None,
}


class Dict2Obj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return None

    def __contains__(self, key):
        return key in self.__dict__


@dataclass
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


@dataclass
class ClassStats:
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    matches: int = 0


@dataclass
class ModelContext:
    model: Any
    postprocessors: Any
    device: Any
    transform: Any


def _resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)


def load_class_names() -> List[str]:
    if not DATA_YAML.is_file(): raise FileNotFoundError(f"[test.py] No se encontró: {DATA_YAML}")
    data = load_yaml(DATA_YAML)
    names = data.get("names") or data.get("classes")
    if isinstance(names, dict): names = list(names.values())
    return [str(n) for n in names]


def load_test_images(image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Tuple[List[Path], Path]:
    images_dir = DATASET_ROOT / "test" / "images"
    labels_dir = DATASET_ROOT / "test" / "labels"
    image_paths: List[Path] = []
    for ext in image_exts: image_paths.extend(sorted(images_dir.glob(f"*{ext}")))
    return image_paths, labels_dir


def build_color_palettes(num_classes: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    return [(0, 255, 0) for _ in range(num_classes)], [(0, 165, 255) for _ in range(num_classes)]


def load_gt_boxes(label_file: Path, img_shape: Tuple[int, int, int]) -> List[Box]:
    h, w = img_shape[:2]
    if not label_file.is_file(): return []
    boxes: List[Box] = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
            x_c *= w
            y_c *= h
            bw *= w
            bh *= h
            boxes.append(
                Box(cls_id=int(cls_id), x1=x_c - bw / 2.0, y1=y_c - bh / 2.0, x2=x_c + bw / 2.0, y2=y_c + bh / 2.0,
                    conf=1.0))
    return boxes


def load_model(weights: Path, variant: str, device_str: str, num_classes: int) -> ModelContext:
    import torch
    from models import build_model
    from engine.bn2gn_patch import replace_bn_with_gn, BN2GNConfig
    from datasets.coco import make_coco_transforms

    device = torch.device(device_str if device_str else "cuda" if torch.cuda.is_available() else "cpu")
    variants_cfg = load_yaml(CONFIGS_ROOT / "model_variants.yaml")

    if variant not in variants_cfg['variants']:
        raise ValueError(f"[test.py] Variante '{variant}' no encontrada en model_variants.yaml")

    v_params = variants_cfg['variants'][variant]

    base_args = DINO_DEFAULTS.copy()
    base_args.update(v_params)

    base_args.update({
        'lr_backbone': 1e-5, 'masks': False, 'frozen_weights': None, 'aux_loss': False, 'set_cost_class': 1.0,
        'set_cost_bbox': 5.0, 'set_cost_giou': 2.0, 'bbox_loss_coef': 5.0, 'giou_loss_coef': 2.0,
        'cls_loss_coef': 1.0, 'focal_alpha': 0.25, 'dataset_file': 'coco',
        'device': str(device), 'num_classes': num_classes,
        'dn_labelbook_size': num_classes
    })

    print(f"[test.py] Cargando pesos desde {weights}...")
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)

    if 'ema_model' in checkpoint:
        print("[test.py] Usando pesos suavizados (EMA)...")
        state_dict = checkpoint['ema_model']
    else:
        state_dict = checkpoint['model']

    # --- AUTO-DETECCIÓN DE ARQUITECTURA DESDE EL CHECKPOINT ---
    if 'transformer.tgt_embed.weight' in state_dict:
        num_queries = state_dict['transformer.tgt_embed.weight'].shape[0]
        base_args['num_queries'] = num_queries

    enc_offset_key = 'transformer.encoder.layers.0.self_attn.sampling_offsets.weight'
    if enc_offset_key in state_dict:
        shape_0 = state_dict[enc_offset_key].shape[0]
        n_heads = base_args.get('nheads', 8)
        num_feature_levels = base_args.get('num_feature_levels', 4)
        enc_n_points = shape_0 // (n_heads * num_feature_levels * 2)
        base_args['enc_n_points'] = enc_n_points

    dec_offset_key = 'transformer.decoder.layers.0.cross_attn.sampling_offsets.weight'
    if dec_offset_key in state_dict:
        shape_0 = state_dict[dec_offset_key].shape[0]
        n_heads = base_args.get('nheads', 8)
        num_feature_levels = base_args.get('num_feature_levels', 4)
        dec_n_points = shape_0 // (n_heads * num_feature_levels * 2)
        base_args['dec_n_points'] = dec_n_points

    model_args = Dict2Obj(base_args)

    print(f"[test.py] Construyendo arquitectura DINO ({variant})...")
    model, criterion, postprocessors = build_model(model_args)
    replace_bn_with_gn(model, BN2GNConfig(policy='on', verbose=0))

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    transform = make_coco_transforms("val")

    return ModelContext(model=model, postprocessors=postprocessors, device=device, transform=transform)


def infer_image(ctx: ModelContext, img_bgr: np.ndarray, conf_thres: float) -> List[Box]:
    import torch
    from util.misc import nested_tensor_from_tensor_list

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    img_pil = Image.fromarray(img_rgb)

    target = {
        "image_id": torch.tensor([0]),
        "annotations": [],
        "orig_size": torch.as_tensor([int(orig_h), int(orig_w)]),
        "size": torch.as_tensor([int(orig_h), int(orig_w)])
    }

    img_tensor, _ = ctx.transform(img_pil, target)
    img_tensor = img_tensor.to(ctx.device)

    nested_tensor = nested_tensor_from_tensor_list([img_tensor])

    with torch.no_grad():
        with MuteStderr():
            outputs = ctx.model(nested_tensor)

    orig_target_sizes = torch.tensor([[orig_h, orig_w]], device=ctx.device)
    results = ctx.postprocessors['bbox'](outputs, orig_target_sizes)[0]

    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()

    boxes_out: List[Box] = []
    for score, label, box in zip(scores, labels, boxes):
        if score >= conf_thres:
            boxes_out.append(
                Box(cls_id=int(label), x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]),
                    conf=float(score)))

    return boxes_out


def box_iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter_area = inter_w * inter_h
    union = (max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)) + (
            max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)) - inter_area
    return float(inter_area / union) if union > 0.0 else 0.0


def evaluate_image(gt_boxes: List[Box], pred_boxes: List[Box], num_classes: int, iou_match: float) -> Tuple[
    Dict[int, ClassStats], Dict[str, float]]:
    stats: Dict[int, ClassStats] = {c: ClassStats() for c in range(num_classes)}
    for box in gt_boxes:
        if 0 <= box.cls_id < num_classes: stats[box.cls_id].n_gt += 1
    for box in pred_boxes:
        if 0 <= box.cls_id < num_classes: stats[box.cls_id].n_pred += 1

    for cls in range(num_classes):
        gt_c = [b for b in gt_boxes if b.cls_id == cls]
        pred_c = sorted([b for b in pred_boxes if b.cls_id == cls], key=lambda b: b.conf, reverse=True)
        used_gt = [False] * len(gt_c)

        for pred in pred_c:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gt_c):
                if used_gt[i]: continue
                iou = box_iou(pred, gt)
                if iou > best_iou: best_iou, best_idx = iou, i
            if best_iou >= iou_match and best_idx >= 0:
                used_gt[best_idx] = True
                stats[cls].tp += 1
                stats[cls].matches += 1
                stats[cls].iou_sum += best_iou
            else:
                stats[cls].fp += 1
        stats[cls].fn += sum(1 for u in used_gt if not u)

    global_metrics: Dict[str, float] = {"P_macro": 0.0, "R_macro": 0.0, "IoU_macro": 0.0}
    p_list, r_list, iou_list = [], [], []
    for cls in range(num_classes):
        s = stats[cls]
        denom_p, denom_r = s.tp + s.fp, s.tp + s.fn
        s.precision = float(s.tp / denom_p) if denom_p > 0 else 0.0
        s.recall = float(s.tp / denom_r) if denom_r > 0 else 0.0
        s.iou_mean = float(s.iou_sum / s.matches) if s.matches > 0 else 0.0
        if s.n_gt > 0 or s.n_pred > 0:
            p_list.append(s.precision)
            r_list.append(s.recall)
            iou_list.append(s.iou_mean)

    if p_list: global_metrics["P_macro"] = float(np.mean(p_list))
    if r_list: global_metrics["R_macro"] = float(np.mean(r_list))
    if iou_list: global_metrics["IoU_macro"] = float(np.mean(iou_list))
    return stats, global_metrics


def draw_boxes(img: np.ndarray, boxes: List[Box], class_names: List[str], colors: List[Tuple[int, int, int]],
               thickness: int = 2, draw_conf: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        if not (0 <= box.cls_id < len(colors)): continue
        color = colors[box.cls_id]
        x1, y1, x2, y2 = int(max(0, min(w - 1, box.x1))), int(max(0, min(h - 1, box.y1))), int(
            max(0, min(w - 1, box.x2))), int(max(0, min(h - 1, box.y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_names[box.cls_id]} {box.conf:.2f}" if draw_conf else class_names[box.cls_id]
        (tw, th), bl = cv2.getTextSize(label, font, 0.5, 1)
        ty1 = y1 - th - bl - 3 if y1 - th - bl - 3 >= 0 else y2 + 3
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 4, ty1 + th + bl + 3), color, -1)
        cv2.putText(img, label, (x1 + 2, ty1 + th + bl), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_legend(split: str, idx: int, num_images: int, model_name: str, class_names: List[str],
                stats: Dict[int, ClassStats], global_metrics: Dict[str, float], show_pred: bool,
                colors_gt: List[Tuple[int, int, int]], colors_pred: List[Tuple[int, int, int]],
                height: int) -> np.ndarray:
    canvas = np.zeros((max(height, 600), 460, 3), dtype=np.uint8)
    canvas[:] = (50, 60, 90)
    font, fs, t, y, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 30, 22

    def put(line: str, color: Tuple[int, int, int] = (255, 255, 255)):
        nonlocal y
        cv2.putText(canvas, line, (10, y), font, fs, color, t, cv2.LINE_AA)
        y += lh

    put(f"Split: {split}", (255, 255, 0))
    put(f"Imagen: {idx + 1}/{num_images}", (255, 255, 0))
    put(f"Modelo: {os.path.basename(model_name)}", (200, 255, 255))
    put("")
    put("Metricas por imagen (macro):", (0, 255, 255))
    put(f"P:   {global_metrics.get('P_macro', 0.0):.3f}")
    put(f"R:   {global_metrics.get('R_macro', 0.0):.3f}")
    put(f"IoU: {global_metrics.get('IoU_macro', 0.0):.3f}")
    put("")
    put("Leyenda bboxes:", (0, 255, 255))
    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_gt[0] if colors_gt else (0, 255, 0), -1)
    cv2.putText(canvas, "GT (etiqueta real)", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA)
    y += lh
    cv2.rectangle(canvas, (10, y - 12), (30, y + 2), colors_pred[0] if colors_pred else (0, 165, 255), -1)
    cv2.putText(canvas, "Prediccion modelo", (40, y), font, fs, (255, 255, 255), t, cv2.LINE_AA)
    y += lh
    put("")
    put("Comandos:", (0, 255, 255))
    put("<- / 'a': imagen anterior")
    put("-> / 'd': imagen siguiente")
    put("'h': mostrar/ocultar pred.")
    put("ESC: salir")
    put("")
    put(f"Predicciones visibles: {'Si' if show_pred else 'No'}")
    put("")
    put("Metricas por clase:", (0, 255, 255))

    for cls_id, s in stats.items():
        if s.n_gt == 0 and s.n_pred == 0: continue
        if y + 36 > canvas.shape[0] - 10: put("...", (200, 200, 200)); break
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        cv2.putText(canvas, f"[{cls_id}] {name}", (10, y), font, 0.45, (255, 255, 0), t, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas,
                    f"GT:{s.n_gt} Pred:{s.n_pred} TP:{s.tp} FP:{s.fp} FN:{s.fn} P:{getattr(s, 'precision', 0.0):.2f} R:{getattr(s, 'recall', 0.0):.2f} IoU:{getattr(s, 'iou_mean', 0.0):.2f}",
                    (10, y), font, 0.45, (220, 220, 220), t, cv2.LINE_AA)
        y += 18
    return canvas


def run_viewer(args: argparse.Namespace) -> None:
    class_names = load_class_names()
    num_classes = len(class_names)
    colors_gt, colors_pred = build_color_palettes(num_classes)
    weights_path = _resolve_path(args.weights, PROJECT_ROOT)
    ctx = load_model(weights_path, args.variant, args.device, num_classes)
    image_paths, labels_dir = load_test_images()

    idx, num_images, show_pred = 0, len(image_paths), True
    while True:
        idx = max(0, min(idx, num_images - 1))
        img_path = image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None: idx += 1; continue

        gt_boxes = load_gt_boxes(labels_dir / f"{img_path.stem}.txt", img.shape)
        pred_boxes = infer_image(ctx, img, args.conf_thres)
        stats, global_metrics = evaluate_image(gt_boxes, pred_boxes, num_classes, args.iou_match)

        img_vis = draw_boxes(img.copy(), gt_boxes, class_names, colors_gt, 2, False)
        if show_pred: img_vis = draw_boxes(img_vis, pred_boxes, class_names, colors_pred, 1, True)

        legend = draw_legend("test", idx, num_images, str(weights_path), class_names, stats, global_metrics, show_pred,
                             colors_gt, colors_pred, img_vis.shape[0])
        if legend.shape[0] != img_vis.shape[0]: legend = cv2.resize(legend, (legend.shape[1], img_vis.shape[0]))

        cv2.imshow("DINO Test Viewer", np.hstack((img_vis, legend)))
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        elif key in (ord("d"), 83):
            idx += 1
        elif key in (ord("a"), 81):
            idx -= 1
        elif key in (ord("h"), ord("p")):
            show_pred = not show_pred
    cv2.destroyAllWindows()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="DINO.test",
        description=(
            "Visor interactivo para testear un modelo DINO entrenado sobre la "
            "partición Dataset/test, mostrando boxes reales vs. predichos y "
            "métricas locales por imagen."
        ),
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Ruta al archivo de pesos .pt del modelo DINO a testear.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="r50_4scale",
        choices=["r50_4scale", "r50_5scale", "swin_l"],
        help="Variante de la arquitectura DINO (ej. r50_4scale, swin_l).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo para inferencia: 'cuda' o 'cpu'.",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Umbral de confianza mínimo para visualizar predicciones.",
    )

    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="IoU mínimo para considerar una predicción como TP frente a un GT.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    print(f"[test.py] Iniciando visor interactivo DINO...")
    args = parse_args(argv)

    # Bootstrap MIOpen sincronizado con valid.yaml
    valid_cfg_path = CONFIGS_ROOT / "valid.yaml"
    if valid_cfg_path.exists():
        valid_cfg = load_yaml(valid_cfg_path)
        mi_cfg = valid_cfg.get('miopen', {})
        user_db_path = mi_cfg.get('user_db_path', None)
    else:
        user_db_path = None

    cfg = MIOpenConfig(
        find_mode="FAST",
        user_db_path=user_db_path,
        disable_cache=True,
        expandable_segments=True,
        verbose=1,
    )
    bootstrap(cfg)

    # Filtros de warnings
    try:
        from engine.warnings import install_global_warning_filters
        install_global_warning_filters()
    except Exception:
        pass

    run_viewer(args)


if __name__ == "__main__":
    main()