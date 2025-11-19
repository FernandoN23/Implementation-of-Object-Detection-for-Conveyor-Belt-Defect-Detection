# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: engine/validator.py
# Descripción: Bucle de validación/evaluación para detección. Ejecuta
#              inferencia, NMS, emparejamiento pred–GT y delega el
#              cómputo de métricas (P/R, mAP@0.5, mAP@[.5:.95], F1,
#              matrices de confusión, curvas) a utility/metrics.py.
#              Soporta "slots" de guardado para organización estándar
#              y una interfaz de validación interna (val_int) con
#              integración opcional a TensorBoard/visualización.
#==============================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torchvision.ops import nms as tv_nms  # preferir si está
except Exception:  # pragma: no cover
    tv_nms = None  # type: ignore

# Métricas oficiales del proyecto (estilo YOLOv11)
try:
    from YOLOv11.utility.metrics import DetMetricsYOLOv11  # type: ignore
except Exception:  # pragma: no cover
    DetMetricsYOLOv11 = None  # type: ignore

__all__ = ["ValConfig", "Validator", "validate", "validate_interna"]


# -------------------------------
# Configuración
# -------------------------------

@dataclass
class ValConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.6
    max_det: int = 300
    agnostic_nms: bool = False
    save_json: bool = False
    save_dir: Optional[str] = None
    names: Optional[List[str]] = None  # nombres de clases
    nc: Optional[int] = None           # número de clases
    device: str = "auto"
    imgsz: int = 640
    plots: bool = False
    verbose: int = 1

    # Para cómputo de métricas AP
    map_iou_lo: float = 0.5
    map_iou_hi: float = 0.95
    map_iou_step: float = 0.05

    # Slots de guardado (estructura estándar del proyecto)
    # phase: "train"|"val"|"test"|"val_int" afecta la ruta base de métricas
    phase: str = "val"
    # slot: "epoch", "tests", "final" o personalizado
    slot: str = "epoch"
    # si slot == "tests", se recomienda proveer run_name (p.ej., fecha o hash)
    run_name: Optional[str] = None
    # etiqueta opcional para el paso/época (p.ej., "epoch_012")
    step_tag: Optional[str] = None


# -------------------------------
# Utilidades
# -------------------------------

def _log(msg: str, cfg: Optional[ValConfig] = None, level: int = 1) -> None:
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(f"[validator] {msg}")


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # En Windows ROCm preview, torch.cuda abstrae HIP; mantenemos fallback a CPU
        return torch.device("cpu")
    return torch.device(spec)


def _box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # box1: [N,4], box2: [M,4]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def _nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    # Implementación sencilla por si no existe torchvision.ops.nms
    keep: List[int] = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.numel() == 1:
            break
        ious = _box_iou_xyxy(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if tv_nms is not None:
        return tv_nms(boxes, scores, iou_thres)
    return _nms_pytorch(boxes, scores, iou_thres)


def _sanitize_phase_tag(tag: str) -> str:
    """Normaliza el nombre de fase para uso en nombres de archivos.

    Elimina espacios y paréntesis, y reemplaza "/" por "_" para evitar
    problemas en sistemas de archivos.
    """
    t = tag.strip()
    for ch in " ()":
        t = t.replace(ch, "")
    t = t.replace("/", "_")
    return t or "metrics"


def _targets_global_to_per_image(
    targets: torch.Tensor,
    bs: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Convierte targets globales [N,6] -> lista por imagen [Ni,5] (cls, x, y, w, h).

    Está adaptado al formato producido por `collate_yolo` en `utility/data_loader.py`:
    `targets` con columnas [img_idx, cls, x, y, w, h] y coordenadas normalizadas.
    Si llega un tensor [N,5], se asume que todas las filas pertenecen a la imagen 0
    y se reparte por índice sintético.
    """

    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets debe ser Tensor en _targets_global_to_per_image")

    targets = targets.to(device)

    if targets.numel() == 0:
        # No hay anotaciones: devolvemos B tensores vacíos [0,5]
        return [targets.new_zeros((0, 5)) for _ in range(bs)]

    if targets.ndim != 2 or targets.size(1) not in (5, 6):
        raise ValueError(
            f"Formato de targets no soportado: shape={tuple(targets.shape)}; "
            "se esperaba [N,6] (img_idx,cls,x,y,w,h) o [N,5] (cls,x,y,w,h)."
        )

    if targets.size(1) == 6:
        img_idx = targets[:, 0].to(torch.long)
        rest = targets[:, 1:]  # [N,5]
    else:  # [N,5], asumimos todo img_idx=0
        img_idx = torch.zeros(targets.size(0), dtype=torch.long, device=device)
        rest = targets

    per_image: List[torch.Tensor] = []
    for i in range(bs):
        mask = img_idx == i
        t_i = rest[mask]
        if t_i.numel() == 0:
            t_i = rest.new_zeros((0, 5))
        per_image.append(t_i)
    return per_image


def _targets_to_list_per_image(
    targets_any: Any,
    bs: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Normaliza distintos formatos de `targets` a lista por imagen [Ni,5].

    Casos soportados:
    - Tensor [N,6] con columnas [img_idx, cls, x, y, w, h] (formato data_loader).
    - Tensor [N,5] con columnas [cls, x, y, w, h] (single-image).
    - Lista de tensores [Ni, K] por imagen, donde K==5 o K==6 (en cuyo caso se
      descarta la primera columna como índice).
    """

    # Caso tensor global [N,6] típico de collate_yolo
    if isinstance(targets_any, torch.Tensor):
        return _targets_global_to_per_image(targets_any, bs=bs, device=device)

    # Caso lista: asumimos que ya viene aproximadamente por imagen
    if isinstance(targets_any, (list, tuple)):
        out: List[torch.Tensor] = []
        for i in range(bs):
            if i < len(targets_any) and isinstance(targets_any[i], torch.Tensor):
                t = targets_any[i].to(device)
                if t.numel() == 0:
                    t = t.new_zeros((0, 5))
                elif t.ndim == 2 and t.size(1) == 6:
                    # [img_idx?, cls, x, y, w, h] -> descartamos primera col.
                    t = t[:, 1:]
                elif t.ndim == 2 and t.size(1) == 5:
                    # ya está en [cls, x, y, w, h]
                    pass
                else:
                    raise ValueError(
                        f"Formato de targets[{i}] no soportado: shape={tuple(t.shape)}"
                    )
            else:
                # No hay entrada para esta imagen: tensor vacío por defecto
                t = torch.zeros((0, 5), dtype=torch.float32, device=device)
            out.append(t)
        return out

    # Fallback: sin etiquetas reconocibles -> lista de vacíos
    return [torch.zeros((0, 5), dtype=torch.float32, device=device) for _ in range(bs)]


# -------------------------------
# Validator
# -------------------------------

class Validator:
    """Validador estilo Ultralytics que delega métricas a utility.metrics.DetMetricsYOLOv11.

    Soporta "slots" de guardado compatibles con utility/metrics.py:
      - metrics/<phase>/tests/<run_name>/
      - metrics/<phase>/final/
      - metrics/<phase>/epoch/<step_tag>/
      - o carpeta personalizada (slot)

    Además, la validación interna (val_int) usa siempre la fase lógica
    "val_int" para separar métricas de las de validación clásica.
    """

    def __init__(self, cfg: Optional[ValConfig] = None) -> None:
        self.cfg = cfg or ValConfig()
        self.device = _select_device(self.cfg.device)
        self.seen: int = 0

        # Raíz para métricas:
        # - Si save_dir apunta a una corrida dentro de runs/, subimos hasta la
        #   raíz del proyecto (carpeta que contiene "metrics/").
        # - Si no se entrega save_dir, usamos por defecto YOLOv11/ (raíz).
        root_default = Path(__file__).resolve().parent.parent  # YOLOv11/
        if self.cfg.save_dir:
            base = Path(self.cfg.save_dir).resolve()
            for p in base.parents:
                if p.name == "runs":
                    base = p.parent
                    break
        else:
            base = root_default
        self.base_dir: Optional[Path] = base
        self.save_dir: Optional[Path] = None  # resuelto por slot/step en validate()

    def _resolve_save_dir(self, *, phase: Optional[str] = None, slot: Optional[str] = None,
                          run_name: Optional[str] = None, step_tag: Optional[str] = None) -> Optional[Path]:
        if self.base_dir is None:
            return None
        phase = phase or self.cfg.phase
        slot = (slot or self.cfg.slot).lower()
        root = self.base_dir / "metrics" / phase
        if slot == "tests":
            rn = run_name or self.cfg.run_name or "unnamed"
            out = root / "tests" / rn
        elif slot == "final":
            out = root / "final"
        elif slot == "epoch":
            tag = step_tag or self.cfg.step_tag or "epoch_000"
            out = root / "epoch" / tag
        else:
            tag = step_tag or self.cfg.step_tag or slot
            out = root / tag
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---------- Modelo/inferencia ----------
    @torch.inference_mode()
    def _model_predict(self, model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
        """Devuelve detecciones por imagen en formato [x1,y1,x2,y2,conf,cls].

        Notas
        -----
        - Si el modelo está envuelto en un contenedor con atributo ``core``,
          se usa siempre dicho ``core`` para inferencia.
        - Se intenta desempaquetar salidas típicas de entrenamiento como
          ``(loss, preds)`` y se descartan elementos que no sean tensores.
        """
        dev = images.device

        # Detectar modelo "core" (por ejemplo, cuando se usa un wrapper de train)
        core = getattr(model, "core", model)
        core.eval()

        # Llamada a la API de inferencia
        if hasattr(core, "predict"):
            out = core.predict(images)
        else:
            try:
                out = core(images)
            except Exception:
                # Fallback para modelos que esperan un batch tipo dict
                out = core({"img": images})

        # Desempaquetar patrones comunes de entrenamiento: (loss, preds)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            first, second = out
            if isinstance(first, torch.Tensor) and first.ndim == 0 and (
                isinstance(second, torch.Tensor)
                or isinstance(second, (list, tuple))
            ):
                out = second

        preds: List[torch.Tensor] = []

        # Normalizar a lista de tensores 2D [Ni, >=6]
        if isinstance(out, torch.Tensor):
            if out.ndim == 3 and out.size(-1) >= 6:
                # [B, N, C] -> lista de [N, C]
                preds = [o for o in out]
            elif out.ndim == 2 and out.size(-1) >= 6:
                preds = [out]
            else:
                raise RuntimeError(
                    f"Tensor de predicción con shape no soportado: {tuple(out.shape)}"
                )
        elif isinstance(out, (list, tuple)):
            for o in out:
                if not isinstance(o, torch.Tensor):
                    # Ignorar elementos no tensor (p.ej., dict de escalas)
                    continue
                if o.ndim == 3 and o.size(-1) >= 6:
                    preds.extend([x for x in o])
                elif o.ndim == 2 and o.size(-1) >= 6:
                    preds.append(o)
                else:
                    raise RuntimeError(
                        f"Tensor de predicción con shape no soportado: {tuple(o.shape)}"
                    )
            if not preds:
                raise RuntimeError(
                    "No se encontraron tensores de predicción válidos en la salida del modelo."
                )
        else:
            raise RuntimeError(
                f"Salida de predicción no reconocida por Validator: {type(out)}"
            )

        # Aplicar NMS + truncado por imagen
        results: List[torch.Tensor] = []
        for p in preds:
            if p.size(-1) > 6:
                # En caso de columnas extra, recortamos a las 6 primeras
                p = p[:, :6]
            if p.ndim != 2 or p.size(-1) < 6:
                raise RuntimeError(
                    f"Predicción con forma inesperada tras normalización: {tuple(p.shape)}"
                )

            # Asumimos formato [x1,y1,x2,y2,conf,cls] tras la fase de inferencia
            boxes_xyxy = p[:, :4]
            scores = p[:, 4]
            classes = p[:, 5].to(boxes_xyxy.dtype)

            keep = _nms(boxes_xyxy, scores, self.cfg.iou_thres)
            det = torch.cat(
                [boxes_xyxy[keep], scores[keep, None], classes[keep, None]], 1
            )
            if det.numel() and self.cfg.max_det > 0:
                det = det[: self.cfg.max_det]
            results.append(det.to(dev))
        return results

    # ---------- Loop principal ----------
    @torch.inference_mode()
    def validate(self,
                 model: nn.Module,
                 loader: Iterable[Dict[str, Any]],
                 *,
                 names: Optional[List[str]] = None,
                 phase: Optional[str] = None,
                 slot: Optional[str] = None,
                 run_name: Optional[str] = None,
                 step_tag: Optional[str] = None) -> Dict[str, Any]:
        if DetMetricsYOLOv11 is None:
            raise ImportError("YOLOv11.utility.metrics.DetMetricsYOLOv11 no disponible")

        dev = self.device
        names_list = names or self.cfg.names or []
        names_dict = {i: n for i, n in enumerate(names_list)} if names_list else {}

        # Resolver directorio de guardado con slots
        self.save_dir = self._resolve_save_dir(
            phase=phase, slot=slot, run_name=run_name, step_tag=step_tag
        )

        met = DetMetricsYOLOv11(
            class_names=names_dict if names_dict else None,
            nc=(len(names_list) if names_list else self.cfg.nc),
            save_dir=self.save_dir,
            iou_thresholds=torch.arange(self.cfg.map_iou_lo, self.cfg.map_iou_hi + 1e-9, self.cfg.map_iou_step).tolist(),
        )

        for batch in loader:
            if isinstance(batch, dict) and "img" in batch:
                imgs = batch["img"]
                targets_any = batch.get("targets", [])
            else:
                imgs, targets_any = batch

            imgs = imgs.to(dev, non_blocking=True).float()
            dets = self._model_predict(model, imgs)

            bs, _, H, W = imgs.shape
            img_hw = [(H, W)] * bs

            preds_list = dets
            targets_per_image = _targets_to_list_per_image(targets_any, bs=bs, device=dev)

            met.add_batch(
                preds_list,
                targets_per_image,
                img_hw,
                labels_is_xywhn=True,
                conf_min_for_cm=self.cfg.conf_thres,
                iou_match_for_cm=0.50,
            )
            self.seen += bs

        det_summary, curves = met.finalize()

        map50 = round(det_summary.map50, 6)
        map50_95 = round(det_summary.map50_95, 6)

        metrics = {
            "precision": round(det_summary.precision, 6),
            "recall": round(det_summary.recall, 6),
            "map50": map50,
            "map50-95": map50_95,  # nombre estándar para mAP@[.5:.95]
            "map": map50_95,       # alias para compatibilidad con Trainer._fitness
            "f1": round(
                (2 * det_summary.precision * det_summary.recall)
                / (det_summary.precision + det_summary.recall + 1e-9),
                6,
            ),
            "seen": int(self.seen),
            "fitness": round(0.1 * map50 + 0.9 * map50_95, 6),
        }

        # Guardado JSON si corresponde
        if self.save_dir and self.cfg.save_json:
            out = {"metrics": metrics, "config": asdict(self.cfg)}
            # Nombre de archivo derivado de la fase lógica (train/val/test/val_int)
            phase_tag_src = phase or self.cfg.phase or "metrics"
            phase_tag = _sanitize_phase_tag(str(phase_tag_src))
            p = self.save_dir / f"{phase_tag}_metrics.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            _log(f"Métricas guardadas en {p}", self.cfg, 1)

        return metrics


# -------------------------------
# Funciones de conveniencia (API de módulo)
# -------------------------------


def validate(model: nn.Module,
             loader: Iterable[Dict[str, Any]],
             names: Optional[List[str]] = None,
             *,
             save_dir: Optional[str] = None,
             conf_thres: float = 0.25,
             iou_thres: float = 0.6,
             max_det: int = 300,
             agnostic_nms: bool = False,
             device: str = "auto",
             plots: bool = False,
             save_json: bool = False,
             # --- parámetros de slot ---
             phase: str = "train",
             slot: str = "epoch",
             run_name: Optional[str] = None,
             step_tag: Optional[str] = None) -> Dict[str, Any]:
    """Wrapper simple para validación completa clásica (train/val/test)."""
    cfg = ValConfig(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        device=device,
        plots=plots,
        save_json=save_json,
        save_dir=save_dir,
        names=names,
        phase=phase,
        slot=slot,
        run_name=run_name,
        step_tag=step_tag,
    )
    v = Validator(cfg)
    return v.validate(
        model,
        loader,
        names=names,
        phase=phase,
        slot=slot,
        run_name=run_name,
        step_tag=step_tag,
    )


def validate_interna(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    names: Optional[List[str]] = None,
    *,
    save_dir: Optional[str] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.6,
    device: str = "auto",
    # --- control de iteraciones/partición ---
    epoch: int = 0,
    max_batches: int = 0,
    split: str = "val",
    use_pivots: bool = True,
    # --- TensorBoard / visualización (desde CLI) ---
    tb_enable: bool = False,
    tb_variant: str = "s",
    tb_run_name: str = "run",
    tb_nrow: int = 3,
    tb_conf_thr: float = 0.25,
    tb_topk: int = 5,
    dataset_base: Optional[str] = None,
    # --- slots/estructura de métricas ---
    phase: str = "val_int",
    slot: str = "epoch",
    run_name: Optional[str] = None,
    step_tag: Optional[str] = None,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Validación interna (val_int) para entrenamiento.

    Esta función está pensada para ser invocada desde `train.py` con la
    configuración proveniente del CLI (`--val-int-*`). Implementa:

    - Construcción de `ValConfig` y `Validator`.
    - Limitación opcional de batches (`max_batches`) sobre el loader
      proporcionado (normalmente train_loader adaptado).
    - Llamada al validador clásico (`Validator.validate`).
    - Integración opcional con `utility/visualization.py` para logging
      en TensorBoard de métricas y, a futuro, overlays de pivotes.

    Nota importante
    ---------------
    Independiente del valor que se pase en `phase`, la validación interna
    usa siempre la fase lógica "val_int" para separar sus métricas de las
    de validación clásica (phase="val").
    """

    # Fase lógica fija para val_int
    val_int_phase = "val_int"

    cfg = ValConfig(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
        save_json=True,
        save_dir=save_dir,
        names=names,
        phase=val_int_phase,
        slot=slot,
        run_name=run_name or tb_run_name,
        step_tag=step_tag,
        verbose=verbose,
    )

    # Limitar número de batches si se solicita
    if max_batches and max_batches > 0:
        def _limited_loader(base_loader: Iterable[Dict[str, Any]]):
            for i, batch in enumerate(base_loader):
                if i >= max_batches:
                    break
                yield batch

        eff_loader: Iterable[Dict[str, Any]] = _limited_loader(loader)
    else:
        eff_loader = loader

    v = Validator(cfg)
    metrics = v.validate(
        model,
        eff_loader,
        names=names,
        phase=val_int_phase,
        slot=slot,
        run_name=run_name or tb_run_name,
        step_tag=step_tag,
    )

    # --- Integración opcional con TensorBoard / visualización ---
    if tb_enable:
        try:
            # Importación perezosa para evitar dependencias fuertes si no se usa TB
            from YOLOv11.utility import visualization as viz  # type: ignore
        except Exception as e:  # pragma: no cover
            _log(f"TensorBoard/visualization no disponible: {e}", cfg, 2)
            return metrics

        # 1) Logging de métricas a TensorBoard (fase val_int)
        try:
            if hasattr(viz, "TBConfig") and hasattr(viz, "TBVisualization"):
                tb_cfg = viz.TBConfig(
                    variant=tb_variant,
                    phase=val_int_phase,
                    run_name=tb_run_name,
                )
                tb = viz.TBVisualization(tb_cfg)
                # Se asume que `log_metrics_epoch` acepta (metrics, epoch, phase)
                tb.log_metrics_epoch(metrics, epoch=epoch, phase=val_int_phase)
                tb.close()
        except Exception as e:  # pragma: no cover
            _log(f"TensorBoard val_int deshabilitado: {e}", cfg, 1)

        # 2) Logging de imágenes pivote (GT / GT+Pred)
        if use_pivots and dataset_base is not None:
            try:
                if hasattr(viz, "log_ref_session_epoch"):
                    viz.log_ref_session_epoch(
                        variant=tb_variant,
                        split=split,
                        run_name=tb_run_name,
                        dataset_base=Path(dataset_base),
                        epoch=epoch,
                        pred_json=None,  # reservado para futuras integraciones
                        conf_thr=tb_conf_thr,
                        topk=tb_topk,
                        nrow=tb_nrow,
                    )
            except Exception as e:  # pragma: no cover
                _log(f"Visualización pivotes val_int deshabilitada: {e}", cfg, 1)

    return metrics
