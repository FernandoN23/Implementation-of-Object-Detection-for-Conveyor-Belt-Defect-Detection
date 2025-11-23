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
#              matrices de confusión y curvas) a utility/metrics.py.
#              Implementa una convención de "slots" para el guardado
#              de resultados y una interfaz de validación interna
#              (val_int) con integración opcional a TensorBoard y
#              utilidades de visualización.
#==============================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJ_ROOT = Path(__file__).resolve().parent.parent  # YOLOv11/

import torch
import torch.nn as nn

from YOLOv11.engine.utils import Validator_Utilities as VU  # type: ignore

# Métricas de detección empleadas en el proyecto (implementación YOLOv11)
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

    # Rango de IoU para el cómputo de mAP
    map_iou_lo: float = 0.5
    map_iou_hi: float = 0.95
    map_iou_step: float = 0.05

    # Parámetros de estructura de guardado
    # phase: "train"|"val"|"test"|"val_int" determina la raíz lógica de métricas
    phase: str = "val"
    # slot: "epoch", "tests", "final" o un identificador personalizado
    slot: str = "epoch"
    # si slot == "tests", se recomienda proveer run_name (por ejemplo, fecha o hash)
    run_name: Optional[str] = None
    # etiqueta opcional para el paso/época (por ejemplo, "epoch_012")
    step_tag: Optional[str] = None


# -------------------------------
# Validator
# -------------------------------

class Validator:
    """Validador para detección que delega el cálculo de métricas a
    utility.metrics.DetMetricsYOLOv11.

    La estructura de guardado de métricas sigue la convención:
      - metrics/<phase>/tests/<run_name>/
      - metrics/<phase>/final/
      - metrics/<phase>/epoch/<step_tag>/
      - o un subdirectorio personalizado definido por "slot".

    La validación interna (val_int) utiliza siempre la fase lógica
    "val_int" para mantener separadas sus métricas de las de validación
    clásica.
    """

    def __init__(self, cfg: Optional[ValConfig] = None) -> None:
        self.cfg = cfg or ValConfig()
        self.device = VU.select_device(self.cfg.device)
        self.seen: int = 0

        # Raíz para métricas:
        # - Si save_dir apunta a una corrida dentro de runs/, se asciende
        #   hasta la raíz del proyecto (carpeta que contiene "metrics/") y
        #   se intenta inferir la variante (n/s/m/l/xl) a partir de la ruta.
        # - Si no se entrega save_dir, se utiliza por defecto YOLOv11/.
        root_default = PROJ_ROOT
        self.variant: Optional[str] = None

        if self.cfg.save_dir:
            base = Path(self.cfg.save_dir).resolve()
            parts = base.parts
            if "runs" in parts:
                idx = parts.index("runs")
                # Inferir variante como el segmento inmediatamente posterior a "runs"
                if idx + 1 < len(parts):
                    self.variant = parts[idx + 1]
                # La raíz del proyecto corresponde al directorio padre de "runs"
                base = Path(*parts[:idx])
        else:
            base = root_default

        self.base_dir: Optional[Path] = base
        self.save_dir: Optional[Path] = None  # resuelto mediante _resolve_save_dir()
        # Estado interno para recolección de predicciones asociadas a pivotes
        self._collect_pivots: bool = False
        self._pivot_files: List[str] = []
        self._pivot_conf_thr: float = self.cfg.conf_thres
        self._pivot_topk: int = self.cfg.max_det
        # Predicciones por archivo para overlays de pivotes (completado en validate)
        self._last_preds_by_file: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def _resolve_save_dir(self, *, phase: Optional[str] = None, slot: Optional[str] = None,
                          run_name: Optional[str] = None, step_tag: Optional[str] = None) -> Optional[Path]:
        """Resuelve y crea, en caso necesario, el directorio de salida para métricas.

        La estructura estándar es:
            <project_root>/metrics/<variant>/<phase>/<slot>/

        Si no es posible inferir la variante, se omite dicho nivel para
        mantener compatibilidad con ejecuciones previas.
        """
        if self.base_dir is None:
            return None

        phase = phase or self.cfg.phase
        slot = (slot or self.cfg.slot).lower()

        root = self.base_dir / "metrics"
        variant = getattr(self, "variant", None)
        if variant:
            root = root / str(variant)
        root = root / phase

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

    def _set_pivots_config(self, files: Optional[List[str]], conf_thr: float, topk: int) -> None:
        """Configura el estado interno para la recolección de predicciones en pivotes.

        Si ``files`` es None o una lista vacía, la recolección se desactiva.
        """
        if files:
            self._collect_pivots = True
            self._pivot_files = list(files)
            self._pivot_conf_thr = float(conf_thr)
            self._pivot_topk = int(topk)
        else:
            self._collect_pivots = False
            self._pivot_files = []

    # ---------- Modelo/inferencia ----------
    @torch.inference_mode()
    def _model_predict(self, model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
        """Realiza inferencia y devuelve detecciones por imagen en formato
        [x1, y1, x2, y2, conf, cls].

        Notas
        -----
        - Si el modelo está envuelto en un contenedor con atributo ``core``,
          se utiliza dicho ``core`` para la inferencia.
        - Si el ``core`` implementa un método ``predict`` (por ejemplo, en
          el caso de YOLOv11), se asume que devuelve un tensor [B, N, 6] con
          [x1, y1, x2, y2, conf, cls] previo a NMS.
        - Se contempla el desempaquetado de salidas típicas de entrenamiento
          de la forma ``(loss, preds)`` y se descartan elementos no tensoriales.
        - Si la salida es un ``dict`` y no existe una ruta de decodificación
          explícita, se devuelve una lista de detecciones vacías por imagen
          (stub controlado) para mantener operativo el pipeline de validación
          interna sin levantar excepciones.
        """
        dev = images.device

        # Selección del modelo "core" en caso de existir un wrapper
        core = getattr(model, "core", model)
        core.eval()

        # Llamada a la API de inferencia
        if hasattr(core, "predict"):
            out = core.predict(images)
        else:
            try:
                out = core(images)
            except Exception:
                # Fallback para modelos que esperan un batch de entrada tipo dict
                out = core({"img": images})

        # Desempaquetado de patrones comunes de entrenamiento: (loss, preds)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            first, second = out
            if isinstance(first, torch.Tensor) and first.ndim == 0 and (
                isinstance(second, torch.Tensor)
                or isinstance(second, (list, tuple))
            ):
                out = second

        # Caso especial: salida como dict (típico del forward de entrenamiento)
        if isinstance(out, dict):
            # En este caso se omite la decodificación específica dict -> cajas + puntajes
            # y se devuelve una estructura de detecciones vacía por imagen.
            bs = images.shape[0]
            empty = images.new_zeros((0, 6), device=dev)
            VU.log(
                "Salida de predicción tipo dict sin decodificación registrada; "
                "se generan detecciones vacías controladas por imagen.",
                self.cfg,
                level=2,
            )
            return [empty for _ in range(bs)]

        preds: List[torch.Tensor] = []

        # Normalización a una lista de tensores 2D [Ni, >=6]
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
                    # Elementos no tensoriales (como diccionarios auxiliares) se omiten
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
                f"Tipo de salida de predicción no soportado por Validator: {type(out)}"
            )

        # Aplicar NMS, filtro de confianza y truncado por imagen
        results: List[torch.Tensor] = []
        for p in preds:
            if p.size(-1) > 6:
                # Si existen columnas adicionales, se conservan solo las 6 primeras
                p = p[:, :6]
            if p.ndim != 2 or p.size(-1) < 6:
                raise RuntimeError(
                    f"Predicción con forma inesperada tras normalización: {tuple(p.shape)}"
                )

            # Se asume formato [x1, y1, x2, y2, conf, cls] tras la inferencia
            boxes_xyxy = p[:, :4]
            scores = p[:, 4]
            classes = p[:, 5].to(boxes_xyxy.dtype)

            keep = VU.nms(boxes_xyxy, scores, self.cfg.iou_thres)
            det = torch.cat(
                [boxes_xyxy[keep], scores[keep, None], classes[keep, None]], 1
            )

            # Filtro por confianza mínima (consistente con conf_thres de ValConfig)
            if det.numel() and self.cfg.conf_thres > 0.0:
                conf_mask = det[:, 4] >= self.cfg.conf_thres
                det = det[conf_mask]

            if det.numel() and self.cfg.max_det > 0:
                det = det[: self.cfg.max_det]
            results.append(det.to(dev))
        return results

    # ---------- Bucle principal ----------
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
        """Ejecuta la validación sobre un conjunto de datos con un modelo dado.

        Esta función recorre el loader, agrega predicciones y anotaciones a la
        estructura de métricas, y devuelve un diccionario con indicadores
        agregados (precisión, recall, mAP, F1 y fitness, entre otros).
        """
        if DetMetricsYOLOv11 is None:
            raise ImportError("YOLOv11.utility.metrics.DetMetricsYOLOv11 no disponible")

        dev = self.device
        names_list = names or self.cfg.names or []
        names_dict = {i: n for i, n in enumerate(names_list)} if names_list else {}

        # Resolver directorio de guardado en función de phase/slot
        self.save_dir = self._resolve_save_dir(
            phase=phase, slot=slot, run_name=run_name, step_tag=step_tag
        )

        met = DetMetricsYOLOv11(
            class_names=names_dict if names_dict else None,
            nc=(len(names_list) if names_list else self.cfg.nc),
            save_dir=self.save_dir,
            iou_thresholds=torch.arange(self.cfg.map_iou_lo, self.cfg.map_iou_hi + 1e-9, self.cfg.map_iou_step).tolist(),
        )

        # Configuración opcional para recolección de predicciones por archivo
        collect_pivots: bool = bool(getattr(self, "_collect_pivots", False))
        pivot_files = set(getattr(self, "_pivot_files", []) or [])
        pivot_conf_thr: float = float(getattr(self, "_pivot_conf_thr", self.cfg.conf_thres))
        pivot_topk: int = int(getattr(self, "_pivot_topk", self.cfg.max_det))
        preds_by_file: Dict[str, List[Dict[str, Any]]] = (
            {} if collect_pivots and pivot_files else {}
        )

        for batch in loader:
            if isinstance(batch, dict) and "img" in batch:
                imgs = batch["img"]
                targets_any = batch.get("targets", [])
                paths_any = batch.get("im_file") or batch.get("im_files") or batch.get("paths")
            else:
                imgs, targets_any = batch
                paths_any = None

            imgs = imgs.to(dev, non_blocking=True).float()
            dets = self._model_predict(model, imgs)

            bs, _, H, W = imgs.shape
            img_hw = [(H, W)] * bs

            preds_list = dets
            targets_per_image = VU.targets_to_list_per_image(targets_any, bs=bs, device=dev)

            met.add_batch(
                preds_list,
                targets_per_image,
                img_hw,
                labels_is_xywhn=True,
                conf_min_for_cm=self.cfg.conf_thres,
                iou_match_for_cm=0.50,
            )

            # Recolección opcional de predicciones asociadas a imágenes pivote
            if collect_pivots and paths_any is not None and pivot_files:
                try:
                    paths_seq = list(paths_any)
                except TypeError:
                    # Si paths_any no es iterable, se replica el mismo valor
                    paths_seq = [paths_any] * bs

                for i in range(bs):
                    if i >= len(preds_list):
                        break
                    fname = Path(str(paths_seq[i])).name
                    if fname not in pivot_files:
                        continue

                    det_i = preds_list[i]
                    if det_i is None or det_i.numel() == 0:
                        continue
                    if det_i.ndim != 2 or det_i.size(-1) < 6:
                        continue

                    H_i, W_i = img_hw[i]
                    boxes_xyxy = det_i[:, :4]
                    confs = det_i[:, 4]
                    clss = det_i[:, 5]

                    # Filtro de confianza específico para overlays de pivotes
                    if pivot_conf_thr > 0.0:
                        mask = confs >= pivot_conf_thr
                        boxes_sel = boxes_xyxy[mask]
                        confs_sel = confs[mask]
                        clss_sel = clss[mask]
                    else:
                        boxes_sel = boxes_xyxy
                        confs_sel = confs
                        clss_sel = clss

                    if boxes_sel.numel() == 0:
                        continue

                    # Limitación a top-k predicciones por imagen
                    if pivot_topk > 0 and boxes_sel.size(0) > pivot_topk:
                        boxes_sel = boxes_sel[:pivot_topk]
                        confs_sel = confs_sel[:pivot_topk]
                        clss_sel = clss_sel[:pivot_topk]

                    x1 = boxes_sel[:, 0]
                    y1 = boxes_sel[:, 1]
                    x2 = boxes_sel[:, 2]
                    y2 = boxes_sel[:, 3]

                    Wf = float(W_i) if W_i else 1.0
                    Hf = float(H_i) if H_i else 1.0

                    cx = ((x1 + x2) / 2.0) / Wf
                    cy = ((y1 + y2) / 2.0) / Hf
                    w = (x2 - x1) / Wf
                    h = (y2 - y1) / Hf

                    entries = preds_by_file.setdefault(fname, [])
                    for j in range(boxes_sel.size(0)):
                        entries.append(
                            {
                                "bbox_xywh": [
                                    float(cx[j]),
                                    float(cy[j]),
                                    float(w[j]),
                                    float(h[j]),
                                ],
                                "conf": float(confs_sel[j]),
                                "cls": int(clss_sel[j]),
                            }
                        )

            self.seen += bs

        # Exponer predicciones por archivo cuando existan pivotes configurados
        if collect_pivots and preds_by_file:
            self._last_preds_by_file = preds_by_file
        else:
            self._last_preds_by_file = None

        det_summary, curves = met.finalize()

        map50 = round(det_summary.map50, 6)
        map50_95 = round(det_summary.map50_95, 6)

        metrics = {
            "precision": round(det_summary.precision, 6),
            "recall": round(det_summary.recall, 6),
            "map50": map50,
            "map50-95": map50_95,  # etiqueta estándar para mAP@[.5:.95]
            "map": map50_95,       # alias para compatibilidad con Trainer._fitness
            "f1": round(
                (2 * det_summary.precision * det_summary.recall)
                / (det_summary.precision + det_summary.recall + 1e-9),
                6,
            ),
            "seen": int(self.seen),
            "fitness": round(0.1 * map50 + 0.9 * map50_95, 6),
        }

        # Guardado en JSON si corresponde
        if self.save_dir and self.cfg.save_json:
            out = {"metrics": metrics, "config": asdict(self.cfg)}
            # Nombre de archivo derivado de la fase lógica (train/val/test/val_int)
            phase_tag_src = phase or self.cfg.phase or "metrics"
            phase_tag = VU.sanitize_phase_tag(str(phase_tag_src))
            p = self.save_dir / f"{phase_tag}_metrics.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            # Mensaje de registro con ruta relativa al proyecto cuando es posible
            try:
                rel = p.relative_to(PROJ_ROOT)
                VU.log(f"metrics -> YOLOv11/{rel.as_posix()}", self.cfg, 1)
            except Exception:
                VU.log(f"Métricas guardadas en {p}", self.cfg, 1)

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
             # --- parámetros de estructura de guardado ---
             phase: str = "train",
             slot: str = "epoch",
             run_name: Optional[str] = None,
             step_tag: Optional[str] = None) -> Dict[str, Any]:
    """Wrapper para validación clásica (train/val/test).

    Construye la configuración, instancia el validador y ejecuta el
    bucle de evaluación en un conjunto de datos.•
    """
    # Inferir nc desde los nombres, cuando estén disponibles, para
    # registrar esta información en las métricas serializadas.
    nc: Optional[int] = len(names) if names else None

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
        nc=nc,
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
    # --- control de iteraciones y partición ---
    epoch: int = 0,
    max_batches: int = 0,
    split: str = "val",
    use_pivots: bool = True,
    # --- parámetros de TensorBoard y visualización ---
    tb_enable: bool = False,
    tb_variant: str = "s",
    tb_run_name: str = "run",
    tb_nrow: int = 3,
    tb_conf_thr: float = 0.25,
    tb_topk: int = 5,
    dataset_base: Optional[str] = None,
    # --- parámetros de estructura de métricas ---
    phase: str = "val_int",
    slot: str = "epoch",
    run_name: Optional[str] = None,
    step_tag: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validación interna (val_int) utilizada durante el entrenamiento.

    Esta función está pensada para ser invocada desde ``train.py`` con la
    configuración derivada del CLI (parámetros ``--val-int-*``). Realiza:

    - Construcción de ``ValConfig`` y ``Validator``.
    - Limitación opcional del número de batches a procesar.
    - Ejecución del validador clásico (``Validator.validate``).
    - Integración opcional con ``utility/visualization.py`` para registro en
      TensorBoard y generación de overlays de imágenes pivote en disco.

    La fase lógica utilizada para almacenar las métricas de validación
    interna es siempre "val_int", independiente del valor recibido en
    ``phase``.
    """

    # Fase lógica fija para val_int
    val_int_phase = "val_int"

    # Inferir número de clases (nc) a partir del nombre de clases o del modelo
    nc: Optional[int] = None
    if names:
        nc = len(names)
    else:
        core = getattr(model, "core", model)
        nc_attr = getattr(core, "nc", None)
        if nc_attr is not None:
            try:
                nc = int(nc_attr)
            except (TypeError, ValueError):  # pragma: no cover
                nc = None

    cfg = ValConfig(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
        save_json=True,
        save_dir=save_dir,
        names=names,
        nc=nc,
        phase=val_int_phase,
        slot=slot,
        run_name=run_name or tb_run_name,
        step_tag=step_tag,
        verbose=verbose,
    )

    # Limitación opcional del número de batches
    if max_batches and max_batches > 0:
        def _limited_loader(base_loader: Iterable[Dict[str, Any]]):
            for i, batch in enumerate(base_loader):
                if i >= max_batches:
                    break
                yield batch

        eff_loader: Iterable[Dict[str, Any]] = _limited_loader(loader)
    else:
        eff_loader = loader

    # Visualización y overlays: obtención de lista de pivotes desde visualization.py
    viz = None
    pivot_files: Optional[List[str]] = None
    if use_pivots:
        try:
            from YOLOv11.utility import visualization as viz_mod  # type: ignore

            viz = viz_mod
            if split == "train":
                pivot_files = list(getattr(viz_mod, "TRAIN_PIVOT_IMAGES", []))
            else:
                pivot_files = list(getattr(viz_mod, "VALID_PIVOT_IMAGES", []))
            if not pivot_files:
                VU.log("No se encontraron imágenes pivote definidas en visualization.py", cfg, 1)
        except Exception as e:  # pragma: no cover
            VU.log(f"Visualización/overlays no disponible: {e}", cfg, 1)
            viz = None
            pivot_files = None
            use_pivots = False

    v = Validator(cfg)

    # Configuración de recolección de predicciones por archivo para pivotes
    v._set_pivots_config(
        files=pivot_files if (use_pivots and pivot_files) else None,
        conf_thr=tb_conf_thr,
        topk=tb_topk,
    )

    metrics = v.validate(
        model,
        eff_loader,
        names=names,
        phase=val_int_phase,
        slot=slot,
        run_name=run_name or tb_run_name,
        step_tag=step_tag,
    )

    # Payload con métricas agregadas y configuración de validación
    payload: Dict[str, Any] = {"metrics": metrics, "config": asdict(cfg)}

    # Exponer ruta de métricas asociada a esta validación interna
    metrics_dir_str: Optional[str] = None
    try:
        if getattr(v, "save_dir", None) is not None:
            metrics_dir_str = str(v.save_dir)
    except Exception:
        metrics_dir_str = None
    if metrics_dir_str:
        payload["metrics_dir"] = metrics_dir_str

    # --- Overlays en disco (PNG) y JSON de predicciones por archivo ---
    pred_json_path: Optional[Path] = None
    overlays_path: Optional[Path] = None

    have_preds = bool(
        use_pivots
        and v.save_dir is not None
        and getattr(v, "_last_preds_by_file", None)
    )

    if have_preds:
        preds_by_file = getattr(v, "_last_preds_by_file", None)
        if isinstance(preds_by_file, dict) and preds_by_file:
            # 1) Guardar JSON con predicciones asociadas a imágenes pivote
            pred_json_path = v.save_dir / "val_int_pivots_pred.json"
            try:
                with open(pred_json_path, "w", encoding="utf-8") as f:
                    json.dump(preds_by_file, f, ensure_ascii=False, indent=2)
                try:
                    rel_json = pred_json_path.relative_to(PROJ_ROOT)
                    VU.log(f"pivots -> YOLOv11/{rel_json.as_posix()}", cfg, 2)
                except Exception:
                    VU.log(f"Predicciones de pivotes guardadas en {pred_json_path}", cfg, 2)
            except Exception as e:  # pragma: no cover
                VU.log(f"No se pudo guardar JSON de pivotes: {e}", cfg, 1)
                pred_json_path = None

            # 2) Generar PNG de overlays de pivotes bajo metrics/<variant>/val_int/epoch/...
            if viz is not None and dataset_base is not None:
                try:
                    overlays_path = v.save_dir / "overlays_pivotes.png"
                    viz.save_reference_overlays_png(
                        out_path=overlays_path,
                        split=split,
                        dataset_base=Path(dataset_base),
                        preds_by_file=preds_by_file,
                        conf_thr=tb_conf_thr,
                        topk=tb_topk,
                        nrow=tb_nrow,
                        size=(640, 640),
                    )
                    try:
                        rel_png = overlays_path.relative_to(PROJ_ROOT)
                        VU.log(f"overlays -> YOLOv11/{rel_png.as_posix()}", cfg, 1)
                    except Exception:
                        VU.log(f"Overlays de pivotes guardados en {overlays_path}", cfg, 1)
                except Exception as e:  # pragma: no cover
                    VU.log(f"No se pudieron generar overlays PNG de pivotes: {e}", cfg, 1)
    elif use_pivots and verbose >= 2:
        # En modo detallado, registrar explícitamente la ausencia de predicciones sobre pivotes
        VU.log(
            "pivots (debug) -> sin predicciones sobre pivotes; no se generó PNG",
            cfg,
            2,
        )

    # --- Integración opcional con TensorBoard y utilidades de visualización ---
    if tb_enable:
        try:
            if viz is None:
                from YOLOv11.utility import visualization as viz_mod  # type: ignore
                viz = viz_mod

            # Registro opcional de métricas en TensorBoard cuando el helper está disponible
            try:
                if hasattr(viz, "log_metrics_epoch_to_tb"):
                    viz.log_metrics_epoch_to_tb(
                        variant=tb_variant,
                        run_name=tb_run_name,
                        epoch=epoch,
                        metrics=metrics,
                        phase=val_int_phase,
                    )
            except Exception as e_tb:  # pragma: no cover
                VU.log(f"No se pudieron registrar métricas en TensorBoard: {e_tb}", cfg, 2)
        except Exception as e:  # pragma: no cover
            VU.log(f"TensorBoard/visualization no disponible: {e}", cfg, 2)

    # Adjuntar información sobre artefactos generados (si aplica)
    extra_paths: Dict[str, Optional[str]] = {}
    if pred_json_path is not None:
        extra_paths["pivots_json"] = str(pred_json_path)
    if overlays_path is not None:
        extra_paths["overlays_png"] = str(overlays_path)
    if extra_paths:
        payload["artifacts"] = extra_paths

    return metrics, payload
