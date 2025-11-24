# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/engine/Tester.py
# Descripción: Módulo de utilidades de prueba (Tester) para el
#              pipeline de detección en el split de test.
#              Permite recorrer el DataLoader de test, obtener
#              predicciones imagen a imagen y guardar casos
#              interesantes en PNG + CSV interactivo.
#==============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import csv
import time

import numpy as np
import torch

try:  # OpenCV es opcional, pero necesario para guardado de PNG
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

# Importes relativos dentro del paquete YOLOv11
from . import utils as ut

try:  # DataLoader real del proyecto (no sintético)
    from ..utility import data_loader as dl  # type: ignore
except Exception:  # pragma: no cover
    dl = None  # type: ignore

__all__ = ["TesterConfig", "ImageMetrics", "SampleResult", "Tester"]


# --------------------------------------------------------------
# Configuración de Tester
# --------------------------------------------------------------


@dataclass
class TesterConfig:
    """Configuración mínima para pruebas sobre el split de test.

    Esta estructura se espera que sea rellenada desde el CLI de
    ``YOLOv11/test.py``. Mantenerla simple facilita el debug.
    """

    # Variante del modelo (n, s, m, l, xl, ...)
    variant: str = "s"

    # Ruta al YAML estructural del modelo (relativa al root del proyecto
    # o absoluta). Ejemplo típico: "configs/yolo11.yaml".
    model_yaml: str = "configs/yolo11.yaml"

    # Ruta a pesos entrenados (.pt). Si es None, Tester no cargará pesos
    # y se utilizarán los pesos por defecto del modelo.
    weights: Optional[str] = None

    # Parámetros de inferencia / dataloader
    imgsz: int = 640
    batch: int = 1
    workers: int = 2
    device: str = "auto"
    split: str = "test"  # por ahora siempre test, pero lo dejamos genérico

    # Umbrales de inferencia
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 300

    # Control de guardado interactivo
    save_interactive_csv: bool = True


@dataclass
class ImageMetrics:
    """Métricas por imagen para debug visual.

    No pretende reemplazar a las métricas agregadas de ``Validator``;
    solo dar contexto local cuando se inspecciona un caso particular.
    """

    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    iou_mean: float
    iou_median: float
    precision: float
    recall: float


@dataclass
class SampleResult:
    """Resultado de una muestra individual (una imagen) en test.

    ``image_bgr`` es la imagen lista para visualizar con OpenCV.
    Las cajas se guardan en formato XYXY en píxeles.
    """

    image_bgr: np.ndarray
    gt_boxes_xyxy: np.ndarray
    gt_cls: np.ndarray
    pred_boxes_xyxy: np.ndarray
    pred_cls: np.ndarray
    pred_conf: np.ndarray
    meta: Dict[str, Any]
    metrics: Optional[ImageMetrics] = None


# --------------------------------------------------------------
# Clase principal Tester
# --------------------------------------------------------------


class Tester:
    """Tester de YOLOv11 para el split de test.

    Responsabilidades principales:

    - Construir el modelo en modo inferencia (core, eval, device).
    - Construir un DataLoader real para ``split=test`` (sin augment).
    - Iterar sobre el DataLoader produciendo ``SampleResult``.
    - Calcular métricas por imagen (TP/FP/FN, IoU medio, etc.).
    - Guardar interactivamente PNG + CSV al ser invocado desde ``test.py``.

    Notas importantes:
    ------------------
    1) Este módulo no realiza métricas agregadas de tipo mAP; para eso
       se debe usar el ``Validator`` del proyecto.
    2) La decodificación final de predicciones asume que el modelo
       devuelve ya detecciones en formato [x1, y1, x2, y2, conf, cls]
       por imagen. Si tu head tiene otro formato, adapta el método
       ``_model_infer`` a tu caso particular.
    """

    def __init__(self, cfg: TesterConfig) -> None:
        self.cfg = cfg

        # Detectar raíz del proyecto (YOLOv11/) a partir de la ruta de este archivo.
        # __file__ → YOLOv11/engine/Tester.py → parents[1] = YOLOv11
        self.project_root = Path(__file__).resolve().parents[1]

        # Dispositivo
        self.device = ut.select_device(cfg.device)

        # Modelo en modo inferencia
        self.model = self._build_model_for_eval()

        # DataLoader de test + nombres de clases
        self.test_loader, self.class_names = self._build_test_loader()

        # Directorio de métricas/artefactos interactivos
        self.metrics_root = (self.project_root / "metrics" / cfg.variant / "test").resolve()
        self.metrics_root.mkdir(parents=True, exist_ok=True)
        self.csv_last_path = self.metrics_root / "interactive_last.csv"
        self._saved_counter = 0

    # ----------------------------------------------------------
    # Construcción de modelo y DataLoader
    # ----------------------------------------------------------

    def _to_abs(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def _build_model_for_eval(self) -> torch.nn.Module:
        """Construye el modelo YOLOv11 en modo inferencia.

        Se apoya en ``engine.utils.build_model`` pero desenrollando el
        wrapper de entrenamiento si es necesario (atributo ``core``).
        """

        model_yaml = self._to_abs(self.cfg.model_yaml)

        # Construcción del modelo según el parser YAML del proyecto.
        raw_model = ut.build_model(str(model_yaml), variant=self.cfg.variant)
        # Si build_model devuelve un wrapper con atributo ``core`` (caso
        # típico del stack de entrenamiento), para inferencia usamos el
        # modelo subyacente.
        model = getattr(raw_model, "core", raw_model)

        model.to(self.device)
        model.eval()

        # Carga de pesos, si se proporcionaron.
        if self.cfg.weights is not None:
            weights_path = self._to_abs(self.cfg.weights)
            if weights_path.is_file():
                sd = torch.load(weights_path, map_location=self.device)
                self._load_state_dict_flex(model, sd)
            else:
                print(f"[tester] Advertencia: pesos no encontrados en {weights_path}, se usarán pesos por defecto.")
        else:
            print("[tester] Advertencia: cfg.weights es None; se usan pesos iniciales del modelo.")

        return model

    @staticmethod
    def _load_state_dict_flex(model: torch.nn.Module, state: Any) -> None:
        """Carga un state_dict intentando cubrir varios formatos comunes.

        Si el formato no encaja, se deja el modelo sin modificar y se
        imprime un aviso (no se levanta excepción dura para facilitar
        el debug en esta fase temprana de test).
        """

        def _try_load(sd: Dict[str, Any]) -> bool:
            try:
                model.load_state_dict(sd, strict=False)
                return True
            except Exception:
                return False

        ok = False
        if isinstance(state, dict):
            # Caso 1: ckpt["state_dict"]
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                ok = _try_load(state["state_dict"])
            # Caso 2: ckpt["model"] tiene state_dict propio
            if not ok and "model" in state and hasattr(state["model"], "state_dict"):
                ok = _try_load(state["model"].state_dict())
            # Caso 3: el propio dict parece un state_dict
            if not ok:
                ok = _try_load(state)

        if not ok:
            print("[tester] Advertencia: no fue posible cargar los pesos; revise el formato del checkpoint.")

    def _build_test_loader(self) -> Tuple[Iterable[Any], List[str]]:
        """Construye el DataLoader de test y los nombres de clases.

        Se apoya en ``utility/data_loader.py``. Para mantener este
        módulo desacoplado de implementaciones específicas, aquí se
        asume una función ``build_train_bundle`` que reciba al menos:

            build_train_bundle(project_root, split, batch, imgsz, workers, augment)

        y devuelva un objeto con atributos ``loader`` o ``train_loader``
        (DataLoader) y ``names`` o ``class_names`` (lista de str).

        Si tu implementación difiere, adapta este método a tu API real.
        """

        if dl is None:
            raise RuntimeError(
                "utility.data_loader no está disponible. "
                "Asegúrate de que '..utility.data_loader' exista y sea importable."
            )

        try:
            bundle = dl.build_train_bundle(  # type: ignore[attr-defined]
                project_root=str(self.project_root),
                split=self.cfg.split,
                batch=self.cfg.batch,
                imgsz=self.cfg.imgsz,
                workers=self.cfg.workers,
                augment=False,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Error al construir el DataLoader de test mediante "
                "utility.data_loader.build_train_bundle(...). "
                "Adapta Tester._build_test_loader a tu API real."
            ) from exc

        loader = getattr(bundle, "loader", None)
        if loader is None:
            loader = getattr(bundle, "train_loader", None)
        if loader is None:
            raise RuntimeError(
                "El objeto devuelto por build_train_bundle no tiene "
                "atributos 'loader' ni 'train_loader'. Adapta este método."
            )

        names = getattr(bundle, "names", None)
        if names is None:
            names = getattr(bundle, "class_names", None)
        if names is None:
            raise RuntimeError(
                "El objeto devuelto por build_train_bundle no tiene "
                "atributos 'names' ni 'class_names'. Adapta este método."
            )

        names_list = list(names)
        return loader, names_list

    # ----------------------------------------------------------
    # Iteración de muestras (viewer) y métricas por imagen
    # ----------------------------------------------------------

    def iter_samples(self) -> Iterator[SampleResult]:
        """Itera sobre el DataLoader de test produciendo ``SampleResult``.

        Se espera que el DataLoader produzca batches con la firma
        típica YOLO:

            batch["img"]     → tensor [B, C, H, W]
            batch["targets"] → tensor [N, 6] con
                                [img_idx, cls, x, y, w, h] en XYWHN
            batch["paths"]   → lista de rutas de imagen por índice

        Si tu DataLoader usa otras claves, adapta este método.
        """

        for batch in self.test_loader:
            imgs = batch.get("img") if isinstance(batch, dict) else batch[0]
            targets = None
            paths = None

            if isinstance(batch, dict):
                targets = batch.get("targets") or batch.get("labels")
                paths = batch.get("paths") or batch.get("im_file") or batch.get("im_files")
            else:
                # Estructura tipo (imgs, targets, paths)
                if len(batch) > 1:
                    targets = batch[1]
                if len(batch) > 2:
                    paths = batch[2]

            if targets is None:
                raise RuntimeError(
                    "Tester.iter_samples espera un batch con 'targets' "
                    "o 'labels'. Adapta este método a tu DataLoader."
                )

            if paths is None:
                # En el peor caso, usamos índices numéricos como identificador.
                paths = [f"sample_{i}" for i in range(imgs.shape[0])]

            imgs = imgs.to(self.device, non_blocking=True)
            preds_per_img = self._model_infer(imgs)

            # Garantizamos que preds_per_img tenga la misma longitud que el batch
            if len(preds_per_img) != imgs.shape[0]:
                raise RuntimeError(
                    "_model_infer debe devolver una lista con una entrada "
                    "por imagen en el batch. Adáptalo a la salida de tu modelo."
                )

            # targets: [M, 6] → numpy para selección por índice
            if isinstance(targets, torch.Tensor):
                targets_np = targets.detach().cpu().numpy()
            else:
                targets_np = np.asarray(targets)

            for b in range(imgs.shape[0]):
                img_tensor = imgs[b]
                img_path = paths[b] if isinstance(paths, (list, tuple)) else paths

                # Seleccionar GT del batch para la imagen b
                mask = targets_np[:, 0] == float(b)
                gt_for_b = targets_np[mask]

                if gt_for_b.size > 0:
                    gt_cls = gt_for_b[:, 1].astype(np.int64)
                    xywhn = gt_for_b[:, 2:6].astype(np.float32)
                else:
                    gt_cls = np.zeros((0,), dtype=np.int64)
                    xywhn = np.zeros((0, 4), dtype=np.float32)

                _, h, w = img_tensor.shape
                gt_xyxy = self._xywhn_to_xyxy_pixels(xywhn, w=w, h=h)

                # Predicciones para la imagen b
                pred = preds_per_img[b]
                if isinstance(pred, torch.Tensor):
                    pred_np = pred.detach().cpu().numpy()
                else:
                    pred_np = np.asarray(pred)

                if pred_np.size > 0:
                    # Se espera [N, 6] = [x1, y1, x2, y2, conf, cls]
                    boxes_pred = pred_np[:, 0:4].astype(np.float32)
                    conf_pred = pred_np[:, 4].astype(np.float32)
                    cls_pred = pred_np[:, 5].astype(np.int64)

                    # Aplicar umbral de confianza y max_det
                    keep = conf_pred >= float(self.cfg.conf_thres)
                    boxes_pred = boxes_pred[keep]
                    conf_pred = conf_pred[keep]
                    cls_pred = cls_pred[keep]

                    if boxes_pred.shape[0] > self.cfg.max_det:
                        order = np.argsort(-conf_pred)
                        order = order[: self.cfg.max_det]
                        boxes_pred = boxes_pred[order]
                        conf_pred = conf_pred[order]
                        cls_pred = cls_pred[order]
                else:
                    boxes_pred = np.zeros((0, 4), dtype=np.float32)
                    conf_pred = np.zeros((0,), dtype=np.float32)
                    cls_pred = np.zeros((0,), dtype=np.int64)

                # Reconstruir imagen en BGR para visualización
                image_bgr = self._tensor_to_bgr(img_tensor)

                # Métricas por imagen
                metrics = self._compute_image_metrics(
                    gt_boxes_xyxy=gt_xyxy,
                    gt_cls=gt_cls,
                    pred_boxes_xyxy=boxes_pred,
                    pred_cls=cls_pred,
                )

                meta = {
                    "img_path": str(img_path),
                    "width": int(w),
                    "height": int(h),
                    "split": self.cfg.split,
                }

                yield SampleResult(
                    image_bgr=image_bgr,
                    gt_boxes_xyxy=gt_xyxy,
                    gt_cls=gt_cls,
                    pred_boxes_xyxy=boxes_pred,
                    pred_cls=cls_pred,
                    pred_conf=conf_pred,
                    meta=meta,
                    metrics=metrics,
                )

    # ----------------------------------------------------------
    # Guardado interactivo: PNG + CSV
    # ----------------------------------------------------------

    def save_sample(self, sample: SampleResult, image_with_overlay: np.ndarray) -> Dict[str, Optional[Path]]:
        """Guarda una muestra (imagen + métricas) en metrics/<variant>/test/.

        Este método está pensado para ser invocado desde ``test.py``
        cuando el usuario presiona una tecla (por ejemplo 's') en un
        viewer OpenCV. ``image_with_overlay`` debe ser la imagen ya
        decorada con GT + pred + textos.
        """

        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) no está disponible; no se pueden guardar PNG. "
                "Instala opencv-python para habilitar esta funcionalidad."
            )

        self._saved_counter += 1
        fname = f"{self.cfg.variant}_test_{self._saved_counter:04d}.png"
        out_png = self.metrics_root / fname

        cv2.imwrite(str(out_png), image_with_overlay)

        if self.cfg.save_interactive_csv and sample.metrics is not None:
            self._write_last_csv(sample, out_png)

        return {"png": out_png, "csv": self.csv_last_path if self.cfg.save_interactive_csv else None}

    def _write_last_csv(self, sample: SampleResult, out_png: Path) -> None:
        """Escribe (sobrescribe) un CSV con la última muestra guardada.

        El archivo se llama ``interactive_last.csv`` y contiene una sola
        fila con información básica de la muestra + métricas por imagen.
        """

        row: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variant": self.cfg.variant,
            "split": self.cfg.split,
            "img_path": sample.meta.get("img_path", ""),
            "save_png": str(out_png),
        }

        if sample.metrics is not None:
            row.update(asdict(sample.metrics))

        fieldnames = list(row.keys())

        with self.csv_last_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    # ----------------------------------------------------------
    # Helpers internos: conversión de cajas, IoU, etc.
    # ----------------------------------------------------------

    @staticmethod
    def _xywhn_to_xyxy_pixels(xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
        """Convierte cajas XYWH normalizadas a XYXY en píxeles.

        xywhn: [N, 4] con x_center, y_center, width, height en [0, 1].
        """

        if xywhn.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        x_c = xywhn[:, 0] * float(w)
        y_c = xywhn[:, 1] * float(h)
        bw = xywhn[:, 2] * float(w)
        bh = xywhn[:, 3] * float(h)

        x1 = x_c - bw / 2.0
        y1 = y_c - bh / 2.0
        x2 = x_c + bw / 2.0
        y2 = y_c + bh / 2.0

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        return boxes

    @staticmethod
    def _tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
        """Convierte un tensor [C, H, W] en imagen BGR uint8.

        Asume que el tensor está en rango [0, 1] o [0, 255].
        """

        img_np = img.detach().cpu().float().numpy()
        # [C, H, W] → [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))

        # Normalización a [0, 255]
        max_val = float(img_np.max()) if img_np.size > 0 else 1.0
        if max_val <= 1.5:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if cv2 is not None and img_np.shape[2] == 3:
            # Asumimos que es RGB → BGR para OpenCV
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_np

    @staticmethod
    def _box_iou_xyxy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calcula IoU entre dos conjuntos de cajas XYXY en numpy.

        boxes1: [N, 4], boxes2: [M, 4] → IoU [N, M].
        """

        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

        b1 = boxes1.astype(np.float32)
        b2 = boxes2.astype(np.float32)

        # Áreas
        area1 = (b1[:, 2] - b1[:, 0]).clip(min=0) * (b1[:, 3] - b1[:, 1]).clip(min=0)
        area2 = (b2[:, 2] - b2[:, 0]).clip(min=0) * (b2[:, 3] - b2[:, 1]).clip(min=0)

        # Intersección
        N = b1.shape[0]
        M = b2.shape[0]
        iou = np.zeros((N, M), dtype=np.float32)

        for i in range(N):
            x1 = np.maximum(b1[i, 0], b2[:, 0])
            y1 = np.maximum(b1[i, 1], b2[:, 1])
            x2 = np.minimum(b1[i, 2], b2[:, 2])
            y2 = np.minimum(b1[i, 3], b2[:, 3])

            inter_w = (x2 - x1).clip(min=0)
            inter_h = (y2 - y1).clip(min=0)
            inter = inter_w * inter_h

            union = area1[i] + area2 - inter
            union = np.where(union > 0, union, 1e-9)
            iou[i, :] = inter / union

        return iou

    def _compute_image_metrics(
        self,
        *,
        gt_boxes_xyxy: np.ndarray,
        gt_cls: np.ndarray,
        pred_boxes_xyxy: np.ndarray,
        pred_cls: np.ndarray,
    ) -> ImageMetrics:
        """Calcula métricas por imagen para debug.

        Estrategia simple tipo matching greedy por IoU (>= cfg.iou_thres)
        y clase igual. No pretende replicar a la perfección la lógica de
        ``DetMetricsYOLOv11``, sino dar una aproximación razonable.
        """

        n_gt = int(gt_boxes_xyxy.shape[0])
        n_pred = int(pred_boxes_xyxy.shape[0])

        if n_gt == 0 and n_pred == 0:
            return ImageMetrics(
                n_gt=0,
                n_pred=0,
                tp=0,
                fp=0,
                fn=0,
                iou_mean=0.0,
                iou_median=0.0,
                precision=0.0,
                recall=0.0,
            )

        if n_gt == 0:
            # Todo lo predicho es FP
            tp = 0
            fp = n_pred
            fn = 0
            return ImageMetrics(
                n_gt=n_gt,
                n_pred=n_pred,
                tp=tp,
                fp=fp,
                fn=fn,
                iou_mean=0.0,
                iou_median=0.0,
                precision=0.0,
                recall=0.0,
            )

        if n_pred == 0:
            # No se predijo nada, todo GT es FN
            tp = 0
            fp = 0
            fn = n_gt
            return ImageMetrics(
                n_gt=n_gt,
                n_pred=n_pred,
                tp=tp,
                fp=fp,
                fn=fn,
                iou_mean=0.0,
                iou_median=0.0,
                precision=0.0,
                recall=0.0,
            )

        iou_mat = self._box_iou_xyxy(pred_boxes_xyxy, gt_boxes_xyxy)

        # Matching greedy por predicción, orden aleatorio (se podría
        # mejorar usando orden por confianza si se expone aquí).
        matched_gt = np.full((n_gt,), False, dtype=bool)
        ious_matched: List[float] = []

        tp = 0
        fp = 0

        for i in range(n_pred):
            # Para cada predicción, buscamos el GT con mejor IoU
            ious_row = iou_mat[i]
            j = int(np.argmax(ious_row))
            iou_ij = float(ious_row[j])

            if iou_ij >= float(self.cfg.iou_thres) and not matched_gt[j] and int(pred_cls[i]) == int(gt_cls[j]):
                tp += 1
                matched_gt[j] = True
                ious_matched.append(iou_ij)
            else:
                fp += 1

        fn = int((~matched_gt).sum())

        if ious_matched:
            iou_mean = float(np.mean(ious_matched))
            iou_median = float(np.median(ious_matched))
        else:
            iou_mean = 0.0
            iou_median = 0.0

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

        return ImageMetrics(
            n_gt=n_gt,
            n_pred=n_pred,
            tp=tp,
            fp=fp,
            fn=fn,
            iou_mean=iou_mean,
            iou_median=iou_median,
            precision=precision,
            recall=recall,
        )

    # ----------------------------------------------------------
    # Inferencia del modelo (decodificación ligera)
    # ----------------------------------------------------------

    def _model_infer(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """Ejecuta el modelo en modo eval y devuelve detecciones por imagen.

        Este método asume que el modelo (o su wrapper) devuelve ya
        detecciones finales por imagen en alguno de los siguientes
        formatos comunes:

        - Lista de tensores, uno por imagen, cada tensor [Ni, 6]
        - Tensor [B, N, 6] con [x1, y1, x2, y2, conf, cls]
        - Tupla (preds, *resto) donde preds es uno de los anteriores.

        Si tu head devuelve un formato distinto (por ejemplo, mapas de
        características o logits crudos), adapta este método a tu
        pipeline de decodificación (anchors, strides, etc.).
        """

        self.model.eval()
        with torch.no_grad():
            out = self.model(imgs)

        preds = out
        if isinstance(out, (list, tuple)):
            # Caso típico: (preds, loss_dict) o similar
            preds = out[0]

        pred_list: List[torch.Tensor]

        if isinstance(preds, list):
            pred_list = [p.detach().cpu() for p in preds]
        elif isinstance(preds, torch.Tensor):
            if preds.ndim == 3 and preds.size(-1) == 6:
                # [B, N, 6]
                pred_list = [p.detach().cpu() for p in preds]
            else:
                raise RuntimeError(
                    "Tester._model_infer recibió un tensor con forma "
                    f"{tuple(preds.shape)}, pero se esperaba [B, N, 6] "
                    "para detecciones. Adapta este método a tu head."
                )
        else:
            raise RuntimeError(
                "Tester._model_infer no reconoce el tipo de salida del modelo. "
                "Devuelve lista o tensor [B, N, 6] o adapta este método."
            )

        return pred_list
