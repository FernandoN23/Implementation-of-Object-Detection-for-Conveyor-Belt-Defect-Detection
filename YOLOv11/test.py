# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: YOLOv11/test.py
# Descripción: Script de pruebas tempranas sobre el split de test.
#              Provee un modo interactivo tipo viewer (GT vs pred)
#              y un modo de evaluación simple agregada usando el
#              módulo ``engine.Tester``.
#==============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import sys

import numpy as np

try:  # OpenCV es necesario para el viewer interactivo
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .engine.Tester import Tester, TesterConfig, SampleResult
from .engine import utils as ut


# --------------------------------------------------------------
# Helpers de CLI
# --------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="YOLOv11.test",
        description=(
            "Pruebas tempranas sobre el split de test: "
            "modo interactivo (viewer) y evaluación simple agregada."
        ),
    )

    # Configuración de modelo / pesos
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        default="s",
        help="Variante del modelo YOLOv11 (n, s, m, l, xl, ...). [por defecto: s]",
    )
    parser.add_argument(
        "--model-yaml",
        type=str,
        default="configs/yolo11.yaml",
        help="Ruta al YAML estructural del modelo (relativa al root YOLOv11).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Ruta a pesos .pt para inferencia. Si se omite, se usarán los "
            "pesos iniciales del modelo (útil solo para pruebas sintéticas)."
        ),
    )

    # Configuración de datos / dispositivo
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen cuadrada para inferencia. [640]",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Tamaño de batch para el DataLoader de test. [1]",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de workers para el DataLoader de test. [2]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo a usar: 'auto', 'cpu', 'cuda:0', etc. [auto]",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split a usar (por defecto 'test').",
    )

    # Umbrales de inferencia
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral de confianza mínima para mostrar predicciones. [0.25]",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Umbral IoU para matching simple TP/FP/FN. [0.45]",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Máximo de detecciones por imagen tras filtrado por confianza. [300]",
    )

    # Modo de operación
    parser.add_argument(
        "--mode",
        type=str,
        choices=["view", "eval"],
        default="view",
        help=(
            "Modo de operación: 'view' = viewer interactivo (GT vs pred), "
            "'eval' = evaluación simple agregada sobre todo el split."
        ),
    )

    # CSV interactivo
    parser.add_argument(
        "--no-interactive-csv",
        action="store_true",
        help="Desactiva la escritura de interactive_last.csv al guardar muestras.",
    )

    # Seed opcional para reproducibilidad en el orden del DataLoader
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed global para reproducibilidad básica. [0]",
    )

    return parser


# --------------------------------------------------------------
# Viewer interactivo: dibujo de GT vs pred y tecla 's' para guardar
# --------------------------------------------------------------


def _draw_overlay(sample: SampleResult) -> np.ndarray:
    """Dibuja GT (verde) y predicciones (rojo) sobre la imagen.

    - BBoxes GT en verde con etiqueta "GT: <clase>".
    - BBoxes pred en rojo con etiqueta "P: <clase> conf=xx%".
    - Panel de texto con métricas por imagen si están disponibles.
    """

    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) no está disponible; el modo 'view' requiere opencv-python."
        )

    img = sample.image_bgr.copy()
    h, w = img.shape[:2]

    # Paleta básica
    color_gt = (0, 255, 0)   # verde
    color_pred = (0, 0, 255)  # rojo

    # Dibujar GT
    for box, cls_id in zip(sample.gt_boxes_xyxy, sample.gt_cls):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_gt, 2)
        label = f"GT: {int(cls_id)}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_gt,
            1,
            cv2.LINE_AA,
        )

    # Dibujar predicciones
    for box, cls_id, conf in zip(sample.pred_boxes_xyxy, sample.pred_cls, sample.pred_conf):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_pred, 2)
        label = f"P: {int(cls_id)} {conf*100:.1f}%"
        cv2.putText(
            img,
            label,
            (x1, min(y2 + 15, h - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_pred,
            1,
            cv2.LINE_AA,
        )

    # Panel de texto con métricas en el lateral derecho
    panel_width = max(int(0.30 * w), 240)
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    y = 20
    dy = 18

    def _put(text: str) -> None:
        nonlocal y
        cv2.putText(
            panel,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += dy

    _put("YOLOv11 test viewer")
    _put("-------------------")
    _put(f"split: {sample.meta.get('split', 'test')}")
    _put(f"img: {Path(sample.meta.get('img_path', '')).name}")

    if sample.metrics is not None:
        m = sample.metrics
        _put("")
        _put(f"GT: {m.n_gt}  Pred: {m.n_pred}")
        _put(f"TP: {m.tp}  FP: {m.fp}  FN: {m.fn}")
        _put(f"IoU mean: {m.iou_mean:.3f}")
        _put(f"Prec: {m.precision:.3f}  Rec: {m.recall:.3f}")

    _put("")
    _put("[d] siguiente  [ESC] salir")
    _put("[s] guardar PNG + CSV")

    # Combinar imagen y panel
    combined = np.hstack((img, panel))
    return combined


def _run_view_mode(tester: Tester) -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) no está disponible; instala opencv-python para usar 'view'."
        )

    window_name = "YOLOv11 Test Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    print("[test] Modo 'view' iniciado. Teclas:")
    print("  d   → siguiente imagen")
    print("  s   → guardar imagen actual (PNG + CSV)")
    print("  ESC → salir")

    sample_iter = tester.iter_samples()

    for sample in sample_iter:
        frame = _draw_overlay(sample)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord("d"), ord("D")):
            continue
        elif key in (ord("s"), ord("S")):
            info = tester.save_sample(sample, frame)
            print(f"[test] Guardado: {info['png']}")
        else:
            # Cualquier otra tecla → siguiente imagen
            continue

    cv2.destroyWindow(window_name)


# --------------------------------------------------------------
# Modo evaluación simple agregada
# --------------------------------------------------------------


def _run_eval_mode(tester: Tester) -> None:
    """Evalúa de forma simple todo el split de test.

    Agrega TP/FP/FN y calcula precisión/recall globales y un IoU
    medio aproximado a partir de las métricas por imagen.
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_sum_weighted = 0.0
    iou_weight = 0

    n_images = 0

    for sample in tester.iter_samples():
        n_images += 1
        m = sample.metrics
        if m is None:
            continue

        total_tp += m.tp
        total_fp += m.fp
        total_fn += m.fn

        # Aproximamos el IoU medio global ponderando por número de TP
        if m.tp > 0:
            iou_sum_weighted += m.iou_mean * m.tp
            iou_weight += m.tp

    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    iou_mean_global = iou_sum_weighted / iou_weight if iou_weight > 0 else 0.0

    print("\n[test] Evaluación simple sobre split de test")
    print("-----------------------------------------")
    print(f"Imágenes procesadas: {n_images}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precisión (TP/(TP+FP)): {precision:.4f}")
    print(f"Recall    (TP/(TP+FN)): {recall:.4f}")
    print(f"IoU medio global (aprox): {iou_mean_global:.4f}")


# --------------------------------------------------------------
# main
# --------------------------------------------------------------


def main(argv: Any = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # Seed global básica (orden del DataLoader, etc.)
    ut.seed_everything(int(args.seed))

    print("[test] Dispositivo disponible:", ut.device_info())

    cfg = TesterConfig(
        variant=args.variant,
        model_yaml=args.model_yaml,
        weights=args.weights,
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        device=str(args.device),
        split=str(args.split),
        conf_thres=float(args.conf),
        iou_thres=float(args.iou),
        max_det=int(args.max_det),
        save_interactive_csv=not bool(args.no_interactive_csv),
    )

    tester = Tester(cfg)

    if args.mode == "view":
        _run_view_mode(tester)
    elif args.mode == "eval":
        _run_eval_mode(tester)
    else:  # pragma: no cover
        raise ValueError(f"Modo desconocido: {args.mode}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
