# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: master.py
# Descripción: Script maestro para evaluación comparativa 2x2.
#              Utiliza un Gestor de Contexto Avanzado (ModelEnv).
#              Incluye overlays dinámicos semitransparentes que
#              evitan solaparse con los bounding boxes.
# ==============================================================

import os
import sys
import time
import argparse
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import yaml

# Intentar importar el silenciador de MIOpen
try:
    from engine.bootstrap_miopen import MIOpenConfig, bootstrap, MuteStderr
except Exception:
    MuteStderr = None

# === CONFIGURACIÓN BASE (RUTAS ABSOLUTAS) ===
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"

# NUEVA RUTA: Se guardará en la raíz del proyecto -> /samples/master_grid
SAMPLES_DIR = PROJECT_ROOT / "samples" / "master_grid"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# === PRESETS DE LOS MEJORES MODELOS ===
BEST_MODELS = {
    "YOLO": {
        "path": "YOLO/weights/detect/s/train/s_yolov5_s_best.pt",
        "variant": "s",
        "imgsz": 640
    },
    "SSD": {
        "path": "SSD/weights/detect/ssd512/train/ssd512/best.pth",
        "variant": "ssd512",
        "imgsz": 512
    },
    "DETR": {
        "path": "DETR/weights/r50/detr_r50_belt_batch_4_best.pt",
        "variant": "r50",
        "imgsz": 800
    },
    "DINO": {
        "path": "DINO/weights/r50_4scale/dino_r50_4s_belt_best.pt",
        "variant": "r50_4scale",
        "imgsz": 800
    }
}

# === 1. GESTOR DE AISLAMIENTO DE ENTORNOS ===
CONFLICTING_NAMESPACES = ['models', 'datasets', 'util', 'utils', 'engine', 'ssd', 'data', 'utility']


class ModelEnv:
    def __init__(self, name, root_path, sub_path=None):
        self.name = name
        self.root_path = str(root_path)
        self.sub_path = str(sub_path) if sub_path else None
        self.modules_snapshot = {}
        self.original_cwd = os.getcwd()

    def __enter__(self):
        self.original_cwd = os.getcwd()
        os.chdir(self.root_path)

        to_delete = []
        for k in sys.modules.keys():
            for c in CONFLICTING_NAMESPACES:
                if k == c or k.startswith(c + '.'):
                    to_delete.append(k)
        for k in set(to_delete):
            del sys.modules[k]
        importlib.invalidate_caches()

        sys.path.insert(0, self.root_path)
        if self.sub_path:
            sys.path.insert(1, self.sub_path)

        sys.modules.update(self.modules_snapshot)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.modules_snapshot = {k: v for k, v in sys.modules.items() if
                                 any(k == c or k.startswith(c + '.') for c in CONFLICTING_NAMESPACES)}
        if self.root_path in sys.path: sys.path.remove(self.root_path)
        if self.sub_path and self.sub_path in sys.path: sys.path.remove(self.sub_path)
        os.chdir(self.original_cwd)


def load_test_module_simple(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# === 2. CLASE WRAPPER PARA UNIFICAR INFERENCIAS ===
class ModelWrapper:
    def __init__(self, name, env, mod, ctx, infer_fn, draw_fn, title):
        self.name = name
        self.env = env
        self.mod = mod
        self.ctx = ctx
        self.infer_fn = infer_fn
        self.draw_fn = draw_fn
        self.title = title
        self.last_time_ms = 0.0

    def infer(self, img, conf_thres):
        with self.env:
            t0 = time.perf_counter()
            if MuteStderr is not None:
                with MuteStderr():
                    boxes = self.infer_fn(self.ctx, img, conf_thres)
            else:
                boxes = self.infer_fn(self.ctx, img, conf_thres)
            t1 = time.perf_counter()
            self.last_time_ms = (t1 - t0) * 1000
            return boxes

    def draw(self, img, boxes, class_names, colors):
        return self.draw_fn(img, boxes, class_names, colors, thickness=2, draw_conf=True)


# === 3. FUNCIONES DE CARGA DE DATOS ===
def load_class_names():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names") or data.get("classes")
    if isinstance(names, dict): names = list(names.values())
    return [str(n) for n in names]


def load_test_images():
    images_dir = DATASET_ROOT / "test" / "images"
    labels_dir = DATASET_ROOT / "test" / "labels"
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    return image_paths, labels_dir


def get_gt_boxes(label_file, w, h):
    """Extrae las coordenadas de las cajas reales sin dibujarlas aún."""
    bboxes = []
    if not label_file.exists(): return bboxes
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            bboxes.append((x1, y1, x2, y2, int(cls_id)))
    return bboxes


def draw_gt_boxes(img, gt_boxes_info, class_names):
    color = (0, 0, 160)  # Rojo oscuro para GT
    for x1, y1, x2, y2, cls_id in gt_boxes_info:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"GT: {class_names[cls_id]}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


# === 4. RENDERIZADO DEL GRID Y OVERLAYS DINÁMICOS ===
def add_overlay_dynamic(img, title, time_ms, all_bboxes):
    """Añade título dinámico y latencia fija con fondo semitransparente."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay = img.copy()

    # --- 1. LATENCIA (Fija Abajo-Derecha, Letra más pequeña) ---
    lat_text = f"Latencia: {time_ms:.1f} ms"
    lat_fs = 0.55
    (lw, lh), _ = cv2.getTextSize(lat_text, font, lat_fs, 2)
    lat_x1, lat_y1 = w - lw - 20, h - lh - 15
    lat_x2, lat_y2 = w, h
    cv2.rectangle(overlay, (lat_x1, lat_y1), (lat_x2, lat_y2), (0, 0, 0), -1)

    # --- 2. TÍTULO DINÁMICO (Busca la esquina más vacía) ---
    title_fs = 0.7
    (tw, th), _ = cv2.getTextSize(title, font, title_fs, 2)
    pad = 12
    box_w, box_h = tw + pad * 2, th + pad * 2

    # Candidatos: Arriba-Izq, Arriba-Der, Abajo-Izq
    candidates = [
        (0, 0, box_w, box_h),
        (w - box_w, 0, w, box_h),
        (0, h - box_h, box_w, h)
    ]

    best_box = candidates[0]
    min_overlap = float('inf')

    for cx1, cy1, cx2, cy2 in candidates:
        overlap = 0
        for bx1, by1, bx2, by2 in all_bboxes:
            ix1 = max(cx1, bx1)
            iy1 = max(cy1, by1)
            ix2 = min(cx2, bx2)
            iy2 = min(cy2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            overlap += (iw * ih)

        if overlap < min_overlap:
            min_overlap = overlap
            best_box = (cx1, cy1, cx2, cy2)
        if overlap == 0:
            break  # Esquina perfecta encontrada

    tx1, ty1, tx2, ty2 = best_box
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)

    # Aplicar transparencia (60% negro, 40% imagen original)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Dibujar textos opacos
    cv2.putText(img, title, (tx1 + pad, ty2 - pad + 4), font, title_fs, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, lat_text, (lat_x1 + 10, lat_y2 - 8), font, lat_fs, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def build_2x2_grid(images):
    target_w, target_h = 640, 360
    resized = [cv2.resize(img, (target_w, target_h)) for img in images]
    top_row = np.hstack((resized[0], resized[1]))
    bot_row = np.hstack((resized[2], resized[3]))
    return np.vstack((top_row, bot_row))


def add_mini_legend(grid):
    """Añade una pequeña leyenda semitransparente en la esquina inferior izquierda del grid."""
    img = grid.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    box_w, box_h = 150, 50
    x1, y1 = 10, h - box_h - 10
    x2, y2 = x1 + box_w, y1 + box_h

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # GT
    cv2.rectangle(img, (x1 + 10, y1 + 10), (x1 + 22, y1 + 22), (0, 0, 160), -1)
    cv2.putText(img, "Ground Truth", (x1 + 30, y1 + 20), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Pred
    cv2.rectangle(img, (x1 + 10, y1 + 28), (x1 + 22, y1 + 40), (0, 200, 0), -1)
    cv2.putText(img, "Prediccion", (x1 + 30, y1 + 38), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_master_legend(idx, num_images, conf_thres, show_pred, show_gt, latencies, height):
    width = 350
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 45, 60)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30

    def put(text, color=(255, 255, 255), scale=0.6, thick=1):
        nonlocal y
        cv2.putText(canvas, text, (15, y), font, scale, color, thick, cv2.LINE_AA)
        y += 25

    put("EVALUACION COMPARATIVA", (0, 255, 255), 0.65, 2)
    y += 10
    put(f"Imagen: {idx + 1} / {num_images}", (255, 255, 0))
    put(f"Confianza global: {conf_thres}")
    y += 20

    put("LEYENDA BBOXES:", (0, 255, 255))
    cv2.rectangle(canvas, (15, y - 15), (35, y + 5), (0, 0, 160), -1)
    cv2.putText(canvas, "Ground Truth (Real)", (45, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += 25
    cv2.rectangle(canvas, (15, y - 15), (35, y + 5), (0, 200, 0), -1)
    cv2.putText(canvas, "Prediccion Modelo", (45, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += 30

    put("LATENCIAS (ms):", (0, 255, 255))
    for name, lat in latencies.items():
        color = (0, 255, 0) if lat < 100 else ((0, 255, 255) if lat < 500 else (0, 0, 255))
        put(f"- {name}: {lat:.1f} ms", color)
    y += 30

    put("COMANDOS:", (0, 255, 255))
    put("[d] / [a] : Sig / Ant")
    put(f"[h] : Predicciones ({'ON' if show_pred else 'OFF'})")
    put(f"[g] : Ground Truth ({'ON' if show_gt else 'OFF'})")
    put("[s] : Guardar Sample")
    put("[ESC] : Salir")

    return canvas


# === 5. BUCLE PRINCIPAL ===
def main():
    parser = argparse.ArgumentParser(description="Master Grid 2x2 Evaluator")
    parser.add_argument("--evaluate-best", action="store_true",
                        help="Usa los pesos predefinidos de los 4 mejores modelos.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Umbral de confianza global.")
    parser.add_argument("--device", type=str, default="", help="Dispositivo (ej. 'cuda:0' o 'cpu').")
    args = parser.parse_args()

    if not args.evaluate_best:
        print("[ERROR] Por ahora, master.py requiere el flag --evaluate-best para cargar los presets.")
        sys.exit(1)

    class_names = load_class_names()
    num_classes = len(class_names)
    pred_colors = [(0, 200, 0) for _ in range(num_classes)]

    envs = {
        "YOLO": ModelEnv("YOLO", PROJECT_ROOT / "YOLO", PROJECT_ROOT / "YOLO" / "yolov5"),
        "SSD": ModelEnv("SSD", PROJECT_ROOT / "SSD"),
        "DETR": ModelEnv("DETR", PROJECT_ROOT / "DETR", PROJECT_ROOT / "DETR" / "detr"),
        "DINO": ModelEnv("DINO", PROJECT_ROOT / "DINO", PROJECT_ROOT / "DINO" / "dino"),
    }

    models = []

    print("\n" + "=" * 50)
    print("[INFO] Iniciando carga secuencial de modelos (Aislados)...")
    print("=" * 50)

    # 1. YOLO
    try:
        print("\n[1/4] Cargando YOLOv5...")
        with envs["YOLO"]:
            mod_yolo = load_test_module_simple(PROJECT_ROOT / "YOLO" / "test.py", "yolo_test")
            weights = PROJECT_ROOT / BEST_MODELS["YOLO"]["path"]
            ctx = mod_yolo.load_model(weights, args.device, BEST_MODELS["YOLO"]["imgsz"])

            def infer_yolo(c, i, conf, m=mod_yolo):
                try:
                    return m.infer_image(c, i, conf, iou_nms=0.6, max_det=300)
                except TypeError as e:
                    if "unexpected keyword argument" in str(e) or "positional arguments" in str(e):
                        return m.infer_image(c, i, conf)
                    raise e

            models.append(ModelWrapper("YOLO", envs["YOLO"], mod_yolo, ctx, infer_yolo, mod_yolo.draw_boxes,
                                       f"YOLOv5-{BEST_MODELS['YOLO']['variant']}"))
    except Exception as e:
        print(f"[ERROR] Fallo al cargar YOLO: {e}")
        models.append(None)

    # 2. SSD
    try:
        print("\n[2/4] Cargando SSD...")
        with envs["SSD"]:
            mod_ssd = load_test_module_simple(PROJECT_ROOT / "SSD" / "test.py", "ssd_test")
            mod_ssd._mock_legacy_coco_dependency()
            weights = PROJECT_ROOT / BEST_MODELS["SSD"]["path"]
            ctx = mod_ssd.load_model_ssd(weights, args.device, BEST_MODELS["SSD"]["imgsz"], num_classes + 1)
            models.append(ModelWrapper("SSD", envs["SSD"], mod_ssd, ctx, mod_ssd.infer_image_ssd, mod_ssd.draw_boxes,
                                       f"SSD-{BEST_MODELS['SSD']['variant']}"))
    except Exception as e:
        print(f"[ERROR] Fallo al cargar SSD: {e}")
        models.append(None)

    # 3. DETR
    try:
        print("\n[3/4] Cargando DETR...")
        with envs["DETR"]:
            mod_detr = load_test_module_simple(PROJECT_ROOT / "DETR" / "test.py", "detr_test")
            weights = PROJECT_ROOT / BEST_MODELS["DETR"]["path"]
            ctx = mod_detr.load_model(weights, BEST_MODELS["DETR"]["variant"], args.device, num_classes)
            models.append(ModelWrapper("DETR", envs["DETR"], mod_detr, ctx, mod_detr.infer_image, mod_detr.draw_boxes,
                                       f"DETR-{BEST_MODELS['DETR']['variant']}"))
    except Exception as e:
        print(f"[ERROR] Fallo al cargar DETR: {e}")
        models.append(None)

    # 4. DINO
    try:
        print("\n[4/4] Cargando DINO...")
        with envs["DINO"]:
            mod_dino = load_test_module_simple(PROJECT_ROOT / "DINO" / "test.py", "dino_test")
            weights = PROJECT_ROOT / BEST_MODELS["DINO"]["path"]
            ctx = mod_dino.load_model(weights, BEST_MODELS["DINO"]["variant"], args.device, num_classes)
            models.append(ModelWrapper("DINO", envs["DINO"], mod_dino, ctx, mod_dino.infer_image, mod_dino.draw_boxes,
                                       f"DINO-{BEST_MODELS['DINO']['variant']}"))
    except Exception as e:
        print(f"[ERROR] Fallo al cargar DINO: {e}")
        models.append(None)

    print("\n[INFO] Todos los modelos cargados exitosamente en VRAM.")

    # --- BUCLE DE INFERENCIA ---
    image_paths, labels_dir = load_test_images()
    idx = 0
    num_images = len(image_paths)
    show_pred = True
    show_gt = True

    window_name = "Master Grid 2x2 - Evaluacion Comparativa"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        idx = max(0, min(idx, num_images - 1))
        img_path = image_paths[idx]
        label_file = labels_dir / f"{img_path.stem}.txt"

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            idx += 1
            continue

        # Extraer coordenadas GT una sola vez
        gt_boxes_info = get_gt_boxes(label_file, img_orig.shape[1], img_orig.shape[0])

        quadrants = []
        latencies = {}
        print(f"\n--- Procesando Imagen {idx + 1}/{num_images}: {img_path.name} ---")

        for i, wrapper in enumerate(models):
            if wrapper is None:
                blank = np.zeros_like(img_orig)
                cv2.putText(blank, "MODELO NO DISPONIBLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                quadrants.append(blank)
                continue

            # 1. Inferencia
            boxes = wrapper.infer(img_orig.copy(), args.conf_thres)
            latencies[wrapper.name] = wrapper.last_time_ms
            print(f"[{wrapper.name}] Latencia: {wrapper.last_time_ms:.2f} ms | Detecciones: {len(boxes)}")

            # 2. Recopilar todas las cajas para evitar solapamientos
            all_bboxes = []

            # Si es el cuadrante inferior izquierdo (índice 2), reservamos espacio virtual
            # para que el título no choque con la futura mini-leyenda.
            if i == 2:
                all_bboxes.append((0, img_orig.shape[0] - 60, 160, img_orig.shape[0]))

            if show_gt:
                for x1, y1, x2, y2, _ in gt_boxes_info:
                    all_bboxes.append((x1, y1, x2, y2))
            if show_pred:
                for b in boxes:
                    all_bboxes.append((b.x1, b.y1, b.x2, b.y2))

            # 3. Dibujo
            img_drawn = img_orig.copy()
            if show_gt:
                img_drawn = draw_gt_boxes(img_drawn, gt_boxes_info, class_names)
            if show_pred:
                img_drawn = wrapper.draw(img_drawn, boxes, class_names, pred_colors)

            # 4. Overlay Dinámico
            img_drawn = add_overlay_dynamic(img_drawn, wrapper.title, wrapper.last_time_ms, all_bboxes)
            quadrants.append(img_drawn)

        # Construir Grid y Leyenda
        grid = build_2x2_grid(quadrants)
        legend = draw_master_legend(idx, num_images, args.conf_thres, show_pred, show_gt, latencies, grid.shape[0])

        # Unir Grid y Leyenda horizontalmente para mostrar en pantalla
        combined = np.hstack((grid, legend))

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('d'), 83):
            idx += 1
        elif key in (ord('a'), 81):
            idx -= 1
        elif key == ord('h'):
            show_pred = not show_pred
        elif key == ord('g'):
            show_gt = not show_gt
        elif key == ord('s'):
            # Al guardar, usamos SOLO el grid y le añadimos la mini-leyenda
            sample_to_save = add_mini_legend(grid)
            save_path = SAMPLES_DIR / f"{img_path.stem}_master_grid.jpg"
            cv2.imwrite(str(save_path), sample_to_save)
            print(f"[INFO] Sample limpio guardado en: {save_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()