"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: valid.py
=============================================================

Validación externa del modelo YOLOv11.
Evalúa métricas y curvas de pérdida con propagación opcional,
manteniendo consistencia total con train.py y TensorBoard.
=============================================================
"""
import os, sys, torch, subprocess, keyboard, socket, psutil
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.yolo11 import YOLOv11
from models.parser_yaml import ModelParser
from utility.data_loader import create_dataloader
from utility.logger import get_logger
from utility.metrics import evaluate_model, measure_fps
from utility.weights import load_checkpoint
from utility.losses import YoloLoss
from utility.visualization import TensorboardVisualizer


# =============================================================
# CONFIGURACIÓN DE ENTORNO Y LOGS
# =============================================================
def setup_environment(model_variant="n"):
    base_dir = "YOLOv11"
    variant = model_variant.lower()
    os.makedirs(os.path.join(base_dir, "logs", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "metrics", variant, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs", variant, "valid"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(log_dir=f"{base_dir}/logs/{variant}/valid", name=f"valid_yolo11_{variant}")
    tb = TensorboardVisualizer(log_dir=f"{base_dir}/runs/{variant}/valid")
    logger.info(f"📦 Dispositivo activo: {device}")
    return device, logger, tb


# =============================================================
# CARGA DE CONFIGS Y MODELO
# =============================================================
def load_model_and_configs(variant_override=None):
    valid_cfg = OmegaConf.load("YOLOv11/configs/valid.yaml")
    variants_cfg = OmegaConf.load("YOLOv11/configs/model_variants.yaml")
    variant_name = variant_override or valid_cfg.get("model_variant", "n")

    if variant_name not in variants_cfg.variants:
        raise ValueError(f"⚠️ Variante '{variant_name}' no existe en model_variants.yaml")

    variant_params = variants_cfg.variants[variant_name]
    print(f"🧩 Configuración YOLOv11-{variant_name.upper()}: {variant_params}")

    model_cfg_path = "YOLOv11/configs/yolo11.yaml"
    parser = ModelParser(model_cfg_path)
    model_cfg = parser.parse_model_config()
    num_classes = model_cfg.get("nc", 1)
    model = YOLOv11(cfg_path=model_cfg_path, num_classes=num_classes)
    return model, valid_cfg, variant_name


# =============================================================
# LOOP DE VALIDACIÓN
# =============================================================
def validate_one_epoch(model, dataloader, device, logger, tb, model_variant, propagate=False):
    """
    Valida una época del modelo YOLOv11 con decodificación explícita multiescala.
    Convierte las salidas del modelo (P3, N4, N5) en coordenadas absolutas (xyxy),
    y calcula las métricas globales y por clase.
    """
    model.eval()
    criterion = YoloLoss()
    total_loss, loss_values = 0.0, []
    all_preds, all_targets = [], []

    torch.set_grad_enabled(propagate)
    pbar = tqdm(dataloader, desc="Validando", leave=False)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        preds = model(images)  # lista [P3, N4, N5]

        # === cálculo de pérdida (solo tracking, sin backprop) ===
        loss, loss_items = criterion(preds, labels)
        total_loss += float(loss.item())
        loss_values.append(float(loss.item()))
        tb.log_metrics({"train_loss": loss_items["total_loss"]}, step, phase="valid")
        pbar.set_postfix(loss=loss_items["total_loss"])

        # === Decodificación explícita multiescala ===
        with torch.no_grad():
            decoded = []
            strides = [8, 16, 32]  # escalas P3, N4, N5
            for i, p in enumerate(preds):
                b, c, h, w = p.shape
                stride = strides[i]
                p = p.view(b, c, h, w)
                # Aplicar activaciones
                xy = torch.sigmoid(p[:, 0:2, :, :])       # centro x, y
                wh = torch.exp(p[:, 2:4, :, :]) * stride  # ancho, alto absolutos
                conf = torch.sigmoid(p[:, 4:5, :, :])     # confianza
                cls_logits = p[:, 5:, :, :]               # clases (logits)

                # Crear grilla espacial
                yv, xv = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing="ij"
                )
                grid = torch.stack((xv, yv), 2).view(1, h, w, 2).permute(0, 3, 1, 2)

                # Desplazar y escalar coordenadas a píxeles absolutos
                x = (xy[:, 0:1, :, :] + grid[:, 0:1, :, :]) * stride
                y = (xy[:, 1:2, :, :] + grid[:, 1:2, :, :]) * stride
                w_abs, h_abs = wh[:, 0:1, :, :], wh[:, 1:2, :, :]
                xywh = torch.cat([x, y, w_abs, h_abs], dim=1)

                # Concatenar [x,y,w,h,conf,cls_logits]
                det = torch.cat([xywh, conf, cls_logits], dim=1)
                det = det.view(b, det.shape[1], -1).permute(0, 2, 1)
                decoded.append(det)

            # Combinar las tres escalas
            P = torch.cat(decoded, dim=1)

            # === Extraer componentes ===
            box_xywh = P[..., :4]
            obj_conf = P[..., 4]
            cls_logits = P[..., 5:]

            # === Convertir xywh -> xyxy ===
            xyxy = box_xywh.clone()
            xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2  # x1
            xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2  # y1
            xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2  # x2
            xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2  # y2

            conf_thr = 0.25
            B = P.shape[0]
            for b in range(B):
                keep = obj_conf[b] > conf_thr
                if keep.any():
                    det_b = torch.cat(
                        [xyxy[b][keep], obj_conf[b][keep].unsqueeze(-1), cls_logits[b][keep]],
                        dim=-1
                    ).cpu()
                else:
                    det_b = torch.empty((0, 5 + cls_logits.shape[-1]))

                # === Preparar ground truth ===
                t = labels[b]
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    cls_id = t[:, 0:1]
                    xywh = t[:, 1:5]
                    t_xyxy = xywh.clone()
                    t_xyxy[:, 0] = xywh[:, 0] * 640 - (xywh[:, 2] * 640) / 2
                    t_xyxy[:, 1] = xywh[:, 1] * 640 - (xywh[:, 3] * 640) / 2
                    t_xyxy[:, 2] = xywh[:, 0] * 640 + (xywh[:, 2] * 640) / 2
                    t_xyxy[:, 3] = xywh[:, 1] * 640 + (xywh[:, 3] * 640) / 2
                    gt_b = torch.cat([t_xyxy, torch.ones((t_xyxy.size(0), 1)), cls_id], dim=1).cpu()
                else:
                    gt_b = torch.empty((0, 6))

                all_preds.append(det_b.numpy())
                all_targets.append(gt_b.numpy())

    torch.set_grad_enabled(False)
    avg_loss = total_loss / max(len(dataloader), 1)

    # === Calcular métricas ===
    try:
        metrics = evaluate_model(
            all_preds, all_targets,
            save_results=True, model_variant=model_variant, phase="valid"
        )
        if isinstance(metrics, tuple):
            global_metrics, per_class_metrics = metrics
        else:
            global_metrics, per_class_metrics = metrics, {}
    except Exception as e:
        logger.warning(f"⚠️ Error al calcular métricas: {e}")
        global_metrics, per_class_metrics = (
            {"mAP": 0.0, "Precision": 0.0, "Recall": 0.0, "IoU": 0.0},
            {},
        )

    # === Medición de FPS y log ===
    fps = measure_fps(model, torch.randn(1, 3, 640, 640), device=device)
    logger.info(f"📉 Loss promedio validación: {avg_loss:.4f} | ⚡ FPS: {fps:.2f}")

    return {"global": global_metrics, "per_class": per_class_metrics}, avg_loss, loss_values

# =============================================================
# INICIALIZACIÓN AUTOMÁTICA DE TENSORBOARD
# =============================================================
def start_tensorboard_if_needed(log_dir, variant):
    def find_free_port(start=6006, end=6015):
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) != 0:
                    return port
        return None

    def is_running(logdir):
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                if "tensorboard" in proc.info["name"].lower():
                    if any(logdir in arg for arg in proc.info["cmdline"]):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    if not is_running(log_dir):
        port = find_free_port()
        if port:
            subprocess.Popen(
                ["tensorboard", "--logdir", log_dir, "--port", str(port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"🔗 TensorBoard iniciado: http://localhost:{port}")
        else:
            print("⚠️ No hay puertos libres para TensorBoard (6006–6015).")
    else:
        print(f"ℹ️ TensorBoard ya activo para {variant}.")


# =============================================================
# MAIN
# =============================================================
def main():
    print("====================================================")
    print("🧪 Validación YOLOv11")
    print("====================================================")
    model_variant = input("👉 Variante a validar [n/s/m/l/x]: ").strip().lower()
    propagate = input("¿Permitir propagación de gradientes? (s/n): ").strip().lower() == "s"

    device, logger, tb = setup_environment(model_variant)
    model, valid_cfg, _ = load_model_and_configs(variant_override=model_variant)
    model.to(device)

    # Cargar último checkpoint
    ckpt_dir = f"YOLOv11/weights/{model_variant}/train"
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"⚠️ Carpeta de checkpoints no encontrada: {ckpt_dir}")
    load_checkpoint(model, path=ckpt_dir, device=device)

    valid_loader = create_dataloader(valid_cfg, phase="valid")

    log_dir = f"YOLOv11/runs/{model_variant}/valid"
    start_tensorboard_if_needed(log_dir, model_variant)

    results = validate_one_epoch(model, valid_loader, device, logger, tb, model_variant, propagate)
    if results is None:
        sys.exit(0)

    metrics, avg_loss, val_loss_history = results
    tb.log_metrics({"train_loss": avg_loss}, 0, phase="valid")  # misma etiqueta que entrenamiento
    tb.close()

    # === Curvas de pérdida combinadas ===
    save_dir = f"YOLOv11/metrics/{model_variant}/valid"
    os.makedirs(save_dir, exist_ok=True)
    train_loss_path = f"YOLOv11/metrics/{model_variant}/train/train_loss_history.pt"
    train_loss_history = torch.load(train_loss_path) if os.path.exists(train_loss_path) else []

    plt.figure(figsize=(8, 5))
    if train_loss_history:
        plt.plot(train_loss_history, label="Entrenamiento", color="blue")
    plt.plot(val_loss_history, label="Validación", color="orange")
    plt.title(f"Curva de pérdida - YOLOv11-{model_variant.upper()}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_vs_valid_loss.png"))
    plt.close()

    logger.info(f"✅ Validación completada. Loss promedio: {avg_loss:.4f}")
    print("\n📊 Métricas finales:", metrics)
    print(f"📈 Curva de pérdida guardada en {save_dir}/train_vs_valid_loss.png")


if __name__ == "__main__":
    main()
