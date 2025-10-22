# -*- coding: utf-8 -*-
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: visualization.py
Módulo de visualización TensorBoard para YOLOv11.
-------------------------------------------------------------
"""

import os
import time
import subprocess
import webbrowser
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorboardVisualizer:
    def __init__(self, log_dir="YOLOv11/runs", model_variant="n", port=6006):
        """
        Inicializa TensorBoard para una variante de modelo YOLOv11 específica.
        Se crearán subcarpetas como YOLOv11/runs/n/, YOLOv11/runs/s/, etc.
        """
        self.variant = model_variant.lower()
        self.log_dir = os.path.join(log_dir, self.variant)
        os.makedirs(self.log_dir, exist_ok=True)

        self.port = port
        self.writer = SummaryWriter(self.log_dir)
        self._launch_tensorboard()

        print(f"📊 TensorBoard activo y registrando en: {self.log_dir}")

    # ----------------------------------------------------------
    # Inicialización del servidor
    # ----------------------------------------------------------
    def _launch_tensorboard(self):
        """Lanza un servidor TensorBoard en segundo plano."""
        try:
            # Cierra instancias previas (Windows)
            if os.name == "nt":
                os.system(
                    f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{self.port}') do taskkill /PID %a /F >nul 2>&1"
                )

            cmd = [
                "tensorboard",
                f"--logdir={self.log_dir}",
                f"--port={self.port}",
                "--reload_interval=5"
            ]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(2)
            url = f"http://localhost:{self.port}"
            print(f"🔗 Visualiza métricas en: {url}")

            try:
                webbrowser.open(url)
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ No se pudo iniciar TensorBoard automáticamente: {e}")

    # ----------------------------------------------------------
    # Registro de métricas escalares
    # ----------------------------------------------------------
    def log_metrics(self, metrics: dict, step: int, phase="train", class_name=None):
        """
        Registra métricas escalares globales o por clase.
        Ejemplo:
          log_metrics({"Precision":0.8,"Recall":0.7}, 1, "valid", class_name="Defecto_A")
        """
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                if class_name:
                    tag = f"{phase}/{class_name}/{key}"
                else:
                    tag = f"{phase}/{key}"
                self.writer.add_scalar(tag, value, step)

    # ----------------------------------------------------------
    # Registro de imágenes en TensorBoard
    # ----------------------------------------------------------
    def log_image_file(self, tag: str, image_path: str, step: int):
        """
        Sube una imagen local (ej: .png) al panel de imágenes.
        Ejemplo: log_image_file("Métricas/Defecto_A", "path/to/img.png", 0)
        """
        if not os.path.exists(image_path):
            print(f"⚠️ Imagen no encontrada: {image_path}")
            return

        try:
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)
            img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            self.writer.add_images(tag, img_tensor, step)
        except Exception as e:
            print(f"⚠️ No se pudo subir imagen a TensorBoard ({image_path}): {e}")

    def log_images_folder(self, folder_path: str, step: int, phase="valid"):
        """
        Sube todas las imágenes de un directorio (ej: gráficas de métricas).
        Ideal para las generadas por metrics.py
        """
        if not os.path.exists(folder_path):
            print(f"⚠️ Carpeta no encontrada: {folder_path}")
            return

        for file in os.listdir(folder_path):
            if file.endswith((".png", ".jpg", ".jpeg")):
                tag = f"{phase}/Metrics/{os.path.splitext(file)[0]}"
                img_path = os.path.join(folder_path, file)
                self.log_image_file(tag, img_path, step)

    # ----------------------------------------------------------
    # Cierre
    # ----------------------------------------------------------
    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
        print(f"🧹 TensorBoard finalizado correctamente para YOLOv11-{self.variant.upper()}.")


# ==============================================================
# Test rápido independiente
# ==============================================================
if __name__ == "__main__":
    vis = TensorboardVisualizer(model_variant="n")
    print("Simulando registro de métricas...")
    vis.log_metrics({"loss": 0.5, "mAP": 0.72}, 0, phase="train")
    vis.log_metrics({"Precision": 0.8, "Recall": 0.75}, 1, phase="valid", class_name="Defecto_A")
    vis.log_image_file("valid/Defecto_A", "YOLOv11/metrics/n/valid/test_0001/Defecto_A_metrics_n.png", 0)
    vis.close()
