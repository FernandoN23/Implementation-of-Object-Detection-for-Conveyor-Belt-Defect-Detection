# -*- coding: utf-8 -*-
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: visualization.py
Módulo de visualización TensorBoard para YOLOv11 (versión mejorada).
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
        Solo lanza un servidor si no hay uno corriendo en el puerto indicado.
        """
        self.variant = model_variant.lower()
        self.log_dir = os.path.join(log_dir, self.variant)
        os.makedirs(self.log_dir, exist_ok=True)

        self.port = port
        self.writer = SummaryWriter(self.log_dir)

        # Lanzar TensorBoard solo si no hay otro en ejecución
        if not self._is_tensorboard_running():
            self._launch_tensorboard()
        else:
            print(f"📊 TensorBoard ya está corriendo en http://localhost:{self.port}")

        print(f"🧭 Registrando métricas en: {self.log_dir}")

    # ----------------------------------------------------------
    # Comprobación de instancia existente
    # ----------------------------------------------------------
    def _is_tensorboard_running(self):
        """Verifica si ya hay un proceso TensorBoard en el puerto."""
        try:
            if os.name == "nt":  # Windows
                output = os.popen(f'netstat -ano | findstr :{self.port}').read()
                return "LISTENING" in output or "ESTABLISHED" in output
            else:  # Linux / macOS
                output = os.popen(f"lsof -i :{self.port}").read()
                return "tensorboard" in output
        except Exception:
            return False

    # ----------------------------------------------------------
    # Inicialización del servidor
    # ----------------------------------------------------------
    def _launch_tensorboard(self):
        """Lanza un servidor TensorBoard en segundo plano."""
        try:
            if os.name == "nt":
                os.system(
                    f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{self.port}') do taskkill /PID %a /F >nul 2>&1"
                )

            cmd = [
                "tensorboard",
                f"--logdir={self.log_dir}",
                f"--port={self.port}",
                "--reload_interval=5",
            ]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            print(f"🔗 TensorBoard iniciado en: http://localhost:{self.port}")
        except Exception as e:
            print(f"⚠️ No se pudo iniciar TensorBoard automáticamente: {e}")

    # ----------------------------------------------------------
    # Registro de métricas
    # ----------------------------------------------------------
    def log_metrics(self, metrics, step: int, phase="train", class_name=None):
        """
        Registra métricas globales o por clase.
        Acepta:
          - dict simple
          - tupla (global_metrics, per_class_metrics)
        """
        if metrics is None:
            print(f"ℹ️ [TensorBoard] Sin métricas válidas ({phase}).")
            return

        # Si viene como tupla (global, por clase)
        if isinstance(metrics, tuple) and len(metrics) == 2:
            global_metrics, per_class_metrics = metrics

            # Globales
            if isinstance(global_metrics, dict):
                for k, v in global_metrics.items():
                    if isinstance(v, (float, int)):
                        self.writer.add_scalar(f"{phase}/Global/{k}", v, step)

            # Por clase
            if isinstance(per_class_metrics, dict):
                for cls_name, cls_data in per_class_metrics.items():
                    if isinstance(cls_data, dict):
                        for k, v in cls_data.items():
                            if isinstance(v, (float, int)):
                                self.writer.add_scalar(f"{phase}/{cls_name}/{k}", v, step)
            return

        # Si viene como dict
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    tag = f"{phase}/{class_name}/{key}" if class_name else f"{phase}/{key}"
                    self.writer.add_scalar(tag, value, step)
        else:
            print(f"⚠️ [TensorBoard] Tipo de métricas no reconocido: {type(metrics)}")

    # ----------------------------------------------------------
    # Subir imágenes
    # ----------------------------------------------------------
    def log_image_file(self, tag: str, image_path: str, step: int):
        if not os.path.exists(image_path):
            print(f"⚠️ Imagen no encontrada: {image_path}")
            return

        try:
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)
            img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            self.writer.add_images(tag, img_tensor, step)
        except Exception as e:
            print(f"⚠️ No se pudo subir imagen ({image_path}): {e}")

    def log_images_folder(self, folder_path: str, step: int, phase="valid"):
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
        """Cierra el writer sin detener el servidor (mantiene la sesión)."""
        self.writer.close()
        print(f"🧹 Writer cerrado correctamente para YOLOv11-{self.variant.upper()}.")
