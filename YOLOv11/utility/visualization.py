# -*- coding: utf-8 -*-
"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: visualization.py
=============================================================

Módulo de visualización de métricas y pérdidas en TensorBoard.
Se lanza automáticamente un servidor TensorBoard local y se
registran métricas, pérdidas e imágenes durante el entrenamiento.
=============================================================
"""

import os
import time
import subprocess
from torch.utils.tensorboard import SummaryWriter
import webbrowser

class TensorboardVisualizer:
    def __init__(self, log_dir="YOLOv11/runs/yolo11_train", port=6006):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.port = port
        self.writer = SummaryWriter(log_dir)
        self._launch_tensorboard()
        print(f"📊 TensorBoard activo y registrando en: {log_dir}")

    def _launch_tensorboard(self):
        """Lanza un servidor TensorBoard en segundo plano."""
        try:
            # Detener instancias previas (solo si es necesario en Windows)
            if os.name == "nt":
                os.system(f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{self.port}') do taskkill /PID %a /F >nul 2>&1")

            # Ejecutar tensorboard
            cmd = [
                "tensorboard",
                f"--logdir={self.log_dir}",
                f"--port={self.port}",
                "--reload_interval=5"
            ]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Esperar un poco para que arranque
            time.sleep(2)
            url = f"http://localhost:{self.port}"
            print(f"🔗 Visualiza métricas en: {url}")
            try:
                webbrowser.open(url)
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ No se pudo iniciar TensorBoard automáticamente: {e}")

    def log_metrics(self, metrics: dict, step: int, phase="train"):
        """
        Registra métricas escalares en TensorBoard.
        Ejemplo: metrics = {'loss': 0.4, 'mAP': 0.72}
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, step)

    def log_images(self, tag, images, step):
        """Registra un lote de imágenes."""
        self.writer.add_images(tag, images, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
        print("🧹 TensorBoard finalizado correctamente.")


# ==============================================================
# Test rápido independiente
# ==============================================================
if __name__ == "__main__":
    vis = TensorboardVisualizer()
    print("Simulando registro de métricas...")
    for e in range(5):
        vis.log_metrics({"loss": 1/(e+1), "accuracy": e/5}, e)
        time.sleep(0.5)
    vis.close()
