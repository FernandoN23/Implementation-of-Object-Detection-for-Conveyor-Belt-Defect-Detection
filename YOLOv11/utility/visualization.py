"""
visualization.py
---------------------------------
Configuración para visualizar métricas y pérdidas en TensorBoard.
Incluye refresco automático durante entrenamiento.
"""

from torch.utils.tensorboard import SummaryWriter
import os
import time

class TensorboardVisualizer:
    def __init__(self, log_dir="runs/yolo11"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard activo en: {log_dir}")

    def log_metrics(self, metrics: dict, epoch: int, phase="train"):
        """
        metrics: dict con valores como {'loss': 0.5, 'mAP': 0.7}
        """
        for k, v in metrics.items():
            self.writer.add_scalar(f"{phase}/{k}", v, epoch)

    def log_images(self, tag, images, epoch):
        self.writer.add_images(tag, images, epoch)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    # Ejemplo de test rápido del visualizador
    vis = TensorboardVisualizer()
    for e in range(5):
        vis.log_metrics({'loss': 1/e if e else 1}, e)
        time.sleep(0.5)
    vis.close()
