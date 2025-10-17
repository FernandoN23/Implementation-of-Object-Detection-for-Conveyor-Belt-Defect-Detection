# -*- coding: utf-8 -*-
"""
=============================================================
 Trabajo de Memoria de Título
 Memorista: Fernando Navarrete
 Modelo actual: YOLOv11
 Código actual: plot_tensorboard_metrics.py
=============================================================

Lee los registros de TensorBoard (runs/*.tfevents) y genera
gráficos comparativos de las métricas de entrenamiento/validación.
=============================================================
"""

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_scalars(log_dir="YOLOv11/runs/yolo11_train", output_dir="YOLOv11/metrics/plots"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Leyendo eventos de TensorBoard desde: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Listar todas las métricas registradas
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print("⚠️ No se encontraron métricas registradas.")
        return

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure(figsize=(7, 4))
        plt.plot(steps, values, marker='o', linewidth=1.5)
        plt.title(f"{tag} vs Steps")
        plt.xlabel("Step / Epoch")
        plt.ylabel(tag)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Guardado gráfico: {filename}")

    print("🎯 Finalizado. Gráficos disponibles en:", output_dir)


if __name__ == "__main__":
    plot_tensorboard_scalars()
