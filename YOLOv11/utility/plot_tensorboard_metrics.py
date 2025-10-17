# -*- coding: utf-8 -*-
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: plot_tensorboard_metrics.py
Analiza los archivos de TensorBoard (.tfevents) y genera
gráficos de evolución de métricas (loss, mAP, precisión, etc.)
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Función: plot_tensorboard_scalars()
#   - Lee eventos de runs/ con event_accumulator.
#   - Extrae las métricas registradas (tags).
#   - Genera gráficos .png organizados en metrics/plots.
#
# Uso:
#   python plot_tensorboard_metrics.py
#
# Conexión:
#   No se ejecuta durante entrenamiento.
#   Es una herramienta analítica para comparar variantes o
#   analizar resultados finales.
# -------------------------------------------------------------


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
