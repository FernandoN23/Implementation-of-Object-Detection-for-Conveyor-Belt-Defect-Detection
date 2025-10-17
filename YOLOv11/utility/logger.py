"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: logger.py
Gestor de registro de eventos y entrenamiento para YOLOv11.
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Función principal: get_logger()
#   - Crea un logger con salida dual (consola + archivo .log)
#   - Formato estándar: [fecha | nivel | mensaje]
#   - Se utiliza durante entrenamiento y validación
#
# Directorio de salida:
#   YOLOv11/logs/
#
# Conexión:
#   Usado en train.py para registrar métricas, pérdidas y
#   estados de checkpoints por cada variante del modelo.
# -------------------------------------------------------------


import logging
import os
import io
import sys

def get_logger(log_dir="YOLOv11/logs", name="train_yolo11"):
    import logging
    import os
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode="a", encoding="utf-8")
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


