"""
logger.py
---------------------------------
Logger simple para entrenamiento y validación.
Guarda métricas, pérdidas y eventos importantes.
"""

import logging
import os
from datetime import datetime

def create_logger(log_dir="logs", name="yolo11"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("🚀 Logger iniciado correctamente")
    return logger
