import logging
import os
import io
import sys

def get_logger(log_dir, name="yolo11"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Archivo de log
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # ✅ Fix de codificación para emojis y caracteres extendidos
    if isinstance(console_handler, logging.StreamHandler):
        console_handler.stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
