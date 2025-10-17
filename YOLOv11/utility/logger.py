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


