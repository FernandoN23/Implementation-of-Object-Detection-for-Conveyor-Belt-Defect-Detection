import yaml
from pathlib import Path

class ModelParser:
    """
    ModelParser
    ------------
    Lector y validador de archivos YAML de configuración del modelo.
    Permite ajustar dinámicamente hiperparámetros, rutas y variantes.
    """

    def __init__(self, cfg_path):
        self.cfg_path = Path(cfg_path)
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {cfg_path}")

    def parse_model_config(self):
        """
        Lee el archivo YAML del modelo (e.g., yolo11.yaml)
        Devuelve un diccionario con los parámetros clave.
        """
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # Validar claves esperadas mínimas
        required_keys = ['nc', 'backbone', 'neck', 'head']
        for key in required_keys:
            if key not in cfg:
                print(f"⚠️ Advertencia: clave '{key}' no encontrada en {self.cfg_path.name}")

        return cfg

    def parse_dataset_config(self, dataset_path):
        """
        Lee un YAML de dataset (dataset.yaml) si se requiere.
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset YAML no encontrado: {dataset_path}")

        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return data


if __name__ == "__main__":
    """
    Prueba rápida para verificar lectura del YAML.
    """
    parser = ModelParser("configs/yolo11.yaml")
    cfg = parser.parse_model_config()
    print("Configuración del modelo YOLOv11:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
