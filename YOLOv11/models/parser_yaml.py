"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: parser_yaml.py
Módulo auxiliar de configuración para YOLOv11.
Permite la lectura, validación y carga dinámica de parámetros
definidos en archivos YAML (modelo y dataset).
-------------------------------------------------------------
"""

import yaml
from pathlib import Path


class ModelParser:
    """
    ModelParser
    ------------
    Lector y validador de archivos YAML de configuración del modelo.
    Permite ajustar dinámicamente hiperparámetros, rutas y variantes.
    """

    def __init__(self, cfg_path: str):
        self.root = Path(__file__).resolve().parents[2]
        self.cfg_path = (self.root / cfg_path).resolve()
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"❌ Archivo de configuración no encontrado: {self.cfg_path}")

    # ---------------------------------------------------------
    # Lectura del modelo
    # ---------------------------------------------------------
    def parse_model_config(self):
        """Lee y valida el archivo YAML del modelo (e.g., yolo11.yaml)."""
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # Claves esenciales, pero flexibles según versión YOLO
        required_keys = ['nc', 'backbone', 'head']
        optional_keys = ['neck']

        # Verificar claves faltantes críticas
        for key in required_keys:
            if key not in cfg:
                raise KeyError(f"❌ Clave obligatoria '{key}' ausente en {self.cfg_path.name}")

        # Claves opcionales (solo avisa una vez si falta)
        for key in optional_keys:
            if key not in cfg:
                cfg[key] = None  # crea placeholder para compatibilidad
                # Comentado: no es necesario mostrar advertencia
                # print(f"ℹ️ Clave opcional '{key}' no encontrada en {self.cfg_path.name}")

        return cfg

    # ---------------------------------------------------------
    # Lectura de dataset
    # ---------------------------------------------------------
    def parse_dataset_config(self, dataset_path: str):
        """Lee un YAML de dataset (dataset.yaml)."""
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"❌ Dataset YAML no encontrado: {dataset_path}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ---------------------------------------------------------
    # Lectura de variantes (n, s, m, l, x)
    # ---------------------------------------------------------
    def load_variant_config(self, variant_file="configs/model_variants.yaml"):
        """Carga los parámetros de escalado depth/width."""
        vf = Path(variant_file)
        if vf.exists():
            with open(vf, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        print(f"ℹ️ Archivo de variantes no encontrado: {vf.name}")
        return {}

    # ---------------------------------------------------------
    # Validación cruzada
    # ---------------------------------------------------------
    def validate_model_structure(self, cfg, variants=None):
        """Verifica consistencia entre modelo y variantes."""
        if variants and 'scales' in cfg:
            missing = [k for k in cfg['scales'] if k not in variants]
            if missing:
                print(f"⚠️ Variantes no definidas en model_variants.yaml: {missing}")

    # ---------------------------------------------------------
    # Resumen visual
    # ---------------------------------------------------------
    def summary(self, cfg):
        """Imprime resumen legible de configuración."""
        print("\n📄 Resumen de configuración del modelo:")
        for k, v in cfg.items():
            print(f"  {k:<15}: {v}")


if __name__ == "__main__":
    parser = ModelParser("YOLOv11/configs/yolo11.yaml")
    cfg = parser.parse_model_config()
    parser.summary(cfg)
