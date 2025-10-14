# models/parser_yaml.py
"""
ModelParser - Constructor dinámico de modelos YOLOv11.
Permite crear modelos desde archivos YAML modulares.
"""

import yaml
import torch
import torch.nn as nn
import importlib
from pathlib import Path


# -----------------------------------------------------------------------------
# 🧩 Clase principal
# -----------------------------------------------------------------------------
class ModelParser:
    """
    Lee los YAMLs y construye el modelo PyTorch correspondiente.
    Permite que los bloques y submódulos estén definidos externamente.
    """

    def __init__(self, model_yaml, parser_yaml, ch_input=3, nc=80, scale='n'):
        """
        Args:
            model_yaml: ruta a yolo11.yaml (define arquitectura)
            parser_yaml: ruta a parser.yaml (define mapeo de bloques)
            ch_input: canales de entrada (por defecto 3)
            nc: número de clases
            scale: escala del modelo ('n', 's', 'm', 'l', 'x')
        """
        self.model_yaml = model_yaml
        self.parser_yaml = parser_yaml
        self.ch_input = ch_input
        self.nc = nc
        self.scale = scale

        # Cargar archivos YAML
        self.model_def = self._load_yaml(model_yaml)
        self.parser_def = self._load_yaml(parser_yaml)

        # Aplicar escalado compuesto si existe
        self.depth_mult, self.width_mult, self.max_channels = self._parse_scale(scale)

    # -----------------------------------------------------------------------------
    # 🔹 Función de carga YAML
    # -----------------------------------------------------------------------------
    def _load_yaml(self, path):
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------------
    # 🔹 Escalado compuesto (depth, width)
    # -----------------------------------------------------------------------------
    def _parse_scale(self, scale):
        scales = self.model_def.get('scales', {})
        if scale in scales:
            return scales[scale]
        else:
            print(f"[WARN] Escala '{scale}' no encontrada en {self.model_yaml}. Usando defaults.")
            return 1.0, 1.0, 1024

    # -----------------------------------------------------------------------------
    # 🔹 Crea un módulo a partir de la definición YAML
    # -----------------------------------------------------------------------------
    def _create_module(self, module_name, args):
        """
        Crea una instancia de un módulo según parser.yaml
        """
        if module_name not in self.parser_def:
            raise ValueError(f"Módulo {module_name} no encontrado en parser.yaml")

        block_info = self.parser_def[module_name]
        import_path = block_info['import']
        class_name = block_info['class']

        # Importa dinámicamente el módulo
        module = importlib.import_module(import_path)
        cls = getattr(module, class_name)

        # Instancia la clase con los argumentos
        if isinstance(args, list):
            return cls(*args)
        elif isinstance(args, dict):
            return cls(**args)
        else:
            return cls(args)

    # -----------------------------------------------------------------------------
    # 🔹 Construcción de una lista de capas secuenciales
    # -----------------------------------------------------------------------------
    def _build_section(self, section_name):
        """
        Construye secuencialmente backbone, neck o head.
        """
        section_def = self.model_def.get(section_name, [])
        layers = nn.ModuleList()
        ch_prev = self.ch_input

        for i, (from_idx, repeat, module_name, args) in enumerate(section_def):
            # Ajustar número de canales si están definidos
            if isinstance(args[0], int):
                args[0] = min(int(args[0] * self.width_mult), self.max_channels)

            # Crea el módulo con los argumentos actualizados
            module = self._create_module(module_name, args)

            # Si el módulo debe repetirse (ej. C3k2 con n bloques)
            if repeat > 1:
                module = nn.Sequential(*[self._create_module(module_name, args) for _ in range(repeat)])

            layers.append(module)
            ch_prev = args[0]

        print(f"✅ Construido módulo {section_name} con {len(layers)} capas.")
        return layers

    # -----------------------------------------------------------------------------
    # 🔹 Construcción completa del modelo (jerárquico)
    # -----------------------------------------------------------------------------
    def build(self):
        """
        Construye el modelo completo (backbone + neck + head)
        """
        model = nn.Module()
        model.backbone = self._build_section("backbone")
        model.neck = self._build_section("neck") if "neck" in self.model_def else None
        model.head = self._build_section("head")

        return model
