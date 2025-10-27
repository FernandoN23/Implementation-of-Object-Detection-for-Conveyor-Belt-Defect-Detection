# yolo11.py
"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: yolo11.py
Definición del modelo completo YOLOv11.
Integra las tres partes principales del detector:
Backbone (extracción), Neck (fusión FPN+PAN) y Head (detección).
-------------------------------------------------------------
"""

# -------------------------------------------------------------
# Estructura general del modelo:
#   1. Se carga el archivo YAML (parser_yaml.py)
#      → Se extraen parámetros como base_channels, anchors,
#        tipo de normalización (BatchNorm, GroupNorm, etc.)
#
#   2. Se construyen los submódulos:
#        • Backbone  → extracción de features jerárquicas
#        • Neck      → fusión de características multiescala
#        • Head      → salida de predicciones (bbox + clases)
#
#   3. En el forward():
#        x → backbone → neck → head → [y3, y4, y5]
#
# Conexiones entre módulos:
#   Backbone produce (x3, x4, x5)
#   Neck combina → (p3, n4, n5)
#   Head entrega las salidas finales del detector.
#
# Compatibilidad:
#   Permite variar tipo de normalización y número de clases
#   directamente desde el archivo YAML de configuración.
# -------------------------------------------------------------
import warnings
import torch
import torch.nn as nn

from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import YOLOv11Head
from .parser_yaml import ModelParser


class YOLOv11(nn.Module):
    def __init__(self, cfg_path=None, num_classes=5):
        super().__init__()

        # -------------------------
        # 1. Cargar configuración YAML
        # -------------------------
        if cfg_path:
            parser = ModelParser(cfg_path)
            cfg = parser.parse_model_config()
            base_channels = cfg.get('base_channels', 64)
            anchors = cfg.get('anchors', 1)
            if anchors > 1:
                warnings.warn(
                    "[YOLOv11] La configuración solicita anchors>1, pero la "
                    "decodificación multi-anchor aún no está habilitada.",
                    UserWarning,
                )
            # nuevos parámetros de normalización
            norm_type = cfg.get('norm', 'bn')
            gn_groups = cfg.get('gn_groups', 32)
        else:
            base_channels = 64
            anchors = 1
            norm_type = 'bn'
            gn_groups = 32

        # -------------------------
        # 2. Construcción de submódulos
        # -------------------------
        self.backbone = YOLOv11Backbone(
            in_channels=3,
            base_channels=base_channels,
            norm_type=norm_type,
            gn_groups=gn_groups
        )

        self.neck = YOLOv11Neck(
            base_channels=base_channels,
            norm_type=norm_type,
            gn_groups=gn_groups
        )

        self.head = YOLOv11Head(
            num_classes=num_classes,
            base_channels=base_channels,
            anchors=anchors,
            norm_type=norm_type,
            gn_groups=gn_groups
        )

        print(f"[YOLOv11] Modelo inicializado con norm='{norm_type}', grupos GN={gn_groups}")

    # -------------------------
    # 3. Forward completo
    # -------------------------
    def forward(self, x):
        """
        Forward completo del modelo:
        1. Extrae features (backbone)
        2. Fusiona escalas (neck)
        3. Predice bounding boxes (head)
        """
        x3, x4, x5 = self.backbone(x)
        p3, n4, n5 = self.neck(x3, x4, x5)
        outputs = self.head(p3, n4, n5)
        return outputs


# ============================
# Test de verificación rápida
# ============================
if __name__ == "__main__":
    model = YOLOv11(cfg_path="configs/yolo11.yaml", num_classes=10)
    dummy_input = torch.randn(1, 3, 640, 640)
    out = model(dummy_input)
    print("Número de salidas:", len(out))
    for i, o in enumerate(out):
        print(f"Salida {i+1}: {list(o.shape)}")
