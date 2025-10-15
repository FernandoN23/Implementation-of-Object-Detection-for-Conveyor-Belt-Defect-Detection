import torch
import torch.nn as nn

from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import YOLOv11Head
from .parser_yaml import ModelParser

class YOLOv11(nn.Module):
    """
    YOLOv11 Full Model
    ------------------
    Combina Backbone, Neck y Head para detección multi-escala.
    Estructura modular compatible con parser YAML y entrenamiento Ultralytics-like.
    """

    def __init__(self, cfg_path=None, num_classes=80):
        super().__init__()

        # Cargar configuración YAML (si se proporciona)
        if cfg_path:
            parser = ModelParser(cfg_path)
            cfg = parser.parse_model_config()
            base_channels = cfg.get('base_channels', 64)
            anchors = cfg.get('anchors', 3)
        else:
            base_channels = 64
            anchors = 3

        # Definir submódulos principales
        self.backbone = YOLOv11Backbone(in_channels=3, base_channels=base_channels)
        self.neck = YOLOv11Neck(base_channels=base_channels)
        self.head = YOLOv11Head(num_classes=num_classes, base_channels=base_channels, anchors=anchors)

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


if __name__ == "__main__":
    """
    Prueba rápida de forward pass para verificar la estructura.
    """
    model = YOLOv11(cfg_path="configs/yolo11.yaml", num_classes=10)
    dummy_input = torch.randn(1, 3, 640, 640)
    out = model(dummy_input)
    print("Número de salidas:", len(out))
    for i, o in enumerate(out):
        print(f"Salida {i+1}: {list(o.shape)}")
