"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando N.

-------------------------------------------------------------
Archivo: yolo11.py
Definición del modelo completo YOLOv11.
Integra Backbone, Neck y Head, con lectura dinámica de YAML.
-------------------------------------------------------------
"""

import warnings
import torch
import torch.nn as nn

from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import YOLOv11Classify
from .parser_yaml import ModelParser


class YOLOv11(nn.Module):
    """
    Modelo YOLOv11 completo
    ------------------------
    Estructura:
        x → Backbone → Neck → Head → [p3, n4, n5]
    """
    def __init__(self, cfg_path="configs/yolo11.yaml", num_classes=5):
        super().__init__()

        # -------------------------
        # 1. Cargar configuración YAML
        # -------------------------
        if cfg_path:
            parser = ModelParser(cfg_path)
            cfg = parser.parse_model_config()
            base_channels = cfg.get("base_channels", 64)
            anchors = cfg.get("anchors", 1)
            norm_type = cfg.get("norm", "bn")
            gn_groups = cfg.get("gn_groups", 32)
        else:
            base_channels, anchors, norm_type, gn_groups = 64, 1, "bn", 32

        if anchors > 1:
            warnings.warn("[YOLOv11] anchors>1 no soportado aún, se usará anchors=1.", UserWarning)

        # -------------------------
        # 2. Construcción de submódulos
        # -------------------------
        self.backbone = YOLOv11Backbone(3, base_channels, norm_type, gn_groups)
        self.neck = YOLOv11Neck(base_channels, norm_type, gn_groups)
        self.head = YOLOv11Classify(num_classes, base_channels, anchors, norm_type, gn_groups)

        # -------------------------
        # 3. Metadatos del modelo
        # -------------------------
        self.model_info = dict(
            backbone="YOLOv11Backbone",
            neck="YOLOv11Neck",
            head="YOLOv11Classify",
            num_classes=num_classes,
            norm=norm_type,
            gn_groups=gn_groups,
            base_channels=base_channels,
        )

        print(f"[YOLOv11] Modelo inicializado con norm='{norm_type}', GN={gn_groups}, clases={num_classes}")
        self.initialize_weights()

    # ---------------------------------------------------------
    # Inicialización de pesos
    # ---------------------------------------------------------
    def initialize_weights(self):
        """Inicializa los pesos de convoluciones y normalizaciones."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, x):
        """Forward completo: Backbone → Neck → Head."""
        x3, x4, x5 = self.backbone(x)
        p3, n4, n5 = self.neck(x3, x4, x5)
        outputs = self.head(p3, n4, n5)
        return outputs if self.training else [o.detach() for o in outputs]

    # ---------------------------------------------------------
    # Información del modelo
    # ---------------------------------------------------------
    def info(self):
        """Muestra resumen estructural del modelo."""
        n_params = sum(p.numel() for p in self.parameters())
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n🧠 YOLOv11 Model Summary")
        for k, v in self.model_info.items():
            print(f"  {k:<15}: {v}")
        print(f"  Parámetros totales : {n_params:,}")
        print(f"  Parámetros entrenables: {n_grad:,}")


# ============================================================
# Test de verificación rápida
# ============================================================
if __name__ == "__main__":
    model = YOLOv11(cfg_path="YOLOv11/configs/yolo11.yaml", num_classes=10)
    dummy_input = torch.randn(1, 3, 640, 640)
    out = model(dummy_input)
    print(f"\nNúmero de salidas: {len(out)}")
    for i, o in enumerate(out):
        print(f"  → Salida {i+1}: {list(o.shape)}")
    model.info()
