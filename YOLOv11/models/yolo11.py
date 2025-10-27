"""
Departamento de Ingeniería Mecánica - Universidad de Chile
Trabajo de Memoria de Título:
"Implementación de algoritmos de reconocimiento de objetos
para la identificación de fallas en correas transportadoras"
Autor: Fernando Navarrete

-------------------------------------------------------------
Archivo: yolo11.py
Definición del modelo completo YOLOv11 (Clasificación)
-------------------------------------------------------------
Estructura general:
    Entrada → Backbone → Neck → Head(Classify) → Logits [B, num_classes]
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
    YOLOv11 (Clasificación)
    ------------------------
    Flujo de procesamiento:
        x → Backbone → Neck → Head → logits
    """

    def __init__(self, cfg_path: str = "configs/yolo11.yaml", num_classes: int = 5):
        super().__init__()

        # -------------------------------------------------
        # 1. Cargar configuración YAML
        # -------------------------------------------------
        if cfg_path:
            parser = ModelParser(cfg_path)
            cfg = parser.parse_model_config()
            base_channels = cfg.get("base_channels", 64)
            norm_type = cfg.get("norm", "bn")
            gn_groups = cfg.get("gn_groups", 32)
            dropout = cfg.get("dropout", 0.0)
        else:
            base_channels, norm_type, gn_groups, dropout = 64, "bn", 32, 0.0

        # -------------------------------------------------
        # 2. Validaciones
        # -------------------------------------------------
        if not isinstance(dropout, (float, int)):
            warnings.warn(f"⚠️ Valor inválido de dropout '{dropout}', usando 0.0", UserWarning)
            dropout = 0.0

        # -------------------------------------------------
        # 3. Construcción de submódulos
        # -------------------------------------------------
        self.backbone = YOLOv11Backbone(3, base_channels, norm_type, gn_groups)
        self.neck = YOLOv11Neck(base_channels, norm_type, gn_groups)

        # ⚙️ Importante: el neck entrega base_channels * 16 canales (1024 por defecto)
        self.head = YOLOv11Classify(
            num_classes=num_classes,
            c_in=base_channels * 16,     # ✅ alineado con salida del neck
            dropout=dropout,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

        # -------------------------------------------------
        # 4. Metadatos del modelo
        # -------------------------------------------------
        self.model_info = dict(
            backbone="YOLOv11Backbone",
            neck="YOLOv11Neck",
            head="YOLOv11Classify",
            num_classes=num_classes,
            norm=norm_type,
            gn_groups=gn_groups,
            base_channels=base_channels,
            dropout=dropout,
        )

        print(f"[YOLOv11] Modelo inicializado con norm='{norm_type}', GN={gn_groups}, clases={num_classes}")
        self.initialize_weights()

    # -------------------------------------------------
    # Inicialización de pesos
    # -------------------------------------------------
    def initialize_weights(self):
        """Inicializa los pesos de convoluciones y normalizaciones."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward completo: Backbone → Neck → Head."""
        features = self.backbone(x)

        # algunos backbones devuelven lista de features
        if isinstance(features, (list, tuple)):
            features = features[-1]

        neck_out = self.neck(features)
        logits = self.head(neck_out)
        return logits

    # -------------------------------------------------
    # Información del modelo
    # -------------------------------------------------
    def info(self):
        """Muestra resumen estructural y de parámetros."""
        n_params = sum(p.numel() for p in self.parameters())
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n🧠 YOLOv11 Model Summary")
        for k, v in self.model_info.items():
            print(f"  {k:<15}: {v}")
        print(f"  Parámetros totales      : {n_params:,}")
        print(f"  Parámetros entrenables  : {n_grad:,}")


# ============================================================
# Test rápido
# ============================================================
if __name__ == "__main__":
    model = YOLOv11(cfg_path="YOLOv11/configs/yolo11.yaml", num_classes=5)
    dummy = torch.randn(1, 3, 640, 640)
    out = model(dummy)
    print(f"\nSalida del modelo: {list(out.shape)}")  # [1, num_classes]
    model.info()
