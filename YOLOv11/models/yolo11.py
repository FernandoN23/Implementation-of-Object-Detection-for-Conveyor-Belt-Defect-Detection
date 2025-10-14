# models/yolo11.py
"""
YOLOv11 - Modelo ensamblado dinámicamente a partir de YAMLs.
Carga:
  - yolo11.yaml  -> define arquitectura (backbone, neck, head)
  - parser.yaml  -> define reglas de mapeo a clases Python
"""

import torch
import torch.nn as nn
from parser_yaml import ModelParser


class YOLOv11(nn.Module):
    """
    Clase principal del modelo YOLOv11.
    Crea toda la arquitectura a partir de archivos YAML modulares.
    """

    def __init__(self, model_cfg="configs/yolo11.yaml", parser_cfg="configs/parser.yaml", nc=80, ch_input=3):
        super().__init__()

        # 🧩 Construye el modelo usando el parser universal
        parser = ModelParser(model_yaml=model_cfg, parser_yaml=parser_cfg, ch_input=ch_input, nc=nc)
        self.model = parser.build()

        # Extrae submódulos
        self.backbone = self.model.backbone
        self.neck = getattr(self.model, "neck", None)
        self.head = self.model.head

        # Guarda información básica
        self.nc = nc
        self.ch_input = ch_input

        print("✅ YOLOv11 inicializado correctamente desde YAML.")

    # -------------------------------------------------------------------------
    # 🔁 Forward completo
    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Pasa un tensor de entrada por backbone → neck → head.
        """
        # 🔹 Backbone: extracción jerárquica
        features = []
        out = x
        for layer in self.backbone:
            out = layer(out)
            features.append(out)

        # Selecciona los últimos 3 mapas como [P3, P4, P5]
        feats = features[-3:]

        # 🔹 Neck: fusión multi-escala (si existe)
        if self.neck is not None:
            feats = self.neck(feats)

        # 🔹 Head: detección final
        preds = self.head(feats)

        return preds


# -------------------------------------------------------------------------
# 🧠 Prueba rápida
# -------------------------------------------------------------------------
if __name__ == "__main__":
    model = YOLOv11("configs/yolo11.yaml", "configs/parser.yaml", nc=80, ch_input=3)
    x = torch.randn(1, 3, 640, 640)
    preds = model(x)
    print("\nTamaños de salida por nivel:")
    for p in preds:
        print(p.shape)
