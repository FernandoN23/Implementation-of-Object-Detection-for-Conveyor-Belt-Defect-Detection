import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
import sys
import yaml
from pathlib import Path


# ===========================================================================
# CARGA DE CONFIGURACIÓN (YAML -> DICT)
# ===========================================================================

def load_model_config():
    """
    Intenta cargar la configuración desde SSD/configs/model_variants.yaml.
    Si falla, retorna configuraciones por defecto (Hardcoded fallback).
    """
    # Ruta relativa: SSD/ssd/ssd.py -> parents[1] = SSD/ -> configs/model_variants.yaml
    current_file = Path(__file__).resolve()
    config_path = current_file.parents[1] / "configs" / "model_variants.yaml"

    configs = {}

    # 1. Intentar cargar desde YAML
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Procesar SSD300
            if 'ssd300' in yaml_data:
                cfg = yaml_data['ssd300']
                cfg['min_dim'] = cfg.pop('image_size', 300)  # Renombrar clave para compatibilidad
                cfg['name'] = 'SSD300'
                configs['300'] = cfg

            # Procesar SSD512
            if 'ssd512' in yaml_data:
                cfg = yaml_data['ssd512']
                cfg['min_dim'] = cfg.pop('image_size', 512)  # Renombrar clave para compatibilidad
                cfg['name'] = 'SSD512'
                configs['512'] = cfg

            print(f"[SSD] Configuración cargada exitosamente desde {config_path}")
            return configs

        except Exception as e:
            print(f"[SSD] Advertencia: Error leyendo {config_path}: {e}")
            print("[SSD] Se usarán valores por defecto (Hardcoded).")

    else:
        print(f"[SSD] Advertencia: No se encontró {config_path}")
        print("[SSD] Se usarán valores por defecto (Hardcoded).")

    # 2. Fallback (Valores por defecto si falla el YAML)
    configs['300'] = {
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'SSD300'
    }

    configs['512'] = {
        'feature_maps': [64, 32, 16, 8, 4, 2, 1],
        'min_dim': 512,
        'steps': [8, 16, 32, 64, 128, 256, 512],
        'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
        'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'SSD512'
    }

    return configs


# Cargar configuraciones al importar el módulo
MODEL_CONFIGS = load_model_config()


class SSD(nn.Module):
    """Single Shot Multibox Architecture"""

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # Selección de configuración basada en el tamaño de entrada
        size_str = str(size)
        if size_str in MODEL_CONFIGS:
            self.cfg = MODEL_CONFIGS[size_str]
        else:
            raise ValueError(f"SSD size {size} not supported. Available: {list(MODEL_CONFIGS.keys())}")

        self.priorbox = PriorBox(self.cfg)

        with torch.no_grad():
            self.register_buffer('priors', self.priorbox.forward())

        self.size = size
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # Asegurar que priors esté en el mismo dispositivo que la entrada
        if self.priors.device != x.device:
            self.priors = self.priors.to(x.device)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v)]
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i

    # Lógica explícita para SSD512
    if size == 512:
        # 1. Conv8_2 (1024 -> 512, stride 2) Output: 32x32 -> 16x16
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # 2. Conv9_2 (512 -> 256, stride 2) Output: 16x16 -> 8x8
        layers += [nn.Conv2d(512, 128, kernel_size=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # 3. Conv10_2 (256 -> 256, stride 2) Output: 8x8 -> 4x4
        layers += [nn.Conv2d(256, 128, kernel_size=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # 4. Conv11_2 (256 -> 256, stride 2) Output: 4x4 -> 2x2
        layers += [nn.Conv2d(256, 128, kernel_size=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # 5. Conv12_2 (256 -> 256, stride 1, valid) Output: 2x2 -> 1x1
        # Nota: Usamos kernel=2, stride=1, padding=0 para reducir 2x2 a 1x1
        layers += [nn.Conv2d(256, 128, kernel_size=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)]

        return layers

    # Lógica estándar para SSD300 (o fallback)
    # Block 1
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 2
    layers += [nn.Conv2d(512, 128, kernel_size=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 3
    layers += [nn.Conv2d(256, 128, kernel_size=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3)]

    # Block 4
    layers += [nn.Conv2d(256, 128, kernel_size=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3)]

    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    # VGG Sources: Conv4_3 y FC7
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    # Extra Sources
    # SSD300: 4 capas extra. SSD512: 5 capas extra.
    # El loop original 'extra_layers[1::2]' asume estructura (1x1, 3x3) y toma la segunda.
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

# Nota: 'extras' se usa solo como referencia de canales en el código original,
# pero la lógica real de construcción está en add_extras.
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4],  # 7 capas para 512
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    # Validar soporte usando las claves cargadas
    size_str = str(size)
    if size_str not in MODEL_CONFIGS:
        print(f"ERROR: You specified size {size}. Supported: {list(MODEL_CONFIGS.keys())}")
        return

    # Obtener configuración para pasarla a multibox si fuera necesario (aunque multibox usa mbox global)
    # cfg = MODEL_CONFIGS[size_str]

    base_, extras_, head_ = multibox(vgg(base[size_str], 3),
                                     add_extras(extras[size_str], 1024, size=size),
                                     mbox[size_str], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)