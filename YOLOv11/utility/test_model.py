# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: test_model.py
# Utilidad para ensamblar el modelo YOLOv11 desde los configs y
# ejecutar un forward de prueba (sin pérdidas ni métricas) para
# verificar coherencia de entradas/salidas y dimensiones por nivel.
# Debe ejecutarse desde la raíz del proyecto (carpeta YOLOv11).
#==============================================================

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Dict, List, Tuple, Any

import torch

# --- Comprobación de ruta de proyecto -------------------------------------------------------------
# Este script puede ejecutarse desde la raíz de *todo* el repo o desde la carpeta YOLOv11/utility.
# Ajustamos sys.path y cwd automáticamente para que apunten a la carpeta YOLOv11 (que contiene /configs y /models).
PROJ_ROOT_CANDIDATES = [
    os.getcwd(),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
]

for _root in PROJ_ROOT_CANDIDATES:
    if os.path.isdir(os.path.join(_root, "configs")) and os.path.isdir(os.path.join(_root, "models")):
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            os.chdir(_root)
        except Exception:
            pass
        break

# Imports del proyecto (después de fijar sys.path)
try:
    from models.parser_yaml import ConfigParserYaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "No se pudo importar models.parser_yaml.ConfigParserYaml.\n"
        "Ejecuta: `python YOLOv11/utility/test_model.py` o `python utility/test_model.py` desde la raíz YOLOv11.\n"
        f"Detalle del error: {e}"
    )


# --- Utilidades -----------------------------------------------------------------------------------

def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def human_size(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params/1_000_000:.2f} M"
    if num_params >= 1_000:
        return f"{num_params/1_000:.2f} K"
    return str(num_params)


def select_device(pref: str | None = None) -> torch.device:
    """Selecciona dispositivo disponible.
    - 'cuda' (incluye ROCm) si está disponible.
    - 'mps' (Apple) si está disponible y no hay CUDA.
    - 'cpu' en caso contrario.
    Si *pref* es 'cpu'|'cuda'|'mps', se intenta respetar.
    """
    pref = (pref or "auto").lower()
    if pref in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _detect_variant(cfg: Any, fallback: str = "m") -> str:
    # Intenta diversas ubicaciones para obtener la variante por defecto.
    for name in ("model", "config", "cfg", "parser"):
        obj = getattr(cfg, name, None)
        if isinstance(obj, dict):
            dv = obj.get("default_variant") or obj.get("default")
            if isinstance(dv, str) and dv:
                return dv
    dv = getattr(cfg, "default_variant", None)
    if isinstance(dv, str) and dv:
        return dv
    return fallback
# -------------------------------------------------------------------------------
def _meta_get(meta, key, default=None):
    """Acceso tolerante a meta: admite dict, dataclass/objeto o anidado."""
    if meta is None:
        return default
    # dict
    if isinstance(meta, dict):
        return meta.get(key, default)
    # dataclass/objeto con atributo directo
    if hasattr(meta, key):
        return getattr(meta, key, default)
    # algunos parsers guardan un subobjeto/attr 'model' como dict
    sub = getattr(meta, "model", None)
    if isinstance(sub, dict):
        return sub.get(key, default)
    return default

# --- Núcleo de la prueba --------------------------------------------------------------------------

def build_from_configs(parser_root: str | None = None,
                       variant: str | None = None,
                       verbose: bool = True):
    """Crea ConfigParserYaml, resuelve variante, construye el modelo y retorna (cfg, model, meta).
    *parser_root*: ruta raíz del proyecto (si None, usa cwd).
    *variant*: fuerza variante 'n|s|m|l|xl' (si None, usa la default disponible).
    *meta*: dict con info útil (nc, in_channels, reg_max, strides, device preferido, imgsz sugerido).
    """
    root = parser_root or os.getcwd()
    cfg = ConfigParserYaml(project_root=root)
    cfg.load()

    # Intento robusto de fijar variante desde CLI
    if variant:
        set_ok = False
        # 1) Método explícito si existiera
        try:
            if hasattr(cfg, "set_variant"):
                cfg.set_variant(variant)  # type: ignore[attr-defined]
                set_ok = True
        except Exception:
            pass
        # 2) Resolver y fijar internamente
        if not set_ok:
            try:
                if hasattr(cfg, "resolve_variant"):
                    cfg.resolve_variant(variant)  # type: ignore[attr-defined]
                    set_ok = True
            except Exception:
                pass
        # 3) Actualizar dict si existe
        if not set_ok and isinstance(getattr(cfg, "model", None), dict):
            try:
                cfg.model["default_variant"] = variant  # type: ignore[index]
                set_ok = True
            except Exception:
                pass

    # Construcción del modelo (pasa la variante explícitamente)
    model = cfg.build_model(variant=variant)

    # Extraemos metadatos (con tolerancia a cambios futuros)
    model_meta = getattr(cfg, "model_meta", {}) or {}
    nc = int(_meta_get(model_meta, "nc", getattr(getattr(model, "head", object()), "nc", 1)))
    in_ch = int(_meta_get(model_meta, "in_channels", 3))
    reg_max = int(_meta_get(model_meta, "reg_max", 16))
    strides = _meta_get(model_meta, "strides", getattr(getattr(model, "head", object()), "strides", None))

    # imgsz sugerido: train.yaml si existe; fallback 640
    imgsz = 640
    train_cfg = getattr(cfg, "train", None)
    if isinstance(train_cfg, dict):
        try:
            imgsz = int(train_cfg.get("imgsz", imgsz))
        except Exception:
            pass

    runtime = getattr(cfg, "runtime", {}) if isinstance(getattr(cfg, "runtime", {}), dict) else {}
    device_pref = str(runtime.get("device", "auto"))

    meta = dict(nc=nc, in_channels=in_ch, reg_max=reg_max, strides=strides, imgsz=imgsz, device_pref=device_pref)

    if verbose:
        dv = _detect_variant(cfg, fallback=variant or "m")
        print("[Config] Variante:", dv)
        print("[Config] nc=", nc, ", in_channels=", in_ch, ", reg_max=", reg_max, ", strides=", strides, ", imgsz=", imgsz)

    return cfg, model, meta

def _randn_with_seed(shape, device, dtype, seed: int = 42):
    """
    Crea un tensor normal con semilla fija y generator en el dispositivo correcto.
    Soporta 'cpu' y 'cuda'. En 'mps' no se soporta generator con device,
    por lo que se usa seed global como fallback.
    """
    # Normaliza device (puede venir como str o torch.device)
    dev_kind = device.type if hasattr(device, "type") else str(device)
    if dev_kind.startswith("cuda"):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)
    elif dev_kind.startswith("cpu"):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return torch.randn(*shape, device="cpu", dtype=dtype, generator=g)
    else:
        # Fallback (p. ej., 'mps'): usar seed global y SIN generator
        torch.manual_seed(seed)
        return torch.randn(*shape, device=device, dtype=dtype)

def forward_sanity(model: torch.nn.Module,
                   meta: Dict,
                   batch: int = 1,
                   imgsz: int | None = None,
                   decode: bool = False,
                   concat_levels: bool = False,
                   device: torch.device | None = None,
                   dtype: torch.dtype = torch.float32,
                   seed: int = 42) -> Dict[str, List[torch.Size] | torch.Size]:
    """Ejecuta un forward de prueba y valida formas esperadas.

    Retorna un diccionario con las shapes por rama ('cls', 'reg') por nivel (o concatenadas), sin alterar parámetros.
    Lanza AssertionError si alguna dimensión no es la esperada.
    """
    g = torch.Generator().manual_seed(seed)

    in_ch = int(meta["in_channels"])  # canales de entrada
    strides: List[int] = list(meta["strides"])  # p.ej. [8,16,32]
    nc = int(meta["nc"])  # clases
    reg_max = int(meta["reg_max"])  # p.ej. 16 → 4*16 canales en rama de bbox
    img = int(imgsz or meta.get("imgsz", 640))

    device = device or select_device(meta.get("device_pref", "auto"))

    # Ajuste de dtype en CPU
    if device.type == "cpu" and dtype == torch.float16:
        print("[Aviso] float16 en CPU no es soportado de forma general. Cambiando a float32.")
        dtype = torch.float32

    # Preparación del modelo
    model.eval()
    model.to(device)

    # Entrada dummy
    x = _randn_with_seed((batch, in_ch, img, img), device=device, dtype=dtype, seed=seed)

    with torch.no_grad():
        out = model(x, decode=decode, concat=concat_levels)  # type: ignore[arg-type]

    # Estructura esperada: dict con claves 'cls' y 'reg'
    if not isinstance(out, dict):
        raise RuntimeError("El forward del modelo no retornó un dict. Revisa yolo11.forward(...).")

    cls_out = out.get("cls")
    reg_out = out.get("reg")

    # Si concat=False → listas por nivel; si concat=True → tensor concatenado
    shapes: Dict[str, List[torch.Size] | torch.Size] = {}

    def _expect_hw(s: int) -> int:
        # Con conv stride-2 y padding 'same', img debe dividir por s sin residuo si img es múltiplo de 32
        return img // s

    if concat_levels:
        # Salida concatenada: [B, C, sum(HW_i)] o similar (según implementación)
        if isinstance(cls_out, torch.Tensor) and isinstance(reg_out, torch.Tensor):
            shapes["cls"] = cls_out.shape
            shapes["reg"] = reg_out.shape
            # Comprobación básica de canales
            assert cls_out.shape[0] == batch, "Batch mismatch en cls (concat)."
            assert reg_out.shape[0] == batch, "Batch mismatch en reg (concat)."
            # Canales esperados
            assert cls_out.shape[1] == nc, f"Canales cls esperados={nc}, obtenidos={cls_out.shape[1]}"
            assert reg_out.shape[1] == 4 * reg_max, f"Canales reg esperados={4*reg_max}, obtenidos={reg_out.shape[1]}"
        else:
            raise AssertionError("Se esperaba salida concatenada tipo Tensor para 'cls' y 'reg'.")
    else:
        # Salida por niveles
        if not (isinstance(cls_out, (list, tuple)) and isinstance(reg_out, (list, tuple))):
            raise AssertionError("Se esperaban listas/tuplas por nivel para 'cls' y 'reg'. Usa concat=False.")
        assert len(cls_out) == len(reg_out) == len(strides), "Número de niveles inconsistente con 'strides'."

        cls_shapes: List[torch.Size] = []
        reg_shapes: List[torch.Size] = []
        for i, s in enumerate(strides):
            cti, rti = cls_out[i], reg_out[i]
            assert isinstance(cti, torch.Tensor) and isinstance(rti, torch.Tensor), "Salida por nivel no es Tensor."
            # Esperado: [B, C, H, W]
            Bh, Ch, Hh, Wh = cti.shape
            Br, Cr, Hr, Wr = rti.shape

            # Comprobaciones básicas
            assert Bh == Br == batch, f"Batch mismatch en nivel {i}."
            assert Ch == nc, f"Nivel {i}: canales cls esperados={nc}, obtenidos={Ch}"
            assert Cr == 4 * reg_max, f"Nivel {i}: canales reg esperados={4*reg_max}, obtenidos={Cr}"

            Hexp = _expect_hw(s)
            Wexp = _expect_hw(s)
            assert (Hh, Wh) == (Hexp, Wexp), (
                f"Nivel {i}: HxW esperado={Hexp}x{Wexp} con stride={s}, obtenido={Hh}x{Wh}"
            )
            assert (Hr, Wr) == (Hexp, Wexp), (
                f"Nivel {i}: HxW reg esperado={Hexp}x{Wexp} con stride={s}, obtenido={Hr}x{Wr}"
            )

            cls_shapes.append(cti.shape)
            reg_shapes.append(rti.shape)

        shapes["cls"] = cls_shapes
        shapes["reg"] = reg_shapes

    return shapes


def pretty_print_shapes(shapes: Dict[str, List[torch.Size] | torch.Size]):
    print("\n=== Formas de salida ===")
    if isinstance(shapes.get("cls"), list):
        cls_list: List[torch.Size] = shapes["cls"]  # type: ignore[assignment]
        reg_list: List[torch.Size] = shapes["reg"]  # type: ignore[assignment]
        for i, (cs, rs) in enumerate(zip(cls_list, reg_list)):
            print(f"Nivel P{i+3}: cls {tuple(cs)}, reg {tuple(rs)}")
    else:
        print(f"cls {tuple(shapes['cls'])}")
        print(f"reg {tuple(shapes['reg'])}")


# --- CLI ------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ensamblar YOLOv11 desde configs y ejecutar un forward de prueba para validar formas de salida.\n"
            "Ejemplo: python utility/test_model.py --variant m --imgsz 640 --batch 2"
        )
    )
    p.add_argument("--variant", type=str, default=None, choices=["n", "s", "m", "l", "xl"],
                   help="Forzar variante del modelo (si se omite, usa la definida en los configs)")
    p.add_argument("--imgsz", type=int, default=None, help="Tamaño de imagen cuadrada (por defecto usa train.yaml o 640)")
    p.add_argument("--batch", type=int, default=1, help="Tamaño de batch para la prueba")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                   help="Dispositivo preferido para la prueba (por defecto auto)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Precisión del tensor de entrada")
    p.add_argument("--decode", action="store_true", help="Si se entrega, solicita salida decodificada (si está implementado)")
    p.add_argument("--concat", action="store_true", help="Si se entrega, concatena niveles en la salida del modelo")
    p.add_argument("--no-assert", action="store_true", help="Desactiva aserciones de coherencia de formas")
    p.add_argument("--summary", action="store_true", help="Imprime resumen básico del modelo y número de parámetros")
    return p.parse_args()


def main():
    args = parse_args()

    # Construcción desde configs
    cfg, model, meta = build_from_configs(variant=args.variant, verbose=True)

    # Ajustes CLI
    if args.imgsz is not None:
        meta["imgsz"] = int(args.imgsz)
    device = select_device(args.device or meta.get("device_pref", "auto"))

    # Precisión
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    # Resumen del modelo
    if args.summary:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n=== Resumen del Modelo ===")
        print(f"Variante: {_detect_variant(cfg, fallback=args.variant or 'm')} | Parámetros: {human_size(total_params)} "
              f"(entrenables: {human_size(trainable)})")
        print(f"Strides: {meta['strides']} | nc: {meta['nc']} | reg_max: {meta['reg_max']} | in_channels: {meta['in_channels']}")
        print(f"Dispositivo seleccionado: {device}")

    # Forward de prueba
    try:
        shapes = forward_sanity(
            model=model,
            meta=meta,
            batch=int(args.batch),
            imgsz=int(meta.get("imgsz", 640)),
            decode=bool(args.decode),
            concat_levels=bool(args.concat),
            device=device,
            dtype=dtype,
        )
    except AssertionError as e:
        if args.no_assert:
            print("[ADVERTENCIA] Aserciones desactivadas --", e)
            # Reintenta sin validar (ya imprimimos el aviso); aún imprimiremos shapes si es posible
            shapes = {"cls": [], "reg": []}  # type: ignore[assignment]
        else:
            raise

    if 'shapes' in locals() and shapes:
        pretty_print_shapes(shapes)

    # Guardar breve log en /logs/
    try:
        os.makedirs("logs", exist_ok=True)
        tag = _now_tag()
        log_path = os.path.join("logs", f"test_model_{tag}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"test_model run @ {tag}\n")
            f.write(f"variant={_detect_variant(cfg, fallback=args.variant or 'm')} imgsz={meta.get('imgsz')} batch={args.batch} "
                    f"device={device} dtype={args.dtype} decode={args.decode} concat={args.concat}\n")
            if isinstance(shapes.get("cls"), list) and shapes.get("reg"):
                for i, (cs, rs) in enumerate(zip(shapes["cls"], shapes["reg"])):  # type: ignore[index]
                    f.write(f"P{i+3}: cls {tuple(cs)} | reg {tuple(rs)}\n")
            elif isinstance(shapes.get("cls"), torch.Size):
                f.write(f"cls {tuple(shapes['cls'])}\n")
                f.write(f"reg {tuple(shapes['reg'])}\n")
        print(f"\n[OK] Log guardado en: {log_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo guardar el log: {e}")


if __name__ == "__main__":
    main()
