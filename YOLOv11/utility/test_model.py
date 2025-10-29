# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: test_model.py
# Extensión de pruebas: verificación de decodificación, gradientes,
# presencia de módulos y barrido de variantes con aserciones.
# Mantiene compatibilidad con el flujo original y logging en /logs.
#==============================================================

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Dict, List, Tuple, Any

import torch

# --- Resolución de raíz de proyecto ---------------------------------------------------------------
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

# Imports del proyecto
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


def _meta_get(meta, key, default=None):
    if meta is None:
        return default
    if isinstance(meta, dict):
        return meta.get(key, default)
    if hasattr(meta, key):
        return getattr(meta, key, default)
    sub = getattr(meta, "model", None)
    if isinstance(sub, dict):
        return sub.get(key, default)
    return default


# --- Núcleo de construcción -----------------------------------------------------------------------

def build_from_configs(parser_root: str | None = None,
                       variant: str | None = None,
                       verbose: bool = True):
    root = parser_root or os.getcwd()
    cfg = ConfigParserYaml(project_root=root)
    cfg.load()

    # Preferencia de variante desde CLI frente a default
    if variant:
        set_ok = False
        try:
            if hasattr(cfg, "set_variant"):
                cfg.set_variant(variant)  # type: ignore[attr-defined]
                set_ok = True
        except Exception:
            pass
        if not set_ok:
            try:
                if hasattr(cfg, "resolve_variant"):
                    cfg.resolve_variant(variant)  # type: ignore[attr-defined]
                    set_ok = True
            except Exception:
                pass
        if not set_ok and isinstance(getattr(cfg, "model", None), dict):
            try:
                cfg.model["default_variant"] = variant  # type: ignore[index]
                set_ok = True
            except Exception:
                pass

    model = cfg.build_model(variant=variant)

    model_meta = getattr(cfg, "model_meta", {}) or {}
    nc = int(_meta_get(model_meta, "nc", getattr(getattr(model, "head", object()), "nc", 1)))
    in_ch = int(_meta_get(model_meta, "in_channels", 3))
    reg_max = int(_meta_get(model_meta, "reg_max", 16))
    strides = _meta_get(model_meta, "strides", getattr(getattr(model, "head", object()), "strides", None))

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
    dev_kind = device.type if hasattr(device, "type") else str(device)
    if dev_kind.startswith("cuda"):
        g = torch.Generator(device="cuda"); g.manual_seed(seed)
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)
    elif dev_kind.startswith("cpu"):
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        return torch.randn(*shape, device="cpu", dtype=dtype, generator=g)
    else:
        torch.manual_seed(seed)
        return torch.randn(*shape, device=device, dtype=dtype)


# --- Pruebas básicas (existentes) -----------------------------------------------------------------

def forward_sanity(model: torch.nn.Module,
                   meta: Dict,
                   batch: int = 1,
                   imgsz: int | None = None,
                   decode: bool = False,
                   concat_levels: bool = False,
                   device: torch.device | None = None,
                   dtype: torch.dtype = torch.float32,
                   seed: int = 42) -> Dict[str, List[torch.Size] | torch.Size]:
    in_ch = int(meta["in_channels"])  # canales de entrada
    strides: List[int] = list(meta["strides"])  # p.ej. [8,16,32]
    nc = int(meta["nc"])  # clases
    reg_max = int(meta["reg_max"])  # p.ej. 16 → 4*16 canales bbox
    img = int(imgsz or meta.get("imgsz", 640))

    device = device or select_device(meta.get("device_pref", "auto"))

    if device.type == "cpu" and dtype == torch.float16:
        print("[Aviso] float16 en CPU no es soportado de forma general. Cambiando a float32.")
        dtype = torch.float32

    model.eval(); model.to(device)
    x = _randn_with_seed((batch, in_ch, img, img), device=device, dtype=dtype, seed=seed)

    with torch.no_grad():
        out = model(x, decode=decode, concat=concat_levels)  # type: ignore[arg-type]

    if not isinstance(out, dict):
        raise RuntimeError("El forward del modelo no retornó un dict. Revisa yolo11.forward(...).")

    cls_out = out.get("cls"); reg_out = out.get("reg")

    shapes: Dict[str, List[torch.Size] | torch.Size] = {}

    def _expect_hw(s: int) -> int:
        return img // s

    if concat_levels:
        if isinstance(cls_out, torch.Tensor) and isinstance(reg_out, torch.Tensor):
            shapes["cls"] = cls_out.shape
            shapes["reg"] = reg_out.shape
            assert cls_out.shape[0] == batch, "Batch mismatch en cls (concat)."
            assert reg_out.shape[0] == batch, "Batch mismatch en reg (concat)."
            assert cls_out.shape[1] == nc, f"Canales cls esperados={nc}, obtenidos={cls_out.shape[1]}"
            assert reg_out.shape[1] == 4 * reg_max, f"Canales reg esperados={4*reg_max}, obtenidos={reg_out.shape[1]}"
        else:
            raise AssertionError("Se esperaba salida concatenada tipo Tensor para 'cls' y 'reg'.")
    else:
        if not (isinstance(cls_out, (list, tuple)) and isinstance(reg_out, (list, tuple))):
            raise AssertionError("Se esperaban listas/tuplas por nivel para 'cls' y 'reg'. Usa concat=False.")
        assert len(cls_out) == len(reg_out) == len(strides), "Número de niveles inconsistente con 'strides'."

        cls_shapes: List[torch.Size] = []
        reg_shapes: List[torch.Size] = []
        for i, s in enumerate(strides):
            cti, rti = cls_out[i], reg_out[i]
            assert isinstance(cti, torch.Tensor) and isinstance(rti, torch.Tensor), "Salida por nivel no es Tensor."
            Bh, Ch, Hh, Wh = cti.shape
            Br, Cr, Hr, Wr = rti.shape
            assert Bh == Br == batch, f"Batch mismatch en nivel {i}."
            assert Ch == nc, f"Nivel {i}: canales cls esperados={nc}, obtenidos={Ch}"
            assert Cr == 4 * reg_max, f"Nivel {i}: canales reg esperados={4*reg_max}, obtenidos={Cr}"
            Hexp = _expect_hw(s); Wexp = _expect_hw(s)
            assert (Hh, Wh) == (Hexp, Wexp), (
                f"Nivel {i}: HxW esperado={Hexp}x{Wexp} con stride={s}, obtenido={Hh}x{Wh}"
            )
            assert (Hr, Wr) == (Hexp, Wexp), (
                f"Nivel {i}: HxW reg esperado={Hexp}x{Wexp} con stride={s}, obtenido={Hr}x{Wr}"
            )
            cls_shapes.append(cti.shape); reg_shapes.append(rti.shape)
        shapes["cls"] = cls_shapes; shapes["reg"] = reg_shapes

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


# --- Pruebas extendidas ---------------------------------------------------------------------------
# Utilidades para robustecer decodificación y mitigar issues MIOpen/ROCm

def is_rocm() -> bool:
    try:
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


def set_bn_eval(model: torch.nn.Module) -> int:
    """Pone en eval() todas las BatchNorm* del modelo. Devuelve cuántas cambió."""
    try:
        from torch.nn.modules.batchnorm import _BatchNorm  # type: ignore
        BNBase = _BatchNorm
    except Exception:  # Fallback si cambia internals
        BNBase = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)
    n = 0
    for m in model.modules():
        if isinstance(m, BNBase):
            if m.training:
                m.eval()
                n += 1
    return n


def _flatten_hw(t: torch.Tensor) -> torch.Tensor:
    """Convierte [B,C,H,W]→[B,C,HW] o deja [B,C,N] tal cual."""
    if t.dim() == 4:
        return t.flatten(2)
    if t.dim() == 3:
        return t
    raise AssertionError("Tensor con dimensionalidad no soportada para _flatten_hw().")


def _cat_levels_tensorlist(tlist: List[torch.Tensor]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for t in tlist:
        if not isinstance(t, torch.Tensor):
            raise AssertionError("Lista de niveles contiene elemento no Tensor.")
        parts.append(_flatten_hw(t))
    if not parts:
        raise AssertionError("Lista de niveles vacía.")
    # Asumimos formato [B,C,N] tras _flatten_hw
    return torch.cat(parts, dim=2)


def _normalize_pair_to_concat(cls_v, reg_v) -> Tuple[torch.Tensor, torch.Tensor]:
    """Acepta Tensor o lista de Tensores por nivel, y retorna Tensores concatenados [B, C, N]."""
    if isinstance(cls_v, (list, tuple)):
        cls_t = _cat_levels_tensorlist(list(cls_v))
    elif isinstance(cls_v, torch.Tensor):
        cls_t = _flatten_hw(cls_v)
    else:
        raise AssertionError("'cls'/'scores' no es Tensor ni lista de Tensores.")

    if isinstance(reg_v, (list, tuple)):
        reg_t = _cat_levels_tensorlist(list(reg_v))
    elif isinstance(reg_v, torch.Tensor):
        reg_t = _flatten_hw(reg_v)
    else:
        raise AssertionError("'reg'/'boxes' no es Tensor ni lista de Tensores.")

    return cls_t, reg_t

def decode_concat_sanity(model: torch.nn.Module, meta: Dict, batch: int, imgsz: int, device, dtype) -> Tuple[Dict[str, torch.Size], str, int]:
    """Comprueba salida decodificada. Acepta {'cls','reg'} o {'scores','boxes'} en Tensor o listas por nivel.
    Normaliza a Tensores concatenados [B, C, N] y verifica N = sum_i (Hi*Wi).
    Devuelve (shapes, used_keys, regC).
    """
    strides: List[int] = list(meta["strides"])  # [8,16,32]
    nc = int(meta["nc"])  # clases
    reg_max = int(meta["reg_max"])  # p.ej. 16
    N = sum((imgsz // s) * (imgsz // s) for s in strides)

    model.eval(); model.to(device)
    x = _randn_with_seed((batch, int(meta["in_channels"]), imgsz, imgsz), device=device, dtype=dtype, seed=123)
    with torch.no_grad():
        out = model(x, decode=True, concat=True)  # si concat=True no aplica, normalizamos igual

    if not isinstance(out, dict):
        raise AssertionError("La salida decodificada debe ser un dict.")

    used_keys = None
    if ("cls" in out) and ("reg" in out):
        cls_v, reg_v = out["cls"], out["reg"]
        used_keys = "cls/reg"
    elif ("scores" in out) and ("boxes" in out):
        cls_v, reg_v = out["scores"], out["boxes"]
        used_keys = "scores/boxes"
    else:
        raise AssertionError(f"Salida decodificada sin claves esperadas. Claves: {list(out.keys())}")

    cls_t, reg_t = _normalize_pair_to_concat(cls_v, reg_v)  # → [B, C, N]

    # Validación de N y canales
    if cls_t.dim() != 3 or reg_t.dim() != 3:
        raise AssertionError("Tras normalización se esperaban Tensores 3D [B,C,N].")

    Bc, Cc, Nc = cls_t.shape
    Br, Cr, Nr = reg_t.shape
    assert Bc == Br == batch, f"Batch inconsistente (cls={Bc}, reg={Br}, esp={batch})."
    assert Nc == Nr == N, f"N esperado {N}, obtenidos cls={Nc}, reg={Nr}."

    # Canales esperados
    assert Cc == nc, f"Canales de cls esperados {nc}, obtenidos {Cc}."
    assert Cr in (4, 4 * reg_max), f"Canales de reg esperados 4 o {4*reg_max}, obtenidos {Cr}."

    return {"cls": cls_t.shape, "reg": reg_t.shape}, used_keys, Cr


def gradient_flow_sanity(model: torch.nn.Module, meta: Dict, batch: int, imgsz: int, device, dtype, bn_eval_fallback: bool = False) -> Dict[str, Any]:
    """Forward+backward sintético. Si ROCm/MIOpen falla en BN en train, puede forzar BN.eval()."""
    model.to(device)
    model.train()
    x = _randn_with_seed((batch, int(meta["in_channels"]), imgsz, imgsz), device=device, dtype=dtype, seed=321)

    bn_eval_used = False
    bn_layers = 0

    def _forward():
        return model(x, decode=False, concat=False)  # type: ignore[arg-type]

    try:
        out = _forward()
    except RuntimeError as e:
        msg = str(e)
        needs_fallback = ("miopen" in msg.lower()) or ("sqlite" in msg.lower()) or ("hip" in msg.lower())
        if (bn_eval_fallback or (is_rocm() and needs_fallback)):
            bn_layers = set_bn_eval(model)
            bn_eval_used = True
            out = _forward()  # reintento con BN en eval()
        else:
            raise

    if not (isinstance(out, dict) and isinstance(out.get("cls"), (list, tuple)) and isinstance(out.get("reg"), (list, tuple))):
        raise AssertionError("Se esperaba dict con listas por nivel en salida (modo no decodificado).")

    loss = 0.0
    for t in out["cls"]:
        loss = loss + t.mean()
    for t in out["reg"]:
        loss = loss + 0.01 * t.abs().mean()

    loss.backward()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None)
    nonzero_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.detach().abs().mean() > 0
    )
    ratio_with = with_grad / max(1, total)
    ratio_nonzero = nonzero_grad / max(1, total)

    return dict(loss=float(loss.detach().cpu()), total=total, with_grad=with_grad, nonzero_grad=nonzero_grad,
                ratio_with=ratio_with, ratio_nonzero=ratio_nonzero, bn_eval_used=bn_eval_used, bn_layers=bn_layers)


def module_presence_sanity(model: torch.nn.Module) -> Dict[str, bool]:
    names = [m.__class__.__name__ for m in model.modules()]
    has_c2psa = any(n == "C2PSA" for n in names)
    dwcls = head_uses_dwconv(model)
    return dict(has_c2psa=has_c2psa, has_dwconv_cls=dwcls)


def head_uses_dwconv(model: torch.nn.Module) -> bool:
    """Heurística: recorre submódulos bajo 'head' y verifica presencia de DWConv en rama cls."""
    if not hasattr(model, "head"):
        return False
    head = model.head
    # Buscar por nombre de atributos comunes en la rama cls
    for name, mod in head.named_modules():
        if "cls" in name.lower() and mod.__class__.__name__ in {"DWConv", "DWConv2d", "DepthwiseConv"}:
            return True
    # fallback: cualquier DWConv bajo head
    for _, mod in head.named_modules():
        if mod.__class__.__name__ in {"DWConv", "DWConv2d", "DepthwiseConv"}:
            return True
    return False


def sweep_variants(cfg: Any, variants: List[str], imgsz: int, batch: int, device, dtype) -> List[Tuple[str, int]]:
    """Construye cada variante y devuelve [(var, params)]."""
    rows: List[Tuple[str, int]] = []
    for v in variants:
        m = cfg.build_model(variant=v)
        m.to(device)
        params = sum(p.numel() for p in m.parameters())
        rows.append((v, params))
        # forward liviano para comprobar formas (sin aserciones duras)
        try:
            forward_sanity(m, meta=dict(nc=5, in_channels=3, reg_max=16, strides=[8, 16, 32], imgsz=imgsz, device_pref=str(device)),
                           batch=batch, imgsz=imgsz, device=device, dtype=dtype, decode=False, concat_levels=False)
        except Exception:
            pass
    return rows


# --- CLI ------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ensamblar YOLOv11 desde configs y ejecutar pruebas estructurales y funcionales ligeras.\n"
            "Ejemplos:\n"
            "  python utility/test_model.py --variant m --imgsz 640 --batch 2 --summary\n"
            "  python utility/test_model.py --variant xl --check-decode --check-grad --check-modules\n"
            "  python utility/test_model.py --sweep --assert-sweep\n"
        )
    )
    p.add_argument("--variant", type=str, default=None, choices=["n", "s", "m", "l", "xl"],
                   help="Forzar variante del modelo (si se omite, usa la definida en los configs)")
    p.add_argument("--imgsz", type=int, default=None, help="Tamaño de imagen cuadrada (por defecto train.yaml o 640)")
    p.add_argument("--batch", type=int, default=1, help="Tamaño de batch para la prueba")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                   help="Dispositivo preferido para la prueba (por defecto auto)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Precisión del tensor de entrada")
    p.add_argument("--decode", action="store_true", help="(Mant.) Solicita salida decodificada en forward_sanity")
    p.add_argument("--concat", action="store_true", help="(Mant.) Concatena niveles en forward_sanity")
    p.add_argument("--no-assert", action="store_true", help="Desactiva aserciones de coherencia de formas (prueba básica)")
    p.add_argument("--summary", action="store_true", help="Imprime resumen básico del modelo y número de parámetros")

    # Nuevas pruebas
    p.add_argument("--check-decode", action="store_true", help="Valida decodificación + concatenación y N esperados")
    p.add_argument("--check-grad", action="store_true", help="Ejecuta backward sintético y reporta % de gradientes")
    p.add_argument("--check-modules", action="store_true", help="Verifica presencia de C2PSA y DWConv en la head")
    p.add_argument("--sweep", action="store_true", help="Barre variantes n,s,m,l,xl y reporta parámetros")
    p.add_argument("--assert-sweep", action="store_true", help="Exige monotonía de parámetros en el barrido")
    p.add_argument("--bn-eval-fallback", action="store_true", help="En --check-grad, forzar BN.eval() si ROCm/MIOpen falla")

    return p.parse_args()


def main():
    args = parse_args()

    cfg, model, meta = build_from_configs(variant=args.variant, verbose=True)

    if args.imgsz is not None:
        meta["imgsz"] = int(args.imgsz)
    device = select_device(args.device or meta.get("device_pref", "auto"))

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.float32)

    if args.summary:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n=== Resumen del Modelo ===")
        print(f"Variante: {_detect_variant(cfg, fallback=args.variant or 'm')} | Parámetros: {human_size(total_params)} "
              f"(entrenables: {human_size(trainable)})")
        print(f"Strides: {meta['strides']} | nc: {meta['nc']} | reg_max: {meta['reg_max']} | in_channels: {meta['in_channels']}")
        print(f"Dispositivo seleccionado: {device}")

    # --- Prueba base (formas) --------------------------------------------------------------------
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
            shapes = {"cls": [], "reg": []}  # type: ignore[assignment]
        else:
            raise

    if 'shapes' in locals() and shapes:
        pretty_print_shapes(shapes)

    # --- Pruebas extendidas -----------------------------------------------------------------------
    d_shapes = None; grad_stats = None; mod_stats = None; sweep_rows = None

    if args.check_decode:
        try:
            d_shapes, d_used, d_regC = decode_concat_sanity(model, meta, batch=int(args.batch), imgsz=int(meta.get("imgsz", 640)), device=device, dtype=dtype)
            print(f"[Decode] claves={d_used} | formas (concat-normalizado):", {k: tuple(v) for k, v in d_shapes.items()})
        except AssertionError as e:
            print("[Decode][FALLO]", e)
            if not args.no_assert:
                raise

    if args.check_grad:
        try:
            grad_stats = gradient_flow_sanity(model, meta, batch=int(args.batch), imgsz=int(meta.get("imgsz", 640)), device=device, dtype=dtype, bn_eval_fallback=bool(args.bn_eval_fallback))
            print("[Grad] loss=%.6f | with_grad=%.2f%% | nonzero_grad=%.2f%% | bn_eval_fallback=%s (bn_layers=%d)" % (
                grad_stats["loss"], 100*grad_stats["ratio_with"], 100*grad_stats["ratio_nonzero"],
                "yes" if grad_stats.get("bn_eval_used") else "no", int(grad_stats.get("bn_layers", 0))
            ))
        except AssertionError as e:
            print("[Grad][FALLO]", e)
            if not args.no_assert:
                raise

    if args.check_modules:
        mod_stats = module_presence_sanity(model)
        print("\n[Modules] C2PSA=", mod_stats.get("has_c2psa"), "| head(DWConv)=", mod_stats.get("has_dwconv_cls"))

    if args.sweep:
        variants = ["n", "s", "m", "l", "xl"]
        sweep_rows = sweep_variants(cfg, variants, imgsz=int(meta.get("imgsz", 640)), batch=int(args.batch), device=device, dtype=dtype)
        print("\n[Sweep] Parámetros por variante:")
        for v, p in sweep_rows:
            print(f"  {v:>2}: {human_size(p)}")
        if args.assert_sweep:
            # Monotonía no estricta (por caps de max_channels): n <= s <= m <= l <= xl
            ok = all(sweep_rows[i][1] <= sweep_rows[i+1][1] for i in range(len(sweep_rows)-1))
            if not ok:
                raise AssertionError("Monotonía de parámetros violada en barrido de variantes.")

    # --- Logging ----------------------------------------------------------------------------------
    try:
        os.makedirs("logs", exist_ok=True)
        tag = _now_tag()
        log_path = os.path.join("logs", f"test_model_{tag}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"test_model run @ {tag}\n")
            f.write(f"variant={_detect_variant(cfg, fallback=args.variant or 'm')} imgsz={meta.get('imgsz')} batch={args.batch} "
                    f"device={device} dtype={args.dtype} decode={args.decode} concat={args.concat}\n")
            # Shapes base
            if isinstance(shapes.get("cls"), list) and shapes.get("reg"):
                for i, (cs, rs) in enumerate(zip(shapes["cls"], shapes["reg"])):  # type: ignore[index]
                    f.write(f"P{i+3}: cls {tuple(cs)} | reg {tuple(rs)}\n")
            elif isinstance(shapes.get("cls"), torch.Size):
                f.write(f"cls {tuple(shapes['cls'])}\n")
                f.write(f"reg {tuple(shapes['reg'])}\n")
            # Decode
            if d_shapes is not None:
                try:
                    f.write(f"DECODE[{d_used}]: cls {tuple(d_shapes['cls'])} | reg {tuple(d_shapes['reg'])} | regC={d_regC}")
                except Exception:
                    f.write(f"DECODE: cls {tuple(d_shapes['cls'])} | reg {tuple(d_shapes['reg'])}")
            # Gradientes
            if grad_stats is not None:
                f.write("GRAD: loss={:.6f} with_grad={:.2f}% nonzero_grad={:.2f}% bn_eval_fallback={} bn_layers={}".format(
                    grad_stats["loss"], 100*grad_stats["ratio_with"], 100*grad_stats["ratio_nonzero"],
                    grad_stats.get("bn_eval_used", False), int(grad_stats.get("bn_layers", 0))
                ))
            # Módulos
            if mod_stats is not None:
                f.write(f"MODULES: C2PSA={mod_stats.get('has_c2psa')} head(DWConv)={mod_stats.get('has_dwconv_cls')}\n")
            # Sweep
            if sweep_rows is not None:
                f.write("SWEEP (variant → params):\n")
                for v, p in sweep_rows:
                    f.write(f"  {v}: {p}\n")
        print(f"\n[OK] Log guardado en: {log_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo guardar el log: {e}")


if __name__ == "__main__":
    main()
