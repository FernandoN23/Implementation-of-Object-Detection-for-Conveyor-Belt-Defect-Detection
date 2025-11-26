# ==============================================================
# Departamento de Ingeniería Mecánica - Universidad de Chile
# Trabajo de Memoria de Título:
# "Implementación de algoritmos de reconocimiento de objetos
#  para la identificación de fallas en correas transportadoras"
# Autor: Fernando N.
# --------------------------------------------------------------
# Archivo: check_dataset.py
# Chequea consistencia del dataset en formato YOLO (imágenes y labels) a partir
# de configs/dataset.yaml o ruta provista. ACEPTA explícitamente imágenes sin
# clases (negativas), ya sea con label vacío o sin archivo de label, y las
# reporta como tales sin marcarlas como error (salvo que se use --strict).
#==============================================================

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LBL_EXT = ".txt"

def find_project_root(start: Path = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()

def load_dataset_yaml(ds_yaml: Path) -> Dict:
    with open(ds_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Normaliza claves
    keys_map = {"val": "valid"}
    for k_old, k_new in keys_map.items():
        if k_old in data and k_new not in data:
            data[k_new] = data[k_old]
    return data

def images_in(path: Path) -> List[Path]:
    if not path.exists():
        return []
    files = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)

def label_path_for(img_path: Path) -> Path:
    # Layout típico: .../images/... -> .../labels/...
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(LBL_EXT)
        return lbl
    except ValueError:
        # Fallback: mismo padre con 'labels' como hermano
        if img_path.parent.name.lower() == "images":
            lbl_dir = img_path.parent.parent / "labels"
            return lbl_dir / (img_path.stem + LBL_EXT)
        return img_path.with_suffix(LBL_EXT)  # último recurso

def parse_label_line(line: str) -> Tuple[int, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Línea inválida (se esperaban 5 valores): '{line.strip()}'")
    c = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    return c, x, y, w, h

def check_split(name: str, images_dir: Path, nc: int, allow_empty: bool, allow_missing: bool) -> Dict:
    imgs = images_in(images_dir)
    total = len(imgs)
    issues = {
        "missing_labels": [],       # sólo si NO permitido
        "empty_labels": [],         # sólo si NO permitido
        "bad_lines": [],
        "bad_ranges": [],
        "class_oob": [],
    }
    negatives = {  # reportes informativos cuando se permiten negativos
        "negatives_missing": [],   # imagen sin archivo de label
        "negatives_empty": [],     # archivo de label vacío
    }

    for img in imgs:
        lbl = label_path_for(img)
        if not lbl.exists():
            if allow_missing:
                negatives["negatives_missing"].append(str(img))
                continue
            else:
                issues["missing_labels"].append(str(img))
                continue
        # Leer label
        try:
            content = lbl.read_text(encoding="utf-8").strip()
        except Exception as e:
            issues["bad_lines"].append(f"{lbl} :: error leyendo archivo: {e}")
            continue
        if content == "":
            if allow_empty:
                negatives["negatives_empty"].append(str(img))
                continue
            else:
                issues["empty_labels"].append(str(lbl))
                continue
        # Hay anotaciones -> validar
        for i, line in enumerate(content.splitlines()):
            try:
                c, x, y, w, h = parse_label_line(line)
            except Exception as e:
                issues["bad_lines"].append(f"{lbl} (línea {i+1}): {e}")
                continue
            if c < 0 or c >= nc:
                issues["class_oob"].append(f"{lbl} (línea {i+1}): clase {c} fuera de [0,{nc-1}]")
            for v, k in zip((x, y, w, h), "xywh"):
                if not (0.0 <= v <= 1.0):
                    issues["bad_ranges"].append(f"{lbl} (línea {i+1}): {k}={v} fuera de [0,1]")

    ok = total > 0 and all(len(v) == 0 for k, v in issues.items())
    return {
        "split": name,
        "images_dir": str(images_dir),
        "total_images": total,
        "ok": ok,
        **issues,
        **negatives,
    }

def pretty_print_report(results: List[Dict], ds_yaml: Path, nc: int, names: Dict, allow_empty: bool, allow_missing: bool) -> int:
    print("=== Chequeo Dataset (YOLO) ===")
    print(f"Dataset YAML: {ds_yaml}")
    print(f"Clases (nc={nc}): {names}")
    print(f"Política negativos -> allow_empty_labels={allow_empty} | allow_missing_labels={allow_missing}")
    exit_code = 0
    for r in results:
        print(f"\n[{r['split']}]  imágenes={r['total_images']}  estado={'OK' if r['ok'] else 'CON ISSUES'}")
        # Reportes informativos (negativos)
        if r.get("negatives_missing"):
            print(f" - negativos (sin label): {len(r['negatives_missing'])}")
        if r.get("negatives_empty"):
            print(f" - negativos (label vacío): {len(r['negatives_empty'])}")
        # Issues
        for k in ("missing_labels", "empty_labels", "bad_lines", "bad_ranges", "class_oob"):
            if r[k]:
                exit_code = 1
                print(f" - {k}: {len(r[k])}")
                for s in r[k][:20]:
                    print(f"   · {s}")
                if len(r[k]) > 20:
                    print(f"   · ... (+{len(r[k]) - 20} más)")
    if exit_code == 0:
        print("\nResultado: ✅ Consistencia básica OK (negativos permitidos contabilizados arriba).")
    else:
        print("\nResultado: ⚠ Se detectaron inconsistencias. Revise el detalle arriba.")
    return exit_code

def main():
    parser = argparse.ArgumentParser(description="Chequeo de consistencia de dataset YOLO (images/labels) con soporte de negativos.")
    parser.add_argument("--dataset-yaml", type=str, default=None,
                        help="Ruta a configs/dataset.yaml (si no se entrega, se intenta resolver automáticamente).")
    parser.add_argument("--strict", action="store_true",
                        help="Modo estricto: considerar como error labels vacíos o ausentes (no se aceptan negativos).")
    parser.add_argument("--disallow-empty-labels", action="store_true",
                        help="Considerar label vacío como error (aunque no se use --strict).")
    parser.add_argument("--disallow-missing-labels", action="store_true",
                        help="Considerar ausencia de label como error (aunque no se use --strict).")
    args = parser.parse_args()

    project_root = find_project_root()
    ds_yaml = Path(args.dataset_yaml) if args.dataset_yaml else (project_root / "configs" / "dataset.yaml")
    if not ds_yaml.exists():
        print(f"ERROR: No se encontró dataset.yaml en: {ds_yaml}")
        sys.exit(2)

    data = load_dataset_yaml(ds_yaml)
    required = ["train", "valid", "test", "nc", "names"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"ERROR: dataset.yaml carece de claves requeridas: {missing}")
        sys.exit(2)

    # Política de negativos
    allow_empty = True
    allow_missing = True
    if args.strict:
        allow_empty = False
        allow_missing = False
    if args.disallow_empty_labels:
        allow_empty = False
    if args.disallow_missing_labels:
        allow_missing = False

    nc = int(data["nc"])
    names = data["names"]
    results = []
    for split_key in ("train", "valid", "test"):
        images_dir = Path(data[split_key])
        results.append(check_split(split_key, images_dir, nc, allow_empty, allow_missing))

    code = pretty_print_report(results, ds_yaml, nc, names, allow_empty, allow_missing)
    sys.exit(code)

if __name__ == "__main__":
    main()
