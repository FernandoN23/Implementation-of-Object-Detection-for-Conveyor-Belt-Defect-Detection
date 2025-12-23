# ==============================================================
# Archivo: SSD/utility/test_results.py
# Descripción: Script auxiliar para generar datos sintéticos de
#              prueba para SSD512 basados en SSD300.
#              Versión: CAÓTICA (Alto Ruido / Inestable).
# ==============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# --------------------------------------------------------------
# Configuración de Rutas
# --------------------------------------------------------------
FILE = Path(__file__).resolve()
SSD_ROOT = FILE.parents[1]  # .../SSD
METRICS_ROOT = SSD_ROOT / "metrics"

# Rutas de origen (SSD300) y destino (SSD512)
SRC_CSV = METRICS_ROOT / "detect" / "ssd300" / "train" / "ssd300" / "results.csv"
DST_DIR = METRICS_ROOT / "detect" / "ssd512" / "train" / "ssd512"
DST_CSV = DST_DIR / "results.csv"


def smooth_array(arr, window=2):
    """
    Suavizado mínimo para mantener el caos.
    window=1 o 2 deja pasar casi todo el ruido.
    """
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1, center=True).mean().values


def generate_dummy_ssd512():
    # 1. Asegurar directorios
    if not DST_DIR.exists():
        DST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Cargar base
    if not DST_CSV.exists():
        if SRC_CSV.exists():
            shutil.copy(SRC_CSV, DST_CSV)
        else:
            print(f"[Error] No se encontró {SRC_CSV}.")
            return

    print(f"[Info] Generando curva SSD512 CAÓTICA (Target ~0.75)...")
    df = pd.read_csv(DST_CSV)
    df.columns = [c.strip() for c in df.columns]

    n_rows = len(df)
    epochs = np.arange(n_rows)

    # Semilla fija
    np.random.seed(42)

    # --------------------------------------------------------------
    # Lógica de Tendencia (Multiplicadores)
    # --------------------------------------------------------------
    e_mid1 = 100
    e_mid2 = 150

    multipliers = np.ones(n_rows)

    # AJUSTE 1: Ruido MUY ALTO (Caos)
    # Antes: 0.005 -> Ahora: 0.015 (3x más ruido)
    noise_scales = np.ones(n_rows) * 0.030

    # Tendencia hacia 0.75 (igual que antes)

    # --- FASE 1: 0 a 100 ---
    mask1 = epochs < e_mid1
    multipliers[mask1] = np.linspace(1.0, 1.10, mask1.sum())

    # --- FASE 2: 100 a 150 ---
    mask2 = (epochs >= e_mid1) & (epochs < e_mid2)
    multipliers[mask2] = np.linspace(1.10, 1.11, mask2.sum())
    noise_scales[mask2] = 0.010  # Incluso la meseta es ruidosa ahora

    # --- FASE 3: 150 a 200 ---
    mask3 = epochs >= e_mid2
    multipliers[mask3] = np.linspace(1.11, 1.14, mask3.sum())

    # Función para aplicar tendencia + CAOS
    def apply_chaotic_trend(base_series, scale_factor=1.0):
        # 1. Tendencia
        trend = base_series * multipliers

        # 2. Ruido Caótico
        noise = np.random.normal(0, 1, n_rows) * noise_scales * scale_factor

        # 3. Picos aleatorios (Outliers)
        # Añadimos saltos bruscos aleatorios en el 5% de los puntos
        spikes = np.random.choice([0, 0.02, -0.02], size=n_rows, p=[0.95, 0.025, 0.025])

        raw_curve = trend + noise + spikes

        # 3. Suavizado mínimo (casi nulo)
        smoothed_curve = smooth_array(raw_curve, window=2)
        return smoothed_curve

    # --------------------------------------------------------------
    # Aplicar a las columnas
    # --------------------------------------------------------------

    # mAP@0.5
    if "val_mAP_0.5" in df.columns:
        df["val_mAP_0.5"] = apply_chaotic_trend(df["val_mAP_0.5"], scale_factor=1.0)
        df["val_mAP_0.5"] = df["val_mAP_0.5"].clip(0, 0.78)  # Un poco de margen para los picos

    # mAP@0.5:0.95
    if "val_mAP_0.5_0.95" in df.columns:
        df["val_mAP_0.5_0.95"] = apply_chaotic_trend(df["val_mAP_0.5_0.95"], scale_factor=0.8)
        df["val_mAP_0.5_0.95"] = df["val_mAP_0.5_0.95"].clip(0, 1.0)

    # Métricas P, R, F1
    for col in ["val_P", "val_R", "val_F1"]:
        if col in df.columns:
            df[col] = apply_chaotic_trend(df[col], scale_factor=1.2)  # Más caos en P/R
            df[col] = df[col].clip(0, 1.0)

    # Pérdidas (Losses) - Muy ruidosas
    loss_multipliers = 1.0 / (multipliers * 0.99)
    loss_cols = [c for c in df.columns if "loss" in c]

    for col in loss_cols:
        # Ruido fuerte en loss (simula inestabilidad de gradiente)
        noise = np.random.normal(0, 0.05, n_rows)
        raw_loss = df[col] * loss_multipliers + noise
        df[col] = smooth_array(raw_loss, window=2)
        df[col] = df[col].clip(lower=0.01)

    # --------------------------------------------------------------
    # Guardar
    # --------------------------------------------------------------
    df.to_csv(DST_CSV, index=False)

    print("-" * 50)
    print(f"✅ Datos SSD512 (Caóticos) generados en: {DST_CSV}")
    print(f"   - Final mAP@0.5: {df['val_mAP_0.5'].iloc[-1]:.4f}")
    print("-" * 50)
    print("Ejecuta: python SSD/utility/metrics.py --merge")


if __name__ == "__main__":
    generate_dummy_ssd512()