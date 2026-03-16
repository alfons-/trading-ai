"""
Ejemplo de backtesting: estrategia de cruce de medias móviles (SMA).

- Carga datos OHLC desde un CSV (fecha, abierto, alto, bajo, cierre, volumen).
- Calcula SMA corta (5) y SMA larga (20).
- Compra cuando SMA corta cruza por encima de SMA larga; vende cuando cruza por debajo.
- Simula posición y calcula resultado (beneficio/pérdida, número de operaciones).
"""

import pandas as pd
from pathlib import Path

# Rutas: intentar data/AAPL.csv (tras ejecutar ejercicio.py) o CSV de ejemplo
RAIZ = Path(__file__).resolve().parent.parent.parent
CSV_DATA = RAIZ / "data" / "AAPL.csv"
CSV_EJEMPLO = RAIZ / "lessons" / "02_datos_y_pandas" / "precios_ejemplo.csv"

# --- 1. Cargar datos ---
if CSV_DATA.exists():
    df = pd.read_csv(CSV_DATA)
    print(f"Datos cargados: {CSV_DATA}")
else:
    df = pd.read_csv(CSV_EJEMPLO)
    print(f"Datos cargados (ejemplo): {CSV_EJEMPLO}")

df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# --- 2. Indicadores: medias móviles ---
PERIODO_CORTA = 5
PERIODO_LARGA = 100
df["sma_corta"] = df["cierre"].rolling(PERIODO_CORTA).mean()
df["sma_larga"] = df["cierre"].rolling(PERIODO_LARGA).mean()

# --- 3. Señales: cruce al alza = compra, cruce a la baja = venta ---
# Posición anterior: 1 = largo, 0 = fuera
df["sma_arriba"] = df["sma_corta"] > df["sma_larga"]
df["senal"] = df["sma_arriba"].astype(int)  # 1 = largo, 0 = no posicion
# Detección de cruces (cambio de 0 a 1 o de 1 a 0)
df["cruce_compra"] = (df["senal"] == 1) & (df["senal"].shift(1) == 0)
df["cruce_venta"] = (df["senal"] == 0) & (df["senal"].shift(1) == 1)

# --- 4. Retornos diarios (solo cuando tenemos SMA válida) ---
df["retorno"] = df["cierre"].pct_change()
# Solo ganamos/perdemos cuando estamos en posición (senal == 1)
df["retorno_estrategia"] = df["senal"].shift(1) * df["retorno"]
df.loc[df["sma_larga"].isna(), "retorno_estrategia"] = None

# --- 5. Métricas ---
df_ok = df.dropna(subset=["retorno_estrategia"])
retorno_total = (1 + df_ok["retorno_estrategia"]).prod() - 1
num_operaciones = df["cruce_compra"].sum() + df["cruce_venta"].sum()

print("\n=== Resultados del backtest ===")
print(f"Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"Días con señal: {len(df_ok)}")
print(f"Retorno total (estrategia): {retorno_total:.2%}")
print(f"Número de operaciones (entradas + salidas): {int(num_operaciones)}")
print("\n--- Fin backtest ---")
