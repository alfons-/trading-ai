"""
Lección 05: Señales y backtesting – Cruce de medias SMA(20) y SMA(50).

Genera señales de compra/venta cuando la SMA rápida cruza a la lenta,
simula operaciones (entrar/salir al cierre) y calcula rentabilidad simple.
"""

import pandas as pd
from pathlib import Path

CARPETA_LECCION = Path(__file__).resolve().parent
RAIZ = CARPETA_LECCION.parent.parent
CARPETA_DATA = RAIZ / "data"
CSV_AAPL = CARPETA_DATA / "AAPL.csv"

# --- 1. Cargar datos (mismo patrón que Lección 04) ---
if CSV_AAPL.exists():
    print(f"Cargando {CSV_AAPL}...")
    df = pd.read_csv(CSV_AAPL)
else:
    print("No hay data/AAPL.csv. Descargando 1 año con yfinance...")
    import yfinance as yf
    hist = yf.Ticker("AAPL").history(period="1y")
    if hist.empty:
        print("Error: no se pudieron descargar datos.")
        exit(1)
    mapeo = {"Open": "abierto", "High": "alto", "Low": "bajo", "Close": "cierre", "Volume": "volumen"}
    df = hist[list(mapeo)].rename(columns=mapeo).reset_index()
    df = df.rename(columns={"Date": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    CARPETA_DATA.mkdir(exist_ok=True)
    df.to_csv(CSV_AAPL, index=False)

df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# --- 2. Calcular SMA(20) y SMA(50) ---
df["sma_20"] = df["cierre"].rolling(window=20).mean()
df["sma_50"] = df["cierre"].rolling(window=50).mean()

# Quitar filas donde no hay SMA(50) aún (necesitamos 50 datos)
df = df.dropna(subset=["sma_20", "sma_50"]).reset_index(drop=True)

# --- 3. Señales de cruce (usando valor anterior con .shift(1)) ---
# Cruce alcista: antes sma_20 <= sma_50, ahora sma_20 > sma_50 → comprar
# Cruce bajista: antes sma_20 >= sma_50, ahora sma_20 < sma_50 → vender
prev_sma20 = df["sma_20"].shift(1)
prev_sma50 = df["sma_50"].shift(1)

df["senal_compra"] = (prev_sma20 <= prev_sma50) & (df["sma_20"] > df["sma_50"])
df["senal_venta"] = (prev_sma20 >= prev_sma50) & (df["sma_20"] < df["sma_50"])

# Mostrar días con señal
compras = df[df["senal_compra"]][["fecha", "cierre", "sma_20", "sma_50"]]
ventas = df[df["senal_venta"]][["fecha", "cierre", "sma_20", "sma_50"]]

print("\n=== Señales de COMPRA (cruce alcista) ===")
print(compras.to_string(index=False))
print("\n=== Señales de VENTA (cruce bajista) ===")
print(ventas.to_string(index=False))

# --- 4. Simulación: emparejar compra → siguiente venta, calcular beneficio por operación ---
operaciones = []
i = 0
while i < len(df):
    if df["senal_compra"].iloc[i]:
        fecha_entrada = df["fecha"].iloc[i]
        precio_entrada = df["cierre"].iloc[i]
        # Buscar la siguiente señal de venta
        j = i + 1
        while j < len(df) and not df["senal_venta"].iloc[j]:
            j += 1
        if j < len(df):
            fecha_salida = df["fecha"].iloc[j]
            precio_salida = df["cierre"].iloc[j]
            retorno_pct = 100 * (precio_salida - precio_entrada) / precio_entrada
            operaciones.append({
                "entrada_fecha": fecha_entrada,
                "entrada_precio": precio_entrada,
                "salida_fecha": fecha_salida,
                "salida_precio": precio_salida,
                "retorno_pct": round(retorno_pct, 2),
            })
            i = j + 1
            continue
    i += 1

# --- 5. Métricas ---
if not operaciones:
    print("\nNo se generaron operaciones completas (compra sin venta posterior en el periodo).")
else:
    ops_df = pd.DataFrame(operaciones)
    print("\n=== Operaciones simuladas (entrada → salida) ===")
    print(ops_df.to_string(index=False))

    retorno_total_pct = sum(o["retorno_pct"] for o in operaciones)
    num_ops = len(operaciones)
    retorno_medio_pct = retorno_total_pct / num_ops

    print("\n--- Resumen del backtest ---")
    print(f"Número de operaciones: {num_ops}")
    print(f"Rentabilidad total (suma de cada operación): {round(retorno_total_pct, 2)} %")
    print(f"Rentabilidad media por operación: {round(retorno_medio_pct, 2)} %")
    print("(Nota: sin costes ni deslizamiento; simplificado para aprendizaje.)")

print("\n--- Fin Lección 05 ---")
print("Siguiente: refactorizar en src/ (módulos data, indicators) o añadir más estrategias.")
