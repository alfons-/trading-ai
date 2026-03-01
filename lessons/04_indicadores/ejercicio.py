"""
Lección 04: Indicadores técnicos (SMA, RSI) y gráfico con matplotlib.

Carga datos de AAPL, calcula SMA(20) y RSI(14), y dibuja precio + SMA.
Si data/AAPL.csv no existe, descarga 1 año con yfinance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas: data/ en raíz del proyecto
CARPETA_LECCION = Path(__file__).resolve().parent
RAIZ = CARPETA_LECCION.parent.parent
CARPETA_DATA = RAIZ / "data"
CSV_AAPL = CARPETA_DATA / "AAPL.csv"

# --- 1. Cargar datos (o descargar si no hay CSV) ---
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
    print("Guardado en", CSV_AAPL)

df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# --- 2. SMA(20) con Pandas (media móvil de los últimos 20 cierres) ---
ventana_sma = 20
df["sma_20"] = df["cierre"].rolling(window=ventana_sma).mean()

# --- 3. RSI(14) con la librería ta ---
from ta.momentum import RSIIndicator

ventana_rsi = 14
rsi_indicator = RSIIndicator(close=df["cierre"], window=ventana_rsi)
df["rsi_14"] = rsi_indicator.rsi()

# Mostrar últimas filas con indicadores
print("\n=== Últimas 5 filas (cierre, sma_20, rsi_14) ===")
print(df[["fecha", "cierre", "sma_20", "rsi_14"]].tail())
print("\nRSI último valor:", round(df["rsi_14"].iloc[-1], 2))

# --- 4. Gráfico: precio y SMA en el mismo eje ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])

# Panel superior: precio y SMA
ax1.plot(df["fecha"], df["cierre"], label="Cierre", color="black", linewidth=1)
ax1.plot(df["fecha"], df["sma_20"], label="SMA(20)", color="blue", linewidth=1.5)
ax1.set_ylabel("Precio (USD)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_title("AAPL – Precio y media móvil simple (20)")

# Panel inferior: RSI
ax2.plot(df["fecha"], df["rsi_14"], label="RSI(14)", color="green", linewidth=1)
ax2.axhline(y=70, color="red", linestyle="--", alpha=0.7, label="Sobrecompra (70)")
ax2.axhline(y=30, color="green", linestyle="--", alpha=0.7, label="Sobreventa (30)")
ax2.set_ylabel("RSI")
ax2.set_xlabel("Fecha")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
grafico_path = CARPETA_LECCION / "grafico_precio_sma.png"
plt.savefig(grafico_path, dpi=150)
print(f"\nGráfico guardado: {grafico_path}")
# Si ejecutas en un entorno con ventanas (no en terminal sin pantalla), descomenta:
plt.show()
plt.close()

print("\n--- Fin Lección 04 ---")
print("Siguiente: Lección 05 - Señales y backtesting (cruce de medias).")
