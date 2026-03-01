"""
Lección 03: Datos de mercado con yfinance.

Descarga precios reales (OHLCV), muestra las primeras filas y guarda en CSV.
Necesitas conexión a internet para la descarga.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# Carpeta data/ en la raíz del proyecto (dos niveles arriba desde esta lección)
RAIZ = Path(__file__).resolve().parent.parent.parent
CARPETA_DATA = RAIZ / "data"
SIMBOLO = "AAPL"
PERIODO = "1y"

# --- 1. Crear carpeta data/ si no existe ---
CARPETA_DATA.mkdir(exist_ok=True)

# --- 2. Descargar histórico con yfinance ---
# Ticker("AAPL") = el "activo"; .history() = precios en el tiempo
print(f"Descargando {PERIODO} de {SIMBOLO}...")
ticker = yf.Ticker(SIMBOLO)
hist = ticker.history(period=PERIODO)

if hist.empty:
    print("No se obtuvieron datos. Revisa la conexión o el símbolo.")
    exit(1)

# --- 3. Estructura: yfinance devuelve columnas en inglés (Open, High, Low, Close, Volume) ---
# El índice del DataFrame son las fechas (DatetimeIndex)
print("\n=== Primeras filas (columnas en inglés) ===")
print(hist.head(10))
print("\nColumnas:", list(hist.columns))
print("Índice (fechas):", type(hist.index))

# --- 4. Renombrar a español (opcional, para coincidir con Lección 02) ---
# Así el CSV queda consistente con el resto del proyecto
mapeo = {
    "Open": "abierto",
    "High": "alto",
    "Low": "bajo",
    "Close": "cierre",
    "Volume": "volumen",
}
df = hist[list(mapeo)].copy()
df = df.rename(columns=mapeo)

# Pasar el índice (fechas) a una columna "fecha" para el CSV
df = df.reset_index()
df = df.rename(columns={"Date": "fecha"})
df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")

print("\n=== DataFrame con columnas en español ===")
print(df.head())
print("\nForma:", df.shape)

# --- 5. Guardar en data/SIMBOLO.csv ---
archivo_csv = CARPETA_DATA / f"{SIMBOLO}.csv"
df.to_csv(archivo_csv, index=False)
print(f"\nGuardado: {archivo_csv}")
print("Cierre medio (periodo descargado):", round(df["cierre"].mean(), 2))

print("\n--- Fin Lección 03 ---")
print("Siguiente: Lección 04 - Indicadores técnicos (SMA, RSI) y gráficos.")
