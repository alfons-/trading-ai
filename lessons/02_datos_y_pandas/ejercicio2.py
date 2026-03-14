import yfinance as yf
import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
CARPETA_DATA = RAIZ / "data"
SIMBOLO = "AAPL"
PERIODO = "1y"

CARPETA_DATA.mkdir(exist_ok=True)

print(f"Descargando {PERIODO} de {SIMBOLO}")
ticker = yf.Ticker(SIMBOLO)
hist = ticker.history(PERIODO)

if hist.empty:
    print("No se obtuvieron datos")
    exit(1)
    
print("primeras filas")
print(hist.head(10))
print("Columnas: ", list(hist.columns))
print("Índice: ", type(hist.index))

indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador, indicador
