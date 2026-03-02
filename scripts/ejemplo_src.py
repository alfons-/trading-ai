"""
Ejemplo de uso de los módulos en src/: datos e indicadores.

Ejecutar desde la raíz del proyecto:
    python scripts/ejemplo_src.py
"""
import sys
from pathlib import Path

# Añadir raíz del proyecto para que se encuentre el paquete src
_raiz = Path(__file__).resolve().parent.parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

from src.data import get_precios
from src.indicators import add_sma, add_rsi

# Cargar o descargar precios (guarda en data/AAPL.csv si descarga)
df = get_precios("AAPL", period="1y")

# Añadir indicadores (modifican el DataFrame in-place)
add_sma(df, window=20)
add_sma(df, window=50)
add_rsi(df, window=14)

print("=== Últimas 5 filas (precio, SMA 20/50, RSI) ===")
print(df[["fecha", "cierre", "sma_20", "sma_50", "rsi_14"]].tail())
print("\nForma:", df.shape)
