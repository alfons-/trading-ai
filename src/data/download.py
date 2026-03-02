"""
Módulo de datos: descarga o carga precios OHLCV desde CSV / yfinance.

Uso:
    from src.data import get_precios
    df = get_precios("AAPL", period="1y")
"""

import pandas as pd
from pathlib import Path

# Raíz del proyecto (tres niveles arriba: download.py -> data -> src -> raíz)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Nombres de columnas en español (igual que en las lecciones)
COLUMNAS_OHLCV = ["fecha", "abierto", "alto", "bajo", "cierre", "volumen"]
MAPEO_YF = {
    "Open": "abierto",
    "High": "alto",
    "Low": "bajo",
    "Close": "cierre",
    "Volume": "volumen",
}


def get_precios(
    symbol: str,
    period: str = "1y",
    data_dir: Path | None = None,
    guardar: bool = True,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """
    Obtiene precios OHLCV: carga desde CSV si existe, si no descarga con yfinance.

    Args:
        symbol: Símbolo del activo (ej. "AAPL", "MSFT").
        period: Periodo para yfinance ("1y", "6mo", "2y", etc.).
        data_dir: Carpeta donde están/guardar los CSV. Si None, usa raíz_proyecto/data.
        guardar: Si True, guarda (o sobrescribe) el CSV tras descargar.
        forzar_descarga: Si True, ignora el CSV y descarga de nuevo.

    Returns:
        DataFrame con columnas: fecha, abierto, alto, bajo, cierre, volumen.
        fecha como datetime, ordenado por fecha ascendente.
    """
    if data_dir is None:
        data_dir = _PROJECT_ROOT / "data"
    data_dir = Path(data_dir)
    archivo = data_dir / f"{symbol.upper()}.csv"

    if not forzar_descarga and archivo.exists():
        df = pd.read_csv(archivo)
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)
        return df

    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=period)
    if hist.empty:
        raise ValueError(f"No se obtuvieron datos para {symbol}. Revisa símbolo o conexión.")

    df = hist[list(MAPEO_YF)].rename(columns=MAPEO_YF).reset_index()
    df = df.rename(columns={"Date": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    df = df[COLUMNAS_OHLCV]

    if guardar:
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(archivo, index=False)

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df
