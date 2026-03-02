"""
Indicadores de momentum (RSI). Añaden columnas al DataFrame.
"""

import pandas as pd
from ta.momentum import RSIIndicator


def add_rsi(
    df: pd.DataFrame,
    column: str = "cierre",
    window: int = 14,
    nombre_columna: str | None = None,
) -> pd.DataFrame:
    """
    Añade el RSI (Relative Strength Index) como nueva columna. Modifica el DataFrame in-place y lo devuelve.

    Args:
        df: DataFrame con columna de precios (ej. cierre).
        column: Nombre de la columna sobre la que calcular el RSI.
        window: Ventana del RSI (típicamente 14).
        nombre_columna: Nombre de la nueva columna. Si None, usa f"rsi_{window}".

    Returns:
        El mismo DataFrame con la columna añadida.
    """
    if nombre_columna is None:
        nombre_columna = f"rsi_{window}"
    indicator = RSIIndicator(close=df[column], window=window)
    df[nombre_columna] = indicator.rsi()
    return df
