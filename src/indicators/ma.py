"""
Media móvil simple (SMA). Añade una columna al DataFrame.
"""

import pandas as pd


def add_sma(
    df: pd.DataFrame,
    column: str = "cierre",
    window: int = 20,
    nombre_columna: str | None = None,
) -> pd.DataFrame:
    """
    Añade la SMA (media móvil simple) como nueva columna. Modifica el DataFrame in-place y lo devuelve.

    Args:
        df: DataFrame con columna de precios (ej. cierre).
        column: Nombre de la columna sobre la que calcular la SMA.
        window: Ventana de la media (número de periodos).
        nombre_columna: Nombre de la nueva columna. Si None, usa f"sma_{window}".

    Returns:
        El mismo DataFrame con la columna añadida.
    """
    if nombre_columna is None:
        nombre_columna = f"sma_{window}"
    df[nombre_columna] = df[column].rolling(window=window).mean()
    return df
