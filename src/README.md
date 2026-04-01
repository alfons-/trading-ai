# Módulos reutilizables (`src/`)

Código común para datos e indicadores. Úsalo desde la raíz del proyecto para que los imports funcionen.

## Estructura

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── download.py    # get_precios()
└── indicators/
    ├── __init__.py
    ├── ma.py         # add_sma()
    └── momentum.py   # add_rsi()
```

## Uso

Ejecuta desde la raíz del proyecto. El script `scripts/ejemplo_src.py` añade la raíz a `sys.path` para que encuentre `src`:

```bash
cd /ruta/a/tradedan
python scripts/ejemplo_src.py
```

Si escribes otro script fuera de la raíz, añade al inicio: `sys.path.insert(0, "/ruta/a/tradedan")` o ejecuta con `PYTHONPATH=. python tu_script.py`.

En tu código:

```python
from src.data import get_precios
from src.indicators import add_sma, add_rsi

df = get_precios("AAPL", period="1y")   # Carga data/AAPL.csv o descarga
add_sma(df, window=20)                   # Añade columna sma_20
add_rsi(df, window=14)                   # Añade columna rsi_14
```

## API resumida

### `get_precios(symbol, period="1y", data_dir=None, guardar=True, forzar_descarga=False)`

- **symbol:** Símbolo (ej. `"AAPL"`).
- **period:** Periodo para yfinance (`"1y"`, `"6mo"`, `"2y"`).
- **data_dir:** Carpeta de CSV (por defecto `data/` en la raíz).
- **guardar:** Guardar CSV tras descargar.
- **forzar_descarga:** Si `True`, no usa el CSV y vuelve a descargar.

Devuelve un `DataFrame` con columnas: `fecha`, `abierto`, `alto`, `bajo`, `cierre`, `volumen`.

### `add_sma(df, column="cierre", window=20, nombre_columna=None)`

Añade la media móvil simple. Por defecto crea la columna `sma_20`. Modifica `df` in-place.

### `add_rsi(df, column="cierre", window=14, nombre_columna=None)`

Añade el RSI. Por defecto crea la columna `rsi_14`. Modifica `df` in-place.
