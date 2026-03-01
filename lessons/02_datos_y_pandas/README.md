# Lección 02: Datos y Pandas

Objetivo: tratar datos como **tablas** (DataFrame). Equivalente mental: un resultado de `SELECT * FROM precios` en MySQL.

## Conceptos

- **DataFrame** = tabla con filas y columnas (como una hoja de cálculo o una tabla SQL).
- **Series** = una sola columna con índice.
- **CSV** = formato de texto; Pandas lo lee y escribe con `read_csv()` y `to_csv()`.

## Cómo ejecutar

Desde la raíz del proyecto (con el entorno virtual activado):

```bash
python lessons/02_datos_y_pandas/ejercicio.py
```

## Contenido

- `precios_ejemplo.csv`: CSV de ejemplo con columnas OHLCV (fecha, abierto, alto, bajo, cierre, volumen).
- `ejercicio.py`: carga el CSV, convierte fechas, añade la columna "rango" (alto - bajo), calcula el cierre medio y muestra filtros básicos.

## Siguiente paso

Lección 03: descargar precios reales con **yfinance** y guardarlos en CSV.
