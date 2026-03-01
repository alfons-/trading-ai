# Lección 03: Datos de mercado con yfinance

Objetivo: **descargar precios reales** (OHLCV) desde Yahoo Finance y guardarlos en CSV.

## Conceptos

- **yfinance**: librería que obtiene datos históricos de acciones (Yahoo Finance).
- **OHLCV**: Open (apertura), High (máximo), Low (mínimo), Close (cierre), Volume (volumen).
- Los datos se guardan en `data/` para reutilizarlos sin depender de la red en cada ejecución.

## Requisito

Necesitas **conexión a internet** para la primera descarga.

## Cómo ejecutar

Desde la raíz del proyecto (con el entorno virtual activado):

```bash
python lessons/03_datos_mercado/ejercicio.py
```

El script crea la carpeta `data/` si no existe y guarda allí un CSV por símbolo (por ejemplo `data/AAPL.csv`).

## Siguiente paso

Lección 04: **indicadores técnicos** (SMA, RSI) y un gráfico con matplotlib.
