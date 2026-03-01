# Lección 04: Indicadores técnicos y gráficos

Objetivo: calcular **SMA** (media móvil simple) y **RSI** sobre precios y dibujar un **gráfico** con matplotlib.

## Conceptos

- **SMA(20)**: media de los últimos 20 cierres; suaviza el precio.
- **RSI(14)**: indicador de momentum (0–100); suele considerarse sobrecompra >70 y sobreventa <30.
- **matplotlib**: librería estándar para gráficos; aquí precio + SMA en el mismo eje.

## Datos

Usa el CSV de AAPL en `data/AAPL.csv`. Si no existe, el script descarga 1 año con yfinance antes de calcular.

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
python lessons/04_indicadores/ejercicio.py
```

Se abre una ventana con el gráfico (o se guarda en `lessons/04_indicadores/grafico_precio_sma.png` si no hay pantalla).

## Siguiente paso

Lección 05: **señales y backtesting** (estrategia de cruce de medias, simulación de operaciones).
