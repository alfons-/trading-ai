# Lección 05: Señales y backtesting

Objetivo: definir una **estrategia de cruce de medias** (SMA 20 / SMA 50), generar **señales** de compra/venta, **simular** las operaciones en histórico y calcular **rentabilidad** simple.

## Conceptos

- **Cruce alcista:** SMA rápida (20) cruza por encima de SMA lenta (50) → señal de **compra**.
- **Cruce bajista:** SMA rápida cruza por debajo de SMA lenta → señal de **venta**.
- **Backtest:** simular que compramos al cierre del día de la señal y vendemos al cierre del día de la señal de venta; calcular beneficio/pérdida sin costes ni deslizamiento (simplificado).

## Datos

Usa `data/AAPL.csv`. Si no existe, el script descarga 1 año con yfinance (como en las lecciones 03 y 04).

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
python lessons/05_senales_backtest/ejercicio.py
```

## Siguiente paso

Refactorizar código reutilizable en `src/` (por ejemplo `src/data/download.py`, `src/indicators/`) y, si quieres, añadir tests con pytest.
