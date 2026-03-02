# Trading AI – Proyecto de aprendizaje

Proyecto paso a paso para aprender **Python** y **trading** desde cero, usando **Git** y **GitHub**. Pensado para desarrolladores con experiencia en PHP/MySQL que quieren pasar a Python aplicado al análisis de mercados.

## Objetivos

- Aprender Python de forma práctica (sintaxis, tipos, funciones, módulos).
- Usar datos de mercados reales (precios, indicadores).
- Trabajar con Git/GitHub (commits, ramas, README).
- Construir poco a poco un sistema de análisis y, opcionalmente, señales de trading.

## Requisitos

- **Python 3.10+** (recomendado 3.12).
- **Git** instalado.
- Cuenta en **GitHub**.

## Inicio rápido

```bash
# Clonar (o ya estás en el repo)
git clone https://github.com/TU_USUARIO/trading-ai.git
cd trading-ai

# Entorno virtual
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependencias
pip install -r requirements.txt

# Ejecutar la primera lección
python lessons/01_python_basico/ejercicio.py
```

## Estructura del proyecto

```
trading-ai/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias Python
├── docs/
│   └── GUIA_APRENDIZAJE.md   # Plan de aprendizaje paso a paso
├── lessons/                  # Lecciones ordenadas (01, 02, 03...)
│   ├── 01_python_basico/
│   ├── 02_datos_y_pandas/
│   └── ...
├── src/                      # Código reutilizable (data, indicators)
│   ├── data/                 # get_precios() – descarga/carga OHLCV
│   └── indicators/          # add_sma(), add_rsi()
├── scripts/                  # Ejemplos que usan src (ej. ejemplo_src.py)
└── notebooks/                # Jupyter para explorar y visualizar
```

## Guía de aprendizaje

Sigue el plan detallado en **[docs/GUIA_APRENDIZAJE.md](docs/GUIA_APRENDIZAJE.md)**. Incluye:

1. **Python básico** – Variables, listas, diccionarios, funciones (equivalencias con PHP).
2. **Datos y Pandas** – Series, DataFrames, fechas (similar a tablas SQL).
3. **Datos de mercado** – yfinance, precios, OHLCV.
4. **Indicadores técnicos** – Medias móviles, RSI, con la librería `ta`.
5. **Git y GitHub** – Commits, ramas, README, buenas prácticas.
6. **Análisis y backtesting** – Evaluar estrategias con datos históricos.

## Recursos

- [Documentación Python (es)](https://docs.python.org/es/3/)
- [Pandas – 10 min](https://pandas.pydata.org/docs/user_guide/10min.html)
- [yfinance](https://pypi.org/project/yfinance/) – Precios de Yahoo Finance
- [Git – Libro Pro Git (es)](https://git-scm.com/book/es/v2)

## Licencia

Uso educativo y personal. No es asesoramiento financiero.
