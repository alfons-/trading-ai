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
├── configs/
│   └── default.yaml          # Config: símbolos, timeframes, XGBoost, backtest
├── src/
│   ├── agents/              # DataAgent, FeatureAgent, ModelAgent, Orchestrator
│   ├── data/                # get_precios() – descarga/carga OHLCV
│   └── indicators/          # add_sma(), add_rsi()
├── scripts/                  # Ejemplos que usan src (ej. ejemplo_src.py)
└── notebooks/                # Jupyter para explorar y visualizar
```

## Marco profesional (riesgo, edge, régimen)

Lectura recomendida (enfoque gestión / supervivencia, no “indicador mágico”):  
**[docs/MARCO_ESTRATEGIA_PROFESIONAL.md](docs/MARCO_ESTRATEGIA_PROFESIONAL.md)**  
Plantilla de parámetros sobria: `configs/strategy_pro_template.yaml`.

## Guía de aprendizaje

Sigue el plan detallado en **[docs/GUIA_APRENDIZAJE.md](docs/GUIA_APRENDIZAJE.md)**. Incluye:

1. **Python básico** – Variables, listas, diccionarios, funciones (equivalencias con PHP).
2. **Datos y Pandas** – Series, DataFrames, fechas (similar a tablas SQL).
3. **Datos de mercado** – yfinance, precios, OHLCV.
4. **Indicadores técnicos** – Medias móviles, RSI, con la librería `ta`.
5. **Git y GitHub** – Commits, ramas, README, buenas prácticas.
6. **Análisis y backtesting** – Evaluar estrategias con datos históricos.

## Multi-timeframe (Bybit)

El sistema descarga, para cada símbolo, **tres CSV** vía `DataAgent`:

| Timeframe | Archivo generado | Uso |
|-----------|-----------------|-----|
| 4h | `data/bybit/BTCUSDT_4h.csv` | TF base: señales de entrada y modelo |
| 1D | `data/bybit/BTCUSDT_1D.csv` | Contexto diario / horizonte de salida |
| 1W | `data/bybit/BTCUSDT_1W.csv` | Filtro de tendencia de mercado |

Configuración en `configs/default.yaml` (sección `timeframe_base` y `higher_timeframes`).

## Estrategia RSI solo cruces en 4h (mínima)

`scripts/strategy_rsi_cross_4h.py` + `configs/strategy_rsi_cross.yaml`:

- **Entrada (long):** en **4h**, cruce **de abajo hacia arriba** del nivel (`RSI_prev < entry_cross_level` y `RSI_actual ≥ entry_cross_level`). Por defecto **`entry_cross_level: 32`** (toque/cruce del 32). Opcional: `entry_touch_lookback_bars` > 0 exige que en esas velas el mínimo del RSI haya sido ≤ ese nivel.
- **Salida:** cruce **de arriba hacia abajo** de `exit_cross_level` (por defecto **50**). Con **`exit_timeframe: "4h"`** (por defecto) se evalúa en 4h. Con **`exit_timeframe: "1D"`** la señal es el cruce en **RSI diario**; la salida se ejecuta en la **primera vela 4h** posterior al cierre del día donde el cruce queda confirmado (motivo `rsi_cross_1d`). El **stop loss** sigue evaluándose en velas **4h**.
- **`position_leverage`:** por defecto **1.0** en el script (sin multiplicador; no hace falta ponerlo en el YAML). Si lo añades al YAML (>1), escala el `retorno_estrategia` en las métricas (apalancamiento simulado; el **buy & hold** no se escala).
- Los **retornos** son **backtest** sobre OHLCV histórico (p. ej. datos tipo exchange vía `DataAgent`), **no** el PnL real de una cuenta en el exchange (faltan comisiones, slippage, ejecución real).
- **`initial_capital_eur`:** capital de referencia (por defecto **10.000**) para imprimir final, ganancia y drawdown en **euros**; opción CLI `--initial-capital-eur`.

```bash
python -m scripts.strategy_rsi_cross_4h --config configs/strategy_rsi_cross.yaml
```

Trades: `data/results/<SYMBOL>_trades_rsi_cross_4h.csv` (salida 4h) o `..._rsi_cross_4h_1dexit.csv` (salida 1D).

## Estrategia long 4h (reglas avanzadas)

`scripts/strategy_long_rules.py` + `configs/strategy_long.yaml`:

- **Entrada (todas):** cruce RSI(14) **hacia arriba** del nivel 30 en 4h (`RSI_prev < 30` y `RSI_now ≥ 30`); cierre **>** EMA(200) 4h; filtro **1D:** cierre **≥** EMA(200) diaria (si no, no operar); opcional **divergencia alcista** RSI (dos mínimos en `bajo`: precio más bajo, RSI más alto); **volumen** en aumento vs vela anterior y vs SMA(volumen).
- **Salida:** stop **-5%** desde entrada (toca `bajo`); take profit **+10%** (toca `alto`); **TP parcial** opcional (+5% cierra una fracción, p. ej. 50%); **RSI(14) diario > 70**. En la misma vela, por defecto se prioriza el **stop** (conservador).

```bash
python -m scripts.strategy_long_rules --config configs/strategy_long.yaml
```

Trades: `data/results/<SYMBOL>_trades_long_rules.csv`. Ajusta `entry.require_divergence` en el YAML (`true` es muy selectivo).

## Estrategia discrecional RSI + MACD (solo indicadores)

Reglas implementadas en `scripts/strategy_rsi_ema.py` (sin modelo ML):

| | |
|--|--|
| **Entrada (4h)** | Solo velas 4h: RSI(14) toca ≤30 en ventana; EMA(14) del RSI cruza desde abajo al RSI; MACD con histograma > 0. |
| **Salida (1D)** | Cuando el RSI(14) de la **vela diaria** ≥ 70. El trade **no** dura “un día fijo”: puede ser mucho más hasta que el RSI diario toque 70. El log muestra la vela 4h donde se detecta y el cierre 4h. |

```bash
python -m scripts.strategy_rsi_ema --symbol BTCUSDT --days 2000
python scripts/plot_rsi_ema_results.py --symbol BTCUSDT
```

Trades: `data/results/<SYMBOL>_trades_rsi_macd.csv`. Gráfico: `data/results/<SYMBOL>_RSI_MACD_backtest_chart.png`.

## Despliegue en Mac de producción

El proyecto se sincroniza automáticamente entre el Mac de desarrollo y el de producción usando GitHub.

```
Mac Desarrollo  ──git push──>  GitHub  ──auto-pull──>  Mac Producción
   (Cursor)                  (origin)                  (bot en ejecución)
```

### Configuración inicial (una sola vez)

```bash
# 1. Clonar el repo
git clone https://github.com/alfons-/trading-ai.git
cd trading-ai

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Crear .env con las API keys de Bybit (no se sincroniza)
cat > .env << 'EOF'
BYBIT_API_KEY=tu_api_key
BYBIT_API_SECRET=tu_api_secret
EOF

# 4. Crear directorio de logs
mkdir -p logs
```

### Auto-deploy con launchd

El script `scripts/deploy.sh` comprueba si hay cambios en GitHub, hace pull e instala dependencias si hace falta, y reinicia el bot.

```bash
# 1. Copiar la plantilla plist
cp scripts/com.trading-ai.deploy.plist ~/Library/LaunchAgents/

# 2. Editar la copia: reemplazar /Users/USUARIO/ por tu ruta real
nano ~/Library/LaunchAgents/com.trading-ai.deploy.plist

# 3. Activar (se ejecutará cada 5 minutos y al iniciar sesión)
launchctl load ~/Library/LaunchAgents/com.trading-ai.deploy.plist

# Verificar que está activo
launchctl list | grep trading-ai

# Para desactivar
launchctl unload ~/Library/LaunchAgents/com.trading-ai.deploy.plist
```

### Configuración por entorno

Cada Mac tiene ficheros propios que **no** se sincronizan (están en `.gitignore`):

| Fichero | Desarrollo | Producción |
|---------|-----------|------------|
| `.env` | Keys de testnet / sin keys | Keys reales de Bybit |
| `logs/` | Local | Local |
| `data/` | Local (CSVs de backtest) | Local (datos live) |

Para usar una config diferente en producción, establece la variable `DEPLOY_CONFIG`:

```bash
export DEPLOY_CONFIG="configs/execution.prod.yaml"
```

### Ejecución manual del bot

```bash
source .venv/bin/activate

# Paper trading (sin dinero real)
python -m scripts.run_live --config configs/execution.yaml --paper

# Live (dinero real — requiere .env con API keys)
python -m scripts.run_live --config configs/execution.yaml
```

### Logs

```bash
# Log del bot
tail -f logs/bot.log

# Log de deploys
tail -f logs/deploy.log
```

## Recursos

- [Documentación Python (es)](https://docs.python.org/es/3/)
- [Pandas – 10 min](https://pandas.pydata.org/docs/user_guide/10min.html)
- [yfinance](https://pypi.org/project/yfinance/) – Precios de Yahoo Finance
- [Git – Libro Pro Git (es)](https://git-scm.com/book/es/v2)

## Licencia

Uso educativo y personal. No es asesoramiento financiero.
