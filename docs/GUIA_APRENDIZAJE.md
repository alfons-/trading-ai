# Guía de aprendizaje: Python + Trading + GitHub

Plan paso a paso para aprender Python con un proyecto de trading. Cada bloque incluye conceptos de Python, práctica con datos/mercado y, cuando toca, uso de Git/GitHub.

---

## Antes de empezar: PHP vs Python (resumen)

| PHP                                     | Python           	 	     | Notas |
|----                                    -|--------|           		     --------|
| `$variable`                            | `variable` |  	   Sin `$`; nombres con snake_case |
| `array()` / `[]` 				| `list` → `[]`, `dict` → `{}` | Listas y diccionarios nativos |
| `function nombre($x)` 			| `def nombre(x):` |	 Indentación obligatoria (4 espacios) |
| `=>` en arrays 					| `:` en dicts 					| `{"clave": "valor"}` |
| `foreach ($arr as $k => $v)` 	| `for k, v in dict.items():`		 | Iteración directa |
| PDO / MySQLi 				| Varios (SQLite, PostgreSQL, etc.) | Aquí usaremos Pandas como “tablas en memoria” |
| `require` / `include`				 | `import modulo` | 				Módulos y paquetes |

---

## Fase 1: Python básico (Lección 01)

**Objetivo:** Escribir y ejecutar Python; variables, listas, diccionarios y funciones.

### Contenido

1. **Ejecutar Python**
   - Script: `python archivo.py`
   - REPL: `python` (entrar y salir con `exit()`)

2. **Variables y tipos**
   - No se declaran tipos: `precio = 100.5`, `simbolo = "AAPL"`
   - Tipos útiles: `int`, `float`, `str`, `bool`, `None`

3. **Listas** (como arrays indexados en PHP)
   - `precios = [100, 102, 98, 105]`
   - Índices: `precios[0]`, `precios[-1]` (último)
   - Slice: `precios[1:3]` → desde índice 1 hasta antes del 3

4. **Diccionarios** (como arrays asociativos)
   - `ticker = {"simbolo": "AAPL", "precio": 150.0}`
   - Acceso: `ticker["simbolo"]` o `ticker.get("precio", 0)`

5. **Funciones**
   - `def nombre(parametro):` con cuerpo indentado
   - `return valor`

6. **Bucles**
   - `for x in lista:` y `for clave, valor in diccionario.items():`
   - `while condicion:` (igual idea que en PHP)

### Práctica

- Carpeta: `lessons/01_python_basico/`
- Ejercicio: definir una lista de precios, una función que calcule el precio medio y otra que devuelva el máximo; imprimir resultados.

### Git

- Hacer el primer commit con el README y la lección 01:
  - `git add .`
  - `git commit -m "Lección 01: Python básico - variables, listas, funciones"`
  - `git push` (si ya tienes el repo en GitHub)

---

## Fase 2: Datos en tablas con Pandas (Lección 02)

**Objetivo:** Tratar datos como “tablas” (DataFrames); equivalente mental a resultados de SQL.

### Contenido

1. **Series** – una columna con índice (fechas o etiquetas).
2. **DataFrame** – varias columnas; pensar en “tabla”.
3. **Leer y escribir**
   - CSV: `pd.read_csv()`, `df.to_csv()`
   - Fechas: `pd.to_datetime()`, índice temporal.

4. **Operaciones típicas**
   - Filtrar filas: `df[df["columna"] > valor]`
   - Columnas: `df["columna"]`, `df[["col1", "col2"]]`
   - Ordenar: `df.sort_values("columna")`
   - Agregar columna: `df["nueva"] = df["a"] + df["b"]`

### Práctica

- Crear un CSV de ejemplo con columnas: fecha, abierto, alto, bajo, cierre, volumen.
- Cargarlo con Pandas, calcular el cierre medio y una columna “rango” = alto - bajo.

### Git

- Rama opcional: `git checkout -b leccion-02-pandas`
- Commit: "Lección 02: introducción a Pandas y CSV"

---

## Fase 3: Datos de mercado con yfinance (Lección 03)

**Objetivo:** Descargar precios reales (OHLCV) y guardarlos en DataFrame.

### Contenido

1. **yfinance**
   - `import yfinance as yf`
   - `ticker = yf.Ticker("AAPL")`
   - `hist = ticker.history(period="1y")` → DataFrame con índice de fechas.

2. **Estructura OHLCV**
   - Open, High, Low, Close, Volume; columnas típicas en `hist`.

3. **Guardar datos**
   - Exportar a CSV para no depender siempre de la API.

### Práctica

- Script que descargue 1 año de AAPL, muestre las primeras filas y guarde el CSV en `data/` (crear la carpeta si no existe).

### Git

- Añadir `data/*.csv` al `.gitignore` si los archivos son grandes; documentar en README cómo descargar datos.
- Commit: "Lección 03: descarga de precios con yfinance"

---

## Fase 4: Indicadores técnicos con `ta` (Lección 04)

**Objetivo:** Calcular medias móviles, RSI, etc., sobre el DataFrame de precios.

### Contenido

1. **Medias móviles**
   - SMA: media de los últimos N cierres.
   - En Pandas: `df["Close"].rolling(window=20).mean()`

2. **Librería `ta`**
   - Indicadores ya implementados (RSI, MACD, Bollinger, etc.).
   - Añadir columnas al DataFrame con los resultados.

3. **Visualización básica**
   - `matplotlib`: gráfico de precios y una media móvil.

### Práctica

- Cargar CSV de AAPL (o descargar de nuevo), calcular SMA(20) y RSI(14), y dibujar precio + SMA en un gráfico.

### Git

- Commit: "Lección 04: indicadores técnicos y gráfico con matplotlib"

---

## Fase 5: Git y GitHub (a lo largo del proyecto)

**Objetivo:** Usar Git de forma cómoda y subir el proyecto a GitHub.

### Conceptos

1. **Repositorio**
   - `git init` (si creas desde cero) o `git clone url` (si partes de GitHub).

2. **Ciclo básico**
   - `git status` → `git add .` o `git add archivo` → `git commit -m "mensaje"` → `git push`.

3. **Ramas**
   - `git checkout -b nombre-rama` para nueva rama.
   - Trabajar por lección en ramas opcionales y luego fusionar a `main`.

4. **README**
   - Descripción del proyecto, cómo clonar, instalar dependencias y ejecutar la lección 01 (y siguientes).
   - En este proyecto: [README.md](../README.md).

5. **.gitignore**
   - Incluir: `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `data/*.csv` (si no quieres subir datos).

### Práctica

- Crear repo en GitHub (si no existe), enlazar remoto con `git remote add origin url`, y hacer push de `main`.
- Una vez por lección (o por bloque lógico), hacer un commit con mensaje claro.

---

## Fase 6: Análisis y backtesting (Lecciones 05+)

**Objetivo:** Definir una estrategia sencilla (por ejemplo: cruce de medias), simular operaciones en histórico y calcular rendimiento.

### Contenido (resumen)

1. **Señales**
   - Columna booleana: comprar cuando SMA corta por encima, vender cuando corta por debajo (o reglas que elijas).

2. **Simulación**
   - Recorrer el DataFrame por fechas, aplicar reglas y anotar “operaciones” (entrada/salida, precio).

3. **Métricas**
   - Rentabilidad total, número de operaciones, beneficio medio por operación (sin considerar costes primero).

4. **Mejoras**
   - Carpeta `src/` con módulos reutilizables: `data/download.py`, `indicators/ma.py`, etc.
   - Tests opcionales con `pytest` para funciones puras.

### Práctica

- Implementar estrategia de cruce de SMA(20) y SMA(50) sobre un CSV anual; imprimir lista de señales y rentabilidad simple.

### Git

- Commits por funcionalidad: "Añadir generación de señales", "Backtest cruce de medias", "Refactor: módulo src.data".

---

## Orden sugerido de lecciones

| Orden 			| Carpeta / tema 			| Enfoque |
|-------			|----------------			|--------|
| 1 | 			`01_python_basico` | 	Variables, listas, dicts, funciones |
| 2 | 			`02_datos_y_pandas` | 	DataFrame, CSV, fechas |
| 3 |			 `03_datos_mercado` |	 yfinance, OHLCV, guardar CSV |
| 4 | 			`04_indicadores` | 		SMA, RSI, ta, matplotlib |
| 5 |			 `05_señales_backtest` | Señales, simulación, métricas |
| 6+ | 			`src/` + notebooks |	 Código reutilizable y experimentos |

---

## Consejos

- **No saltes fases:** la 01 y 02 son la base para leer y manipular datos en el resto.
- **Equivalencias PHP:** cuando dudes, piensa en “¿cómo lo haría en PHP?” y busca el equivalente en Python (listas, dicts, funciones).
- **REPL:** usa `python` en la terminal para probar una línea (igual que `php -a`).
- **Git:** haz commits pequeños y mensajes descriptivos; así ves el progreso y puedes volver atrás si algo se rompe.
- **Documentación:** la documentación oficial de Python y Pandas está en inglés, pero con tu nivel de PHP te resultará familiar; usa [docs.python.org](https://docs.python.org/es/3/) en español cuando necesites.

Cuando termines la Lección 01, continúa con la 02 y así sucesivamente. Si quieres, en cada lección puedes crear una rama en Git para aislar cambios y fusionar a `main` cuando esté listo.
