"""
Lección 02: Datos y Pandas – DataFrame como "tabla".

Equivalencia mental con MySQL:
- DataFrame ≈ resultado de SELECT (filas y columnas)
- df["columna"] ≈ una columna
- df[df["columna"] > x] ≈ WHERE columna > x
"""

import pandas as pd
from pathlib import Path

# Ruta al CSV de esta lección (funciona desde raíz del proyecto o desde esta carpeta)
CARPETA = Path(__file__).resolve().parent
CSV_EJEMPLO = CARPETA / "precios_ejemplo.csv"

# --- 1. Leer CSV (como cargar un resultado de consulta en memoria) ---
# En PHP/MySQL pensarías: $rows = $pdo->query("SELECT * FROM precios")->fetchAll(PDO::FETCH_ASSOC);
df = pd.read_csv(CSV_EJEMPLO)

print("=== DataFrame cargado (primeras 5 filas) ===")
print(df.head())
print()

# --- 2. Tipo y forma ---
print("Tipo de df:", type(df))
print("Forma (filas, columnas):", df.shape)
print("Columnas:", list(df.columns))
print()

# --- 3. Fechas: convertir texto a datetime ---
# Así Pandas entiende la columna como fechas y podemos ordenar/filtrar por tiempo
df["fecha"] = pd.to_datetime(df["fecha"])
print("Columna 'fecha' como datetime:")
print(df["fecha"].head())
print()

# --- 4. Añadir columna: rango = alto - bajo ---
# Igual que en SQL: SELECT *, (alto - bajo) AS rango FROM ...
df["rango"] = df["alto"] - df["bajo"]
print("=== DataFrame con columna 'rango' (alto - bajo) ===")
print(df[["fecha", "alto", "bajo", "rango"]].head())
print()

# --- 5. Calcular el cierre medio (una sola columna = Series) ---
# En PHP: array_sum(array_column($rows, 'cierre')) / count($rows)
cierre_medio = df["cierre"].mean()
print("Cierre medio:", round(cierre_medio, 2))
print("Cierre mínimo:", df["cierre"].min())
print("Cierre máximo:", df["cierre"].max())
print()

# --- 6. Filtrar filas (equivalente a WHERE) ---
# Solo días donde el cierre fue mayor que 186
mayores_186 = df[df["cierre"] > 186]
print("Filas con cierre > 186:")
print(mayores_186[["fecha", "cierre"]])
print()

# --- 7. Ordenar por una columna ---
# ORDER BY volumen DESC
por_volumen = df.sort_values("volumen", ascending=False)
print("Día con mayor volumen (primera fila después de ordenar):")
print(por_volumen[["fecha", "volumen"]].head(1))
print()

# --- 8. Guardar resultado en otro CSV (opcional) ---
# Añadimos la columna 'rango' y guardamos; útil para exportar datos procesados
salida = CARPETA / "precios_con_rango.csv"
df.to_csv(salida, index=False)
print("Guardado DataFrame con 'rango' en:", salida)
print()

print("--- Fin Lección 02 ---")
print("Siguiente: Lección 03 - Descargar precios reales con yfinance.")
