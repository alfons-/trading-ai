import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
CSV_DATA = RAIZ / "data" / "APPL.csv"
CSV_EJEMPLO = RAIZ / "lessons" / "02_datos_y_pandas" / "precios_ejemplo.csv"

if CSV_DATA.exists():
    df = pd.read_csv(CSV_DATA)
    print(f"Descargados datos {CSV_DATA}")
else:
    df = pd.read_csv(CSV_EJEMPLO)
    print (f"Descargando datos de {CSV_EJEMPLO}")
    
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

PERIODO_CORTA = 5
PERIODO_LARGA = 20

df["sma_corta"] = df["cierre"].rolling(PERIODO_CORTA).mean()
df["sma_larga"] = df["cierre"].rolling(PERIODO_LARGA).mean()
df["sma_arriba"] = df["sma_corta"] > df["sma_larga"]
df["senal"] = df["sma_arriba"].astype(int)
df["cruce_compra"] = (df["senal"] == 1) & (df["senal"].shift(1) == 0)
df["cruce_venta"] = (df["senal"] == 0) & (df["senal"].shift(1) == 1)

df["retorno"] = df["cierre"].pct_change()
df['retorno_estrategia'] = df["senal"].shift(1) * df["retorno"]





print(f"\n\nsma_corta: \n{df['sma_corta'].tolist()}\n\n señal larga:\n{df['sma_larga'].tolist()}")
print(f"\n\nsma_arriba:\n{df['sma_arriba'].tolist()}\n\nsenal:\n{df['senal'].tolist()}")
print(f"\n\ncruce compra:\n{df['cruce_compra'].tolist()}\n\ncruce venta:\n{df['cruce_venta'].tolist()}")
print(f"\n\nretorno: \n{df['retorno'].tolist()}\n\nretorno estrategia:\n{df['retorno_estrategia'].tolist()}")




# --- 5. Métricas ---
df_ok = df.dropna(subset=["retorno_estrategia"])
retorno_total = (1 + df_ok["retorno_estrategia"]).prod() - 1
num_operaciones = df["cruce_compra"].sum() + df["cruce_venta"].sum()

print("\n=== Resultados del backtest ===")
print(f"\n\nPeríodo: \n{df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"\n\nDías con señal: \n{len(df_ok)}")
print(f"\n\nRetorno total (estrategia): \n{retorno_total:.2%}")
print(f"\n\nNúmero de operaciones (entradas + salidas): \n{int(num_operaciones)}")
print("\n\n\n--- Fin backtest ---")
