"""
Lección 01: Python básico – Variables, listas, diccionarios y funciones.

Equivalencia rápida con PHP:
- No hay $ delante de variables
- list ≈ array() indexado
- dict ≈ array() asociativo
- def nombre(): ≈ function nombre() {}
"""

# --- 1. Variables (como en PHP, pero sin $) ---
simbolo = "AAPL"
precio_actual = 175.50
cantidad = 100

print("Símbolo:", simbolo)
print("Precio:", precio_actual)
print()

# --- 2. Lista de precios (como un array de números) ---
# En PHP: $precios = [100, 102, 98, 105, 110];
precios = [100.0, 102.5, 98.0, 3000, 110.0, 108.0, 112.0]

print("Lista de precios:", precios)
print("Primer precio:", precios[0])   # índice 0
print("Último precio:", precios[-1])  # -1 = último en Python
print("Tres primeros:", precios[0:3]) # slice [inicio:fin]
print()

# --- 3. Diccionario (clave -> valor, como array asociativo en PHP) ---
# En PHP: $ticker = ["simbolo" => "AAPL", "precio" => 175.5];
ticker = {
    "simbolo": "AAPL",
    "precio": 200.00,
    "volumen": 50_000_000,
}
print("Diccionario ticker:", ticker)
print("Símbolo:", ticker["simbolo"])
print("Precio (con .get):", ticker.get("precio", "no existe"))  # 0 si no existe
print()

# --- 4. Funciones (def nombre(parametros): ... return) ---
# En PHP: function precio_medio($lista) { return array_sum($lista) / count($lista); }


def precio_medio(lista_precios):
    """Calcula el precio medio de una lista de precios."""
    if not lista_precios:
        return 0.0
    total = sum(lista_precios)  # sum() es built-in
    return total / len(lista_precios)  # len() = longitud


def precio_maximo(lista_precios):
    """Devuelve el precio máximo."""
    if not lista_precios:
        return 0.0
    return max(lista_precios)


def precio_minimo(lista_precios):
    """Devuelve el precio mínimo."""
    if not lista_precios:
        return 0.0
    return min(lista_precios)


# Usar las funciones
media = precio_medio(precios)
maximo = precio_maximo(precios)
minimo = precio_minimo(precios)

print("Precio medio:", round(media, 2))
print("Precio máximo:", maximo)
print("Precio mínimo:", minimo)
print()

# --- 5. Bucle for (recorrer lista) ---
# En PHP: foreach ($precios as $p) { ... }
print("Precios uno a uno:")
for p in precios:
    print("  ", p)

print()

# --- 6. Resumen en un diccionario (útil para después con Pandas/JSON) ---
resumen = {
    "simbolo": simbolo,
    "precio_medio": round(media, 2),
    "precio_maximo": maximo,
    "precio_minimo": minimo,
    "num_precios": len(precios),
}
print("Resumen:", resumen)
print()
print("--- Fin Lección 01 ---")
print("Siguiente: Lección 02 - Pandas y datos en tabla (CSV, DataFrame).")
