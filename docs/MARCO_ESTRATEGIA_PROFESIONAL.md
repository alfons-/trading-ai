# Marco de estrategia (visión de gestión profesional)

> **Aviso:** Esto es material educativo. No es asesoramiento financiero. Ninguna estrategia garantiza beneficios.

Después de décadas en mercados, lo que separa a quien **sobrevive** de quien desaparece no es un indicador mágico, sino **cómo** se define la ventaja, el riesgo y la disciplina operativa.

---

## 1. Qué significa “ganar” de verdad

- **Ganar** no es un mes bueno: es **sobrevivir** años, drawdowns y cambios de régimen (alcista, bajista, lateral, crisis de liquidez).
- Una estrategia “ganadora” en papel suele **morir** por: costes, slippage, sobreajuste al pasado, sobreoperar y romper reglas en vivo.
- El objetivo profesional es un **edge pequeño pero repetible** + **tamaño de posición coherente** + **límite de ruina** cercano a cero.

---

## 2. Los tres pilares (más importantes que la entrada)

### A) Ventaja (edge)

Debe poder explicarse en una frase, sin tautología:

- *“Solo opero pullbacks con tendencia cuando el mercado mayor filtra régimen alcista y el riesgo por trade está acotado.”*

Si no sabes **por qué** podría existir ineficiencia (flujo, comportamiento repetido, estructura), es hobby, no proceso.

### B) Riesgo y supervivencia

Reglas típicas en gestión seria (orientativas):

| Concepto | Idea |
|----------|------|
| Riesgo por trade | Fracción muy pequeña del capital (orden habitual: **0,25 % – 1 %** según estilo y volatilidad). |
| R múltiple | Objetivo de **beneficio esperado vs riesgo** (p. ej. buscar escenarios donde el TP lógico esté a **≥ 2R** si el stop es 1R). |
| Drawdown máximo | Definir cuándo **parar** de operar y revisar el sistema (no “revenge trading”). |
| Correlación | No acumular 10 trades que son en realidad **la misma apuesta** (misma beta cripto). |

Tu código ya acota **stop -5 %** y **TP +10 %** (≈ 2R si el riesgo es 5 % del precio de entrada en spot simulado). En real, el riesgo en **cuenta** debe calcularse con **tamaño de posición**, no solo con % de precio.

### C) Régimen de mercado

La mayoría de sistemas “rompen” cuando el régimen cambia:

- **Filtro de timeframe superior** (precio vs EMA 200 diario, como en tu estrategia long) es una forma clásica de decir: *no compro contra la marea mayor*.
- Profesionales suelen **reducir tamaño** o **no operar** en: alta volatilidad anómala, eventos de noticias, gaps de liquidez.

---

## 3. Entradas y salidas: diseño sensato (encaje con tu proyecto)

Lo que ya tienes encaja con ideas **razonables** si no se sobrecalienta:

| Pieza | Lógica profesional |
|--------|---------------------|
| **RSI + cruce 30** | Busca **reversión desde sobreventa**; solo tiene sentido si el contexto mayor no es claramente bajista (tu filtro 1D ayuda). |
| **Precio > EMA 200 (4h)** | Evita “cuchillo cayendo” en marco operativo. |
| **EMA 200 diario** | Filtro de **régimen**; coherente con “no nadar contra la corriente”. |
| **Divergencia** | Señal **discrecional** válida pero fácil de ver donde no existe; mejor como **confluencia opcional**, no dogma. |
| **Volumen creciente** | Confirma **participación**; en cripto, cuidado con wash y exchanges distintos. |
| **SL -5 % / TP +10 %** | Perfil **2:1** en precio; en cuenta real hace falta **posición** para que 5 % de precio ≠ 5 % de cuenta. |
| **TP parcial** | Muy usado para **sacar incertidumbre** y dejar correr parte con trailing o señal de salida (RSI diario > 70 es una salida por **sobrecompra mayor**, no óptima en tendencias fuertes). |

---

## 4. Lo que haría un gestor con 30 años (lista corta)

1. **Definir el universo:** spot vs perpetuo, par, liquidez mínima, horarios.
2. **Congelar reglas** y backtestear con **walk-forward** (entrenar en un tramo, validar en el siguiente), no solo un único periodo.
3. **Incluir costes** (comisión, slippage) en el simulador antes de fiarse del PnL.
4. **Journal:** cada trade con motivo de entrada/salida; revisar violaciones de plan.
5. **Reducir parámetros** “libres”; cada parámetro extra es una oportunidad de **sobreajuste**.
6. **Diversificar** el riesgo (no solo más indicadores en el mismo activo).

---

## 5. Cómo usar este repo con mentalidad “pro”

- **`strategy_long_rules.py`:** buen laboratorio de **reglas explícitas**; trata `require_divergence` y umbrales como **hipótesis**, no verdades.
- **`run_experiment` (XGBoost):** es **otro animal** (aprendizaje estadístico); exige control de **leakage**, costes y degradación fuera de muestra.
- **Próximo paso técnico recomendado:** simular **comisión + slippage** y **tamaño de posición** en función del riesgo por trade en cuenta.

---

## 6. Frase final

La “estrategia ganadora” de un veterano no es un secreto: es **un sistema aburrido** que respeta el riesgo, opera poco cuando no hay edge y sobrevive lo suficiente para que las matemáticas del edge pequeño trabajen a su favor.
