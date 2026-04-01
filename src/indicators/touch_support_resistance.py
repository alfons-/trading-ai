"""
Soporte / resistencia por pivotes (swings) y recuento de toques.

Un pivote bajista es un mínimo local estricto; la resistencia, un máximo local.
Un "toque" es una vela cuyo bajo/cierre (soporte) o alto/cierre (resistencia)
cae dentro de touch_tolerance_pct del nivel.
"""

from __future__ import annotations

import numpy as np


def pivot_low_mask(low: np.ndarray, order: int) -> np.ndarray:
    """True donde low[i] es estrictamente menor que order velas a cada lado."""
    n = len(low)
    out = np.zeros(n, dtype=bool)
    if order < 1 or n <= 2 * order:
        return out
    for i in range(order, n - order):
        li = low[i]
        left = low[i - order : i]
        right = low[i + 1 : i + order + 1]
        if np.all(li < left) and np.all(li < right):
            out[i] = True
    return out


def pivot_high_mask(high: np.ndarray, order: int) -> np.ndarray:
    """True donde high[i] es estrictamente mayor que order velas a cada lado."""
    n = len(high)
    out = np.zeros(n, dtype=bool)
    if order < 1 or n <= 2 * order:
        return out
    for i in range(order, n - order):
        hi = high[i]
        left = high[i - order : i]
        right = high[i + 1 : i + order + 1]
        if np.all(hi > left) and np.all(hi > right):
            out[i] = True
    return out


def _count_touches_support(levels: np.ndarray, lows_w: np.ndarray, closes_w: np.ndarray, tol: float) -> np.ndarray:
    """Toques por nivel: bajo o cierre cerca del soporte (vectorizado)."""
    if levels.size == 0:
        return np.array([], dtype=np.int32)
    d_low = np.abs(lows_w[:, None] - levels[None, :]) / levels[None, :]
    d_cl = np.abs(closes_w[:, None] - levels[None, :]) / levels[None, :]
    return ((d_low <= tol) | (d_cl <= tol)).sum(axis=0).astype(np.int32)


def _count_touches_resistance(levels: np.ndarray, highs_w: np.ndarray, closes_w: np.ndarray, tol: float) -> np.ndarray:
    if levels.size == 0:
        return np.array([], dtype=np.int32)
    d_hi = np.abs(highs_w[:, None] - levels[None, :]) / levels[None, :]
    d_cl = np.abs(closes_w[:, None] - levels[None, :]) / levels[None, :]
    return ((d_hi <= tol) | (d_cl <= tol)).sum(axis=0).astype(np.int32)


def compute_touch_support_resistance(
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    *,
    lookback: int,
    swing_order: int,
    touch_tolerance_pct: float,
    min_touches: int,
    fallback_minmax: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Por índice t devuelve nivel de soporte y resistencia elegidos.

    Soporte: entre pivotes bajistas confirmados bajo close[t], con
    al menos min_touches en la ventana, se elige el de más toques;
    empate → el más cercano por debajo del cierre (nivel más alto).

    Resistencia: análogo por encima del cierre.

    Si no hay candidato válido y fallback_minmax: min/max de la ventana.
    """
    n = len(close)
    support = np.full(n, np.nan, dtype=np.float64)
    resistance = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return support, resistance

    pl = pivot_low_mask(low.astype(np.float64), swing_order)
    ph = pivot_high_mask(high.astype(np.float64), swing_order)
    lb = max(lookback, 2 * swing_order + 1)

    for t in range(0, min(lb - 1, n)):
        if fallback_minmax:
            support[t] = float(np.min(low[: t + 1]))
            resistance[t] = float(np.max(high[: t + 1]))

    tol = float(touch_tolerance_pct)
    mt = int(min_touches)

    for t in range(lb - 1, n):
        lo = t - lb + 1
        hi = t + 1
        lows_w = low[lo:hi]
        highs_w = high[lo:hi]
        closes_w = close[lo:hi]
        ct = float(close[t])

        i_max_pivot = t - swing_order
        if i_max_pivot < lo:
            if fallback_minmax:
                support[t] = float(np.min(lows_w))
                resistance[t] = float(np.max(highs_w))
            continue

        # Pivotes bajistas válidos en [lo, i_max_pivot]
        idx_range = np.arange(lo, i_max_pivot + 1, dtype=np.int32)
        mask_pl = pl[lo : i_max_pivot + 1]
        cand_i = idx_range[mask_pl]
        if cand_i.size > 0:
            levels_s = low[cand_i].astype(np.float64)
            below = levels_s < ct
            levels_s = levels_s[below]
            if levels_s.size > 0:
                touches = _count_touches_support(levels_s, lows_w, closes_w, tol)
                good = touches >= mt
                if np.any(good):
                    levels_s = levels_s[good]
                    touches = touches[good]
                    mx = int(touches.max())
                    cand = levels_s[touches == mx]
                    support[t] = float(cand.max())

        if np.isnan(support[t]) and fallback_minmax:
            support[t] = float(np.min(lows_w))

        mask_ph = ph[lo : i_max_pivot + 1]
        cand_i = idx_range[mask_ph]
        if cand_i.size > 0:
            levels_r = high[cand_i].astype(np.float64)
            above = levels_r > ct
            levels_r = levels_r[above]
            if levels_r.size > 0:
                touches = _count_touches_resistance(levels_r, highs_w, closes_w, tol)
                good = touches >= mt
                if np.any(good):
                    levels_r = levels_r[good]
                    touches = touches[good]
                    mx = int(touches.max())
                    cand = levels_r[touches == mx]
                    resistance[t] = float(cand.min())

        if np.isnan(resistance[t]) and fallback_minmax:
            resistance[t] = float(np.max(highs_w))

    return support, resistance
