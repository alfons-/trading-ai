"""
Genera gráficos para la estrategia RSI + MACD (4h) / salida RSI 1D:
- Precio 4h + entradas/salidas
- Equity curve vs Buy & Hold
- Retorno por trade y distribución

Uso:
    python scripts/plot_rsi_ema_results.py
    python scripts/plot_rsi_ema_results.py --symbol ETHUSDT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gráficos estrategia RSI+MACD")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo (ej. BTCUSDT)")
    args = parser.parse_args()

    symbol = args.symbol
    trades_csv = OUTPUT_DIR / f"{symbol}_trades_rsi_macd.csv"
    price_csv = PROJECT_ROOT / "data" / "bybit" / f"{symbol}_4h.csv"

    df_trades = pd.read_csv(trades_csv)
    df_trades["entrada_fecha"] = pd.to_datetime(df_trades["entrada_fecha"])
    df_trades["salida_fecha"] = pd.to_datetime(df_trades["salida_fecha"])
    df_trades["retorno_pct"] = df_trades["retorno"] * 100

    df_price = pd.read_csv(price_csv)
    df_price["fecha"] = pd.to_datetime(df_price["fecha"])
    df_price = df_price.sort_values("fecha").reset_index(drop=True)

    # Periodo que cubren los trades RSI-EMA
    start = df_trades["entrada_fecha"].min()
    end = df_trades["salida_fecha"].max()
    df_price_win = df_price[
        (df_price["fecha"] >= start) & (df_price["fecha"] <= end + pd.Timedelta(days=1))
    ]

    # Equity curve de la estrategia (capital relativo)
    equity = (1 + df_trades["retorno"]).cumprod()
    equity_dates = df_trades["salida_fecha"]

    # Buy & Hold normalizado al mismo intervalo
    bh_start_price = df_price_win.iloc[0]["cierre"]
    df_price_win = df_price_win.copy()
    df_price_win["bh_equity"] = df_price_win["cierre"] / bh_start_price

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: Precio + entradas/salidas ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        df_price_win["fecha"],
        df_price_win["cierre"],
        color="gray",
        linewidth=0.8,
        alpha=0.7,
        label=f"{symbol} 4h",
    )
    wins = df_trades[df_trades["retorno"] > 0]
    losses = df_trades[df_trades["retorno"] <= 0]
    ax1.scatter(
        wins["entrada_fecha"],
        wins["entrada_precio"],
        marker="^",
        color="green",
        s=60,
        zorder=5,
        label=f"Entrada ganadora ({len(wins)})",
    )
    ax1.scatter(
        losses["entrada_fecha"],
        losses["entrada_precio"],
        marker="^",
        color="red",
        s=60,
        zorder=5,
        label=f"Entrada perdedora ({len(losses)})",
    )
    ax1.scatter(
        df_trades["salida_fecha"],
        df_trades["salida_precio"],
        marker="v",
        color="blue",
        s=40,
        zorder=5,
        alpha=0.7,
        label="Salida",
    )
    ax1.set_title(
        f"{symbol} – RSI+MACD (4h) / salida RSI 1D≥70: precio y operaciones",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Precio (USD)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # ── Panel 2: Equity curve ──
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(equity_dates, equity, color="blue", linewidth=2, label="RSI+MACD (acumulado)")
    ax2.plot(
        df_price_win["fecha"],
        df_price_win["bh_equity"],
        color="gray",
        linewidth=1,
        alpha=0.6,
        label="Buy & Hold",
    )
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    ax2.fill_between(
        equity_dates,
        1.0,
        equity,
        where=(equity >= 1.0),
        alpha=0.15,
        color="green",
    )
    ax2.fill_between(
        equity_dates,
        1.0,
        equity,
        where=(equity < 1.0),
        alpha=0.15,
        color="red",
    )
    ax2.set_title("Equity Curve – RSI+MACD vs Buy & Hold", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Capital relativo (1.0 = inicio)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # ── Panel 3: Retorno por trade ──
    ax3 = fig.add_subplot(gs[2, 0])
    colors = ["green" if r > 0 else "red" for r in df_trades["retorno_pct"]]
    ax3.bar(range(1, len(df_trades) + 1), df_trades["retorno_pct"], color=colors, width=0.8)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_title("RSI+MACD – Retorno por trade (%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Trade #")
    ax3.set_ylabel("Retorno (%)")
    ax3.grid(True, alpha=0.2, axis="y")

    # ── Panel 4: Distribución de retornos ──
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(
        df_trades["retorno_pct"],
        bins=10,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )
    ax4.axvline(x=0, color="black", linewidth=0.8)
    ax4.axvline(
        x=df_trades["retorno_pct"].mean(),
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Media: {df_trades['retorno_pct'].mean():.2f}%",
    )
    ax4.axvline(
        x=df_trades["retorno_pct"].median(),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mediana: {df_trades['retorno_pct'].median():.2f}%",
    )
    ax4.set_title("Distribución de retornos (RSI+MACD)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Retorno (%)")
    ax4.set_ylabel("Frecuencia")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2, axis="y")

    # ── Panel 5: Resumen ──
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")

    n_trades = len(df_trades)
    n_wins = len(wins)
    n_losses = len(losses)
    ret_total = (equity.iloc[-1] - 1) * 100
    bh_ret = (df_price_win.iloc[-1]["bh_equity"] - 1) * 100
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0
    avg_win = wins["retorno_pct"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["retorno_pct"].mean() if len(losses) > 0 else 0.0
    best = df_trades["retorno_pct"].max()
    worst = df_trades["retorno_pct"].min()
    cum_returns = (1 + df_trades["retorno"]).cumprod()
    peak = cum_returns.cummax()
    max_dd = ((cum_returns - peak) / peak).min() * 100

    summary = (
        f"{'RESUMEN RSI+MACD (4h) / salida 1D':^80}\n"
        f"{'─' * 80}\n"
        f"  Trades: {n_trades}  |  Ganadores: {n_wins}  |  Perdedores: {n_losses}  |  Win Rate: {win_rate:.1f}%\n"
        f"  Retorno estrategia: {ret_total:+.2f}%  |  Buy & Hold (mismo periodo): {bh_ret:+.2f}%  |  Alpha: {ret_total - bh_ret:+.2f}%\n"
        f"  Media ganadores: {avg_win:+.2f}%  |  Media perdedores: {avg_loss:+.2f}%  |  Ratio: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}x\n"
        f"  Mejor trade: {best:+.2f}%  |  Peor trade: {worst:+.2f}%  |  Max Drawdown: {max_dd:.2f}%"
    )
    ax5.text(
        0.5,
        0.5,
        summary,
        transform=ax5.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="lightyellow",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{symbol}_RSI_MACD_backtest_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico RSI+MACD guardado: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

