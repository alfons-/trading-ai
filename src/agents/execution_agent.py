"""
ExecutionAgent: ejecuta órdenes en Bybit (testnet o producción) via pybit.

Soporta:
  - Consulta de balance y posiciones abiertas
  - Órdenes market y limit (long / close)
  - Stop-loss y take-profit automáticos al abrir posición
  - Modo paper (local, sin conexión) para testing rápido
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OrderSide = Literal["Buy", "Sell"]
OrderType = Literal["Market", "Limit"]

_TESTNET_URL = "https://api-testnet.bybit.com"
_MAINNET_URL = "https://api.bybit.com"


class ExecutionAgent:
    """Gestiona la ejecución de órdenes en Bybit v5 (unified trading account)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = True,
        category: str = "linear",
        tld: str = "com",
        log_dir: Path | str | None = None,
    ):
        from pybit.unified_trading import HTTP

        self.testnet = testnet
        self.category = category.strip().lower()

        self._session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
            tld=tld,
        )

        self._log_dir = Path(log_dir) if log_dir else _PROJECT_ROOT / "data" / "execution_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        env_label = "TESTNET" if testnet else "MAINNET"
        region = f" [{tld.upper()}]" if tld != "com" else ""
        print(f"[ExecutionAgent] Conectado a Bybit {env_label}{region} ({self.category})")

    # ------------------------------------------------------------------
    # Información de cuenta
    # ------------------------------------------------------------------

    def get_balance(self, coin: str = "USDT") -> dict[str, Any]:
        """Devuelve balance disponible y total para una moneda."""
        resp = self._session.get_wallet_balance(accountType="UNIFIED")
        for acct in resp["result"]["list"]:
            for c in acct.get("coin", []):
                if c["coin"] == coin:
                    return {
                        "coin": coin,
                        "equity": float(c.get("equity", 0)),
                        "available": float(c.get("availableToWithdraw", 0)),
                        "wallet_balance": float(c.get("walletBalance", 0)),
                        "unrealised_pnl": float(c.get("unrealisedPnl", 0)),
                    }
        return {"coin": coin, "equity": 0, "available": 0, "wallet_balance": 0, "unrealised_pnl": 0}

    def get_positions(self, symbol: str | None = None) -> list[dict]:
        """Lista posiciones abiertas (filtradas por symbol si se indica)."""
        params: dict[str, Any] = {"category": self.category}
        if symbol:
            params["symbol"] = symbol
        resp = self._session.get_positions(**params)
        positions = []
        for p in resp["result"]["list"]:
            size = float(p.get("size", 0))
            if size == 0:
                continue
            positions.append({
                "symbol": p["symbol"],
                "side": p["side"],
                "size": size,
                "entry_price": float(p.get("avgPrice", 0)),
                "mark_price": float(p.get("markPrice", 0)),
                "unrealised_pnl": float(p.get("unrealisedPnl", 0)),
                "leverage": p.get("leverage", "1"),
                "stop_loss": p.get("stopLoss", ""),
                "take_profit": p.get("takeProfit", ""),
            })
        return positions

    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Precio actual (last, bid, ask) para un símbolo."""
        resp = self._session.get_tickers(category=self.category, symbol=symbol)
        tick = resp["result"]["list"][0]
        return {
            "last": float(tick.get("lastPrice", 0)),
            "bid": float(tick.get("bid1Price", 0)),
            "ask": float(tick.get("ask1Price", 0)),
            "volume_24h": float(tick.get("volume24h", 0)),
        }

    # ------------------------------------------------------------------
    # Órdenes
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType = "Market",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        """
        Coloca una orden en Bybit.

        Args:
            symbol: par (e.g. BTCUSDT)
            side: "Buy" o "Sell"
            qty: cantidad en unidades del activo
            order_type: "Market" o "Limit"
            price: precio límite (solo para Limit)
            stop_loss: precio de stop-loss
            take_profit: precio de take-profit
            reduce_only: True para cerrar posición existente
            time_in_force: "GTC", "IOC", "FOK"
        """
        params: dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
        }

        if order_type == "Limit" and price is not None:
            params["price"] = str(price)

        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        if reduce_only:
            params["reduceOnly"] = True

        resp = self._session.place_order(**params)
        result = resp.get("result", {})

        order_info = {
            "order_id": result.get("orderId", ""),
            "order_link_id": result.get("orderLinkId", ""),
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "qty": qty,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reduce_only": reduce_only,
            "ret_code": resp.get("retCode", -1),
            "ret_msg": resp.get("retMsg", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._log_order(order_info)

        status = "OK" if resp.get("retCode") == 0 else "ERROR"
        print(
            f"[ExecutionAgent] {status} | {side} {qty} {symbol} @ {order_type}"
            f"{f' price={price}' if price else ''}"
            f"{f' SL={stop_loss}' if stop_loss else ''}"
            f"{f' TP={take_profit}' if take_profit else ''}"
            f" → orderId={order_info['order_id']}"
        )

        return order_info

    def open_long(
        self,
        symbol: str,
        qty: float,
        *,
        order_type: OrderType = "Market",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Abre posición long (Buy)."""
        return self.place_order(
            symbol=symbol,
            side="Buy",
            qty=qty,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def close_long(
        self,
        symbol: str,
        qty: float,
        *,
        order_type: OrderType = "Market",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Cierra posición long (Sell, reduceOnly)."""
        return self.place_order(
            symbol=symbol,
            side="Sell",
            qty=qty,
            order_type=order_type,
            price=price,
            reduce_only=True,
        )

    def open_short(
        self,
        symbol: str,
        qty: float,
        *,
        order_type: OrderType = "Market",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Abre posición short (Sell)."""
        return self.place_order(
            symbol=symbol,
            side="Sell",
            qty=qty,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def close_short(
        self,
        symbol: str,
        qty: float,
        *,
        order_type: OrderType = "Market",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Cierra posición short (Buy, reduceOnly)."""
        return self.place_order(
            symbol=symbol,
            side="Buy",
            qty=qty,
            order_type=order_type,
            price=price,
            reduce_only=True,
        )

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Ajusta el apalancamiento para un símbolo."""
        resp = self._session.set_leverage(
            category=self.category,
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )
        print(f"[ExecutionAgent] Leverage {symbol} → {leverage}x (retCode={resp.get('retCode')})")
        return resp

    def get_order_history(self, symbol: str, limit: int = 20) -> list[dict]:
        """Últimas órdenes ejecutadas."""
        resp = self._session.get_order_history(
            category=self.category, symbol=symbol, limit=limit
        )
        return resp.get("result", {}).get("list", [])

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Órdenes abiertas (pendientes)."""
        params: dict[str, Any] = {"category": self.category}
        if symbol:
            params["symbol"] = symbol
        resp = self._session.get_open_orders(**params)
        return resp.get("result", {}).get("list", [])

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancela todas las órdenes abiertas para un símbolo."""
        resp = self._session.cancel_all_orders(category=self.category, symbol=symbol)
        print(f"[ExecutionAgent] Canceladas todas las órdenes de {symbol}")
        return resp

    # ------------------------------------------------------------------
    # Instrument info (para calcular qty mínima)
    # ------------------------------------------------------------------

    def get_instrument_info(self, symbol: str) -> dict[str, Any]:
        """Devuelve info del instrumento (minQty, qtyStep, tickSize, etc.)."""
        resp = self._session.get_instruments_info(category=self.category, symbol=symbol)
        items = resp.get("result", {}).get("list", [])
        if not items:
            return {}
        inst = items[0]
        lot_filter = inst.get("lotSizeFilter", {})
        price_filter = inst.get("priceFilter", {})
        return {
            "symbol": inst.get("symbol", symbol),
            "min_qty": float(lot_filter.get("minOrderQty", 0)),
            "max_qty": float(lot_filter.get("maxOrderQty", 0)),
            "qty_step": float(lot_filter.get("qtyStep", 0)),
            "tick_size": float(price_filter.get("tickSize", 0)),
            "min_price": float(price_filter.get("minPrice", 0)),
        }

    def calculate_qty(
        self,
        symbol: str,
        capital_usdt: float,
        leverage: int = 1,
        price: float | None = None,
    ) -> float:
        """
        Calcula la cantidad de contratos/monedas para un capital dado.

        Redondea al qtyStep del instrumento.
        """
        if price is None:
            ticker = self.get_ticker(symbol)
            price = ticker["last"]

        info = self.get_instrument_info(symbol)
        qty_step = info.get("qty_step", 0.001)
        min_qty = info.get("min_qty", 0.001)

        raw_qty = (capital_usdt * leverage) / price
        qty = max(min_qty, round(raw_qty / qty_step) * qty_step)
        return round(qty, 8)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_order(self, order_info: dict) -> None:
        log_file = self._log_dir / "orders.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(order_info, default=str) + "\n")

    def get_execution_log(self) -> pd.DataFrame:
        """Carga el log de ejecuciones como DataFrame."""
        log_file = self._log_dir / "orders.jsonl"
        if not log_file.exists():
            return pd.DataFrame()
        records = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def print_status(self, symbol: str) -> None:
        """Imprime resumen de cuenta y posición para un símbolo."""
        balance = self.get_balance()
        positions = self.get_positions(symbol)
        ticker = self.get_ticker(symbol)

        env = "TESTNET" if self.testnet else "MAINNET"
        print(f"\n{'='*55}")
        print(f"  Estado Bybit ({env}) — {symbol}")
        print(f"{'='*55}")
        print(f"  Balance USDT: {balance['wallet_balance']:.2f}")
        print(f"  Disponible:   {balance['available']:.2f}")
        print(f"  PnL no real.: {balance['unrealised_pnl']:.2f}")
        print(f"  Precio:       {ticker['last']:.2f} (bid={ticker['bid']:.2f} ask={ticker['ask']:.2f})")

        if positions:
            for p in positions:
                print(
                    f"  Posición: {p['side']} {p['size']} @ {p['entry_price']:.2f}"
                    f"  PnL={p['unrealised_pnl']:.2f}"
                    f"  lev={p['leverage']}x"
                    f"  SL={p['stop_loss'] or 'N/A'} TP={p['take_profit'] or 'N/A'}"
                )
        else:
            print("  Sin posiciones abiertas")
        print()


class PaperExecutionAgent:
    """
    Simulador local de ejecución (sin conexión a exchange).

    Misma interfaz que ExecutionAgent para testing rápido de la
    lógica de señales sin necesidad de API keys ni testnet.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        category: str = "linear",
        log_dir: Path | str | None = None,
    ):
        self.testnet = True
        self.category = category
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._order_counter = 0

        self._log_dir = Path(log_dir) if log_dir else _PROJECT_ROOT / "data" / "paper_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trades_log_file = self._log_dir / "paper_trades.jsonl"

        print(f"[PaperExecution] Simulador local iniciado | Balance: {initial_balance:.2f} USDT")

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"paper-{self._order_counter:06d}-{uuid.uuid4().hex[:8]}"

    def get_balance(self, coin: str = "USDT") -> dict[str, Any]:
        unrealised = sum(p.get("unrealised_pnl", 0) for p in self._positions.values())
        return {
            "coin": coin,
            "equity": self._balance + unrealised,
            "available": self._balance,
            "wallet_balance": self._balance,
            "unrealised_pnl": unrealised,
        }

    def get_positions(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            p = self._positions.get(symbol)
            return [p] if p else []
        return list(self._positions.values())

    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Obtiene precio real de Bybit (público, sin auth)."""
        import httpx
        resp = httpx.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": self.category, "symbol": symbol},
            timeout=10,
        )
        resp.raise_for_status()
        tick = resp.json()["result"]["list"][0]
        return {
            "last": float(tick.get("lastPrice", 0)),
            "bid": float(tick.get("bid1Price", 0)),
            "ask": float(tick.get("ask1Price", 0)),
            "volume_24h": float(tick.get("volume24h", 0)),
        }

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType = "Market",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        exec_price = price if price else self.get_ticker(symbol)["last"]
        order_id = self._next_order_id()
        cost = qty * exec_price

        if side == "Buy" and not reduce_only:
            self._balance -= cost
            self._positions[symbol] = {
                "symbol": symbol,
                "side": "Buy",
                "size": qty,
                "entry_price": exec_price,
                "mark_price": exec_price,
                "unrealised_pnl": 0.0,
                "leverage": "1",
                "stop_loss": str(stop_loss) if stop_loss else "",
                "take_profit": str(take_profit) if take_profit else "",
            }
        elif side == "Sell" and reduce_only:
            pos = self._positions.pop(symbol, None)
            if pos:
                pnl = (exec_price - pos["entry_price"]) * qty
                self._balance += cost + pnl
                trade = {
                    "symbol": symbol,
                    "entry_price": pos["entry_price"],
                    "exit_price": exec_price,
                    "qty": qty,
                    "pnl": pnl,
                    "retorno": exec_price / pos["entry_price"] - 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._trades.append(trade)
                self._log_trade(trade)
        elif side == "Sell" and not reduce_only:
            self._positions[symbol] = {
                "symbol": symbol,
                "side": "Sell",
                "size": qty,
                "entry_price": exec_price,
                "mark_price": exec_price,
                "unrealised_pnl": 0.0,
                "leverage": "1",
                "stop_loss": str(stop_loss) if stop_loss else "",
                "take_profit": str(take_profit) if take_profit else "",
            }
        elif side == "Buy" and reduce_only:
            pos = self._positions.pop(symbol, None)
            if pos:
                pnl = (pos["entry_price"] - exec_price) * qty
                self._balance += (qty * pos["entry_price"]) + pnl
                trade = {
                    "symbol": symbol,
                    "entry_price": pos["entry_price"],
                    "exit_price": exec_price,
                    "qty": qty,
                    "pnl": pnl,
                    "retorno": pos["entry_price"] / exec_price - 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._trades.append(trade)
                self._log_trade(trade)

        order_info = {
            "order_id": order_id,
            "order_link_id": "",
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "qty": qty,
            "price": exec_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reduce_only": reduce_only,
            "ret_code": 0,
            "ret_msg": "paper_ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._log_order(order_info)
        print(
            f"[PaperExecution] {side} {qty} {symbol} @ {exec_price:.2f}"
            f"{f' SL={stop_loss}' if stop_loss else ''}"
            f"{f' TP={take_profit}' if take_profit else ''}"
            f" | Balance: {self._balance:.2f} USDT"
        )
        return order_info

    def open_long(self, symbol: str, qty: float, **kwargs) -> dict[str, Any]:
        return self.place_order(symbol=symbol, side="Buy", qty=qty, **kwargs)

    def close_long(self, symbol: str, qty: float, **kwargs) -> dict[str, Any]:
        return self.place_order(symbol=symbol, side="Sell", qty=qty, reduce_only=True, **kwargs)

    def open_short(self, symbol: str, qty: float, **kwargs) -> dict[str, Any]:
        return self.place_order(symbol=symbol, side="Sell", qty=qty, **kwargs)

    def close_short(self, symbol: str, qty: float, **kwargs) -> dict[str, Any]:
        return self.place_order(symbol=symbol, side="Buy", qty=qty, reduce_only=True, **kwargs)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        print(f"[PaperExecution] Leverage {symbol} → {leverage}x (simulado)")
        return {"retCode": 0}

    def get_instrument_info(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "min_qty": 0.001, "max_qty": 100, "qty_step": 0.001, "tick_size": 0.01, "min_price": 0.01}

    def calculate_qty(self, symbol: str, capital_usdt: float, leverage: int = 1, price: float | None = None) -> float:
        if price is None:
            price = self.get_ticker(symbol)["last"]
        raw_qty = (capital_usdt * leverage) / price
        return round(raw_qty, 3)

    def get_order_history(self, symbol: str, limit: int = 20) -> list[dict]:
        return self._trades[-limit:]

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return []

    def cancel_all_orders(self, symbol: str) -> dict:
        return {"retCode": 0}

    def _log_order(self, order_info: dict) -> None:
        log_file = self._log_dir / "paper_orders.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(order_info, default=str) + "\n")

    def _log_trade(self, trade_info: dict) -> None:
        with open(self._trades_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade_info, default=str) + "\n")

    def get_execution_log(self) -> pd.DataFrame:
        log_file = self._log_dir / "paper_orders.jsonl"
        if not log_file.exists():
            return pd.DataFrame()
        records = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def print_status(self, symbol: str) -> None:
        balance = self.get_balance()
        positions = self.get_positions(symbol)
        pnl_total = self._balance - self._initial_balance
        pnl_pct = pnl_total / self._initial_balance * 100

        print(f"\n{'='*60}")
        print(f"  Paper Trading — {symbol}")
        print(f"{'='*60}")

        print(f"\n  ── Balance ──")
        print(f"  Capital inicial: {self._initial_balance:,.2f} USDT")
        print(f"  Balance actual:  {balance['wallet_balance']:,.2f} USDT")
        print(f"  Equity:          {balance['equity']:,.2f} USDT")
        print(f"  PnL realizado:   {pnl_total:+,.2f} USDT ({pnl_pct:+.2f}%)")
        if balance["unrealised_pnl"] != 0:
            print(f"  PnL no real.:    {balance['unrealised_pnl']:+,.2f} USDT")

        print(f"\n  ── Posiciones abiertas ──")
        if positions:
            for p in positions:
                try:
                    mark = self.get_ticker(symbol)["last"]
                    if p["side"] == "Buy":
                        upnl = (mark - p["entry_price"]) * p["size"]
                    else:
                        upnl = (p["entry_price"] - mark) * p["size"]
                    upnl_pct = upnl / (p["entry_price"] * p["size"]) * 100
                    print(
                        f"  {p['side']} {p['size']} @ {p['entry_price']:.2f}"
                        f"  →  mark={mark:.2f}"
                        f"  PnL={upnl:+,.2f} USDT ({upnl_pct:+.2f}%)"
                    )
                    if p.get("stop_loss"):
                        print(f"       SL={p['stop_loss']}  TP={p.get('take_profit', 'N/A')}")
                except Exception:
                    print(f"  {p['side']} {p['size']} @ {p['entry_price']:.2f}")
        else:
            print("  (ninguna)")

        print(f"\n  ── Historial de trades ({len(self._trades)}) ──")
        if self._trades:
            wins = sum(1 for t in self._trades if t["pnl"] > 0)
            losses = sum(1 for t in self._trades if t["pnl"] <= 0)
            total_pnl = sum(t["pnl"] for t in self._trades)
            print(f"  Win/Loss: {wins}W / {losses}L  |  Win rate: {wins/len(self._trades):.1%}  |  PnL acum: {total_pnl:+,.2f} USDT")
            print(f"  {'#':>3}  {'Símbolo':<10}  {'Entrada':>10}  {'Salida':>10}  {'Qty':>8}  {'PnL':>12}  {'Retorno':>9}  {'Fecha'}")
            print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*9}  {'─'*19}")
            for i, t in enumerate(self._trades, 1):
                ts = t.get("timestamp", "")[:19]
                print(
                    f"  {i:>3}  {t['symbol']:<10}"
                    f"  {t['entry_price']:>10.2f}"
                    f"  {t['exit_price']:>10.2f}"
                    f"  {t['qty']:>8.4f}"
                    f"  {t['pnl']:>+12.2f}"
                    f"  {t['retorno']:>+8.2%}"
                    f"  {ts}"
                )
        else:
            print("  (sin trades todavía)")

        print(f"\n{'='*60}\n")
