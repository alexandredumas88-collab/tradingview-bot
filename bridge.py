"""
MT5 REST bridge — exposes POST /order and GET /health for the Node.js bot.

Requirements:
    pip install flask MetaTrader5

Usage:
    python bridge.py

MT5 must be installed, running, and logged in before starting this script.
"""

import sys
from flask import Flask, request, jsonify
import MetaTrader5 as mt5

app = Flask(__name__)


def connect():
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialize() failed: {mt5.last_error()}", flush=True)
        sys.exit(1)
    info = mt5.account_info()
    print(f"[INFO] Connected to MT5 — account {info.login} ({info.server})", flush=True)


@app.get("/health")
def health():
    connected = mt5.terminal_info() is not None
    return jsonify({"status": "ok" if connected else "disconnected"}), 200


@app.post("/order")
def place_order():
    data = request.get_json(force=True)

    symbol = data.get("symbol")
    action = data.get("action", "").upper()   # BUY or SELL
    volume = float(data.get("volume", 0.01))
    comment = data.get("comment", "TradingView")

    if not symbol or action not in ("BUY", "SELL"):
        return jsonify({"error": "symbol and action (BUY/SELL) are required"}), 400

    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        return jsonify({"error": f"Symbol not found: {symbol}"}), 400

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return jsonify({"error": f"No tick data for {symbol}"}), 400

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if action == "BUY" else tick.bid

    order_request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      volume,
        "type":        order_type,
        "price":       price,
        "deviation":   20,
        "magic":       20250521,
        "comment":     comment,
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(order_request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else -1
        comment_out = result.comment if result else str(mt5.last_error())
        return jsonify({"error": "Order failed", "retcode": retcode, "detail": comment_out}), 502

    return jsonify({
        "order":   result.order,
        "volume":  result.volume,
        "price":   result.price,
        "symbol":  symbol,
        "action":  action,
        "comment": result.comment,
    }), 200


@app.post("/close")
def close_position():
    data = request.get_json(force=True)

    symbol = data.get("symbol")
    ticket = data.get("ticket")

    if not symbol and not ticket:
        return jsonify({"error": "Provide symbol (closes all) or ticket (closes one)"}), 400

    if ticket:
        positions = mt5.positions_get(ticket=int(ticket))
    else:
        positions = mt5.positions_get(symbol=symbol)

    if not positions:
        return jsonify({"error": "No open positions found"}), 404

    results = []
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            results.append({"ticket": pos.ticket, "error": "No tick data"})
            continue

        # Close is the opposite direction of the open
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        close_request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    20,
            "magic":        20250521,
            "comment":      "Close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(close_request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else -1
            detail = result.comment if result else str(mt5.last_error())
            results.append({"ticket": pos.ticket, "error": "Close failed", "retcode": retcode, "detail": detail})
        else:
            results.append({"ticket": pos.ticket, "closed": True, "volume": result.volume, "price": result.price})

    status = 200 if all(r.get("closed") for r in results) else 502
    return jsonify({"results": results}), status


if __name__ == "__main__":
    connect()
    app.run(host="127.0.0.1", port=5000)
