#!/usr/bin/env python3
"""
ULTIMATE DAY + SWING TRADING BOT - FINAL VERSION
Fixed all bugs + Professional Web Dashboard
"""

import yfinance as yf
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
import requests
import os
import logging
import uuid
import numpy as np
import json
import sys
import pytz
import threading

app = Flask(__name__, static_folder='static')
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('trading_log.txt'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIG
# ==============================
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
TELEGRAM_BOT_TOKEN = '7'
TELEGRAM_CHAT_IDS = ['78970', '79803']

MAX_POS_DAY = 3
STATE_FILE = 'state.json'
NY = pytz.timezone('America/New_York')

stocks = [
    {"name": "Galaxy Digital", "symbol": "GLXY.TO", "flag": "Canada"},
    {"name": "Canopy Growth", "symbol": "WEED.TO", "flag": "Canada"},
    {"name": "Franco-Nevada", "symbol": "FNV.TO", "flag": "Canada"},
    {"name": "Hut 8", "symbol": "HUT.TO", "flag": "Canada"},
    {"name": "Shopify", "symbol": "SHOP.TO", "flag": "Canada"},
    {"name": "Robinhood", "symbol": "HOOD", "flag": "USA"},
    {"name": "Enbridge", "symbol": "ENB.TO", "flag": "Canada"},
    {"name": "Canadian Natural", "symbol": "CNQ.TO", "flag": "Canada"},
    {"name": "Palantir", "symbol": "PLTR", "flag": "USA"},
    {"name": "Tesla", "symbol": "TSLA", "flag": "USA"}
]

# ==============================
# STATE
# ==============================
def get_default():
    return {
        "day_positions": [], "swing_position": None,
        "prev_day": "Hold", "prev_swing": "Hold",
        "reason_day": "Monitoring", "last_signal_time": 0,
        "last_swing_check": None, "last_sl_tp": ""
    }

prev_prices = {s["symbol"]: 0.0 for s in stocks}
signal_history = {s["symbol"]: get_default() for s in stocks}
day_trades = {s["symbol"]: [] for s in stocks}
swing_trades = {s["symbol"]: [] for s in stocks}

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                for sym, h in data.get("history", {}).items():
                    if sym in signal_history:
                        signal_history[sym].update(h)
        except: pass

def save_state():
    try:
        safe = {sym: {k: v for k, v in h.items() if k != "log"} for sym, h in signal_history.items()}
        with open(STATE_FILE, 'w') as f:
            json.dump({"history": safe}, f, default=str)
    except: pass

load_state()

# ==============================
# HELPERS
# ==============================
def send_telegram(msg):
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}, timeout=10)
        except: pass

def is_market_open():
    now = datetime.now(NY)
    t = now.time()
    return now.weekday() < 5 and dt_time(9, 30) <= t <= dt_time(16, 0)

# ==============================
# INDICATORS
# ==============================
def calc_5min(df):
    if len(df) < 20: return df
    df = df.copy()
    df['SMA15'] = df['Close'].rolling(15).mean()
    df['Std'] = df['Close'].rolling(15).std()
    df['UpperBB'] = df['SMA15'] + 0.7 * df['Std']
    df['LowerBB'] = df['SMA15'] - 0.7 * df['Std']
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = -delta.where(delta < 0, 0).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)
    df['TR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(10).mean()
    df['VolMA20'] = df['Volume'].rolling(20).mean()
    return df

def calc_swing(df):
    df = df.copy()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)
    df['VolMA20'] = df['Volume'].rolling(20).mean()
    return df

# ==============================
# MAIN ENGINE
# ==============================
def get_signal(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist5 = ticker.history(period="5d", interval="5m", prepost=False)
        if hist5.empty or len(hist5) < 20:
            return {"error": "No data"}

        hist5.index = hist5.index.tz_convert(NY) if hist5.index.tzinfo else hist5.index.tz_localize('UTC').tz_convert(NY)
        df5 = calc_5min(hist5)
        price = round(df5['Close'].iloc[-1], 2)
        prev = prev_prices.get(symbol, price)
        r = df5.iloc[-1]
        h = signal_history[symbol]

        # === DAY TRADE ===
        day_sig = "Hold"
        day_reason = "Monitoring"
        day_pnl = sum(t.get("profit", 0) for t in day_trades[symbol])

        # Exit
        to_close = []
        for pos in h["day_positions"]:
            if price <= pos["sl"] or price >= pos["tp"]:
                profit = price - pos["entry"]
                day_trades[symbol].append({"profit": profit})
                to_close.append(pos["id"])
                day_sig = "Sell"
                day_reason = "SL/TP Hit"
        h["day_positions"] = [p for p in h["day_positions"] if p["id"] not in to_close]

        # Buy
        reasons = []
        if price <= r['LowerBB']: reasons.append("Oversold")
        if r['RSI'] < 70: reasons.append("RSI<70")
        if r['Volume'] > 0.7 * r['VolMA20']: reasons.append("Volume Surge")
        if price > prev * 0.995: reasons.append("Price Stable")

        if len(h["day_positions"]) < MAX_POS_DAY and len(reasons) >= 3:
            atr = max(r['ATR'], 0.01)
            sl = price - 2 * atr
            tp = price + 4 * atr
            h["day_positions"].append({"id": str(uuid.uuid4()), "entry": price, "sl": sl, "tp": tp})
            day_sig = "Buy"
            day_reason = " + ".join(reasons)

        # === SWING ===
        swing_sig = "Hold"
        swing_reason = ""
        swing_pnl = sum(t.get("profit", 0) for t in swing_trades[symbol]) if swing_trades[symbol] else 0.0
        today = datetime.now(NY).date().isoformat()

        if h["last_swing_check"] != today:
            daily = ticker.history(period="6mo", interval="1d")
            if len(daily) >= 50:
                daily = calc_swing(daily)
                d = daily.iloc[-1]
                reasons_s = []
                if d['EMA50'] > d['EMA200']: reasons_s.append("Bull Trend")
                if daily['Close'].pct_change().tail(5).sum() < -0.08: reasons_s.append("Deep Pullback")
                if d['Volume'] > 1.5 * d['VolMA20']: reasons_s.append("Volume Spike")
                if d['RSI'] < 65: reasons_s.append("Not Overbought")

                if len(reasons_s) >= 4 and not h.get("swing_position"):
                    swing_sig = "Swing Buy"
                    swing_reason = " + ".join(reasons_s)
                    h["swing_position"] = {"entry": price, "trail": price * 0.88, "date": today}

                if h.get("swing_position"):
                    pos = h["swing_position"]
                    profit_pct = (price - pos["entry"]) / pos["entry"]
                    new_trail = price * 0.88
                    if price > new_trail: pos["trail"] = new_trail
                    days = (datetime.now(NY).date() - datetime.fromisoformat(pos["date"])).days
                    if price <= pos["trail"] or profit_pct >= 0.30 or days > 15:
                        swing_sig = "Swing Sell"
                        profit = price - pos["entry"]
                        swing_trades[symbol].append({"profit": profit})
                        h.pop("swing_position", None)

            h["last_swing_check"] = today

        # === ALERTS ===
        now = time.time()
        if day_sig != h["prev_day"] and day_sig in ["Buy", "Sell"] and (now - h["last_signal_time"] > 60):
            send_telegram(f"<b>{symbol} {day_sig}</b>\n<i>{day_reason}</i>\nPrice: ${price}\nP&L: ${day_pnl:.2f}")
            h["last_signal_time"] = now

        if swing_sig != "Hold" and swing_sig != h.get("prev_swing"):
            emoji = "Chart Increasing" if "Buy" in swing_sig else "Chart Decreasing"
            send_telegram(f"<b>{symbol} {swing_sig}</b>\n{emoji} <i>{swing_reason}</i>\nPrice: ${price}\nHold: 2–15 Days")
            h["prev_swing"] = swing_sig

        h["prev_day"] = day_sig
        prev_prices[symbol] = price

        return {
            "price": price,
            "open": round(hist5['Open'].iloc[0], 2),
            "high": round(hist5['High'].max(), 2),
            "low": round(hist5['Low'].min(), 2),
            "change": round(((price / hist5['Close'].iloc[0]) - 1) * 100, 2),
            "day_signal": day_sig,
            "day_reason": day_reason,
            "day_pnl": round(day_pnl, 2),
            "swing_signal": swing_sig,
            "swing_pnl": round(swing_pnl, 2),
            "swing_status": "OPEN" if h.get("swing_position") else "None"
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")
        return {"error": str(e)[:50]}

# ==============================
# ROUTES
# ==============================
@app.route('/stocks')
def get_stocks():
    result = []
    for s in stocks:
        data = get_signal(s["symbol"])
        if "error" not in data:
            result.append({
                "flag": s["flag"],
                "name": s["name"],
                "symbol": s["symbol"],
                **data
            })
        else:
            result.append({
                "flag": s["flag"], "name": s["name"], "symbol": s["symbol"],
                "price": "N/A", "open": "N/A", "high": "N/A", "low": "N/A",
                "change": 0, "day_signal": "Error", "swing_signal": "Error",
                "day_pnl": 0, "swing_pnl": 0, "swing_status": "Error"
            })
    save_state()
    return jsonify(result)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ==============================
# RUN
# ==============================
def run():
    while True:
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except:
            time.sleep(5)

if __name__ == '__main__':
    threading.Thread(target=run, daemon=True).start()
    logger.info("PRO TRADING BOT STARTED - LIVE DASHBOARD READY")

    while True: time.sleep(60)
