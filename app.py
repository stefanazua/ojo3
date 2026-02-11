from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# ✅ Permitir llamadas desde cualquier origen (Google Apps Script incluido)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================
# Helpers
# =========================================================

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analizar_ticker(ticker: str):
    ticker = ticker.upper().strip()

    # =========================================================
    # 1) Descargar data 1h (últimos 30 días)
    # =========================================================
    df_1h = yf.download(
        tickers=ticker,
        period="1mo",
        interval="1h",
        progress=False
    )

    if df_1h is None or df_1h.empty:
        return {"error": "No se pudo obtener data 1H (ticker inválido o sin datos).", "ticker": ticker}

    df_1h = df_1h.dropna()

    # =========================================================
    # 2) Crear 4h desde 1h (IMPORTANTE: '4h' minúscula)
    # =========================================================
    df_4h = df_1h.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # =========================================================
    # 3) Indicadores simples
    # =========================================================
    df_1h["SMA20"] = df_1h["Close"].rolling(20).mean()
    df_1h["SMA50"] = df_1h["Close"].rolling(50).mean()
    df_1h["RSI14"] = calcular_rsi(df_1h["Close"], 14)

    df_4h["SMA20"] = df_4h["Close"].rolling(20).mean()
    df_4h["SMA50"] = df_4h["Close"].rolling(50).mean()
    df_4h["RSI14"] = calcular_rsi(df_4h["Close"], 14)

    # =========================================================
    # 4) Lecturas finales
    # =========================================================
    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]

    precio = safe_float(last_1h["Close"])
    rsi_1h = safe_float(last_1h["RSI14"])
    rsi_4h = safe_float(last_4h["RSI14"])

    sma20_1h = safe_float(last_1h["SMA20"])
    sma50_1h = safe_float(last_1h["SMA50"])
    sma20_4h = safe_float(last_4h["SMA20"])
    sma50_4h = safe_float(last_4h["SMA50"])

    # =========================================================
    # 5) Señal simple
    # =========================================================
    tendencia_4h = None
    if sma20_4h and sma50_4h:
        tendencia_4h = "ALCISTA" if sma20_4h > sma50_4h else "BAJISTA"

    # Pullback simple: precio sobre SMA50 y RSI entre 40-60
    pullback_ok = False
    if precio and sma50_4h and rsi_4h:
        if precio > sma50_4h and 40 <= rsi_4h <= 60:
            pullback_ok = True

    # =========================================================
    # 6) Respuesta JSON
    # =========================================================
    return {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat() + "Z",

        "precio": precio,

        "tendencia_4h": tendencia_4h,
        "pullback_ok": pullback_ok,

        "rsi_1h": rsi_1h,
        "rsi_4h": rsi_4h,

        "sma20_1h": sma20_1h,
        "sma50_1h": sma50_1h,
        "sma20_4h": sma20_4h,
        "sma50_4h": sma50_4h,

        "velas_1h": int(len(df_1h)),
        "velas_4h": int(len(df_4h))
    }


# =========================================================
# Rutas API
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "API Trading Pullback activa 🚀",
        "endpoints": {
            "/analyze": "GET /analyze?ticker=AAPL"
        }
    })


@app.route("/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"error": "Falta el parámetro ticker. Ej: /analyze?ticker=AAPL"}), 400

    try:
        result = analizar_ticker(ticker)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


# =========================================================
# Render requiere escuchar en PORT
# =========================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
