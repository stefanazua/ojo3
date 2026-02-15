import os
import io
import time
import math
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ============================================================
# CONFIG
# ============================================================

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

app = FastAPI(title="IA Trading Pullback API", version="1.0")


# ============================================================
# UTILS
# ============================================================

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default


def _now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


# ============================================================
# DATA DOWNLOAD (ALPHA VANTAGE)
# ============================================================

def descargar_intraday_60min_alpha_vantage(ticker: str) -> pd.DataFrame:
    """
    Descarga velas intraday 60min desde Alpha Vantage.
    Retorna DataFrame con columnas:
    datetime, open, high, low, close, volume
    """

    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("Falta ALPHAVANTAGE_API_KEY en variables de Railway.")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "60min",
        "outputsize": "compact",  # últimas ~100 velas
        "apikey": ALPHAVANTAGE_API_KEY
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"AlphaVantage HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()

    # Errores típicos
    if "Error Message" in data:
        raise RuntimeError(f"AlphaVantage Error: {data['Error Message']}")
    if "Note" in data:
        # rate limit
        raise RuntimeError(f"AlphaVantage RateLimit: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"AlphaVantage Info: {data['Information']}")

    key = "Time Series (60min)"
    if key not in data:
        raise RuntimeError(f"No viene '{key}' en respuesta. Respuesta: {str(data)[:250]}")

    rows = []
    for dt_str, values in data[key].items():
        rows.append({
            "datetime": pd.to_datetime(dt_str),
            "open": _safe_float(values.get("1. open")),
            "high": _safe_float(values.get("2. high")),
            "low": _safe_float(values.get("3. low")),
            "close": _safe_float(values.get("4. close")),
            "volume": _safe_float(values.get("5. volume")),
        })

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    # limpieza
    df = df.dropna(subset=["open", "high", "low", "close"])

    if len(df) < 50:
        raise RuntimeError("Muy pocas velas descargadas (menos de 50).")

    return df


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
# STRATEGY (PULLBACK SIMPLE)
# ============================================================

def analizar_pullback(df: pd.DataFrame) -> dict:
    """
    Estrategia simple:
    - Tendencia: EMA20 > EMA50 (bull)
    - Pullback: precio toca/cae cerca EMA20 y RSI se recupera
    - Stop: close - 1.5*ATR
    - TP: close + 2.5*ATR
    """

    df = df.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    last = df.iloc[-1]

    close = float(last["close"])
    ema20v = float(last["ema20"])
    ema50v = float(last["ema50"])
    rsiv = float(last["rsi14"]) if not math.isnan(last["rsi14"]) else None
    atrv = float(last["atr14"]) if not math.isnan(last["atr14"]) else None

    tendencia = "ALCISTA" if ema20v > ema50v else "BAJISTA"

    # Pullback simple
    cerca_ema20 = abs(close - ema20v) / close < 0.01  # dentro de 1%
    rsi_ok = (rsiv is not None) and (rsiv > 45)

    señal = "NO"
    motivo = []

    if tendencia == "ALCISTA":
        motivo.append("EMA20 > EMA50 (tendencia alcista)")
        if cerca_ema20:
            motivo.append("Precio cerca de EMA20 (pullback)")
        else:
            motivo.append("Precio NO está cerca de EMA20")
        if rsi_ok:
            motivo.append("RSI > 45 (momentum recuperando)")
        else:
            motivo.append("RSI NO confirma (>45)")

        if cerca_ema20 and rsi_ok:
            señal = "LONG"

    else:
        motivo.append("EMA20 <= EMA50 (tendencia bajista)")

    # niveles
    if atrv is None or math.isnan(atrv):
        stop = None
        tp = None
    else:
        stop = close - 1.5 * atrv
        tp = close + 2.5 * atrv

    return {
        "close": close,
        "ema20": ema20v,
        "ema50": ema50v,
        "rsi14": rsiv,
        "atr14": atrv,
        "tendencia": tendencia,
        "senal": señal,
        "motivo": motivo,
        "stop_loss": stop,
        "take_profit": tp,
        "timestamp": _now_utc_str()
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "IA Trading Pullback API running", "timestamp": _now_utc_str()}


@app.get("/analizar")
def analizar(ticker: str = Query(..., description="Ej: AAPL, SPY, QQQ")):
    try:
        ticker = _normalize_ticker(ticker)

        df = descargar_intraday_60min_alpha_vantage(ticker)
        result = analizar_pullback(df)

        return {
            "ticker": ticker,
            "fuente": "AlphaVantage TIME_SERIES_INTRADAY 60min",
            "velas": len(df),
            "resultado": result
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/pdf")
def pdf(ticker: str = Query(..., description="Ej: AAPL, SPY, QQQ")):
    try:
        ticker = _normalize_ticker(ticker)

        df = descargar_intraday_60min_alpha_vantage(ticker)
        result = analizar_pullback(df)

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)

        y = 750
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"IA Trading Pullback Report - {ticker}")
        y -= 25

        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Fecha: {result['timestamp']}")
        y -= 20
        c.drawString(50, y, f"Fuente: AlphaVantage (60min)")
        y -= 30

        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Resumen")
        y -= 18

        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Señal: {result['senal']}")
        y -= 15
        c.drawString(50, y, f"Tendencia: {result['tendencia']}")
        y -= 15
        c.drawString(50, y, f"Close: {result['close']:.4f}")
        y -= 15
        c.drawString(50, y, f"EMA20: {result['ema20']:.4f}")
        y -= 15
        c.drawString(50, y, f"EMA50: {result['ema50']:.4f}")
        y -= 15

        if result["rsi14"] is not None:
            c.drawString(50, y, f"RSI14: {result['rsi14']:.2f}")
            y -= 15

        if result["atr14"] is not None:
            c.drawString(50, y, f"ATR14: {result['atr14']:.4f}")
            y -= 15

        y -= 10
        if result["stop_loss"] is not None:
            c.drawString(50, y, f"Stop Loss (1.5 ATR): {result['stop_loss']:.4f}")
            y -= 15
        if result["take_profit"] is not None:
            c.drawString(50, y, f"Take Profit (2.5 ATR): {result['take_profit']:.4f}")
            y -= 15

        y -= 20
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Motivos")
        y -= 18

        c.setFont("Helvetica", 10)
        for m in result["motivo"]:
            c.drawString(60, y, f"- {m}")
            y -= 14
            if y < 60:
                c.showPage()
                y = 750
                c.setFont("Helvetica", 10)

        c.showPage()
        c.save()

        buffer.seek(0)

        filename = f"reporte_{ticker}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'}
        )

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
