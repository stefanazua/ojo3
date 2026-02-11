from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

app = FastAPI(title="IA Trading Pullback API", version="1.0")


# =========================================================
# UTILIDADES
# =========================================================
def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_round(x, dec=2):
    if x is None:
        return None
    try:
        return round(float(x), dec)
    except:
        return None


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =========================================================
# DESCARGA DATOS (1H + 4H, 1 mes)
# =========================================================
def fetch_data(ticker: str):
    # 1H (últimos 30 días)
    df_1h = yf.download(ticker, period="1mo", interval="1h", progress=False)
    if df_1h is None or df_1h.empty:
        return None, None

    df_1h = df_1h.dropna().copy()

    # 4H se construye resampleando 1H
    df_4h = df_1h.resample("4H").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return df_1h, df_4h


# =========================================================
# LÓGICA PULLBACK SWING (PRO)
# =========================================================
def analyze_pullback(df_1h, df_4h):
    close1 = df_1h["Close"]
    close4 = df_4h["Close"]

    # Indicadores
    df_4h["EMA20"] = ema(close4, 20)
    df_4h["EMA50"] = ema(close4, 50)
    df_4h["RSI14"] = rsi(close4, 14)
    df_4h["ATR14"] = atr(df_4h, 14)

    df_1h["EMA20"] = ema(close1, 20)
    df_1h["EMA50"] = ema(close1, 50)
    df_1h["RSI14"] = rsi(close1, 14)

    current_price = float(close1.iloc[-1])
    atr_now = float(df_4h["ATR14"].iloc[-1]) if not np.isnan(df_4h["ATR14"].iloc[-1]) else current_price * 0.01

    # Tendencia 4H
    ema20_4h = df_4h["EMA20"].iloc[-1]
    ema50_4h = df_4h["EMA50"].iloc[-1]

    if ema20_4h > ema50_4h:
        trend = "ALCISTA"
        direction = "LONG (pullback)"
    else:
        trend = "BAJISTA"
        direction = "SHORT (pullback)"

    # Pullback zone 1H
    ema20_1h = df_1h["EMA20"].iloc[-1]
    ema50_1h = df_1h["EMA50"].iloc[-1]

    # Entrada: cerca EMA20/EMA50 según tendencia
    if trend == "ALCISTA":
        entry = min(ema20_1h, ema50_1h)
        sl1 = entry - atr_now * 0.8
        sl2 = entry - atr_now * 1.2
        tp1 = entry + atr_now * 1.2
        tp2 = entry + atr_now * 2.0
    else:
        entry = max(ema20_1h, ema50_1h)
        sl1 = entry + atr_now * 0.8
        sl2 = entry + atr_now * 1.2
        tp1 = entry - atr_now * 1.2
        tp2 = entry - atr_now * 2.0

    # RR aproximado
    risk = abs(entry - sl1)
    reward = abs(tp1 - entry)
    rr = reward / (risk + 1e-9)

    # Ajustes finales
    result = {
        "timestamp": now_utc_str(),
        "precio_actual": safe_round(current_price, 4),
        "direccion": direction,
        "tendencia_4h": trend,
        "entrada_sugerida": safe_round(entry, 4),
        "sl1": safe_round(sl1, 4),
        "sl2": safe_round(sl2, 4),
        "tp1": safe_round(tp1, 4),
        "tp2": safe_round(tp2, 4),
        "rr_aprox_tp1": safe_round(rr, 2),
        "nota": "Estrategia swing pullback: tendencia 4H (EMA20/EMA50) + zona de entrada 1H + ATR como distancia dinámica."
    }
    return result


# =========================================================
# PDF
# =========================================================
def create_pdf(data, ticker):
    filename = f"resumen_{ticker}.pdf"

    c = canvas.Canvas(filename, pagesize=letter)
    w, h = letter
    y = h - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Resumen IA Trading (Pullback Swing)")
    y -= 22

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Ticker: {ticker}")
    y -= 18
    c.drawString(50, y, f"Generado: {data['timestamp']}")
    y -= 26

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Plan sugerido")
    y -= 18

    c.setFont("Helvetica", 11)
    lines = [
        f"Dirección: {data['direccion']}",
        f"Tendencia 4H: {data['tendencia_4h']}",
        f"Precio actual: {data['precio_actual']}",
        f"Entrada sugerida: {data['entrada_sugerida']}",
        f"SL1: {data['sl1']}   |   SL2: {data['sl2']}",
        f"TP1: {data['tp1']}   |   TP2: {data['tp2']}",
        f"RR aprox (TP1): {data['rr_aprox_tp1']}",
    ]

    for line in lines:
        c.drawString(50, y, line)
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Nota")
    y -= 16
    c.setFont("Helvetica", 9)
    c.drawString(50, y, data["nota"])

    y -= 25
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, y, "Aviso: Esto es apoyo educativo. No es asesoría financiera.")

    c.save()
    return filename


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/analisis")
def analisis(ticker: str = Query(..., description="Ej: BTC-USD, AAPL, TSLA")):
    df_1h, df_4h = fetch_data(ticker)
    if df_1h is None:
        return JSONResponse({"error": "No se pudieron descargar datos. Revisa ticker."}, status_code=400)

    data = analyze_pullback(df_1h, df_4h)
    data["ticker"] = ticker.upper()
    return data


@app.get("/pdf")
def pdf(ticker: str = Query(...)):
    df_1h, df_4h = fetch_data(ticker)
    if df_1h is None:
        return JSONResponse({"error": "No se pudieron descargar datos. Revisa ticker."}, status_code=400)

    data = analyze_pullback(df_1h, df_4h)
    pdf_file = create_pdf(data, ticker.upper())

    return FileResponse(
        pdf_file,
        media_type="application/pdf",
        filename=pdf_file
    )
