from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import io
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yfinance as yf

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# ============================================================
# APP FASTAPI
# ============================================================
app = FastAPI(title="IA Trading Pullback API", version="1.0")

# CORS (para Google Apps Script, Hostgator, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas (luego puedes restringir)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================
def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default


def download_data(ticker: str):
    """
    Descarga datos 1H y 4H del último mes.
    yfinance a veces limita 4H, entonces se reconstruye 4H desde 1H.
    """
    ticker = ticker.strip().upper()

    # 1H directo
    df_1h = yf.download(
        ticker,
        period="1mo",
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df_1h is None or df_1h.empty:
        raise ValueError(f"No se pudo descargar datos 1H para {ticker}")

    df_1h = df_1h.reset_index()
    df_1h.rename(columns={"Datetime": "Date"}, inplace=True)
    if "Date" not in df_1h.columns:
        # fallback por si yfinance entrega otra columna
        df_1h.rename(columns={df_1h.columns[0]: "Date"}, inplace=True)

    df_1h["Date"] = pd.to_datetime(df_1h["Date"])
    df_1h = df_1h.sort_values("Date").dropna()

    # Construcción 4H desde 1H
    df_1h_indexed = df_1h.set_index("Date")

    df_4h = df_1h_indexed.resample("4H").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna().reset_index()

    return df_1h, df_4h


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def detect_pullback_plan(df_4h: pd.DataFrame, df_1h: pd.DataFrame):
    """
    Estrategia swing pullback:
    - Tendencia en 4H por EMA20 vs EMA50
    - Pullback: precio cerca EMA20 y RSI recuperándose
    - Entrada: ruptura del máximo 1H reciente (gatillo)
    - SL: bajo swing low + ATR
    - TP: R:R 1.5 y 2.5 aprox + resistencia reciente
    """

    # ---------------------------
    # Indicadores 4H
    # ---------------------------
    df4 = df_4h.copy()
    df4["EMA20"] = ema(df4["Close"], 20)
    df4["EMA50"] = ema(df4["Close"], 50)
    df4["RSI14"] = rsi(df4["Close"], 14)

    # ---------------------------
    # Indicadores 1H
    # ---------------------------
    df1 = df_1h.copy()
    df1["EMA20"] = ema(df1["Close"], 20)
    df1["EMA50"] = ema(df1["Close"], 50)
    df1["RSI14"] = rsi(df1["Close"], 14)
    df1["ATR14"] = atr(df1, 14)

    # Último precio
    last_price = float(df1["Close"].iloc[-1])
    last_time = df1["Date"].iloc[-1]

    # ---------------------------
    # Dirección por 4H
    # ---------------------------
    ema20 = df4["EMA20"].iloc[-1]
    ema50 = df4["EMA50"].iloc[-1]
    rsi4 = df4["RSI14"].iloc[-1]

    if ema20 > ema50:
        trend = "ALCISTA"
        direction = "LONG"
    else:
        trend = "BAJISTA"
        direction = "SHORT"

    # ---------------------------
    # Pullback zone (aprox)
    # ---------------------------
    # LONG: precio cerca EMA20 4H (pullback sano)
    # SHORT: precio cerca EMA20 4H pero por debajo
    last_4h_close = df4["Close"].iloc[-1]
    pullback_distance = abs(last_4h_close - ema20) / (ema20 + 1e-9)

    pullback_ok = pullback_distance < 0.02  # 2% del EMA20

    # ---------------------------
    # Gatillo 1H
    # ---------------------------
    recent = df1.tail(20)  # últimas 20 velas 1H (~1 día)
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())

    # Swing low (para SL)
    swing_low = float(df1.tail(60)["Low"].min())   # ~2.5 días
    swing_high = float(df1.tail(60)["High"].max())

    atr_now = float(df1["ATR14"].iloc[-1]) if not np.isnan(df1["ATR14"].iloc[-1]) else 0.0

    # ---------------------------
    # Definir entrada / SL / TP
    # ---------------------------
    if direction == "LONG":
        entry = recent_high  # entrada por ruptura
        sl1 = swing_low - (0.3 * atr_now)
        sl2 = swing_low - (0.8 * atr_now)

        risk = entry - sl1
        tp1 = entry + (1.5 * risk)
        tp2 = entry + (2.5 * risk)

        # Ajuste: no dejar TP ridículo si el mercado está comprimido
        tp1 = max(tp1, swing_high * 1.005)
        tp2 = max(tp2, swing_high * 1.015)

    else:
        entry = recent_low  # ruptura hacia abajo
        sl1 = swing_high + (0.3 * atr_now)
        sl2 = swing_high + (0.8 * atr_now)

        risk = sl1 - entry
        tp1 = entry - (1.5 * risk)
        tp2 = entry - (2.5 * risk)

        tp1 = min(tp1, swing_low * 0.995)
        tp2 = min(tp2, swing_low * 0.985)

    rr = abs((tp1 - entry) / (entry - sl1 + 1e-9))

    # ---------------------------
    # Texto final
    # ---------------------------
    nota = []
    nota.append(f"Tendencia 4H: {trend} (EMA20 vs EMA50).")
    nota.append(f"RSI 4H: {rsi4:.1f}.")
    if pullback_ok:
        nota.append("Pullback válido: precio cerca EMA20 4H.")
    else:
        nota.append("Pullback débil: precio lejos de EMA20 4H (precaución).")

    nota.append("Entrada sugerida por ruptura 1H (gatillo swing).")
    nota.append("SL basado en swing + ATR para evitar barridas.")

    return {
        "ticker": None,
        "fecha_generacion": now_utc_str(),
        "hora_sugerencia": str(last_time),
        "direccion": direction,
        "tendencia_4h": trend,
        "precio_actual": round(last_price, 6),
        "entrada": round(entry, 6),
        "sl1": round(sl1, 6),
        "sl2": round(sl2, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "rr_aprox_tp1": round(rr, 2),
        "nota": " ".join(nota),
    }


def build_pdf(plan: dict) -> bytes:
    """
    PDF tipo tarjeta (bonito, simple, limpio).
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter

    # Fondo
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "IA Trading Pullback (Swing)")

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.7 * cm, f"Generado: {plan['fecha_generacion']}")

    # Caja tarjeta
    x = 2 * cm
    y = height - 15 * cm
    w = width - 4 * cm
    h = 11.5 * cm

    c.roundRect(x, y, w, h, 12, stroke=1, fill=0)

    # Contenido
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 0.7 * cm, y + h - 1.2 * cm, f"Ticker: {plan['ticker']}")

    c.setFont("Helvetica", 11)
    c.drawString(x + 0.7 * cm, y + h - 2.2 * cm, f"Dirección: {plan['direccion']}  |  Tendencia 4H: {plan['tendencia_4h']}")

    c.setFont("Helvetica", 11)
    c.drawString(x + 0.7 * cm, y + h - 3.2 * cm, f"Precio actual: {plan['precio_actual']}")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 0.7 * cm, y + h - 4.4 * cm, f"Entrada sugerida: {plan['entrada']}")

    c.setFont("Helvetica", 11)
    c.drawString(x + 0.7 * cm, y + h - 5.6 * cm, f"SL1: {plan['sl1']}   |   SL2: {plan['sl2']}")

    c.drawString(x + 0.7 * cm, y + h - 6.8 * cm, f"TP1: {plan['tp1']}   |   TP2: {plan['tp2']}")

    c.drawString(x + 0.7 * cm, y + h - 8.0 * cm, f"RR aprox (TP1): {plan['rr_aprox_tp1']}")

    c.setFont("Helvetica", 9)
    c.drawString(x + 0.7 * cm, y + 1.2 * cm, f"Hora sugerencia (última vela 1H): {plan['hora_sugerencia']}")

    # Nota
    c.setFont("Helvetica", 9)
    text = c.beginText(x + 0.7 * cm, y + 3.0 * cm)
    text.setLeading(12)
    text.textLines("Nota:")
    for line in split_text(plan["nota"], 95):
        text.textLine(line)
    c.drawText(text)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        2 * cm,
        1.2 * cm,
        "Aviso: Esto es apoyo estadístico/automatizado. No es asesoría financiera.",
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


def split_text(text, max_len=90):
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= max_len:
            current += (" " if current else "") + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "API IA Trading Pullback funcionando"}


@app.get("/analyze")
def analyze(ticker: str = Query(..., description="Ticker ejemplo: AAPL, TSLA, NVDA, SPY")):
    try:
        df_1h, df_4h = download_data(ticker)
        plan = detect_pullback_plan(df_4h, df_1h)
        plan["ticker"] = ticker.upper()
        return JSONResponse(plan)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "ticker": ticker},
        )


@app.get("/pdf")
def pdf(ticker: str = Query(..., description="Ticker ejemplo: AAPL, TSLA, NVDA, SPY")):
    try:
        df_1h, df_4h = download_data(ticker)
        plan = detect_pullback_plan(df_4h, df_1h)
        plan["ticker"] = ticker.upper()

        pdf_bytes = build_pdf(plan)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="IA_Trading_{ticker.upper()}.pdf"'
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "ticker": ticker},
        )
