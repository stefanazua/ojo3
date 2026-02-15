from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import tempfile
import time
import requests


# ============================================================
# APP
# ============================================================

app = FastAPI(title="IA Trading Pullback API", version="2.0 (TwelveData)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIG
# ============================================================

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

# Cache simple en memoria (para no gastar créditos)
CACHE = {}
CACHE_TTL_SECONDS = 60 * 10  # 10 minutos


# ============================================================
# INDICADORES
# ============================================================

def ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ============================================================
# PIVOTS
# ============================================================

def pivots(df, left=3, right=3):
    highs = df["High"].values
    lows  = df["Low"].values
    idxs = df.index

    ph = []
    pl = []

    for i in range(left, len(df)-right):
        wh = highs[i-left:i+right+1]
        wl = lows[i-left:i+right+1]

        if highs[i] == wh.max():
            ph.append((idxs[i], float(highs[i])))

        if lows[i] == wl.min():
            pl.append((idxs[i], float(lows[i])))

    return ph, pl

def last_swing_low(df):
    _, pl = pivots(df, 3, 3)
    return pl[-1][1] if pl else None

def last_swing_high(df):
    ph, _ = pivots(df, 3, 3)
    return ph[-1][1] if ph else None


# ============================================================
# SOPORTE / RESISTENCIA
# ============================================================

def pivot_levels(df, left=3, right=3):
    ph, pl = pivots(df, left, right)
    levels = [p for _, p in ph] + [p for _, p in pl]
    return levels

def cluster_levels(levels, current_price, tolerance_pct=0.004):
    if not levels:
        return []

    levels = sorted(levels)
    clustered = []
    bucket = [levels[0]]

    for lvl in levels[1:]:
        ref = np.mean(bucket)
        tol = ref * tolerance_pct
        if abs(lvl - ref) <= tol:
            bucket.append(lvl)
        else:
            clustered.append(np.mean(bucket))
            bucket = [lvl]

    clustered.append(np.mean(bucket))
    filtered = [x for x in clustered if (current_price*0.65 <= x <= current_price*1.35)]
    return sorted(filtered)


# ============================================================
# TWELVEDATA DOWNLOAD
# ============================================================

def _cache_get(key):
    obj = CACHE.get(key)
    if not obj:
        return None
    if time.time() - obj["ts"] > CACHE_TTL_SECONDS:
        return None
    return obj["value"]

def _cache_set(key, value):
    CACHE[key] = {"ts": time.time(), "value": value}

def download_1h_data_twelvedata(symbol: str, outputsize: int = 500):
    """
    Descarga velas 1H desde TwelveData.
    Retorna df con columnas: Open High Low Close Volume
    Index datetime (UTC)
    """
    if not TWELVEDATA_API_KEY:
        return None, "Falta TWELVEDATA_API_KEY en variables de Railway."

    symbol = symbol.strip().upper()
    cache_key = f"td_1h_{symbol}_{outputsize}"

    cached = _cache_get(cache_key)
    if cached is not None:
        return cached, None

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": str(outputsize),
        "format": "JSON",
        "apikey": TWELVEDATA_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        return None, f"Error de red al conectar a TwelveData: {str(e)}"

    if r.status_code != 200:
        return None, f"TwelveData HTTP {r.status_code}: {r.text}"

    try:
        data = r.json()
    except Exception:
        return None, f"TwelveData devolvió respuesta no-JSON: {r.text[:250]}"

    # Errores típicos
    if isinstance(data, dict) and "status" in data and data["status"] == "error":
        msg = data.get("message", "Error desconocido en TwelveData")
        return None, f"TwelveData error: {msg}"

    values = data.get("values")
    if not values or not isinstance(values, list):
        return None, f"No se pudieron descargar velas 1H. Respuesta TwelveData: {str(data)[:250]}"

    # Construcción DataFrame
    df = pd.DataFrame(values)

    # TwelveData entrega: datetime, open, high, low, close, volume
    required_cols = ["datetime", "open", "high", "low", "close"]
    for c in required_cols:
        if c not in df.columns:
            return None, f"TwelveData no entregó columna '{c}'."

    # Convertir
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = 0

    df = df.dropna(subset=["open", "high", "low", "close"])

    # Orden cronológico
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    # Validación mínima
    if df.empty or len(df) < 80:
        return None, f"Datos insuficientes (solo {len(df)} velas)."

    _cache_set(cache_key, df)
    return df, None


def resample_to_4h(df_1h):
    o = df_1h["Open"].resample("4H").first()
    h = df_1h["High"].resample("4H").max()
    l = df_1h["Low"].resample("4H").min()
    c = df_1h["Close"].resample("4H").last()
    v = df_1h["Volume"].resample("4H").sum()

    df4 = pd.concat([o, h, l, c, v], axis=1)
    df4.columns = ["Open","High","Low","Close","Volume"]
    df4 = df4.dropna()
    return df4


# ============================================================
# TENDENCIA 4H
# ============================================================

def trend_4h(df4):
    df = df4.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)

    last = df.iloc[-1]
    close = float(last["Close"])
    e20 = float(last["EMA20"])
    e50 = float(last["EMA50"])

    if e20 > e50 and close > e20:
        return "ALCISTA"
    if e20 < e50 and close < e20:
        return "BAJISTA"
    return "LATERAL"


# ============================================================
# BOS 1H
# ============================================================

def detect_bos_1h(df1, direction):
    df = df1.copy()
    close = float(df["Close"].iloc[-1])

    ph, pl = pivots(df, 3, 3)

    if direction == "LONG":
        if not ph:
            return False, None
        last_pivot_high = ph[-1][1]
        return close > last_pivot_high, last_pivot_high

    if direction == "SHORT":
        if not pl:
            return False, None
        last_pivot_low = pl[-1][1]
        return close < last_pivot_low, last_pivot_low

    return False, None


# ============================================================
# ENTRADA PULLBACK
# ============================================================

def compute_pullback_entry(df1, direction):
    df = df1.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["ATR14"] = atr(df, 14)
    df["RSI14"] = rsi(df["Close"], 14)

    last = df.iloc[-1]
    price = float(last["Close"])
    atr_now = float(last["ATR14"])

    if np.isnan(atr_now) or atr_now == 0:
        atr_now = price * 0.007

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi_now = float(last["RSI14"])

    if direction == "LONG":
        entry = ema20
        if rsi_now > 68:
            entry = min(entry, ema20 - 0.25*atr_now)
        if abs(price - entry) <= 0.25*atr_now:
            entry = price
    else:
        entry = ema20
        if rsi_now < 32:
            entry = max(entry, ema20 + 0.25*atr_now)
        if abs(price - entry) <= 0.25*atr_now:
            entry = price

    return entry, atr_now, ema20, ema50, rsi_now


# ============================================================
# PLAN PRO
# ============================================================

def build_swing_plan_pro(df1, df4, ticker):
    df1 = df1.copy()
    df4 = df4.copy()

    current_price = float(df1["Close"].iloc[-1])
    tendencia = trend_4h(df4)

    if tendencia == "ALCISTA":
        direction = "LONG"
    elif tendencia == "BAJISTA":
        direction = "SHORT"
    else:
        e50_1h = ema(df1["Close"], 50).iloc[-1]
        direction = "LONG" if current_price > e50_1h else "SHORT"

    entry, atr_now, ema20, ema50, rsi_now = compute_pullback_entry(df1, direction)
    bos_ok, bos_level = detect_bos_1h(df1, direction)

    levels = cluster_levels(
        pivot_levels(df1, 3, 3) + pivot_levels(df4, 2, 2),
        current_price,
        tolerance_pct=0.004
    )

    supports = sorted([x for x in levels if x < current_price])
    resistances = sorted([x for x in levels if x > current_price])

    swing_low = last_swing_low(df1)
    swing_high = last_swing_high(df1)

    if swing_low is None:
        swing_low = current_price - 2.2*atr_now
    if swing_high is None:
        swing_high = current_price + 2.2*atr_now

    if direction == "LONG":
        sl1 = entry - 1.4*atr_now
        sl2 = min(swing_low - 0.25*atr_now, entry - 2.2*atr_now)

        tp1 = resistances[0] if resistances else entry + 2.2*atr_now
        tp2 = resistances[1] if len(resistances) >= 2 else entry + 3.4*atr_now

        risk = entry - sl1
        if risk <= 0:
            sl1 = entry - 1.4*atr_now
            risk = entry - sl1

        if (tp1 - entry) < 1.6*risk:
            tp1 = entry + 1.6*risk
        if tp2 <= tp1:
            tp2 = tp1 + 1.0*risk

        rr = (tp1 - entry) / risk

    else:
        sl1 = entry + 1.4*atr_now
        sl2 = max(swing_high + 0.25*atr_now, entry + 2.2*atr_now)

        tp1 = supports[-1] if supports else entry - 2.2*atr_now
        tp2 = supports[-2] if len(supports) >= 2 else entry - 3.4*atr_now

        risk = sl1 - entry
        if risk <= 0:
            sl1 = entry + 1.4*atr_now
            risk = sl1 - entry

        if (entry - tp1) < 1.6*risk:
            tp1 = entry - 1.6*risk
        if tp2 >= tp1:
            tp2 = tp1 - 1.0*risk

        rr = (entry - tp1) / risk

    confirmacion = "CONFIRMADO (BOS detectado en 1H)" if bos_ok else "NO confirmado (esperar BOS en 1H)"

    def fmt(x):
        if x is None:
            return None
        x = float(x)
        if x >= 10:
            return round(x, 2)
        return round(x, 5)

    now = datetime.now(timezone.utc)

    plan = {
        "ticker": ticker,
        "fecha_hora": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tendencia_4h": tendencia,
        "direccion": direction,
        "confirmacion": confirmacion,
        "nivel_bos": fmt(bos_level) if bos_level else None,
        "precio_actual": fmt(current_price),
        "entrada_sugerida": fmt(entry),
        "sl1": fmt(sl1),
        "sl2": fmt(sl2),
        "tp1": fmt(tp1),
        "tp2": fmt(tp2),
        "rr_aprox_tp1": round(float(rr), 2),
        "nota": "Estrategia PRO: tendencia 4H + pullback a EMA20 1H + SL doble + TP por resistencias + RR mínimo.",
        "fuente_datos": "TwelveData 1H"
    }

    return plan


# ============================================================
# PDF
# ============================================================

def export_pdf(plan, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    w, h = letter

    y = h - 55
    lh = 18

    def write(text, bold=False, size=11):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(50, y, str(text))
        y -= lh

    write("📈 Resumen IA SwingTrading PRO (Pullback)", bold=True, size=15)
    write(f"Generado: {plan['fecha_hora']}")
    write(f"Fuente: {plan.get('fuente_datos', 'N/A')}")
    y -= 10

    write(f"Ticker: {plan['ticker']}", bold=True)
    write(f"Tendencia 4H: {plan['tendencia_4h']}")
    write(f"Dirección sugerida: {plan['direccion']}", bold=True)
    write(f"Confirmación: {plan['confirmacion']}")
    if plan["nivel_bos"]:
        write(f"Nivel BOS (referencia): {plan['nivel_bos']}")
    y -= 10

    write("📌 Plan sugerido", bold=True)
    write(f"Precio actual: {plan['precio_actual']}")
    write(f"Entrada sugerida: {plan['entrada_sugerida']}")
    write(f"SL1: {plan['sl1']}")
    write(f"SL2: {plan['sl2']}")
    write(f"TP1: {plan['tp1']}")
    write(f"TP2: {plan['tp2']}")
    write(f"RR aprox (TP1): {plan['rr_aprox_tp1']}")
    y -= 10

    write("🧠 Nota", bold=True)
    write(plan["nota"], size=9)
    y -= 8
    write("Aviso: Esto NO es asesoría financiera. Es un apoyo automatizado.", size=8)

    c.showPage()
    c.save()


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "IA Trading Pullback API running",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.get("/analizar")
def analizar(ticker: str):
    try:
        ticker = ticker.strip().upper()

        df1, err = download_1h_data_twelvedata(ticker, outputsize=500)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        if df1 is None or df1.empty:
            return JSONResponse({"error": "No se pudieron descargar datos 1H. Revisa ticker."}, status_code=400)

        df4 = resample_to_4h(df1)
        if df4 is None or df4.empty:
            return JSONResponse({"error": "No se pudo generar 4H desde 1H."}, status_code=400)

        plan = build_swing_plan_pro(df1, df4, ticker)
        return plan

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pdf")
def pdf(ticker: str):
    try:
        ticker = ticker.strip().upper()

        df1, err = download_1h_data_twelvedata(ticker, outputsize=500)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        if df1 is None or df1.empty:
            return JSONResponse({"error": "No se pudieron descargar datos 1H. Revisa ticker."}, status_code=400)

        df4 = resample_to_4h(df1)
        if df4 is None or df4.empty:
            return JSONResponse({"error": "No se pudo generar 4H desde 1H."}, status_code=400)

        plan = build_swing_plan_pro(df1, df4, ticker)

        tmpdir = tempfile.gettempdir()
        filename = os.path.join(tmpdir, f"Resumen_{ticker}_SwingPRO.pdf")

        export_pdf(plan, filename)

        return FileResponse(
            filename,
            media_type="application/pdf",
            filename=f"Resumen_{ticker}_SwingPRO.pdf"
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
