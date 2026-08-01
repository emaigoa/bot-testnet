# -*- coding: utf-8 -*-
"""
Bot de paper trading para la estrategia FINAL 4H (EMA 21/89 + RSI + filtro ATR
+ brackets 2.5/4.0 xATR + cortos espejados + circuit breaker 25%).

DOS MODOS (elige solo segun config_testnet.json):
  1) PAPEL LOCAL  — sin claves: simula los fills localmente y lleva un equity
     ficticio en state_paper.json. Ideal para arrancar hoy.
  2) TESTNET      — con claves de https://testnet.binancefuture.com: manda
     ordenes reales (MARKET + brackets STOP_MARKET / TAKE_PROFIT_MARKET) con
     dinero ficticio del testnet.

Las SEÑALES siempre se calculan con datos PUBLICOS del mercado real (fapi de
Binance), igual que el backtest: señal al cierre de la vela 4H, ejecucion al
instante siguiente.

Uso:
  python bot_paper.py --once     # un ciclo y sale (para probar)
  python bot_paper.py            # loop continuo: espera cada cierre de 4H
"""
import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt

# ───────────────────────── Configuracion ─────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(DIR, "config_testnet.json")
STATE_FILE = os.path.join(DIR, "state_paper.json")
LOG_FILE = os.path.join(DIR, "bot_log.txt")
TRADES_FILE = os.path.join(DIR, "trades_paper.csv")

SYMBOL = "ETHUSDT"
INTERVAL = "4h"
BARS_MS = 4 * 3600 * 1000

P = dict(
    emaF=21, emaS=89, rsiLen=14, rsiLo=55.0, rsiHi=75.0,
    atrLen=14, atrAvg=20, slMult=2.5, tpMult=4.0,
    sizePct=0.40,          # fraccion del equity por trade
    shorts=True,           # cortos espejados (validados en ETH)
    brakeDD=0.25,          # circuit breaker: umbral de drawdown realizado
    brakeFactor=0.25,      # tamaño durante el freno
    feeLocal=0.00075,      # comision simulada en modo papel local
    qtyStep=0.001,         # step de cantidad ETHUSDT futuros
    priceStep=0.01,        # tick de precio
)

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"


def log(msg):
    linea = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}"
    print(linea, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


# ───────────────────────── Datos publicos (mainnet) ─────────────────────────

def klines(limit=600):
    """Velas del mercado real. Si el runner esta en una region bloqueada por
    fapi.binance.com (p.ej. GitHub Actions en EE.UU. -> HTTP 451), cae al
    espejo publico de datos spot data-api.binance.vision (diferencia de
    precios con el perpetuo: despreciable para las señales)."""
    urls = [
        f"{MAINNET}/fapi/v1/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={limit}",
        f"https://data-api.binance.vision/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={limit}",
    ]
    data, err = None, None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=20) as r:
                data = json.loads(r.read())
            if "binance.vision" in u:
                log("Aviso: usando datos spot (espejo binance.vision) por bloqueo regional de fapi.")
            break
        except urllib.error.HTTPError as e:
            err = e
            continue
    if data is None:
        raise RuntimeError(f"No pude obtener velas de ninguna fuente: {err}")
    df = pd.DataFrame([dict(
        open_time=int(k[0]), open=float(k[1]), high=float(k[2]), low=float(k[3]),
        close=float(k[4]), volume=float(k[5]), close_time=int(k[6])) for k in data])
    # descartar la vela aun abierta
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] <= now_ms].reset_index(drop=True)
    return df


def senal(df):
    """Evalua las condiciones sobre la ULTIMA vela cerrada. Devuelve (lado, atr, close)."""
    c = df["close"]
    ef = bt.ema(c, P["emaF"])
    es = bt.ema(c, P["emaS"])
    r = bt.rsi(c, P["rsiLen"])
    a = bt.atr(df.rename(columns=str), P["atrLen"])
    aProm = a.rolling(P["atrAvg"]).mean()
    i = len(df) - 1
    volOk = a.iloc[i] > aProm.iloc[i]
    lado = None
    if ef.iloc[i] > es.iloc[i] and c.iloc[i] > ef.iloc[i] and P["rsiLo"] <= r.iloc[i] <= P["rsiHi"] and volOk:
        lado = "LONG"
    elif P["shorts"] and ef.iloc[i] < es.iloc[i] and c.iloc[i] < ef.iloc[i] \
            and (100 - P["rsiHi"]) <= r.iloc[i] <= (100 - P["rsiLo"]) and volOk:
        lado = "SHORT"
    detalle = (f"close={c.iloc[i]:.2f} ema21={ef.iloc[i]:.2f} ema89={es.iloc[i]:.2f} "
               f"rsi={r.iloc[i]:.1f} atr={a.iloc[i]:.2f} volOk={bool(volOk)}")
    return lado, float(a.iloc[i]), float(c.iloc[i]), detalle


# ───────────────────────── Estado local ─────────────────────────

def cargar_estado():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return dict(equity=100.0, peak=100.0, pos=0, qty=0.0, entry=0.0,
                sl=0.0, tp=0.0, entry_time="", last_bar=0)


def guardar_estado(st):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def registrar_trade(fila):
    nuevo = not os.path.exists(TRADES_FILE)
    pd.DataFrame([fila]).to_csv(TRADES_FILE, mode="a", header=nuevo, index=False)


# ───────────────────────── Testnet (ordenes firmadas) ─────────────────────────

def cargar_claves():
    # 1) variables de entorno (GitHub Actions / nube)
    k = os.environ.get("BINANCE_TESTNET_KEY", "")
    s = os.environ.get("BINANCE_TESTNET_SECRET", "")
    if k and s:
        return k, s
    # 2) archivo local
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    k, s = cfg.get("api_key", ""), cfg.get("api_secret", "")
    if not k or "PEGA_TU" in k:
        return None
    return k, s


def firmar(params, secret):
    q = urllib.parse.urlencode(params)
    sig = hmac.new(secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    return q + "&signature=" + sig


_offset_ms = None

def hora_servidor():
    """Offset entre el reloj local y el del servidor (evita error -1021)."""
    global _offset_ms
    if _offset_ms is None:
        with urllib.request.urlopen(f"{TESTNET}/fapi/v1/time", timeout=15) as r:
            server = json.loads(r.read())["serverTime"]
        _offset_ms = server - int(time.time() * 1000)
        log(f"Reloj sincronizado con Binance (desfase local: {_offset_ms} ms)")
    return int(time.time() * 1000) + _offset_ms


def req_testnet(metodo, path, params, claves):
    key, secret = claves
    params = dict(params, timestamp=hora_servidor(), recvWindow=15000)
    q = firmar(params, secret)
    url = f"{TESTNET}{path}?{q}"
    r = urllib.request.Request(url, method=metodo, headers={"X-MBX-APIKEY": key})
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        cuerpo = e.read().decode(errors="replace")
        raise RuntimeError(f"Binance testnet rechazo {metodo} {path}: HTTP {e.code} — {cuerpo}") from None


def testnet_equity(claves):
    for b in req_testnet("GET", "/fapi/v2/balance", {}, claves):
        if b["asset"] == "USDT":
            return float(b["balance"])
    return 0.0


def testnet_posicion(claves):
    for p in req_testnet("GET", "/fapi/v2/positionRisk", {"symbol": SYMBOL}, claves):
        amt = float(p["positionAmt"])
        if amt != 0:
            return amt, float(p["entryPrice"])
    return 0.0, 0.0


def rnd(x, step):
    return round(round(x / step) * step, 8)


def testnet_ciclo(claves, st):
    equity = testnet_equity(claves)
    st["peak"] = max(st.get("peak", equity), equity)
    amt, entry = testnet_posicion(claves)

    if amt != 0:
        log(f"TESTNET: posicion abierta {amt:+.3f} ETH @ {entry:.2f} — los brackets del exchange la manejan.")
        return

    # sin posicion: limpiar brackets huerfanos
    try:
        req_testnet("DELETE", "/fapi/v1/allOpenOrders", {"symbol": SYMBOL}, claves)
    except Exception:
        pass

    df = klines()
    lado, atr, close, detalle = senal(df)
    log(f"TESTNET equity={equity:.2f} | {detalle} | señal={lado}")
    if lado is None:
        return

    dd = (st["peak"] - equity) / st["peak"] if st["peak"] > 0 else 0
    factor = P["brakeFactor"] if dd > P["brakeDD"] else 1.0
    if factor < 1:
        log(f"CIRCUIT BREAKER activo (DD realizado {dd*100:.1f}%) — tamaño x{factor}")
    qty = rnd(equity * P["sizePct"] * factor / close, P["qtyStep"])
    if qty < P["qtyStep"]:
        log("Cantidad menor al minimo; no se opera.")
        return

    side = "BUY" if lado == "LONG" else "SELL"
    cierre = "SELL" if lado == "LONG" else "BUY"
    sgn = 1 if lado == "LONG" else -1
    sl = rnd(close - sgn * P["slMult"] * atr, P["priceStep"])
    tp = rnd(close + sgn * P["tpMult"] * atr, P["priceStep"])

    o = req_testnet("POST", "/fapi/v1/order",
                    dict(symbol=SYMBOL, side=side, type="MARKET", quantity=qty), claves)
    log(f"ORDEN {lado} enviada: {qty} ETH (orderId {o.get('orderId')})")
    req_testnet("POST", "/fapi/v1/order",
                dict(symbol=SYMBOL, side=cierre, type="STOP_MARKET",
                     stopPrice=sl, closePosition="true", workingType="MARK_PRICE"), claves)
    req_testnet("POST", "/fapi/v1/order",
                dict(symbol=SYMBOL, side=cierre, type="TAKE_PROFIT_MARKET",
                     stopPrice=tp, closePosition="true", workingType="MARK_PRICE"), claves)
    log(f"Brackets colocados: SL {sl} | TP {tp}")
    registrar_trade(dict(fecha=datetime.now(timezone.utc).isoformat(), modo="testnet",
                         lado=lado, qty=qty, entrada_aprox=close, sl=sl, tp=tp,
                         equity_antes=equity))


# ───────────────────────── Papel local ─────────────────────────

def local_ciclo(st):
    df = klines()
    ultima = df.iloc[-1]
    if st.get("last_bar", 0) == int(ultima["open_time"]):
        log("Sin vela nueva; nada que hacer.")
        return
    fee = P["feeLocal"]

    # 1) si hay posicion, chequear SL/TP contra la vela recien cerrada (SL primero, pesimista)
    if st["pos"] != 0:
        sgn = st["pos"]
        salida, motivo = None, ""
        if (sgn == 1 and ultima["low"] <= st["sl"]) or (sgn == -1 and ultima["high"] >= st["sl"]):
            salida, motivo = st["sl"], "stop"
        elif (sgn == 1 and ultima["high"] >= st["tp"]) or (sgn == -1 and ultima["low"] <= st["tp"]):
            salida, motivo = st["tp"], "take profit"
        if salida:
            pnl = st["qty"] * (salida - st["entry"]) * sgn - st["qty"] * salida * fee
            st["equity"] += pnl
            st["peak"] = max(st["peak"], st["equity"])
            log(f"PAPEL: cierre por {motivo} @ {salida:.2f} | PnL {pnl:+.2f} | equity {st['equity']:.2f}")
            registrar_trade(dict(fecha=datetime.now(timezone.utc).isoformat(), modo="papel",
                                 lado="LONG" if sgn == 1 else "SHORT", qty=st["qty"],
                                 entrada=st["entry"], salida=salida, motivo=motivo,
                                 pnl=round(pnl, 2), equity=round(st["equity"], 2)))
            st.update(pos=0, qty=0.0, entry=0.0, sl=0.0, tp=0.0)

    # 2) si quedo flat, evaluar señal de la vela cerrada
    if st["pos"] == 0:
        lado, atr, close, detalle = senal(df)
        log(f"PAPEL equity={st['equity']:.2f} | {detalle} | señal={lado}")
        if lado:
            dd = (st["peak"] - st["equity"]) / st["peak"] if st["peak"] > 0 else 0
            factor = P["brakeFactor"] if dd > P["brakeDD"] else 1.0
            if factor < 1:
                log(f"CIRCUIT BREAKER activo (DD {dd*100:.1f}%) — tamaño x{factor}")
            sgn = 1 if lado == "LONG" else -1
            qty = st["equity"] * P["sizePct"] * factor / close
            st["equity"] -= qty * close * fee
            st.update(pos=sgn, qty=qty, entry=close,
                      sl=close - sgn * P["slMult"] * atr,
                      tp=close + sgn * P["tpMult"] * atr,
                      entry_time=datetime.now(timezone.utc).isoformat())
            log(f"PAPEL: ENTRADA {lado} {qty:.4f} ETH @ {close:.2f} | SL {st['sl']:.2f} | TP {st['tp']:.2f}")
    else:
        log(f"PAPEL: en posicion ({'LONG' if st['pos']==1 else 'SHORT'} @ {st['entry']:.2f}, "
            f"SL {st['sl']:.2f}, TP {st['tp']:.2f})")

    st["last_bar"] = int(ultima["open_time"])
    guardar_estado(st)


# ───────────────────────── Loop principal ─────────────────────────

def un_ciclo():
    st = cargar_estado()
    claves = cargar_claves()
    if claves:
        testnet_ciclo(claves, st)
        guardar_estado(st)
    else:
        local_ciclo(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="un ciclo y salir")
    args = ap.parse_args()

    if os.environ.get("GITHUB_ACTIONS") and not cargar_claves():
        log("ERROR: faltan los secrets BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET en el repo.")
        sys.exit(1)

    modo = "TESTNET (ordenes reales con dinero ficticio)" if cargar_claves() else \
           "PAPEL LOCAL (sin claves; simulacion pura)"
    log(f"Bot iniciado — {SYMBOL} {INTERVAL} — modo: {modo}")
    un_ciclo()
    if args.once:
        return
    while True:
        now_ms = int(time.time() * 1000)
        prox = (now_ms // BARS_MS + 1) * BARS_MS + 20_000  # 20 s tras el cierre
        espera = (prox - now_ms) / 1000
        log(f"Durmiendo {espera/60:.0f} min hasta el proximo cierre de 4H...")
        time.sleep(espera)
        try:
            un_ciclo()
        except Exception as e:
            log(f"ERROR en ciclo: {e} — reintento en el proximo cierre.")


if __name__ == "__main__":
    main()
