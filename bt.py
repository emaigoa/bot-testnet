# -*- coding: utf-8 -*-
"""
Motor de backtest para futuros de BTC (USDT-M, estilo Binance).
Modela: comisión taker por lado, slippage, funding cada 8h, tamaño de posición
por riesgo fijo, apalancamiento efectivo con tope, y liquidación.

Ejecución realista: la señal se evalúa al CIERRE de la vela y la orden se
ejecuta en la APERTURA de la vela siguiente (con slippage). Los stops son
stop-market intrabar.
"""
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\Emanuel\Desktop\Programacion\Proyectos\TradingML"

_cache = {}

def load(tf):
    if tf not in _cache:
        df = pd.read_csv(f"{DATA_DIR}\\btc_{tf}.csv", parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        _cache[tf] = df
    return _cache[tf]

# ---------------- Indicadores (compatibles con TradingView: RMA/EMA) ----------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)

def adx(df, n=14):
    h, l = df["high"], df["low"]
    up = h.diff()
    dn = -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    trur = atr(df, n)
    pdi = 100 * rma(plus, n) / trur
    mdi = 100 * rma(minus, n) / trur
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n)

def rsi(s, n):
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

# ---------------- Estrategias: generan señales evaluadas al cierre ----------------

def sig_trend(df, emaF=30, emaS=100, adxTh=23, atrN=14, mult=3.0):
    """Tendencia: EMA rapida/lenta + filtro ADX + trailing ATR."""
    c = df["close"]
    ef, es = ema(c, emaF), ema(c, emaS)
    a = atr(df, atrN)
    up, dn = (ef > es), (ef < es)
    if adxTh > 0:
        okA = adx(df, 14) > adxTh
    else:
        okA = pd.Series(True, index=df.index)
    return dict(
        long_e=(up & okA & (c > ef)).values,
        short_e=(dn & okA & (c < ef)).values,
        long_x=dn.values,
        short_x=up.values,
        stop_dist=(mult * a).values,
        trail_dist=(mult * a).values,
        warmup=max(emaS, 30) * 3,
    )

def sig_donchian(df, N=40, mult=3.0, emaFilt=200, atrN=14):
    """Ruptura de canal Donchian + filtro EMA + trailing ATR."""
    c = df["close"]
    hh = df["high"].rolling(N).max().shift(1)
    ll = df["low"].rolling(N).min().shift(1)
    mid = (hh + ll) / 2
    a = atr(df, atrN)
    if emaFilt > 0:
        f = ema(c, emaFilt)
        fl, fs = (c > f), (c < f)
    else:
        fl = fs = pd.Series(True, index=df.index)
    return dict(
        long_e=((c > hh) & fl).values,
        short_e=((c < ll) & fs).values,
        long_x=(c < mid).values,
        short_x=(c > mid).values,
        stop_dist=(mult * a).values,
        trail_dist=(mult * a).values,
        warmup=max(N, emaFilt, 50) * 2,
    )

def sig_meanrev(df, rsiN=2, buyTh=10, exitTh=60, emaFilt=200, mult=2.0, atrN=14):
    """Reversion a la media: RSI corto + filtro EMA de regimen + stop ATR."""
    c = df["close"]
    r = rsi(c, rsiN)
    f = ema(c, emaFilt)
    a = atr(df, atrN)
    return dict(
        long_e=((c > f) & (r < buyTh)).values,
        short_e=((c < f) & (r > 100 - buyTh)).values,
        long_x=(r > exitTh).values,
        short_x=(r < 100 - exitTh).values,
        stop_dist=(mult * a).values,
        trail_dist=(mult * a).values,
        warmup=max(emaFilt, 50) * 2,
    )

STRATS = {"trend": sig_trend, "donchian": sig_donchian, "meanrev": sig_meanrev}

# ---------------- Motor ----------------

def run(df, sig, cfg, i0=None, i1=None, collect_trades=False):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    ts = df["timestamp"]
    hour = ts.dt.hour.values
    minute = ts.dt.minute.values
    is_funding = (minute == 0) & ((hour % 8) == 0)

    fee = cfg.get("fee", 0.0005)        # taker por lado
    slip = cfg.get("slip", 0.0002)      # slippage por lado
    frate = cfg.get("funding", 0.0001)  # 0.01% cada 8h (long paga, short cobra)
    risk = cfg.get("risk", 0.02)        # % del equity arriesgado por trade
    maxlev = cfg.get("maxlev", 10.0)
    allow_short = cfg.get("allow_short", True)
    mmr = cfg.get("mmr", 0.005)         # margen de mantenimiento aprox.
    eq0 = cfg.get("capital", 100.0)
    brake_dd = cfg.get("brake_dd", 0.0)       # p.ej. 0.15: umbral de DD realizado
    brake_factor = cfg.get("brake_factor", 0.25)

    n = len(df)
    if i0 is None:
        i0 = int(sig.get("warmup", 300))
    if i1 is None:
        i1 = n

    long_e, short_e = sig["long_e"], sig["short_e"]
    long_x, short_x = sig["long_x"], sig["short_x"]
    stop_dist, trail_dist = sig["stop_dist"], sig["trail_dist"]
    tp_arr = sig.get("tp_dist")          # distancia de take profit fijo (opcional)
    size_frac = cfg.get("size_frac")     # sizing por fraccion fija del equity (opcional)

    equity = eq0
    eq_peak = eq0
    pos = 0
    qty = entry = stop = 0.0
    entry_i = 0
    entry_fees = fund_paid = 0.0
    eq_at_entry = lev_val = 0.0
    pend = 0            # 1 abrir long, -1 abrir short, 2 solo cerrar
    pend_dist = 0.0
    pend_tpd = 0.0
    tp_price = 0.0
    busted = False
    trades = []
    ec = np.empty(i1 - i0)

    def close_trade(i, px, reason):
        nonlocal equity, pos, qty
        fee_x = qty * px * fee
        pnl_bruto = qty * (px - entry) * pos
        equity += pnl_bruto - fee_x
        if collect_trades:
            trades.append(dict(
                side="LONG" if pos == 1 else "SHORT", ei=entry_i, xi=i,
                entry=entry, exit=px, qty=qty, notional=qty * entry,
                eq_entry=eq_at_entry, lev=lev_val,
                fees=entry_fees + fee_x, funding=fund_paid,
                pnl_neto=pnl_bruto - fee_x - entry_fees - fund_paid,
                eq_after=equity, reason=reason,
            ))
        pos = 0
        qty = 0.0

    for i in range(i0, i1):
        # ---- apertura: ejecutar ordenes pendientes de la vela anterior
        if pend != 0 and not busted:
            if pos != 0:
                px = o[i] * (1 - slip) if pos == 1 else o[i] * (1 + slip)
                close_trade(i, px, "senal")
            if pend in (1, -1) and equity > 0.5:
                side = pend
                px = o[i] * (1 + slip) if side == 1 else o[i] * (1 - slip)
                d = pend_dist
                if d > 0 and np.isfinite(d):
                    if size_frac:
                        q = equity * size_frac / px
                    else:
                        q = equity * risk / d
                    # circuit breaker: con drawdown realizado > umbral, opera a fraccion
                    if brake_dd > 0 and (eq_peak - equity) / eq_peak > brake_dd:
                        q *= brake_factor
                    if q * px > maxlev * equity:
                        q = maxlev * equity / px
                    qty = q
                    entry = px
                    pos = side
                    entry_i = i
                    eq_at_entry = equity
                    lev_val = qty * px / equity
                    entry_fees = qty * px * fee
                    equity -= entry_fees
                    fund_paid = 0.0
                    stop = entry - d if side == 1 else entry + d
                    tp_price = 0.0
                    if pend_tpd > 0 and np.isfinite(pend_tpd):
                        tp_price = entry + pend_tpd if side == 1 else entry - pend_tpd
            pend = 0

        if pos != 0:
            # ---- funding (si la vela arranca en las 00/08/16 UTC y ya veniamos posicionados)
            if is_funding[i] and entry_i < i:
                fp = qty * c[i] * frate * pos
                equity -= fp
                fund_paid += fp
            # ---- stop intrabar
            if pos == 1:
                if o[i] <= stop:
                    close_trade(i, o[i] * (1 - slip), "stop-gap")
                elif l[i] <= stop:
                    close_trade(i, stop * (1 - slip), "stop")
            else:
                if o[i] >= stop:
                    close_trade(i, o[i] * (1 + slip), "stop-gap")
                elif h[i] >= stop:
                    close_trade(i, stop * (1 + slip), "stop")
            # ---- take profit fijo (orden limite: sin slippage, con comision)
            if pos != 0 and tp_price > 0:
                if pos == 1:
                    if o[i] >= tp_price:
                        close_trade(i, o[i], "tp-gap")
                    elif h[i] >= tp_price:
                        close_trade(i, tp_price, "tp")
                else:
                    if o[i] <= tp_price:
                        close_trade(i, o[i], "tp-gap")
                    elif l[i] <= tp_price:
                        close_trade(i, tp_price, "tp")
            # ---- liquidacion (si el stop no salvo la posicion)
            if pos != 0:
                worst = l[i] if pos == 1 else h[i]
                if equity + qty * (worst - entry) * pos <= mmr * qty * worst:
                    equity = 0.0
                    busted = True
                    if collect_trades:
                        trades.append(dict(
                            side="LONG" if pos == 1 else "SHORT", ei=entry_i, xi=i,
                            entry=entry, exit=worst, qty=qty, notional=qty * entry,
                            eq_entry=eq_at_entry, lev=lev_val,
                            fees=entry_fees, funding=fund_paid,
                            pnl_neto=-eq_at_entry, eq_after=0.0, reason="LIQUIDACION",
                        ))
                    pos = 0
                    qty = 0.0
            # ---- actualizar trailing al cierre
            if pos == 1:
                ns = h[i] - trail_dist[i]
                if ns > stop:
                    stop = ns
            elif pos == -1:
                ns = l[i] + trail_dist[i]
                if ns < stop:
                    stop = ns

        # ---- señales al cierre
        if not busted:
            tpd = tp_arr[i] if tp_arr is not None else 0.0
            if pos == 1:
                if short_e[i] and allow_short:
                    pend, pend_dist, pend_tpd = -1, stop_dist[i], tpd
                elif long_x[i]:
                    pend = 2
            elif pos == -1:
                if long_e[i]:
                    pend, pend_dist, pend_tpd = 1, stop_dist[i], tpd
                elif short_x[i]:
                    pend = 2
            else:
                if long_e[i]:
                    pend, pend_dist, pend_tpd = 1, stop_dist[i], tpd
                elif short_e[i] and allow_short:
                    pend, pend_dist, pend_tpd = -1, stop_dist[i], tpd

        if equity > eq_peak:
            eq_peak = equity
        ec[i - i0] = equity + (qty * (c[i] - entry) * pos if pos != 0 else 0.0)

    return _metrics(df, ec, trades, i0, i1, eq0, busted, collect_trades)


def _metrics(df, ec, trades, i0, i1, eq0, busted, collect_trades):
    ts = df["timestamp"]
    years = max((ts.iloc[i1 - 1] - ts.iloc[i0]).days / 365.25, 1e-9)
    final = ec[-1] if len(ec) else eq0
    ret_tot = final / eq0 - 1
    cagr = (max(final, 1e-9) / eq0) ** (1 / years) - 1 if final > 0 else -1.0
    peak = np.maximum.accumulate(np.maximum(ec, 1e-9))
    dd = 1 - ec / peak
    maxdd = float(dd.max()) if len(dd) else 0.0
    r = np.diff(ec) / np.maximum(ec[:-1], 1e-9)
    bpy = len(ec) / years
    sharpe = float(r.mean() / r.std() * np.sqrt(bpy)) if len(r) > 1 and r.std() > 0 else 0.0
    out = dict(final=final, ret=ret_tot, cagr=cagr, maxdd=maxdd, sharpe=sharpe,
               mar=cagr / max(maxdd, 0.05), busted=busted, years=years)
    if collect_trades:
        out["trades"] = trades
        out["ec"] = ec
        out["i0"], out["i1"] = i0, i1
    if trades or not collect_trades:
        pass
    # stats de trades (si el motor los recolecto)
    if collect_trades and trades:
        pnl = np.array([t["pnl_neto"] for t in trades])
        wins = pnl[pnl > 0]
        loss = pnl[pnl <= 0]
        out.update(ntrades=len(trades),
                   winrate=len(wins) / len(trades),
                   pf=wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() != 0 else np.inf,
                   fees=sum(t["fees"] for t in trades),
                   funding=sum(t["funding"] for t in trades),
                   avg_lev=float(np.mean([t["lev"] for t in trades])),
                   max_lev=float(np.max([t["lev"] for t in trades])))
    return out


def run_counted(df, sig, cfg, i0=None, i1=None):
    """Como run() pero siempre recolecta trades para poder contar y filtrar."""
    return run(df, sig, cfg, i0, i1, collect_trades=True)
