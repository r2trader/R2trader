"""R2trader: crypto paper-trading simulator. Simulated trades only; no real orders, no API keys."""
import json, os, sys, time
from datetime import datetime, timezone
import requests

SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK", "AVAX", "LTC"]
CFG = dict(start=10000.0, risk=0.01, max_pos_pct=0.25, max_open=3, daily_loss=0.03,
           fee=0.001, slip=0.0005, atr_stop=2.0, atr_tp=3.0, max_bars=48, vol_mult=2.0, ema=50)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data")
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api.binance.us"]
HOUR = 3600000


def fetch(sym):
    """Hourly candles: [open_time, open, high, low, close, volume, close_time] in ms."""
    for h in HOSTS:
        try:
            r = requests.get(h + "/api/v3/klines", timeout=20,
                             params={"symbol": sym + "USDT", "interval": "1h", "limit": 1000})
            if r.ok:
                return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[6])]
                        for k in r.json()]
        except Exception:
            pass
    try:
        r = requests.get("https://api.exchange.coinbase.com/products/%s-USD/candles" % sym, timeout=20,
                         params={"granularity": 3600}, headers={"User-Agent": "R2trader"})
        if r.ok:
            return [[k[0] * 1000, k[3], k[2], k[1], k[4], k[5], k[0] * 1000 + HOUR - 1] for k in sorted(r.json())]
    except Exception:
        pass
    return None


def ema(vals, n):
    k, e = 2 / (n + 1), vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e


def atr(c, n=14):
    trs = [max(c[i][2] - c[i][3], abs(c[i][2] - c[i - 1][4]), abs(c[i][3] - c[i - 1][4]))
           for i in range(len(c) - n, len(c))]
    return sum(trs) / n


def signal(c):
    """Breakout: close above the prior 20-bar high, volume >= 2x average, price above EMA50."""
    if len(c) < 60:
        return None
    last, prev = c[-1], c[-21:-1]
    hh = max(b[2] for b in prev)
    va = sum(b[5] for b in prev) / 20
    ratio = last[5] / va if va else 0
    trend = last[4] > ema([b[4] for b in c[-120:]], CFG["ema"])
    brk = last[4] > hh
    return dict(price=last[4], vol_ratio=ratio, breakout=brk, trend=trend, atr=atr(c),
                fire=bool(brk and trend and ratio >= CFG["vol_mult"]))


def new_state():
    return dict(cash=CFG["start"], open=[], closed=[], curve=[], day=dict(date="", eq=CFG["start"]), t=0)


def equity(st, last):
    return st["cash"] + sum(p["qty"] * last.get(p["sym"], p["entry"]) for p in st["open"])


def close_pos(st, p, px, t, why):
    px *= 1 - CFG["slip"]
    proceeds = p["qty"] * px * (1 - CFG["fee"])
    st["cash"] += proceeds
    pnl = proceeds - p["qty"] * p["entry"] * (1 + CFG["fee"])
    st["closed"].append(dict(sym=p["sym"], entry=p["entry"], exit=px, qty=p["qty"], pnl=pnl,
                             why=why, t_in=p["t"], t_out=t))


def step(st, cl, t):
    """Advance the paper account by one closed hourly bar. cl = closed candles per symbol."""
    last = {s: c[-1][4] for s, c in cl.items() if c}
    for p in list(st["open"]):
        for b in cl.get(p["sym"], []):
            if b[0] <= p["bar"]:
                continue
            p["bar"], p["n"] = b[0], p["n"] + 1
            if b[3] <= p["stop"]:
                px, why = min(p["stop"], b[1]), "stop"
            elif b[2] >= p["tp"]:
                px, why = p["tp"], "target"
            elif p["n"] >= CFG["max_bars"]:
                px, why = b[4], "time"
            else:
                continue
            close_pos(st, p, px, b[6], why)
            st["open"].remove(p)
            break
    eq = equity(st, last)
    day = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")
    if st["day"]["date"] != day:
        st["day"] = dict(date=day, eq=eq)
    halted = eq < st["day"]["eq"] * (1 - CFG["daily_loss"])
    sigs = {s: g for s, g in ((s, signal(c)) for s, c in cl.items()) if g}
    if not halted:
        for s, g in sorted(sigs.items(), key=lambda x: -x[1]["vol_ratio"]):
            if not g["fire"] or len(st["open"]) >= CFG["max_open"] or any(p["sym"] == s for p in st["open"]):
                continue
            entry, stopd = g["price"] * (1 + CFG["slip"]), CFG["atr_stop"] * g["atr"]
            if stopd <= 0:
                continue
            qty = min(eq * CFG["risk"] / stopd, eq * CFG["max_pos_pct"] / entry)
            cost = qty * entry * (1 + CFG["fee"])
            if cost > st["cash"]:
                qty = st["cash"] / (entry * (1 + CFG["fee"]))
                cost = st["cash"]
            if qty * entry < 10:
                continue
            st["cash"] -= cost
            st["open"].append(dict(sym=s, entry=entry, qty=qty, stop=entry - stopd,
                                   tp=entry + CFG["atr_tp"] * g["atr"], t=t, bar=cl[s][-1][0], n=0))
    eq = equity(st, last)
    st["curve"].append([t, round(eq, 2)])
    return sigs, halted, eq


def backtest(data):
    """Replay the same rules over the fetched history, bar by bar."""
    st, ptr = new_state(), {s: 0 for s in data}
    times = sorted({b[6] for c in data.values() for b in c})
    first = {}
    for i, t in enumerate(times):
        cl = {}
        for s, c in data.items():
            while ptr[s] < len(c) and c[ptr[s]][6] <= t:
                ptr[s] += 1
            cl[s] = c[max(0, ptr[s] - 130):ptr[s]]
        if i < 60:
            continue
        for s, c in cl.items():
            if c and s not in first:
                first[s] = c[-1][4]
        step(st, {s: c for s, c in cl.items() if len(c) >= 60}, t)
        st["t"] = t
    if not st["curve"]:
        return None
    eqs = [e for _, e in st["curve"]]
    peak, dd = eqs[0], 0.0
    for e in eqs:
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak)
    wins = sum(1 for x in st["closed"] if x["pnl"] > 0)
    bh = [(data[s][-1][4] / first[s] - 1) * 100 for s in first]
    n = len(st["closed"])
    return dict(trades=n, win_rate=round(100 * wins / n, 1) if n else 0,
                return_pct=round((eqs[-1] / CFG["start"] - 1) * 100, 2), max_drawdown_pct=round(dd * 100, 2),
                buy_hold_pct=round(sum(bh) / len(bh), 2) if bh else 0,
                days=round((times[-1] - times[60]) / 86400000, 1))


def llm_note(payload):
    k, u, m = os.getenv("LLM_API_KEY"), os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
    if not (k and u and m):
        return None
    try:
        r = requests.post(u.rstrip("/") + "/chat/completions", timeout=40,
                          headers={"Authorization": "Bearer " + k},
                          json={"model": m, "max_tokens": 300, "messages": [
                              {"role": "system", "content": "Summarise this paper-trading simulation in 4 short sentences. "
                               "Use only the numbers given. Give no trading advice."},
                              {"role": "user", "content": json.dumps(payload)}]})
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(OUT, "state.json")
    st = json.load(open(sp)) if os.path.exists(sp) else new_state()
    data = {}
    for s in SYMBOLS:
        c = fetch(s)
        if c:
            data[s] = c
        time.sleep(0.2)
    now = int(time.time() * 1000)
    cl = {s: [b for b in c if b[6] <= now][-130:] for s, c in data.items()}
    cl = {s: c for s, c in cl.items() if len(c) >= 60}
    if not cl:
        print("No market data could be fetched.")
        sys.exit(1)
    t = max(c[-1][6] for c in cl.values())
    if t > st["t"]:
        sigs, halted, eq = step(st, cl, t)
        st["t"] = t
    else:
        sigs = {s: g for s, g in ((s, signal(c)) for s, c in cl.items()) if g}
        halted = False
        eq = st["curve"][-1][1] if st["curve"] else st["cash"]
    bt = backtest({s: [b for b in c if b[6] <= now] for s, c in data.items()})
    closed = st["closed"]
    wins = sum(1 for x in closed if x["pnl"] > 0)
    live = dict(equity=round(eq, 2), return_pct=round((eq / CFG["start"] - 1) * 100, 2), trades=len(closed),
                win_rate=round(100 * wins / len(closed), 1) if closed else 0,
                open_positions=len(st["open"]), halted_today=halted)
    table = []
    for s, g in sigs.items():
        c = cl[s]
        chg = (c[-1][4] / c[-25][4] - 1) * 100 if len(c) >= 25 else 0
        table.append(dict(sym=s, price=g["price"], chg24=round(chg, 2), vol_ratio=round(g["vol_ratio"], 2),
                          breakout=g["breakout"], trend=g["trend"], fire=g["fire"]))
    table.sort(key=lambda x: -x["vol_ratio"])
    fired = [x["sym"] for x in table if x["fire"]]
    note = "Live paper account: %s equity, %s%% since start, %d closed trades. Signals firing now: %s." % (
        live["equity"], live["return_pct"], live["trades"], ", ".join(fired) or "none")
    if bt:
        note += " Backtest over %s days: %s%% vs %s%% buy-and-hold, max drawdown %s%%." % (
            bt["days"], bt["return_pct"], bt["buy_hold_pct"], bt["max_drawdown_pct"])
    note = llm_note(dict(live=live, backtest=bt, firing=fired)) or note
    dash = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), bar_time=t, config=CFG,
                live=live, backtest=bt, signals=table, open=st["open"], closed=closed[-50:][::-1],
                curve=st["curve"][-2000:], note=note)
    st["closed"], st["curve"] = closed[-500:], st["curve"][-5000:]
    json.dump(dash, open(os.path.join(OUT, "dashboard.json"), "w"))
    json.dump(st, open(sp, "w"))
    print(note)


if __name__ == "__main__":
    main()
