"""R2trader research harness: tests a small fixed set of strategies on public candles, with costs and an out-of-sample split.
Paper research only. Writes docs/data/research.json."""
import json, os, sys, time
from datetime import datetime, timezone
import requests

SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK", "AVAX", "LTC"]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api.binance.us"]
FEE, SLIP, START, RISK, MAXPOS, MAXOPEN, WARM = 0.001, 0.0005, 10000.0, 0.01, 0.25, 3, 210
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data")
NOEXIT = 10 ** 9


def fetch(sym, tf):
    for h in HOSTS:
        try:
            r = requests.get(h + "/api/v3/klines", timeout=20,
                             params={"symbol": sym + "USDT", "interval": tf, "limit": 1000})
            if r.ok:
                rows = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[6])]
                        for k in r.json()]
                return rows[:-1]
        except Exception:
            pass
    return None


def ema_s(v, n):
    k, e, out = 2 / (n + 1), v[0], []
    for x in v:
        e = x * k + e * (1 - k)
        out.append(e)
    return out


def rsi_s(cl, n=14):
    out, ag, al = [None] * len(cl), 0.0, 0.0
    for i in range(1, len(cl)):
        d = cl[i] - cl[i - 1]
        g, l = max(d, 0), max(-d, 0)
        if i <= n:
            ag, al = ag + g / n, al + l / n
        else:
            ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l) / n
        if i >= n:
            out[i] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def atr_s(b, n=14):
    out, a = [None] * len(b), 0.0
    for i in range(1, len(b)):
        tr = max(b[i][2] - b[i][3], abs(b[i][2] - b[i - 1][4]), abs(b[i][3] - b[i - 1][4]))
        a = a + tr / n if i <= n else (a * (n - 1) + tr) / n
        if i >= n:
            out[i] = a
    return out


def build_breakout(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    e50, at = ema_s(cl, 50), atr_s(b)
    en = [False] * n
    for i in range(60, n):
        va = sum(x[5] for x in b[i - 20:i]) / 20
        en[i] = cl[i] > max(x[2] for x in b[i - 20:i]) and va > 0 and b[i][5] >= 2 * va and cl[i] > e50[i]
    return en, [False] * n, at


def build_pullback(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    e, r, at = ema_s(cl, 100), rsi_s(cl), atr_s(b)
    en = [bool(r[i] is not None and cl[i] > e[i] and r[i] < 35) for i in range(n)]
    ex = [bool(r[i] is not None and r[i] > 55) for i in range(n)]
    return en, ex, at


def build_ema_cross(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    f, s, t, at = ema_s(cl, 20), ema_s(cl, 50), ema_s(cl, 200), atr_s(b)
    en = [i > 0 and f[i] > s[i] and f[i - 1] <= s[i - 1] and cl[i] > t[i] for i in range(n)]
    return en, [f[i] < s[i] for i in range(n)], at


def build_boll(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    e, at = ema_s(cl, 100), atr_s(b)
    en, ex = [False] * n, [False] * n
    for i in range(20, n):
        w = cl[i - 19:i + 1]
        m = sum(w) / 20
        sd = (sum((x - m) ** 2 for x in w) / 20) ** 0.5
        en[i] = cl[i] < m - 2 * sd and cl[i] > e[i]
        ex[i] = cl[i] > m
    return en, ex, at


def build_donchian(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    at, reg = atr_s(b), ctx["regime"]
    en, ex = [False] * n, [False] * n
    for i in range(60, n):
        en[i] = cl[i] > max(x[2] for x in b[i - 55:i]) and reg.get(b[i][0], False)
        ex[i] = cl[i] < min(x[3] for x in b[i - 20:i])
    return en, ex, at


STRATS = {
    "breakout20_vol (current)": (build_breakout, 2.0, 3.0, 48),
    "pullback_rsi_in_uptrend": (build_pullback, 2.5, None, 30),
    "ema20_50_cross_trend": (build_ema_cross, 3.0, None, NOEXIT),
    "bollinger_dip_in_uptrend": (build_boll, 2.5, None, 30),
    "donchian55_btc_regime": (build_donchian, 3.0, None, NOEXIT),
}


def sim(data, name, ctx):
    build, stop_atr, tp_atr, maxbars = STRATS[name]
    sig = {s: build(b, ctx) for s, b in data.items()}
    idx = {s: {x[0]: i for i, x in enumerate(b)} for s, b in data.items()}
    times = sorted({x[0] for b in data.values() for x in b})[WARM:]
    if len(times) < 100:
        return None
    k = int(0.7 * len(times))
    cash, pos, closed, curve, last, snap = START, [], [], [], {}, {}
    for ti, t in enumerate(times):
        for s, b in data.items():
            i = idx[s].get(t)
            if i is not None:
                last[s] = b[i][4]
        for p in list(pos):
            i = idx[p["s"]].get(t)
            if i is None:
                continue
            x, p["n"] = data[p["s"]][i], p["n"] + 1
            px = None
            if x[3] <= p["stop"]:
                px = min(p["stop"], x[1])
            elif p["tp"] and x[2] >= p["tp"]:
                px = p["tp"]
            elif sig[p["s"]][1][i] or p["n"] >= maxbars:
                px = x[4]
            if px is not None:
                px *= 1 - SLIP
                pr = p["q"] * px * (1 - FEE)
                cash += pr
                closed.append(pr - p["q"] * p["e"] * (1 + FEE))
                pos.remove(p)
        eq = cash + sum(p["q"] * last.get(p["s"], p["e"]) for p in pos)
        for s in data:
            i = idx[s].get(t)
            if i is None or not sig[s][0][i] or len(pos) >= MAXOPEN or any(p["s"] == s for p in pos):
                continue
            a = sig[s][2][i]
            if not a:
                continue
            e = data[s][i][4] * (1 + SLIP)
            sd = stop_atr * a
            q = min(eq * RISK / sd, eq * MAXPOS / e)
            c = q * e * (1 + FEE)
            if c > cash:
                q, c = cash / (e * (1 + FEE)), cash
            if q * e < 10:
                continue
            cash -= c
            pos.append(dict(s=s, e=e, q=q, stop=e - sd, tp=(e + tp_atr * a) if tp_atr else None, n=0))
        eq = cash + sum(p["q"] * last.get(p["s"], p["e"]) for p in pos)
        curve.append(eq)
        if ti in (0, k, len(times) - 1):
            snap[ti] = dict(last)
    peak, dd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)

    def bh(a, z):
        r = [snap[z][s] / snap[a][s] - 1 for s in snap[a] if s in snap[z]]
        return 100 * sum(r) / len(r) if r else 0

    end = len(times) - 1
    days = (times[-1] - times[0]) / 86400000
    wins = sum(1 for x in closed if x > 0)
    res = dict(trades=len(closed), trades_per_day=round(len(closed) / days, 2) if days else 0,
               win_rate=round(100 * wins / len(closed), 1) if closed else 0,
               ret=round((curve[-1] / START - 1) * 100, 2), ret_in=round((curve[k] / START - 1) * 100, 2),
               ret_out=round((curve[-1] / curve[k] - 1) * 100, 2), bh=round(bh(0, end), 2),
               bh_out=round(bh(k, end), 2), max_dd=round(dd * 100, 2), days=round(days))
    res["passes"] = bool(res["trades"] >= 30 and res["ret_out"] > 0 and res["ret_out"] > res["bh_out"] and res["max_dd"] < 25)
    return res


def run(fetcher=fetch):
    rows = []
    for tf in ("1h", "4h", "1d"):
        data = {}
        for s in SYMBOLS:
            b = fetcher(s, tf)
            if b and len(b) > WARM + 100:
                data[s] = b
            time.sleep(0.2)
        if "BTC" not in data:
            continue
        bc = [x[4] for x in data["BTC"]]
        e200 = ema_s(bc, 200)
        ctx = dict(regime={x[0]: x[4] > e200[i] for i, x in enumerate(data["BTC"])})
        for name in STRATS:
            r = sim(data, name, ctx)
            if r:
                rows.append(dict(tf=tf, strategy=name, coins=len(data), **r))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = run()
    if not rows:
        print("No data")
        sys.exit(1)
    rows.sort(key=lambda r: (not r["passes"], -r["ret_out"]))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               note="Fees 0.1%/side, slippage 0.05%. ret_out = last 30% of history, never used to pick rules. "
                    "15 variants were tested, so the best one may be luck.", rows=rows)
    json.dump(out, open(os.path.join(OUT, "research.json"), "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
