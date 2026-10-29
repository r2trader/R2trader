"""R2trader forward test: three paper accounts run side by side on new candles only (no look-back trades).
Paper only. Writes docs/data/forward.json and forward_state.json."""
import json, os, time
from datetime import datetime, timezone
import research as R
import research_hf as H

TF_MS = {"1h": 3600000, "4h": 14400000}
ACC = {
    "A": dict(title="A: EMA 20/50 trend (4h)", tf="4h", build=R.build_ema_cross, stop=3.0, tp=None, maxbars=R.NOEXIT,
              rule="Buy when EMA20 crosses above EMA50 and price is above EMA200. Sell on the reverse cross or a 3x ATR stop."),
    "B": dict(title="B: Donchian 55 + BTC regime (4h)", tf="4h", build=R.build_donchian, stop=3.0, tp=None, maxbars=R.NOEXIT,
              rule="Buy a 55-bar high only when BTC is above its EMA200. Sell on a 20-bar low or a 3x ATR stop."),
    "C": dict(title="C: High-frequency RSI dip (1h)", tf="1h", build=H.rsi_dip(25), stop=1.5, tp=None, maxbars=12,
              rule="Experiment aimed at 3-4 trades/day: buy when RSI(7) < 25, sell at RSI > 50, a 1.5x ATR stop or after 12 hours."),
}


def advance(st, data, a, ctx):
    sig = {s: a["build"](b, ctx) for s, b in data.items()}
    idx = {s: {x[0]: i for i, x in enumerate(b)} for s, b in data.items()}
    times = sorted({x[0] for b in data.values() for x in b})
    if not st["t"]:
        st["t"], st["start"], st["curve"] = times[-1], times[-1], [[times[-1] + TF_MS[a["tf"]] - 1, R.START]]
        return
    last = {}
    for s, b in data.items():
        prev = [x for x in b if x[0] <= st["t"]]
        if prev:
            last[s] = prev[-1][4]
    for t in [x for x in times if x > st["t"]]:
        for s, b in data.items():
            i = idx[s].get(t)
            if i is not None:
                last[s] = b[i][4]
        for p in list(st["pos"]):
            i = idx[p["s"]].get(t)
            if i is None:
                continue
            x, p["n"] = data[p["s"]][i], p["n"] + 1
            px = why = None
            if x[3] <= p["stop"]:
                px, why = min(p["stop"], x[1]), "stop"
            elif p["tp"] and x[2] >= p["tp"]:
                px, why = p["tp"], "target"
            elif sig[p["s"]][1][i]:
                px, why = x[4], "signal"
            elif p["n"] >= a["maxbars"]:
                px, why = x[4], "time"
            if px is not None:
                px *= 1 - R.SLIP
                pr = p["q"] * px * (1 - R.FEE)
                pnl = pr - p["q"] * p["e"] * (1 + R.FEE)
                st["cash"] += pr
                st["nc"], st["nw"] = st["nc"] + 1, st["nw"] + (pnl > 0)
                st["closed"].append(dict(sym=p["s"], entry=p["e"], exit=px, pnl=pnl, why=why, t_out=x[6]))
                st["pos"].remove(p)
        eq = st["cash"] + sum(p["q"] * last.get(p["s"], p["e"]) for p in st["pos"])
        for s in data:
            i = idx[s].get(t)
            if i is None or not sig[s][0][i] or len(st["pos"]) >= R.MAXOPEN or any(p["s"] == s for p in st["pos"]):
                continue
            at = sig[s][2][i]
            if not at:
                continue
            e = data[s][i][4] * (1 + R.SLIP)
            sd = a["stop"] * at
            q = min(eq * R.RISK / sd, eq * R.MAXPOS / e)
            c = q * e * (1 + R.FEE)
            if c > st["cash"]:
                q, c = st["cash"] / (e * (1 + R.FEE)), st["cash"]
            if q * e < 10:
                continue
            st["cash"] -= c
            st["pos"].append(dict(s=s, e=e, q=q, stop=e - sd, tp=(e + a["tp"] * at) if a["tp"] else None, n=0,
                                  t=data[s][i][6]))
        eq = st["cash"] + sum(p["q"] * last.get(p["s"], p["e"]) for p in st["pos"])
        st["curve"].append([t + TF_MS[a["tf"]] - 1, round(eq, 2)])
        st["t"] = t


def main():
    os.makedirs(R.OUT, exist_ok=True)
    sp = os.path.join(R.OUT, "forward_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    cache = {}
    for tf in sorted({a["tf"] for a in ACC.values()}):
        data = {}
        for s in R.SYMBOLS:
            b = R.fetch(s, tf)
            if b and len(b) > R.WARM:
                data[s] = b
            time.sleep(0.2)
        if "BTC" in data:
            e200 = R.ema_s([x[4] for x in data["BTC"]], 200)
            cache[tf] = (data, dict(regime={x[0]: x[4] > e200[i] for i, x in enumerate(data["BTC"])}))
    accounts = []
    for k, a in ACC.items():
        if a["tf"] not in cache:
            continue
        st = state.setdefault(k, dict(cash=R.START, pos=[], closed=[], curve=[], t=0, start=0, nc=0, nw=0))
        data, ctx = cache[a["tf"]]
        advance(st, data, a, ctx)
        st["closed"], st["curve"] = st["closed"][-500:], st["curve"][-3000:]
        eq = st["curve"][-1][1] if st["curve"] else R.START
        days = (st["curve"][-1][0] - st["start"]) / 86400000 if st["curve"] else 0
        accounts.append(dict(id=k, title=a["title"], rule=a["rule"], tf=a["tf"], equity=eq,
                             ret_pct=round((eq / R.START - 1) * 100, 2), trades=st["nc"],
                             win_rate=round(100 * st["nw"] / st["nc"], 1) if st["nc"] else 0,
                             days=round(days, 1), trades_per_day=round(st["nc"] / days, 2) if days >= 1 else None,
                             open=st["pos"], closed=st["closed"][-30:][::-1], curve=st["curve"]))
    if not accounts:
        raise SystemExit("No data")
    json.dump(dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), accounts=accounts),
              open(os.path.join(R.OUT, "forward.json"), "w"))
    json.dump(state, open(sp, "w"))
    for x in accounts:
        print(x["title"], x["equity"], x["trades"])


if __name__ == "__main__":
    main()
