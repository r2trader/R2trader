"""R2trader lab engine: long/short paper simulator with funding cost, VWAP rules, and a research run.
Paper research only. Writes docs/data/research_lab.json when run directly."""
import json, os, sys, time
from datetime import datetime, timezone
import research as R

TF_MS = {"1h": 3600000, "4h": 14400000, "1d": 86400000}
FUND_8H = 0.0001


def vwap_s(b):
    out, day, pv, vv = [], None, 0.0, 0.0
    for x in b:
        d = x[0] // 86400000
        if d != day:
            day, pv, vv = d, 0.0, 0.0
        pv += (x[2] + x[3] + x[4]) / 3 * x[5]
        vv += x[5]
        out.append(pv / vv if vv else x[4])
    return out


def new_day(b, i):
    return i > 0 and b[i][0] // 86400000 == b[i - 1][0] // 86400000


def build_ema_cross_short(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    f, s, t, at = R.ema_s(cl, 20), R.ema_s(cl, 50), R.ema_s(cl, 200), R.atr_s(b)
    en = [i > 0 and f[i] < s[i] and f[i - 1] >= s[i - 1] and cl[i] < t[i] for i in range(n)]
    return en, [f[i] > s[i] for i in range(n)], at


def build_vwap_reclaim(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    vw, e50, at = vwap_s(b), R.ema_s(cl, 50), R.atr_s(b)
    en, ex = [False] * n, [cl[i] < vw[i] for i in range(n)]
    for i in range(61, n):
        va = sum(x[5] for x in b[i - 20:i]) / 20
        en[i] = (new_day(b, i) and cl[i] > vw[i] and cl[i - 1] <= vw[i - 1] and cl[i] > e50[i] and b[i][5] >= va)
    return en, ex, at


def build_vwap_dip(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    vw, e100, at = vwap_s(b), R.ema_s(cl, 100), R.atr_s(b)
    en, ex = [False] * n, [cl[i] >= vw[i] for i in range(n)]
    for i in range(101, n):
        en[i] = bool(at[i] and new_day(b, i) and cl[i] > e100[i] and cl[i] < vw[i] - at[i])
    return en, ex, at


def build_vwap_reject(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    vw, e50, at = vwap_s(b), R.ema_s(cl, 50), R.atr_s(b)
    en, ex = [False] * n, [cl[i] > vw[i] for i in range(n)]
    for i in range(61, n):
        va = sum(x[5] for x in b[i - 20:i]) / 20
        en[i] = (new_day(b, i) and cl[i] < vw[i] and cl[i - 1] >= vw[i - 1] and cl[i] < e50[i] and b[i][5] >= va)
    return en, ex, at


def new_state():
    return dict(cash=R.START, pos=[], closed=[], curve=[], t=0, start=0, nc=0, nw=0)


def value(p, px):
    return p["q"] * px if p["side"] == "long" else p["q"] * (p["e"] - px)


def close_pos(st, p, px, why, t_out):
    if p["side"] == "long":
        px *= 1 - R.SLIP
        pr = p["q"] * px * (1 - R.FEE)
        st["cash"] += pr
        pnl = pr - p["q"] * p["e"] * (1 + R.FEE)
    else:
        px *= 1 + R.SLIP
        pr = p["q"] * (p["e"] - px) - p["q"] * px * R.FEE
        st["cash"] += pr
        pnl = pr - p["q"] * p["e"] * R.FEE - p["fund"]
    st["nc"], st["nw"] = st["nc"] + 1, st["nw"] + (pnl > 0)
    st["closed"].append(dict(sym=p["s"], side=p["side"], entry=p["e"], exit=px, pnl=pnl, why=why, t_out=t_out))
    st["pos"].remove(p)


def advance(st, data, a, ctx):
    """Process every closed candle after st['t']. Works for a live forward test and for a full backtest."""
    side, hrs = a.get("side", "long"), TF_MS[a["tf"]] / 3600000
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
            if p["side"] == "short":
                fc = p["q"] * x[4] * FUND_8H * hrs / 8
                st["cash"] -= fc
                p["fund"] += fc
            px = why = None
            if p["side"] == "long":
                if x[3] <= p["stop"]:
                    px, why = min(p["stop"], x[1]), "stop"
                elif p["tp"] and x[2] >= p["tp"]:
                    px, why = p["tp"], "target"
            else:
                if x[2] >= p["stop"]:
                    px, why = max(p["stop"], x[1]), "stop"
                elif p["tp"] and x[3] <= p["tp"]:
                    px, why = p["tp"], "target"
            if px is None and sig[p["s"]][1][i]:
                px, why = x[4], "signal"
            if px is None and p["n"] >= a["maxbars"]:
                px, why = x[4], "time"
            if px is not None:
                close_pos(st, p, px, why, x[6])
        eq = st["cash"] + sum(value(p, last.get(p["s"], p["e"])) for p in st["pos"])
        for s in data:
            i = idx[s].get(t)
            if i is None or not sig[s][0][i] or len(st["pos"]) >= R.MAXOPEN or any(p["s"] == s for p in st["pos"]):
                continue
            at = sig[s][2][i]
            if not at:
                continue
            c0 = data[s][i][4]
            e = c0 * (1 + R.SLIP) if side == "long" else c0 * (1 - R.SLIP)
            sd = a["stop"] * at
            q = min(eq * R.RISK / sd, eq * R.MAXPOS / e)
            if side == "long":
                c = q * e * (1 + R.FEE)
                if c > st["cash"]:
                    q, c = st["cash"] / (e * (1 + R.FEE)), st["cash"]
            else:
                c = q * e * R.FEE
            if q * e < 10:
                continue
            st["cash"] -= c
            st["pos"].append(dict(s=s, side=side, e=e, q=q, fund=0.0,
                                  stop=(e - sd) if side == "long" else (e + sd),
                                  tp=((e + a["tp"] * at) if side == "long" else (e - a["tp"] * at)) if a["tp"] else None,
                                  n=0, t=data[s][i][6]))
        eq = st["cash"] + sum(value(p, last.get(p["s"], p["e"])) for p in st["pos"])
        st["curve"].append([t + TF_MS[a["tf"]] - 1, round(eq, 2)])
        st["t"] = t


def backtest(data, a, ctx):
    times = sorted({x[0] for b in data.values() for x in b})
    if len(times) < R.WARM + 100:
        return None
    st = new_state()
    t0 = times[R.WARM - 1]
    st["t"], st["start"], st["curve"] = t0, t0, [[t0 + TF_MS[a["tf"]] - 1, R.START]]
    advance(st, data, a, ctx)
    eqs = [c[1] for c in st["curve"]]
    k = int(0.7 * len(eqs))
    peak, dd = eqs[0], 0.0
    for v in eqs:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    idx = {s: {x[0]: x[4] for x in b} for s, b in data.items()}
    ts = times[R.WARM:]
    kk = int(0.7 * len(ts))

    def bh(i0, i1):
        r = [idx[s][ts[i1]] / idx[s][ts[i0]] - 1 for s in idx if ts[i0] in idx[s] and ts[i1] in idx[s]]
        return 100 * sum(r) / len(r) if r else 0

    days = (ts[-1] - ts[0]) / 86400000
    res = dict(side=a.get("side", "long"), trades=st["nc"], trades_per_day=round(st["nc"] / days, 2) if days else 0,
               win_rate=round(100 * st["nw"] / st["nc"], 1) if st["nc"] else 0,
               ret=round((eqs[-1] / R.START - 1) * 100, 2), ret_out=round((eqs[-1] / eqs[k] - 1) * 100, 2),
               bh=round(bh(0, len(ts) - 1), 2), bh_out=round(bh(kk, len(ts) - 1), 2),
               max_dd=round(dd * 100, 2), days=round(days))
    beat = res["side"] == "short" or res["ret_out"] > res["bh_out"]
    res["passes"] = bool(res["trades"] >= 30 and res["ret_out"] > 0 and beat and res["max_dd"] < 25)
    return res


VARIANTS = [
    ("ema20_50_short", dict(build=build_ema_cross_short, stop=3.0, tp=None, maxbars=R.NOEXIT, side="short"), ("1h", "4h", "1d")),
    ("vwap_reclaim_long", dict(build=build_vwap_reclaim, stop=1.5, tp=None, maxbars=24, side="long"), ("1h", "4h")),
    ("vwap_dip_long", dict(build=build_vwap_dip, stop=2.0, tp=None, maxbars=24, side="long"), ("1h", "4h")),
    ("vwap_reject_short", dict(build=build_vwap_reject, stop=1.5, tp=None, maxbars=24, side="short"), ("1h", "4h")),
]


def main():
    os.makedirs(R.OUT, exist_ok=True)
    rows = []
    for tf in ("1h", "4h", "1d"):
        data = {}
        for s in R.SYMBOLS:
            b = R.fetch(s, tf)
            if b and len(b) > R.WARM + 100:
                data[s] = b
            time.sleep(0.2)
        if "BTC" not in data:
            continue
        for name, cfg, tfs in VARIANTS:
            if tf in tfs:
                r = backtest(data, dict(cfg, tf=tf), {})
                if r:
                    rows.append(dict(tf=tf, strategy=name, coins=len(data), **r))
    if not rows:
        print("No data")
        sys.exit(1)
    rows.sort(key=lambda r: (not r["passes"], -r["ret_out"]))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               note="Shorts and VWAP rules. Fees 0.1%/side, slippage 0.05%, shorts pay 0.01% funding per 8h (assumed). "
                    "Shorts ignore margin and liquidation, so real results would be worse. ret_out = last 30% of history. "
                    "9 variants tested, so the best may be luck.", rows=rows)
    json.dump(out, open(os.path.join(R.OUT, "research_lab.json"), "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
