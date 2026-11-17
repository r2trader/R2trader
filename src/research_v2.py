"""R2trader research v2: every strategy rebuilt on engine v2 and red-teamed the same way.
Per strategy: train/validation/test split (50/25/25, test read once), then on the frozen (as deployed) and the selected configuration:
doubled costs, one extra candle of delay, neighbouring parameters, best coin removed, top-3 trades removed, random-entry control with the same
exits and sizing, a future-data (causality) check and a first-half/second-half test. Also reports average monthly return and the share of
months at or above +5% against the 5%-a-month target. Paper research only. Writes docs/data/research_v2.json."""
import json, os, random, sys, time
from datetime import datetime, timezone
import research as R
import research_fibma as FB
import v2engine as E

BARS = {"15m": 105000, "1h": 5000, "4h": 5000, "1d": 3000}
SEEDS = 8
mk = E.mksig


def b_ema(side, f, s_, t, stop):
    sg = 1 if side == "long" else -1

    def build(data):
        out = {}
        for sym, b in data.items():
            n, cl = len(b), [x[4] for x in b]
            ef, es, et, at = R.ema_s(cl, f), R.ema_s(cl, s_), R.ema_s(cl, t), R.atr_s(b)
            m = mk(n)
            for i in range(1, n):
                m["exl"][i], m["exs"][i] = ef[i] < es[i], ef[i] > es[i]
                if at[i]:
                    m["sd"][i] = stop * at[i]
                    if sg * (ef[i] - es[i]) > 0 and sg * (ef[i - 1] - es[i - 1]) <= 0 and sg * (cl[i] - et[i]) > 0:
                        m["sig"][i] = sg
            out[sym] = m
        return out
    return build


def b_donchian(n_in, n_out, stop):
    def build(data):
        bc = [x[4] for x in data["BTC"]]
        e200 = R.ema_s(bc, 200)
        reg = {x[0]: x[4] > e200[i] for i, x in enumerate(data["BTC"])}
        out = {}
        for sym, b in data.items():
            n = len(b)
            at = R.atr_s(b)
            m = mk(n)
            for i in range(max(n_in, n_out) + 1, n):
                if at[i]:
                    m["sd"][i] = stop * at[i]
                    if b[i][4] > max(x[2] for x in b[i - n_in:i]) and reg.get(b[i][0], False):
                        m["sig"][i] = 1
                m["exl"][i] = b[i][4] < min(x[3] for x in b[i - n_out:i])
            out[sym] = m
        return out
    return build


def b_rsi(thr, stop):
    def build(data):
        out = {}
        for sym, b in data.items():
            n = len(b)
            r, at = R.rsi_s([x[4] for x in b], 7), R.atr_s(b)
            m = mk(n)
            for i in range(n):
                if at[i] and r[i] is not None:
                    m["sd"][i] = stop * at[i]
                    if r[i] < thr:
                        m["sig"][i] = 1
                    m["exl"][i] = r[i] > 50
            out[sym] = m
        return out
    return build


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


def b_vwap(mult, stop):
    def build(data):
        out = {}
        for sym, b in data.items():
            n, cl = len(b), [x[4] for x in b]
            vw, e100, at = vwap_s(b), R.ema_s(cl, 100), R.atr_s(b)
            m = mk(n)
            for i in range(101, n):
                if at[i]:
                    m["sd"][i] = stop * at[i]
                    if b[i][0] // 86400000 == b[i - 1][0] // 86400000 and cl[i] > e100[i] and cl[i] < vw[i] - mult * at[i]:
                        m["sig"][i] = 1
                m["exl"][i] = cl[i] >= vw[i]
            out[sym] = m
        return out
    return build


def b_fib(side, mm, touch, stop_pct):
    sg = 1 if side == "long" else -1

    def build(data):
        out = {}
        for sym, b in data.items():
            n = len(b)
            en = FB.signals(b, sg, mm, touch, True, FB.trend_gate(b, sg))
            m = mk(n)
            for i in range(n):
                m["sd"][i] = stop_pct / 100 * b[i][4]
                m["sig"][i] = sg if en[i] else 0
            out[sym] = m
        return out
    return build


def b_mom(tf, lb_d, rb_d, k, stop, ls, levmode):
    bpd = 6 if tf == "4h" else 1
    lb, rb = lb_d * bpd, rb_d * bpd

    def build(data):
        syms = list(data)
        times = sorted({x[0] for b in data.values() for x in b})
        idx = {s: {x[0]: i for i, x in enumerate(b)} for s, b in data.items()}
        ema = {s: R.ema_s([x[4] for x in b], 100) for s, b in data.items()}
        atr = {s: R.atr_s(b) for s, b in data.items()}
        out = {s: mk(len(b)) for s, b in data.items()}
        volser = []
        for t in times:
            v = [atr[s][idx[s][t]] / data[s][idx[s][t]][4] for s in syms if t in idx[s] and atr[s][idx[s][t]]]
            volser.append(sum(v) / len(v) if v else None)
        for j, t in enumerate(times):
            if j < R.WARM or j % rb or j < lb:
                continue
            cl_, cs_ = [], []
            for s in syms:
                i = idx[s].get(t)
                if i is None or i < lb or not atr[s][i]:
                    continue
                c0, ret = data[s][i][4], data[s][i][4] / data[s][i - lb][4] - 1
                if c0 > ema[s][i] and ret > 0:
                    cl_.append((ret, s))
                elif ls and c0 < ema[s][i] and ret < 0:
                    cs_.append((-ret, s))
            cl_.sort(reverse=True)
            cs_.sort(reverse=True)
            SL, SS = {s for _, s in cl_[:k]}, {s for _, s in cs_[:k]}
            cur = volser[j]
            hist = [v for v in volser[max(0, j - 365 * bpd):j] if v]
            pct = sum(1 for v in hist if v <= cur) / len(hist) if (cur and len(hist) >= 30) else 1.0
            lev = (2.0 if pct > 0.5 else 5.0) if levmode == "dyn" else (2.0 if levmode == "2x" else 1.0)
            alloc = (0.95 if lev == 1.0 else lev) / (2 * k if ls else k)
            for s in syms:
                i = idx[s].get(t)
                if i is None:
                    continue
                m = out[s]
                m["exl"][i], m["exs"][i] = s not in SL, s not in SS
                if atr[s][i]:
                    m["sd"][i], m["alloc"][i] = stop * atr[s][i], alloc
                    if s in SL:
                        m["sig"][i] = 1
                    elif s in SS:
                        m["sig"][i] = -1
        return out
    return build


def nb(p, keys):
    out = []
    for k in keys:
        for f in (0.75, 1.33):
            q = dict(p)
            v = p[k] * f
            q[k] = int(round(v)) if isinstance(p[k], int) else round(v, 3)
            if q[k] != p[k]:
                out.append(q)
    return out


BASE = dict(risk=0.01, maxpos=0.25, maxopen=3)
STRATS = {
    "A": dict(title="A: EMA 20/50 trend long (4h)", tf="4h", cfg=dict(BASE), mk=lambda p: b_ema("long", p["f"], p["s"], 200, p["stop"]),
              frozen=dict(f=20, s=50, stop=3.0), grid=[dict(f=f, s=s, stop=st) for f, s in ((20, 50), (10, 30)) for st in (2.0, 3.0)], keys=["f", "s", "stop"]),
    "B": dict(title="B: Donchian 55 + BTC regime long (4h)", tf="4h", cfg=dict(BASE), mk=lambda p: b_donchian(p["n"], 20, p["stop"]),
              frozen=dict(n=55, stop=3.0), grid=[dict(n=n, stop=st) for n in (40, 55) for st in (2.0, 3.0)], keys=["n", "stop"]),
    "C": dict(title="C: RSI(7) dip long (1h)", tf="1h", cfg=dict(BASE, maxbars=12), mk=lambda p: b_rsi(p["thr"], p["stop"]),
              frozen=dict(thr=25, stop=1.5), grid=[dict(thr=t, stop=1.5) for t in (20, 25, 30)], keys=["thr", "stop"]),
    "D": dict(title="D: EMA 20/50 trend short (4h)", tf="4h", cfg=dict(BASE), mk=lambda p: b_ema("short", p["f"], p["s"], 200, p["stop"]),
              frozen=dict(f=20, s=50, stop=3.0), grid=[dict(f=f, s=s, stop=st) for f, s in ((20, 50), (10, 30)) for st in (2.0, 3.0)], keys=["f", "s", "stop"]),
    "E": dict(title="E: VWAP dip long (4h)", tf="4h", cfg=dict(BASE, maxbars=24), mk=lambda p: b_vwap(p["mult"], p["stop"]),
              frozen=dict(mult=1.0, stop=2.0), grid=[dict(mult=m, stop=2.0) for m in (0.5, 1.0, 1.5)], keys=["mult", "stop"]),
    "F": dict(title="F: EMA 9/21 + EMA100 long (1d)", tf="1d", cfg=dict(BASE), mk=lambda p: b_ema("long", 9, 21, 100, p["stop"]),
              frozen=dict(stop=2.0), grid=[dict(stop=s) for s in (2.0, 3.0)], keys=["stop"]),
    "G": dict(title="G: Fibonacci pullback + 50 MA long (15m, BTC+ETH)", tf="15m", syms=["BTC", "ETH"], cfg=dict(BASE, tp=2.0),
              mk=lambda p: b_fib("long", p["m"], p["touch"], p["stop"]), frozen=dict(m=4, touch=0.5, stop=1.0),
              grid=[dict(m=m, touch=0.5, stop=s) for m in (4, 8) for s in (1.0, 2.0)], keys=["m", "stop"]),
    "H": dict(title="H: Fibonacci pullback + 50 MA short (15m, BTC+ETH)", tf="15m", syms=["BTC", "ETH"], cfg=dict(BASE, tp=2.0),
              mk=lambda p: b_fib("short", p["m"], p["touch"], p["stop"]), frozen=dict(m=4, touch=0.5, stop=1.0),
              grid=[dict(m=m, touch=0.5, stop=s) for m in (4, 8) for s in (1.0, 2.0)], keys=["m", "stop"]),
    "I": dict(title="I: Momentum long-only (1d)", tf="1d", cfg=dict(mode="alloc", maxopen=3, lev=1.0),
              mk=lambda p: b_mom("1d", p["lb"], p["rb"], 3, p["stop"], False, "1x"), frozen=dict(lb=30, rb=7, stop=3.5),
              grid=[dict(lb=lb, rb=rb, stop=3.5) for lb in (14, 30) for rb in (7, 14)], keys=["lb", "rb", "stop"]),
    "J": dict(title="J: Momentum long/short, 1x control (1d)", tf="1d", cfg=dict(mode="alloc", maxopen=4, lev=1.0),
              mk=lambda p: b_mom("1d", p["lb"], p["rb"], 2, p["stop"], True, "1x"), frozen=dict(lb=30, rb=7, stop=3.5),
              grid=[dict(lb=lb, rb=7, stop=3.5) for lb in (14, 30)], keys=["lb", "rb", "stop"]),
    "K": dict(title="K: Momentum long/short, dynamic 2x/5x leverage (1d)", tf="1d", cfg=dict(mode="alloc", maxopen=4, lev=5.0),
              mk=lambda p: b_mom("1d", p["lb"], p["rb"], 2, p["stop"], True, "dyn"), frozen=dict(lb=30, rb=7, stop=3.5),
              grid=[dict(lb=lb, rb=7, stop=3.5) for lb in (14, 30)], keys=["lb", "rb", "stop"]),
}


def run(S, data, p, cfg_over=None, delay=0, sigs=None):
    sg = sigs if sigs is not None else S["mk"](p)(data)
    cfg = dict(S["cfg"], **(cfg_over or {}))
    cfg["delay"] = delay
    sim = E.simulate(data, sg, S["tf"], cfg)
    return E.report(sim, data, S["tf"]), sim, sg


def causal(S, data, p):
    full = S["mk"](p)(data)
    times = sorted({x[0] for b in data.values() for x in b})
    tcut = times[int(0.7 * len(times))]
    cut = {s: [x for x in b if x[0] <= tcut] for s, b in data.items()}
    part = S["mk"](p)(cut)
    bad = 0
    for s in part:
        n = len(cut[s]) - 3
        for key in ("sig", "exl", "exs"):
            bad += sum(1 for a, b in zip(full[s][key][:n], part[s][key][:n]) if bool(a) != bool(b))
    return bad


def redteam(S, data, p):
    base, sim, sg = run(S, data, p)
    t0 = base["_test_from"]
    f0, s0 = R.FEE, R.SLIP
    R.FEE, R.SLIP = f0 * 2, s0 * 2
    try:
        cost2 = run(S, data, p)[0]["test"]
    finally:
        R.FEE, R.SLIP = f0, s0
    delay1 = run(S, data, p, delay=1)[0]["test"]
    nbs = [dict(params=q, test=run(S, data, q)[0]["test"]) for q in nb(p, S["keys"])]
    tr = [x for x in sim["closed"] if x["t_out"] > t0]
    by = {}
    for x in tr:
        by[x["sym"]] = by.get(x["sym"], 0) + x["pnl"]
    drop = None
    if len(data) >= 2 and by:
        best = max(by, key=by.get)
        drop = dict(coin=best, test=run(S, {s: b for s, b in data.items() if s != best}, p)[0]["test"])
    pn = sorted((x["pnl"] for x in tr), reverse=True)
    top3 = round(sum(pn[3:]) / S["cfg"].get("start", 1000.0) * 100, 2)
    hold = max(1, sorted(x["bars"] for x in sim["closed"])[len(sim["closed"]) // 2]) if sim["closed"] else 6
    tot = sum(len(b) for b in data.values())
    pr = len(sim["closed"]) / max(1, tot)
    sides = sorted({x["side"] for x in sim["closed"]}) or [1]
    amax = max((max(m["alloc"]) for m in sg.values()), default=0.0)
    rnd = []
    for seed in range(SEEDS):
        rg = {}
        for s, m in sg.items():
            r = random.Random(seed * 7919 + len(m["sig"]) + sum(map(ord, s)))
            q = mk(len(m["sig"]))
            q["sd"] = list(m["sd"])
            q["alloc"] = [amax if v else 0.0 for v in m["sd"]]
            q["sig"] = [r.choice(sides) if (m["sd"][i] > 0 and r.random() < pr) else 0 for i in range(len(m["sig"]))]
            rg[s] = q
        rnd.append(run(S, data, p, cfg_over=dict(maxbars=hold), sigs=rg)[0]["test"]["ret"])
    bad = causal(S, data, p)
    t = base["test"]
    npos = sum(1 for x in nbs if (x["test"]["sharpe"] or 0) > 0)
    beats = round(sum(1 for r in rnd if t["ret"] > r) / len(rnd), 2)
    fails = []
    if not ((t["sharpe"] or 0) > 0 and t["ret"] > 0):
        fails.append("test not positive")
    if t["trades"] < 20:
        fails.append("fewer than 20 test trades")
    if cost2["ret"] <= 0:
        fails.append("loses at double costs")
    if delay1["ret"] <= 0:
        fails.append("loses with one extra candle of delay")
    if nbs and npos < 0.6 * len(nbs):
        fails.append("neighbouring settings mostly negative")
    if drop and drop["test"]["ret"] <= 0:
        fails.append("loses without its best coin")
    if top3 <= 0:
        fails.append("loses without its top 3 trades")
    if beats < 0.75:
        fails.append("does not clearly beat random entries")
    if bad:
        fails.append("uses future data (%d mismatches)" % bad)
    if base["liqs"]:
        fails.append("liquidated %d times" % base["liqs"])
    out = dict(params=p, base={k: v for k, v in base.items() if not k.startswith("_")}, cost_x2=cost2, delay_1=delay1, neighbours=nbs, neighbours_positive=npos,
               drop_best_coin=drop, pnl_without_top3_pct=top3, random_test_ret=rnd, beats_random=beats, causality_mismatches=bad, fails=fails,
               survives=not fails, meets_5pct_month=bool(not fails and (t["avg_month"] or 0) >= 5.0))
    return out


def load(cache, S):
    key = (S["tf"], tuple(S.get("syms", R.SYMBOLS)))
    if key not in cache:
        d = {}
        for s in S.get("syms", R.SYMBOLS):
            b = FB.fetch_long(s, S["tf"], BARS[S["tf"]])
            if len(b) > R.WARM + 400:
                d[s] = b
        cache[key] = d
    return cache[key]


def main():
    os.makedirs(R.OUT, exist_ok=True)
    cache, results, board = {}, {}, []
    for k, S in STRATS.items():
        try:
            data = load(cache, S)
            if len(data) < min(2, len(S.get("syms", R.SYMBOLS))) or (k not in ("G", "H") and len(data) < 5):
                results[k] = dict(title=S["title"], error="not enough data")
                continue
            rows = []
            for p in S["grid"]:
                rep = run(S, data, p)[0]
                rows.append(dict(params=p, **{a: b for a, b in rep.items() if not a.startswith("_")}))
            cand = [r for r in rows if r["train"]["trades"] >= 15 and (r["train"]["sharpe"] or 0) > 0 and (r["val"]["sharpe"] or 0) > 0]
            sel = max(cand, key=lambda r: r["train"]["sharpe"])["params"] if cand else None
            res = dict(title=S["title"], tf=S["tf"], coins=len(data), grid=rows, selected_params=sel, frozen=redteam(S, data, S["frozen"]))
            if sel and sel != S["frozen"]:
                res["selected"] = redteam(S, data, sel)
            results[k] = res
            for which in ("frozen", "selected"):
                if which in res:
                    r = res[which]
                    t = r["base"]["test"]
                    board.append(dict(id=k, which=which, params=r["params"], test_ret=t["ret"], sharpe=t["sharpe"], avg_month=t["avg_month"], months_ge5=t["months_ge5"],
                                      max_dd=t["max_dd"], trades=t["trades"], win=t["win"], bh_test=r["base"]["bh_test"], survives=r["survives"],
                                      meets_5pct_month=r["meets_5pct_month"], fails=r["fails"]))
        except Exception as ex:
            results[k] = dict(title=S["title"], error="%s: %s" % (type(ex).__name__, ex))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), engine="v2",
               note="Engine v2: entries and signal exits fill at the next open, stops are checked before targets, funding on shorts and on levered longs, "
                    "liquidation when worst-case equity falls below 3% of gross exposure. Fees 0.1%/side, slippage 0.05%, start 1,000. Coins are today's survivors, "
                    "so long results are flattered. Test = last 25%, read once.", scoreboard=board, strategies=results)
    json.dump(out, open(os.path.join(R.OUT, "research_v2.json"), "w"), indent=1)
    for b in board:
        print(b["id"], b["which"], b["test_ret"], b["sharpe"], b["avg_month"], "SURVIVES" if b["survives"] else "; ".join(b["fails"]))


if __name__ == "__main__":
    main()
