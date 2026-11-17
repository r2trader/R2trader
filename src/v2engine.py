"""R2trader engine v2: one portfolio simulator for long/short, leverage, margin, funding and liquidation.
Signals are read at a candle close; entries and signal exits fill at the NEXT candle open (optionally delayed further).
Stops and targets trade inside the candle (stop first when both are touched, gaps fill at the open).
Paper research only."""
import math
from datetime import datetime, timezone
import research as R

TF_MS = {"15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}
FUND_8H = 0.0001
MAINT = 0.03
LIQ_PEN = 0.005
DEFAULT = dict(start=1000.0, risk=0.01, maxpos=0.25, maxopen=3, lev=1.0, tp=None, maxbars=None, mode="risk", delay=0)


def mksig(n):
    return dict(sig=[0] * n, exl=[False] * n, exs=[False] * n, sd=[0.0] * n, alloc=[0.0] * n, score=[0.0] * n)


def sharpe(rets, per_year):
    if len(rets) < 30:
        return None
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
    return round(m / sd * math.sqrt(per_year), 2) if sd else None


def simulate(data, sigs, tf, cfg=None):
    c = dict(DEFAULT)
    c.update(cfg or {})
    fee, slip = R.FEE, R.SLIP
    hrs = TF_MS[tf] / 3600000
    times = sorted({x[0] for b in data.values() for x in b})
    ts = times[R.WARM:]
    idx = {s: {x[0]: i for i, x in enumerate(b)} for s, b in data.items()}
    st = dict(cash=c["start"])
    pos, pend_in, pend_out, last = {}, [], set(), {}
    closed, curve = [], []
    stats = dict(liqs=0, ni=0, nb=0)

    def upl(p, px):
        return p["q"] * (px - p["e"]) if p["side"] > 0 else p["q"] * (p["e"] - px)

    def equity():
        return st["cash"] + sum(upl(p, last.get(s, p["e"])) for s, p in pos.items())

    def close(s, raw, why, t_out):
        p = pos.pop(s)
        px = raw * (1 - slip) if p["side"] > 0 else raw * (1 + slip)
        gross = upl(p, px)
        fx = p["q"] * px * fee
        st["cash"] += gross - fx
        closed.append(dict(sym=s, side=p["side"], pnl=gross - fx - p["fee_in"] - p["fund"], notional=p["q"] * p["e"],
                           bars=p["n"], t_out=t_out, why=why, entry=p["e"], exit=px))

    def open_(q, x):
        s, side, sd = q["s"], q["side"], q["sd"]
        E = equity()
        e = x[1] * (1 + slip) if side > 0 else x[1] * (1 - slip)
        if E <= 0 or sd <= 0:
            return
        notional = min(E * c["risk"] * e / sd, E * c["maxpos"]) if c["mode"] == "risk" else E * q["alloc"]
        gross = sum(p["q"] * p["e"] for p in pos.values())
        notional = min(notional, c["lev"] * E - gross)
        if notional < 10:
            return
        fee_in = notional * fee
        st["cash"] -= fee_in
        pos[s] = dict(side=side, q=notional / e, e=e, stop=e - side * sd, n=0, fee_in=fee_in, fund=0.0, lev=notional / E,
                      tp=(e + side * c["tp"] * sd) if c["tp"] else None)

    for t in ts:
        tc = t + TF_MS[tf] - 1
        for s in list(pend_out):
            i = idx[s].get(t)
            if s not in pos:
                pend_out.discard(s)
            elif i is not None:
                close(s, data[s][i][1], "signal", data[s][i][6])
                pend_out.discard(s)
        ready, keep = [], []
        for q in pend_in:
            q["wait"] -= 1
            (ready if q["wait"] <= 0 else keep).append(q)
        pend_in = keep
        for q in sorted(ready, key=lambda z: -z["score"]):
            i = idx[q["s"]].get(t)
            if i is None or q["s"] in pos or len(pos) >= c["maxopen"]:
                continue
            open_(q, data[q["s"]][i])
        for s in list(pos):
            i = idx[s].get(t)
            if i is None:
                continue
            p, x = pos[s], data[s][i]
            p["n"] += 1
            share = 1.0 if p["side"] < 0 else max(0.0, 1 - 1 / p["lev"])
            if share > 0:
                fc = p["q"] * x[4] * FUND_8H * hrs / 8 * share
                st["cash"] -= fc
                p["fund"] += fc
            if p["side"] > 0:
                if x[3] <= p["stop"]:
                    close(s, min(p["stop"], x[1]), "stop", x[6])
                elif p["tp"] and x[2] >= p["tp"]:
                    close(s, p["tp"], "target", x[6])
            else:
                if x[2] >= p["stop"]:
                    close(s, max(p["stop"], x[1]), "stop", x[6])
                elif p["tp"] and x[3] <= p["tp"]:
                    close(s, p["tp"], "target", x[6])
        if pos:
            worst, gross, wpx = st["cash"], 0.0, {}
            for s, p in pos.items():
                i = idx[s].get(t)
                if i is None:
                    pw = pc = last.get(s, p["e"])
                else:
                    x = data[s][i]
                    pw, pc = (x[3] if p["side"] > 0 else x[2]), x[4]
                wpx[s] = pw
                worst += upl(p, pw)
                gross += p["q"] * pc
            if worst < MAINT * gross:
                stats["liqs"] += 1
                for s in list(pos):
                    close(s, wpx[s] * (1 - pos[s]["side"] * LIQ_PEN), "liquidated", tc)
                st["cash"] = max(st["cash"], 0.0)
        for s, b in data.items():
            i = idx[s].get(t)
            if i is not None:
                last[s] = b[i][4]
        eq = equity()
        stats["nb"] += 1
        stats["ni"] += 1 if pos else 0
        curve.append([tc, round(eq, 2)])
        waiting = {q["s"] for q in pend_in}
        for s, sg in sigs.items():
            i = idx[s].get(t)
            if i is None:
                continue
            if s in pos:
                p = pos[s]
                if (p["side"] > 0 and sg["exl"][i]) or (p["side"] < 0 and sg["exs"][i]) or (c["maxbars"] and p["n"] >= c["maxbars"]):
                    pend_out.add(s)
            elif sg["sig"][i] and s not in waiting and sg["sd"][i] > 0:
                pend_in.append(dict(s=s, side=sg["sig"][i], sd=sg["sd"][i], alloc=sg["alloc"][i], score=sg["score"][i], wait=1 + c["delay"]))
    return dict(curve=curve, closed=closed, stats=stats, final=curve[-1][1] if curve else c["start"])


def seg_stats(curve, closed, lo, hi, per_year):
    e = [x[1] for x in curve[lo:hi + 1]]
    t0, t1 = curve[lo][0], curve[hi][0]
    tr = [x for x in closed if t0 < x["t_out"] <= t1]
    peak, dd = e[0], 0.0
    for v in e:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak if peak > 0 else 0)
    month_last, order = {}, []
    for t, v in curve[lo:hi + 1]:
        d = datetime.fromtimestamp(t / 1000, timezone.utc)
        k = (d.year, d.month)
        if k not in month_last:
            order.append(k)
            month_last[k] = [t, t, v]
        month_last[k][1], month_last[k][2] = t, v
    mrets, prev = [], e[0]
    for k in order:
        first_t, last_t, v = month_last[k]
        if last_t - first_t >= 20 * 86400000 and prev > 0:
            mrets.append(v / prev - 1)
        prev = v
    nets = [(x["pnl"] / x["notional"]) for x in tr if x["notional"] > 0]
    ret = (e[-1] / e[0] - 1) * 100 if e[0] > 0 else -100.0
    avg_m = (math.exp(sum(math.log(max(1e-9, 1 + r)) for r in mrets) / len(mrets)) - 1) * 100 if mrets else None
    return dict(ret=round(ret, 2), sharpe=sharpe([e[i] / e[i - 1] - 1 for i in range(1, len(e)) if e[i - 1] > 0], per_year),
                max_dd=round(dd * 100, 1), trades=len(tr), win=round(100 * sum(1 for x in tr if x["pnl"] > 0) / len(tr), 1) if tr else None,
                exp=round(100 * sum(nets) / len(nets), 3) if nets else None,
                avg_month=round(avg_m, 2) if avg_m is not None else None, months=len(mrets),
                months_ge5=round(100 * sum(1 for r in mrets if r >= 0.05) / len(mrets), 0) if mrets else None)


def report(sim, data, tf):
    curve, closed = sim["curve"], sim["closed"]
    n = len(curve)
    c1, c2 = int(0.5 * n), int(0.75 * n)
    py = 8760 / (TF_MS[tf] / 3600000)
    times = sorted({x[0] for b in data.values() for x in b})
    ts = times[R.WARM:]
    kk = int(0.75 * len(ts))
    idx = {s: {x[0]: x[4] for x in b} for s, b in data.items()}
    r = [idx[s][ts[-1]] / idx[s][ts[kk]] - 1 for s in idx if ts[kk] in idx[s] and ts[-1] in idx[s]]
    rep = dict(train=seg_stats(curve, closed, 0, c1, py), val=seg_stats(curve, closed, c1, c2, py), test=seg_stats(curve, closed, c2, n - 1, py),
               full=seg_stats(curve, closed, 0, n - 1, py), expo=round(100 * sim["stats"]["ni"] / max(1, sim["stats"]["nb"]), 1),
               liqs=sim["stats"]["liqs"], bh_test=round(100 * sum(r) / len(r), 2) if r else None, days=round((ts[-1] - ts[0]) / 86400000))
    rep["_test_from"] = curve[c2][0]
    return rep
