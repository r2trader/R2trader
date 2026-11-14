"""R2trader research: Cross-Sectional Momentum + Volatility Sizing ($1,000 capital, $0 fixed overhead).
Ranks the 10-coin basket by lookback return, buys the top-K strongest coins above EMA100, and sizes by inverse volatility.
Split: 50% train, 25% validation, 25% final test (read once). Red-team: doubled costs, neighbouring params, top-3 trades removed.
Paper research only. Writes docs/data/research_momentum.json."""
import json, math, os, sys, time
from datetime import datetime, timezone
import requests
import research as R
import lab as L

START_CAP = 1000.0
BARS = {"4h": 5000, "1d": 3000}


def fetch_long(sym, tf, bars):
    out, end = [], None
    while len(out) < bars:
        params = {"symbol": sym + "USDT", "interval": tf, "limit": 1000}
        if end:
            params["endTime"] = end
        rows = None
        for h in R.HOSTS:
            try:
                r = requests.get(h + "/api/v3/klines", params=params, timeout=20)
                if r.ok:
                    rows = r.json()
                    break
            except Exception:
                pass
        if not rows:
            break
        batch = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[6])] for k in rows]
        out = batch + out
        end = batch[0][0] - 1
        if len(batch) < 1000:
            break
    return out[:-1]


def sim_mom(data, tf, lookback_days, rebal_days, top_k, stop_atr):
    bpd = 6 if tf == "4h" else 1
    lb = lookback_days * bpd
    rb = rebal_days * bpd
    times = sorted({x[0] for b in data.values() for x in b})
    if len(times) < R.WARM + 200:
        return None
    idx = {s: {x[0]: i for i, x in enumerate(b)} for s, b in data.items()}
    ema100 = {s: R.ema_s([x[4] for x in b], 100) for s, b in data.items()}
    atr14 = {s: R.atr_s(b, 14) for s, b in data.items()}
    cash = START_CAP
    pos = {}
    curve = []
    closed = []
    last = {}
    ni = nb = 0
    ts = times[R.WARM:]
    for ti, t in enumerate(ts):
        for s, b in data.items():
            i = idx[s].get(t)
            if i is not None:
                last[s] = b[i][4]
        for s in list(pos):
            i = idx[s].get(t)
            if i is None:
                continue
            x = data[s][i]
            if x[3] <= pos[s]["stop"]:
                px = min(pos[s]["stop"], x[1]) * (1 - R.SLIP)
                pr = pos[s]["q"] * px * (1 - R.FEE)
                cash += pr
                pnl = pr - pos[s]["q"] * pos[s]["e"] * (1 + R.FEE)
                closed.append(dict(sym=s, pnl=pnl, t_out=x[6]))
                del pos[s]
        if ti % rb == 0 and ti >= lb:
            scores = []
            for s, b in data.items():
                i = idx[s].get(t)
                if i is None or i < lb or atr14[s][i] is None:
                    continue
                ret = b[i][4] / b[i - lb][4] - 1
                if b[i][4] > ema100[s][i] and ret > 0:
                    vol = atr14[s][i] / b[i][4]
                    scores.append((ret, s, b[i][4], atr14[s][i], vol))
            scores.sort(reverse=True)
            target_syms = {x[1] for x in scores[:top_k]}
            for s in list(pos):
                if s not in target_syms:
                    px = last[s] * (1 - R.SLIP)
                    pr = pos[s]["q"] * px * (1 - R.FEE)
                    cash += pr
                    pnl = pr - pos[s]["q"] * pos[s]["e"] * (1 + R.FEE)
                    closed.append(dict(sym=s, pnl=pnl, t_out=t))
                    del pos[s]
            eq = cash + sum(p["q"] * last[s] for s, p in pos.items())
            need = [x for x in scores[:top_k] if x[1] not in pos]
            if need:
                inv_vols = [1.0 / max(1e-4, x[4]) for x in need]
                tot_iv = sum(inv_vols)
                avail = cash * 0.98
                for (_, s, px, at, _), iv in zip(need, inv_vols):
                    alloc = min(avail * (iv / tot_iv), eq * 0.40)
                    e = px * (1 + R.SLIP)
                    q = alloc / (e * (1 + R.FEE))
                    if q * e >= 10:
                        cash -= q * e * (1 + R.FEE)
                        pos[s] = dict(q=q, e=e, stop=e - stop_atr * at)
        eq = cash + sum(p["q"] * last[s] for s, p in pos.items())
        nb += 1
        ni += 1 if pos else 0
        curve.append([t, round(eq, 2)])
    n = len(curve)
    c1, c2 = int(0.5 * n), int(0.75 * n)
    per_year = 8760 / (L.TF_MS[tf] / 3600000)

    def seg(lo, hi):
        e = [c[1] for c in curve[lo:hi + 1]]
        tr = sum(1 for x in closed if curve[lo][0] < x["t_out"] <= curve[hi][0])
        return dict(ret=round((e[-1] / e[0] - 1) * 100, 2),
                    sharpe=L.sharpe([e[i] / e[i - 1] - 1 for i in range(1, len(e))], per_year),
                    trades=tr)

    res = dict(train=seg(0, c1), val=seg(c1, c2), test=seg(c2, n - 1),
               expo=round(100 * ni / max(1, nb), 1),
               days=round((ts[-1] - ts[0]) / 86400000))
    res["_trades"] = [x for x in closed if curve[c2][0] < x["t_out"] <= curve[n - 1][0]]
    return res


def redteam(data, tf, cfg):
    base = sim_mom(data, tf, cfg["lb"], cfg["rb"], cfg["k"], cfg["stop"])
    f0, s0 = R.FEE, R.SLIP
    R.FEE, R.SLIP = f0 * 2, s0 * 2
    try:
        stress = sim_mom(data, tf, cfg["lb"], cfg["rb"], cfg["k"], cfg["stop"])
    finally:
        R.FEE, R.SLIP = f0, s0
    nb = []
    for dl, dr in ((-7, 0), (7, 0), (0, -3), (0, 7)):
        lb2, rb2 = max(7, cfg["lb"] + dl), max(3, cfg["rb"] + dr)
        r = sim_mom(data, tf, lb2, rb2, cfg["k"], cfg["stop"])
        nb.append(dict(lb=lb2, rb=rb2, test=r["test"]))
    tr = base["_trades"]
    pn = sorted((x["pnl"] for x in tr), reverse=True)
    top3 = round(sum(pn[3:]), 2)
    out = dict(cost_x2_test=stress["test"], neighbours=nb,
               neighbours_positive=sum(1 for x in nb if (x["test"]["sharpe"] or 0) > 0),
               pnl_without_top3_trades=top3)
    t = base["test"]
    out["survives"] = bool((t["sharpe"] or 0) > 0 and t["ret"] > 0 and t["trades"] >= 10
                           and stress["test"]["ret"] > 0 and out["neighbours_positive"] >= 3 and top3 > 0)
    return out


def main():
    os.makedirs(R.OUT, exist_ok=True)
    rows, store = [], {}
    for tf in ("4h", "1d"):
        data = {}
        for s in R.SYMBOLS:
            b = fetch_long(s, tf, BARS[tf])
            if len(b) > R.WARM + 200:
                data[s] = b
            time.sleep(0.2)
        if len(data) < 5:
            continue
        store[tf] = data
        for lb in (14, 30):
            for rb in (7, 14):
                for k in (2, 3):
                    for stop in (2.5, 3.5):
                        r = sim_mom(data, tf, lb, rb, k, stop)
                        if r:
                            r.pop("_trades")
                            rows.append(dict(tf=tf, lb=lb, rb=rb, k=k, stop=stop, **r))
    if not rows:
        print("No data")
        sys.exit(1)
    cand = [r for r in rows if r["train"]["trades"] >= 20 and (r["train"]["sharpe"] or 0) > 0 and (r["val"]["sharpe"] or 0) > 0]
    selected = None
    if cand:
        best = max(cand, key=lambda r: r["train"]["sharpe"])
        best["selected"] = True
        cfg = dict(lb=best["lb"], rb=best["rb"], k=best["k"], stop=best["stop"])
        selected = dict(tf=best["tf"], config=cfg, test=best["test"], redteam=redteam(store[best["tf"]], best["tf"], cfg))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               capital=START_CAP, fixed_overhead=0.0, configs=len(rows),
               selected=selected, rows=rows)
    json.dump(out, open(os.path.join(R.OUT, "research_momentum.json"), "w"), indent=1)
    print(json.dumps(selected, indent=1))


if __name__ == "__main__":
    main()
