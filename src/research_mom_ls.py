"""R2trader research: Long + Short Momentum with Dynamic Leverage (2x when risk > 50%, 5x when risk < 50%).
Capital: $1,000, $0 fixed overhead. Split: 50% train, 25% validation, 25% final test (read once).
Paper research only. Writes docs/data/research_mom_ls.json."""
import json, os, sys, time
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


def sim_ls(data, tf, lb_days, rb_days, stop_atr):
    bpd = 6 if tf == "4h" else 1
    lb, rb = lb_days * bpd, rb_days * bpd
    times = sorted({x[0] for b in data.values() for x in b})
    if len(times) < R.WARM + 200:
        return None
    idx = {s: {x[0]: i for i, x in enumerate(b)} for s, b in data.items()}
    ema100 = {s: R.ema_s([x[4] for x in b], 100) for s, b in data.items()}
    atr14 = {s: R.atr_s(b, 14) for s, b in data.items()}
    cash, pos, curve, closed, last = START_CAP, {}, [], [], {}
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
            x, p = data[s][i], pos[s]
            hit = (x[3] <= p["stop"]) if p["side"] == "long" else (x[2] >= p["stop"])
            if hit:
                px = (min(p["stop"], x[1]) * (1 - R.SLIP)) if p["side"] == "long" else (max(p["stop"], x[1]) * (1 + R.SLIP))
                pnl = (p["q"] * (px - p["e"]) if p["side"] == "long" else p["q"] * (p["e"] - px)) - p["q"] * (p["e"] + px) * R.FEE
                cash += p["margin"] + pnl
                closed.append(dict(sym=s, pnl=pnl, t_out=x[6]))
                del pos[s]
        if ti % rb == 0 and ti >= lb:
            longs, shorts, vols = [], [], []
            for s, b in data.items():
                i = idx[s].get(t)
                if i is None or i < lb or atr14[s][i] is None:
                    continue
                ret = b[i][4] / b[i - lb][4] - 1
                vol = atr14[s][i] / b[i][4]
                vols.append(vol)
                if b[i][4] > ema100[s][i] and ret > 0:
                    longs.append((ret, s, b[i][4], atr14[s][i]))
                elif b[i][4] < ema100[s][i] and ret < 0:
                    shorts.append((-ret, s, b[i][4], atr14[s][i]))
            longs.sort(reverse=True)
            shorts.sort(reverse=True)
            targets = {x[1]: ("long", x[2], x[3]) for x in longs[:2]}
            for x in shorts[:2]:
                targets[x[1]] = ("short", x[2], x[3])
            for s in list(pos):
                if s not in targets or pos[s]["side"] != targets[s][0]:
                    p, px = pos[s], last[s]
                    px = px * (1 - R.SLIP) if p["side"] == "long" else px * (1 + R.SLIP)
                    pnl = (p["q"] * (px - p["e"]) if p["side"] == "long" else p["q"] * (p["e"] - px)) - p["q"] * (p["e"] + px) * R.FEE
                    cash += p["margin"] + pnl
                    closed.append(dict(sym=s, pnl=pnl, t_out=t))
                    del pos[s]
            avg_vol = sum(vols) / len(vols) if vols else 0.05
            lev = 2.0 if avg_vol > 0.04 else 5.0  # 2x when risk > 50% regime, 5x when risk < 50%
            eq = cash + sum(p["margin"] + (p["q"] * (last[s] - p["e"]) if p["side"] == "long" else p["q"] * (p["e"] - last[s])) for s, p in pos.items())
            need = [(s, v) for s, v in targets.items() if s not in pos]
            if need and eq > 50:
                per_margin = min(cash * 0.90 / len(need), eq * 0.25)
                for s, (side, px, at) in need:
                    e = px * (1 + R.SLIP) if side == "long" else px * (1 - R.SLIP)
                    q = (per_margin * lev) / e
                    if q * e >= 20:
                        cash -= per_margin
                        pos[s] = dict(side=side, q=q, e=e, margin=per_margin, stop=(e - stop_atr * at) if side == "long" else (e + stop_atr * at))
        eq = max(0.0, cash + sum(p["margin"] + (p["q"] * (last[s] - p["e"]) if p["side"] == "long" else p["q"] * (p["e"] - last[s])) for s, p in pos.items()))
        curve.append([t, round(eq, 2)])
    n = len(curve)
    c1, c2 = int(0.5 * n), int(0.75 * n)
    per_year = 8760 / (L.TF_MS[tf] / 3600000)

    def seg(lo, hi):
        e = [c[1] for c in curve[lo:hi + 1]]
        tr = sum(1 for x in closed if curve[lo][0] < x["t_out"] <= curve[hi][0])
        ret = round((e[-1] / e[0] - 1) * 100, 2) if e[0] > 0 else -100.0
        return dict(ret=ret, sharpe=L.sharpe([e[i] / e[i - 1] - 1 for i in range(1, len(e)) if e[i - 1] > 0], per_year), trades=tr)

    return dict(train=seg(0, c1), val=seg(c1, c2), test=seg(c2, n - 1), days=round((ts[-1] - ts[0]) / 86400000))


def main():
    os.makedirs(R.OUT, exist_ok=True)
    rows = []
    for tf in ("4h", "1d"):
        data = {}
        for s in R.SYMBOLS:
            b = fetch_long(s, tf, BARS[tf])
            if len(b) > R.WARM + 200:
                data[s] = b
            time.sleep(0.2)
        if len(data) < 5:
            continue
        for lb in (14, 30):
            for rb in (7, 14):
                for stop in (2.5, 3.5):
                    r = sim_ls(data, tf, lb, rb, stop)
                    if r:
                        rows.append(dict(tf=tf, lb=lb, rb=rb, stop=stop, **r))
    json.dump(dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), capital=START_CAP, rows=rows),
              open(os.path.join(R.OUT, "research_mom_ls.json"), "w"), indent=1)
    print(f"Saved {len(rows)} rows")


if __name__ == "__main__":
    main()
