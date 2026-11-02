"""R2trader research: the "VWAP + EMA9 cross" intraday idea adapted to crypto.
Choose the configuration on the first 60% of history only, then report the untouched last 40% once.
Paper research only. Writes docs/data/research_vwap9.json."""
import json, os, sys, time
from datetime import datetime, timezone
import requests
import research as R
import lab as L

L.TF_MS.setdefault("15m", 900000)
BARS = 5000
SKIP_HOURS = 4
CHOP_WINDOW, CHOP_MAX = 20, 3


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


def build_vwap9(side, entry, trend, skip_bars):
    sg = 1 if side == "long" else -1

    def build(b, ctx):
        n, cl = len(b), [x[4] for x in b]
        vw, e9, e200, at = L.vwap_s(b), R.ema_s(cl, 9), R.ema_s(cl, 200), R.atr_s(b)
        day = [x[0] // 86400000 for x in b]
        first = [i == 0 or day[i] != day[i - 1] for i in range(n)]
        since, k = [], 0
        for i in range(n):
            k = 0 if first[i] else k + 1
            since.append(k)
        cross = [i > 0 and not first[i] and (e9[i] - vw[i]) * (e9[i - 1] - vw[i - 1]) < 0 for i in range(n)]
        cum = [0] * (n + 1)
        for i in range(n):
            cum[i + 1] = cum[i] + (1 if cross[i] else 0)
        en = [False] * n
        armed = 0
        for i in range(1, n):
            chop = cum[i + 1] - cum[max(0, i + 1 - CHOP_WINDOW)] >= CHOP_MAX
            ok = since[i] >= skip_bars and not chop and (not trend or sg * (cl[i] - e200[i]) > 0)
            up = sg * (cl[i] - vw[i]) > 0 and sg * (cl[i] - e9[i]) > 0
            fresh = sg * (e9[i] - vw[i]) > 0 and sg * (e9[i - 1] - vw[i - 1]) <= 0 and not first[i]
            if entry == "cross":
                en[i] = bool(fresh and up and ok)
            else:
                if fresh and ok:
                    armed = 6
                elif armed:
                    armed -= 1
                    if sg * (cl[i] - vw[i]) <= 0 or first[i]:
                        armed = 0
                    elif (sg > 0 and b[i][3] <= e9[i] or sg < 0 and b[i][2] >= e9[i]) and up and ok:
                        en[i] = True
                        armed = 0
        ex = [bool(first[i] or sg * (cl[i] - e9[i]) < 0) for i in range(n)]
        return en, ex, at
    return build


def run(data, a):
    times = sorted({x[0] for b in data.values() for x in b})
    if len(times) < R.WARM + 400:
        return None
    st = L.new_state()
    t0 = times[R.WARM - 1]
    st["t"], st["start"], st["curve"] = t0, t0, [[t0 + L.TF_MS[a["tf"]] - 1, R.START]]
    L.advance(st, data, a, {})
    eqs = [c[1] for c in st["curve"]]
    cut = int(0.6 * len(eqs))
    per_year = 8760 / (L.TF_MS[a["tf"]] / 3600000)

    def seg(lo, hi):
        e = eqs[lo:hi + 1]
        tr = sum(1 for x in st["closed"] if st["curve"][lo][0] < x["t_out"] <= st["curve"][hi][0])
        return dict(ret=round((e[-1] / e[0] - 1) * 100, 2), sharpe=L.sharpe([e[i] / e[i - 1] - 1 for i in range(1, len(e))], per_year),
                    trades=tr)

    idx = {s: {x[0]: x[4] for x in b} for s, b in data.items()}
    ts = times[R.WARM:]
    kk = int(0.6 * len(ts))
    r_out = [idx[s][ts[-1]] / idx[s][ts[kk]] - 1 for s in idx if ts[kk] in idx[s] and ts[-1] in idx[s]]
    peak, dd = eqs[0], 0.0
    for v in eqs:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    return dict(insample=seg(0, cut), holdout=seg(cut, len(eqs) - 1), bh_holdout=round(100 * sum(r_out) / len(r_out), 2),
                expo=round(100 * st["ni"] / max(1, st["nb"]), 1), max_dd=round(dd * 100, 2),
                days=round((ts[-1] - ts[0]) / 86400000))


def main():
    os.makedirs(R.OUT, exist_ok=True)
    rows = []
    for tf in ("15m", "1h"):
        data = {}
        for s in R.SYMBOLS:
            b = fetch_long(s, tf, BARS)
            if len(b) > R.WARM + 400:
                data[s] = b
            time.sleep(0.2)
        if len(data) < 5:
            continue
        skip = int(SKIP_HOURS * 3600000 / L.TF_MS[tf])
        for side in ("long", "short"):
            for entry in ("cross", "pullback"):
                for trend in (False, True):
                    a = dict(tf=tf, build=build_vwap9(side, entry, trend, skip), stop=1.5, tp=None, maxbars=R.NOEXIT, side=side)
                    r = run(data, a)
                    if r:
                        rows.append(dict(tf=tf, side=side, entry=entry, trend_filter=trend, coins=len(data), **r))
    if not rows:
        print("No data")
        sys.exit(1)
    for side in ("long", "short"):
        cand = [r for r in rows if r["side"] == side and r["insample"]["trades"] >= 30 and r["insample"]["sharpe"] is not None]
        if cand:
            best = max(cand, key=lambda r: r["insample"]["sharpe"])
            best["selected"] = True
            h = best["holdout"]
            best["verdict"] = ("candidate" if (h["sharpe"] or 0) > 0 and h["ret"] > 0 and h["trades"] >= 15
                               else "no edge: fails the untouched holdout")
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), configs=len(rows),
               note="VWAP (UTC-day anchor) + EMA9 cross adapted to crypto spot. Stop 1.5x ATR, exit on a close through EMA9 "
                    "or at the next UTC day start, skip the first 4h after the VWAP reset, skip chop (3+ crosses in 20 bars). "
                    "Chosen by in-sample Sharpe on the first 60% of history only; the last 40% is the holdout. Fees 0.1%/side, "
                    "slippage 0.05%, fills at the next candle open.", rows=rows)
    json.dump(out, open(os.path.join(R.OUT, "research_vwap9.json"), "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
