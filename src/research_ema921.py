"""R2trader research: EMA 9/21 cross with an EMA100 trend filter, optimised on train/validation and red-teamed on a final test.
Split: first 50% train, next 25% validation, last 25% final test (read once). Paper research only. Writes docs/data/research_ema921.json."""
import json, os, sys, time
from datetime import datetime, timezone
import requests
import research as R
import lab as L

BARS = {"1h": 5000, "4h": 5000, "1d": 3000}
NEIGH = [(8, 21, 100), (10, 21, 100), (9, 18, 100), (9, 24, 100), (9, 21, 80), (9, 21, 120)]


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


def adx_s(b, n=14):
    N = len(b)
    out, dx = [None] * N, [None] * N
    tr, pdm, mdm = [0.0] * N, [0.0] * N, [0.0] * N
    for i in range(1, N):
        up, dn = b[i][2] - b[i - 1][2], b[i - 1][3] - b[i][3]
        pdm[i] = up if up > dn and up > 0 else 0.0
        mdm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(b[i][2] - b[i][3], abs(b[i][2] - b[i - 1][4]), abs(b[i][3] - b[i - 1][4]))
    atr = sp = sm = 0.0
    for i in range(1, N):
        if i <= n:
            atr, sp, sm = atr + tr[i], sp + pdm[i], sm + mdm[i]
        else:
            atr, sp, sm = atr - atr / n + tr[i], sp - sp / n + pdm[i], sm - sm / n + mdm[i]
        if i >= n and atr:
            pdi, mdi = 100 * sp / atr, 100 * sm / atr
            dx[i] = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
    a, cnt = 0.0, 0
    for i in range(n, N):
        if dx[i] is None:
            continue
        cnt += 1
        a = a + dx[i] / n if cnt <= n else (a * (n - 1) + dx[i]) / n
        if cnt >= n:
            out[i] = a
    return out


def build_ema(side, exit_mode, adx_min, f=9, s=21, t=100):
    sg = 1 if side == "long" else -1

    def build(b, ctx):
        n, cl = len(b), [x[4] for x in b]
        ef, es, et, at = R.ema_s(cl, f), R.ema_s(cl, s), R.ema_s(cl, t), R.atr_s(b)
        ax = adx_s(b) if adx_min else None
        en = [False] * n
        for i in range(1, n):
            cross = sg * (ef[i] - es[i]) > 0 and sg * (ef[i - 1] - es[i - 1]) <= 0
            en[i] = bool(cross and sg * (cl[i] - et[i]) > 0 and (not adx_min or (ax[i] or 0) >= adx_min))
        if exit_mode == "cross":
            ex = [sg * (ef[i] - es[i]) < 0 for i in range(n)]
        else:
            ex = [sg * (cl[i] - es[i]) < 0 for i in range(n)]
        return en, ex, at
    return build


def mk(tf, cfg, f=9, s=21, t=100):
    return dict(tf=tf, build=build_ema(cfg["side"], cfg["exit"], cfg["adx"], f, s, t), stop=cfg["stop"], tp=None,
                maxbars=R.NOEXIT, side=cfg["side"])


def run(data, a):
    times = sorted({x[0] for b in data.values() for x in b})
    if len(times) < R.WARM + 600:
        return None
    st = L.new_state()
    t0 = times[R.WARM - 1]
    st["t"], st["start"], st["curve"] = t0, t0, [[t0 + L.TF_MS[a["tf"]] - 1, R.START]]
    L.advance(st, data, a, {})
    eqs = [c[1] for c in st["curve"]]
    n = len(eqs)
    c1, c2 = int(0.5 * n), int(0.75 * n)
    per_year = 8760 / (L.TF_MS[a["tf"]] / 3600000)

    def seg(lo, hi):
        e = eqs[lo:hi + 1]
        tr = sum(1 for x in st["closed"] if st["curve"][lo][0] < x["t_out"] <= st["curve"][hi][0])
        return dict(ret=round((e[-1] / e[0] - 1) * 100, 2), sharpe=L.sharpe([e[i] / e[i - 1] - 1 for i in range(1, len(e))], per_year),
                    trades=tr)

    idx = {s: {x[0]: x[4] for x in b} for s, b in data.items()}
    ts = times[R.WARM:]
    kk = int(0.75 * len(ts))
    r_t = [idx[s][ts[-1]] / idx[s][ts[kk]] - 1 for s in idx if ts[kk] in idx[s] and ts[-1] in idx[s]]
    res = dict(train=seg(0, c1), val=seg(c1, c2), test=seg(c2, n - 1), expo=round(100 * st["ni"] / max(1, st["nb"]), 1),
              bh_test=round(100 * sum(r_t) / len(r_t), 2), days=round((ts[-1] - ts[0]) / 86400000))
    lo, hi = st["curve"][c2][0], st["curve"][n - 1][0]
    res["_trades"] = [x for x in st["closed"] if lo < x["t_out"] <= hi]
    return res


def redteam(data, tf, cfg):
    base = run(data, mk(tf, cfg))
    f0, s0 = R.FEE, R.SLIP
    R.FEE, R.SLIP = f0 * 2, s0 * 2
    try:
        stress = run(data, mk(tf, cfg))
    finally:
        R.FEE, R.SLIP = f0, s0
    nb = []
    for f, s, t in NEIGH:
        r = run(data, mk(tf, cfg, f, s, t))
        nb.append(dict(params=[f, s, t], test=r["test"]))
    tr = base["_trades"]
    by_coin = {}
    for x in tr:
        by_coin[x["sym"]] = round(by_coin.get(x["sym"], 0) + x["pnl"], 2)
    pn = sorted((x["pnl"] for x in tr), reverse=True)
    top3 = round(sum(pn[3:]), 2)
    out = dict(cost_x2_test=stress["test"], neighbours=nb, neighbours_positive=sum(1 for x in nb if (x["test"]["sharpe"] or 0) > 0),
               pnl_by_coin_test=by_coin, coins_positive=sum(1 for v in by_coin.values() if v > 0), pnl_without_top3_trades=top3,
               exposure_matched_bh=round(base["bh_test"] * base["expo"] / 100, 2))
    t = base["test"]
    out["survives"] = bool((t["sharpe"] or 0) > 0 and t["ret"] > 0 and t["trades"] >= 20 and stress["test"]["ret"] > 0
                           and out["neighbours_positive"] >= 4 and top3 > 0)
    return out


def main():
    os.makedirs(R.OUT, exist_ok=True)
    rows, store = [], {}
    for tf in ("1h", "4h", "1d"):
        data = {}
        for s in R.SYMBOLS:
            b = fetch_long(s, tf, BARS[tf])
            if len(b) > R.WARM + 600:
                data[s] = b
            time.sleep(0.2)
        if len(data) < 5:
            continue
        store[tf] = data
        for side in ("long", "short"):
            for exit_mode in ("cross", "ema21"):
                for stop in (2.0, 3.0):
                    for adx in (0, 20):
                        cfg = dict(side=side, exit=exit_mode, stop=stop, adx=adx)
                        r = run(data, mk(tf, cfg))
                        if r:
                            r.pop("_trades")
                            rows.append(dict(tf=tf, coins=len(data), **cfg, **r))
    if not rows:
        print("No data")
        sys.exit(1)
    selected = []
    for side in ("long", "short"):
        cand = [r for r in rows if r["side"] == side and r["train"]["trades"] >= 30 and (r["train"]["sharpe"] is not None)
                and (r["val"]["sharpe"] or 0) > 0]
        if not cand:
            selected.append(dict(side=side, result="no configuration passed train (30+ trades) and validation (Sharpe > 0)"))
            continue
        best = max(cand, key=lambda r: r["train"]["sharpe"])
        best["selected"] = True
        cfg = dict(side=side, exit=best["exit"], stop=best["stop"], adx=best["adx"])
        selected.append(dict(side=side, tf=best["tf"], config=cfg, test=best["test"], redteam=redteam(store[best["tf"]], best["tf"], cfg)))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), configs=len(rows),
               note="EMA 9/21 cross with EMA100 filter. Tuned only on the first 50% (train) and checked on the next 25% (validation); "
                    "the last 25% (test) is read once. Red-team: doubled costs, neighbouring EMA lengths, removal of the top 3 trades, "
                    "per-coin spread, exposure-matched buy-and-hold. Fees 0.1%/side, slippage 0.05%, next-open fills.",
               selected=selected, rows=rows)
    json.dump(out, open(os.path.join(R.OUT, "research_ema921.json"), "w"), indent=1)
    for sel in selected:
        print(json.dumps(sel))


if __name__ == "__main__":
    main()
