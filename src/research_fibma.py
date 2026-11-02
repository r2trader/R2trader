"""R2trader research: the "Fibonacci pullback + 50 MA" strategy from a social-media post, rebuilt and red-teamed.
Rules as posted: 4h trend from swing structure (higher high + higher low = up), 15m Fibonacci pullback to 0.5 (cancelled by a close beyond 0.786),
then a 15m close back through the 50 MA; stop 1%, target 2%; BTC and ETH. Sell is the mirror image.
Method: tune only on the first 50% (train), require a positive Sharpe on the next 25% (validation), read the last 25% (final test) once.
Red team: doubled costs, neighbouring settings, ablations (no Fibonacci, no trend), random entries inside the same trend, top-3 trades removed.
Paper research only. Writes docs/data/research_fibma.json."""
import json, os, random, sys, time
from datetime import datetime, timezone
import requests
import research as R
import lab as L

L.TF_MS.setdefault("15m", 900000)
SYMS = ["BTC", "ETH"]
BARS = 105000
H4 = 14400000
MA_N = 50
CANCEL = 0.786
SEEDS = 12
PER_YEAR = 8760 * 4


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
        time.sleep(0.1)
    return out[:-1]


def trend_gate(b, sg):
    """True where the 4h swing structure points the trade's way. Uses only 4h candles that are fully closed."""
    n = len(b)
    out = [False] * n
    H, Lo, phs, pls = [], [], [], []
    cur, state = None, 0
    for i, x in enumerate(b):
        s0 = x[0] // H4 * H4
        if cur is None or cur[0] != s0:
            cur = [s0, x[2], x[3]]
        else:
            cur[1], cur[2] = max(cur[1], x[2]), min(cur[2], x[3])
        if x[6] + 1 >= s0 + H4:
            H.append(cur[1])
            Lo.append(cur[2])
            c, cur = len(H) - 1, None
            if c >= 4:
                j = c - 2
                if H[j] > max(H[j - 2:j]) and H[j] >= max(H[j + 1:j + 3]):
                    phs.append(H[j])
                if Lo[j] < min(Lo[j - 2:j]) and Lo[j] <= min(Lo[j + 1:j + 3]):
                    pls.append(Lo[j])
            if len(phs) >= 2 and len(pls) >= 2:
                if phs[-1] > phs[-2] and pls[-1] > pls[-2]:
                    state = 1
                elif phs[-1] < phs[-2] and pls[-1] < pls[-2]:
                    state = -1
                else:
                    state = 0
        out[i] = state == sg
    return out


def signals(b, sg, m, touch, use_fib, gate):
    n = len(b)
    if sg > 0:
        hi, lo, cl = [x[2] for x in b], [x[3] for x in b], [x[4] for x in b]
    else:
        hi, lo, cl = [-x[3] for x in b], [-x[2] for x in b], [-x[4] for x in b]
    ma, run_sum = [None] * n, 0.0
    for i in range(n):
        run_sum += cl[i]
        if i >= MA_N:
            run_sum -= cl[i - MA_N]
        if i >= MA_N - 1:
            ma[i] = run_sum / MA_N
    en = [False] * n
    sw, last_low = None, None
    for i in range(max(MA_N, 2 * m + 1), n):
        cross = cl[i] > ma[i] and cl[i - 1] <= ma[i - 1]
        if not use_fib:
            en[i] = bool(cross and (gate is None or gate[i]))
            continue
        j = i - m
        if hi[j] > max(hi[j - m:j]) and hi[j] >= max(hi[j + 1:i + 1]) and last_low is not None and last_low < hi[j]:
            sw = [last_low, hi[j], False]
        if lo[j] < min(lo[j - m:j]) and lo[j] <= min(lo[j + 1:i + 1]):
            last_low = lo[j]
        if sw is None:
            continue
        top, rng = sw[1], sw[1] - sw[0]
        if hi[i] > top or cl[i] < top - CANCEL * rng:
            sw = None
            continue
        if lo[i] <= top - touch * rng:
            sw[2] = True
        if sw[2] and cross:
            en[i] = bool(gate is None or gate[i])
            sw = None
    return en


def build_fib(side, m, touch, stop_pct, use_fib=True, use_trend=True):
    sg = 1 if side == "long" else -1

    def build(b, ctx):
        gate = trend_gate(b, sg) if use_trend else None
        return signals(b, sg, m, touch, use_fib, gate), [False] * len(b), [stop_pct / 100 * x[4] for x in b]
    return build


def build_rand(side, p, seed, stop_pct):
    sg = 1 if side == "long" else -1

    def build(b, ctx):
        rng, gate = random.Random(seed * 7919 + b[0][0] % 10007), trend_gate(b, sg)
        return [bool(gate[i] and rng.random() < p) for i in range(len(b))], [False] * len(b), [stop_pct / 100 * x[4] for x in b]
    return build


def mk(cfg, **kw):
    c = dict(cfg, **kw)
    return dict(tf="15m", build=build_fib(c["side"], c["m"], c["touch"], c["stop"], c.get("fib", True), c.get("trend", True)),
                stop=1.0, tp=2.0, maxbars=R.NOEXIT, side=c["side"])


def net_pct(x):
    g = x["exit"] / x["entry"] - 1 if x["side"] == "long" else 1 - x["exit"] / x["entry"]
    return g - 2 * R.FEE


def run(data, a):
    times = sorted({x[0] for b in data.values() for x in b})
    if len(times) < R.WARM + 2000:
        return None
    st = L.new_state()
    t0 = times[R.WARM - 1]
    st["t"], st["start"], st["curve"] = t0, t0, [[t0 + L.TF_MS["15m"] - 1, R.START]]
    L.advance(st, data, a, {})
    eqs, cv = [c[1] for c in st["curve"]], st["curve"]
    n = len(eqs)
    c1, c2 = int(0.5 * n), int(0.75 * n)

    def seg(lo, hi):
        e = eqs[lo:hi + 1]
        tr = [x for x in st["closed"] if cv[lo][0] < x["t_out"] <= cv[hi][0]]
        nets = [net_pct(x) for x in tr]
        return dict(ret=round((e[-1] / e[0] - 1) * 100, 2), sharpe=L.sharpe([e[i] / e[i - 1] - 1 for i in range(1, len(e))], PER_YEAR),
                    trades=len(tr), win=round(100 * sum(1 for x in tr if x["pnl"] > 0) / len(tr), 1) if tr else None,
                    exp=round(100 * sum(nets) / len(nets), 3) if nets else None)

    idx = {s: {x[0]: x[4] for x in b} for s, b in data.items()}
    ts = times[R.WARM:]
    kk = int(0.75 * len(ts))
    r_t = [idx[s][ts[-1]] / idx[s][ts[kk]] - 1 for s in idx if ts[kk] in idx[s] and ts[-1] in idx[s]]
    res = dict(train=seg(0, c1), val=seg(c1, c2), test=seg(c2, n - 1), full=seg(0, n - 1),
               expo=round(100 * st["ni"] / max(1, st["nb"]), 1), bh_test=round(100 * sum(r_t) / len(r_t), 2),
               days=round((ts[-1] - ts[0]) / 86400000))
    res["_trades"] = [x for x in st["closed"] if cv[c2][0] < x["t_out"] <= cv[n - 1][0]]
    return res


def neighbours(cfg):
    return [dict(cfg, m=max(2, cfg["m"] // 2)), dict(cfg, m=cfg["m"] * 2), dict(cfg, touch=round(cfg["touch"] - 0.118, 3)),
            dict(cfg, touch=round(cfg["touch"] + 0.118, 3)), dict(cfg, stop=round(cfg["stop"] * 0.75, 2)),
            dict(cfg, stop=round(cfg["stop"] * 1.5, 2))]


def redteam(data, cfg):
    base = run(data, mk(cfg))
    f0, s0 = R.FEE, R.SLIP
    R.FEE, R.SLIP = f0 * 2, s0 * 2
    try:
        stress = run(data, mk(cfg))
    finally:
        R.FEE, R.SLIP = f0, s0
    nb = []
    for c in neighbours(cfg):
        r = run(data, mk(c))
        nb.append(dict(m=c["m"], touch=c["touch"], stop=c["stop"], test=r["test"]))
    abl = {}
    for name, kw in (("no_fibonacci", dict(fib=False)), ("no_trend_filter", dict(trend=False))):
        abl[name] = run(data, mk(cfg, **kw))["test"]
    sg = 1 if cfg["side"] == "long" else -1
    n_en = sum(sum(build_fib(cfg["side"], cfg["m"], cfg["touch"], cfg["stop"])(b, {})[0]) for b in data.values())
    n_gate = sum(sum(trend_gate(b, sg)) for b in data.values())
    p = n_en / max(1, n_gate)
    rnd = []
    for seed in range(SEEDS):
        a = dict(mk(cfg), build=build_rand(cfg["side"], p, seed, cfg["stop"]))
        rnd.append(run(data, a)["test"]["ret"])
    tr = base["_trades"]
    by_coin = {}
    for x in tr:
        by_coin[x["sym"]] = round(by_coin.get(x["sym"], 0) + x["pnl"], 2)
    pn = sorted((x["pnl"] for x in tr), reverse=True)
    out = dict(cost_x2_test=stress["test"], neighbours=nb, neighbours_positive=sum(1 for x in nb if (x["test"]["sharpe"] or 0) > 0),
               ablations=abl, random_in_trend_test_ret=rnd, random_median=round(sorted(rnd)[len(rnd) // 2], 2),
               beats_random_share=round(sum(1 for r in rnd if base["test"]["ret"] > r) / len(rnd), 2),
               pnl_by_coin_test=by_coin, pnl_without_top3_trades=round(sum(pn[3:]), 2))
    t = base["test"]
    out["survives"] = bool((t["sharpe"] or 0) > 0 and t["ret"] > 0 and t["trades"] >= 30 and stress["test"]["ret"] > 0
                           and out["neighbours_positive"] >= 4 and out["pnl_without_top3_trades"] > 0 and out["beats_random_share"] >= 0.8)
    return out


def main():
    os.makedirs(R.OUT, exist_ok=True)
    data = {}
    for s in SYMS:
        b = fetch_long(s, "15m", BARS)
        if len(b) > R.WARM + 5000:
            data[s] = b
    if len(data) < 2:
        print("No data")
        sys.exit(1)
    rows = []
    for side in ("long", "short"):
        for m in (4, 8):
            for touch in (0.5, 0.382):
                for stop in (1.0, 2.0):
                    cfg = dict(side=side, m=m, touch=touch, stop=stop)
                    r = run(data, mk(cfg))
                    if r:
                        r.pop("_trades")
                        rows.append(dict(cfg, as_posted=bool(touch == 0.5 and stop == 1.0), **r))
    selected = []
    for side in ("long", "short"):
        cand = [r for r in rows if r["side"] == side and r["train"]["trades"] >= 40 and r["train"]["sharpe"] is not None
                and (r["val"]["sharpe"] or 0) > 0]
        if not cand:
            selected.append(dict(side=side, result="no configuration passed train (40+ trades) and validation (Sharpe > 0)"))
            continue
        best = max(cand, key=lambda r: r["train"]["sharpe"])
        best["selected"] = True
        cfg = dict(side=side, m=best["m"], touch=best["touch"], stop=best["stop"])
        selected.append(dict(side=side, config=cfg, test=best["test"], redteam=redteam(data, cfg)))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), configs=len(rows), coins=list(data),
               note="Fibonacci pullback + 50 MA as posted (4h swing trend, 15m entry, 1% stop, 2% target), BTC and ETH, about 3 years of 15m candles. "
                    "m = 15m swing width in candles. as_posted rows use touch 0.5 and stop 1%. Tuned on the first 50%, validated on the next 25%, "
                    "final test read once. Stops are checked before targets inside a candle. Fees 0.1%/side, slippage 0.05%, next-open fills. "
                    "Position size is capped at 25% of equity, so account returns are smaller than the per-trade 'exp' figures suggest.",
               selected=selected, rows=rows)
    json.dump(out, open(os.path.join(R.OUT, "research_fibma.json"), "w"), indent=1)
    for sel in selected:
        print(json.dumps(sel))


if __name__ == "__main__":
    main()
