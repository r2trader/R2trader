"""R2trader high-frequency experiment: candidates aimed at 3-4 trades/day, tested honestly on 1h candles.
Paper research only. Writes docs/data/research_hf.json."""
import json, os, sys, time
from datetime import datetime, timezone
import research as R


def rsi_dip(th):
    def build(b, ctx):
        cl, n = [x[4] for x in b], len(b)
        r, at = R.rsi_s(cl, 7), R.atr_s(b)
        en = [bool(r[i] is not None and r[i] < th) for i in range(n)]
        ex = [bool(r[i] is not None and r[i] > 50) for i in range(n)]
        return en, ex, at
    return build


def fast_breakout(lb):
    def build(b, ctx):
        cl, n = [x[4] for x in b], len(b)
        e20, e10, at = R.ema_s(cl, 20), R.ema_s(cl, 10), R.atr_s(b)
        en, ex = [False] * n, [bool(cl[i] < e10[i]) for i in range(n)]
        for i in range(lb + 1, n):
            en[i] = cl[i] > max(x[2] for x in b[i - lb:i]) and cl[i] > e20[i]
        return en, ex, at
    return build


def boll_any(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    at = R.atr_s(b)
    en, ex = [False] * n, [False] * n
    for i in range(20, n):
        w = cl[i - 19:i + 1]
        m = sum(w) / 20
        sd = (sum((x - m) ** 2 for x in w) / 20) ** 0.5
        en[i] = cl[i] < m - 2 * sd
        ex[i] = cl[i] > m
    return en, ex, at


def vol_thrust(b, ctx):
    cl, n = [x[4] for x in b], len(b)
    e20, e10, at = R.ema_s(cl, 20), R.ema_s(cl, 10), R.atr_s(b)
    en, ex = [False] * n, [bool(cl[i] < e10[i]) for i in range(n)]
    for i in range(21, n):
        va = sum(x[5] for x in b[i - 20:i]) / 20
        en[i] = cl[i] > b[i][1] and va > 0 and b[i][5] >= 1.5 * va and cl[i] > e20[i]
    return en, ex, at


HF = {
    "rsi7_dip<25": (rsi_dip(25), 1.5, None, 12),
    "rsi7_dip<30": (rsi_dip(30), 1.5, None, 12),
    "rsi7_dip<35": (rsi_dip(35), 1.5, None, 12),
    "breakout10_ema20": (fast_breakout(10), 1.5, None, 24),
    "breakout20_ema20": (fast_breakout(20), 1.5, None, 24),
    "bollinger_dip_any_trend": (boll_any, 2.0, None, 12),
    "volume_thrust_1.5x": (vol_thrust, 1.5, None, 6),
}


def main():
    R.STRATS.clear()
    R.STRATS.update(HF)
    data = {}
    for s in R.SYMBOLS:
        b = R.fetch(s, "1h")
        if b and len(b) > R.WARM + 100:
            data[s] = b
        time.sleep(0.2)
    if "BTC" not in data:
        print("No data")
        sys.exit(1)
    rows = []
    for name in HF:
        r = R.sim(data, name, dict(regime={}))
        if r:
            rows.append(dict(tf="1h", strategy=name, coins=len(data), **r))
    rows.sort(key=lambda r: -r["ret"])
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               note="Goal: 3-4 trades/day. Fees 0.1%/side, slippage 0.05%. About 33 days of 1h data only, so results are very noisy. "
                    "7 variants tested.", rows=rows)
    os.makedirs(R.OUT, exist_ok=True)
    json.dump(out, open(os.path.join(R.OUT, "research_hf.json"), "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
