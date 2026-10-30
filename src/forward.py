"""R2trader forward test: five paper accounts run side by side on new candles only (no look-back trades).
Paper only. Writes docs/data/forward.json and forward_state.json."""
import json, os, time
from datetime import datetime, timezone
import research as R
import research_hf as H
import lab as L

ACC = {
    "A": dict(title="A: EMA 20/50 trend (4h)", tf="4h", build=R.build_ema_cross, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="long", rule="Buy when EMA20 crosses above EMA50 and price is above EMA200. Sell on the reverse cross or a 3x ATR stop."),
    "B": dict(title="B: Donchian 55 + BTC regime (4h)", tf="4h", build=R.build_donchian, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="long", rule="Buy a 55-bar high only when BTC is above its EMA200. Sell on a 20-bar low or a 3x ATR stop."),
    "C": dict(title="C: High-frequency RSI dip (1h)", tf="1h", build=H.rsi_dip(25), stop=1.5, tp=None, maxbars=12,
              side="long", rule="Experiment aimed at 3-4 trades/day: buy when RSI(7) < 25, sell at RSI > 50, a 1.5x ATR stop or after 12 hours."),
    "D": dict(title="D: EMA 20/50 SHORT (4h, experiment)", tf="4h", build=L.build_ema_cross_short, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="short", rule="UNPROVEN EXPERIMENT. Short when EMA20 crosses below EMA50 and price is below EMA200. Cover on the reverse cross "
                                 "or a 3x ATR stop. Assumes 0.01% funding per 8h; ignores margin, liquidation and futures pricing."),
    "E": dict(title="E: VWAP dip buy (4h, experiment)", tf="4h", build=L.build_vwap_dip, stop=2.0, tp=None, maxbars=24,
              side="long", rule="UNPROVEN EXPERIMENT. Buy, after the first candle of each UTC day, when price is above its EMA100 but more than 1x ATR "
                                "below the daily VWAP (resets 00:00 UTC). Sell when price returns to VWAP, on a 2x ATR stop, or after 24 candles."),
}


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
        st = state.setdefault(k, L.new_state())
        for p in st["pos"]:
            p.setdefault("side", "long")
            p.setdefault("fund", 0.0)
        data, ctx = cache[a["tf"]]
        L.advance(st, data, a, ctx)
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
