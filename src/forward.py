"""R2trader forward test: seven paper accounts (engine v2) on new candles only. Rules for A-E are frozen as of 2026-10-30; F added 2026-11-02; G added 2026-11-14.
Paper only. Writes docs/data/forward.json and forward_state.json."""
import json, os, time
from datetime import datetime, timezone
import research as R
import research_hf as H
import research_ema921 as E
import lab as L

ACC = {
    "A": dict(title="A: EMA 20/50 trend (4h)", tf="4h", build=R.build_ema_cross, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="long", rule="Buy when EMA20 crosses above EMA50 and price is above EMA200. Sell on the reverse cross or a 3x ATR stop."),
    "B": dict(title="B: Donchian 55 + BTC regime (4h)", tf="4h", build=R.build_donchian, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="long", rule="Buy a 55-bar high only when BTC is above its EMA200. Sell on a 20-bar low or a 3x ATR stop."),
    "C": dict(title="C: High-frequency RSI dip (1h, experiment)", tf="1h", build=H.rsi_dip(25), stop=1.5, tp=None, maxbars=12,
              side="long", rule="Experiment aimed at 3-4 trades/day: buy when RSI(7) < 25, sell at RSI > 50, a 1.5x ATR stop or after 12 hours."),
    "D": dict(title="D: EMA 20/50 SHORT (4h, experiment)", tf="4h", build=L.build_ema_cross_short, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="short", rule="UNPROVEN EXPERIMENT. Short when EMA20 crosses below EMA50 and price is below EMA200. Cover on the reverse cross "
                                 "or a 3x ATR stop. Assumes 0.01% funding per 8h; ignores margin, liquidation and futures pricing."),
    "E": dict(title="E: VWAP dip buy (4h, experiment)", tf="4h", build=L.build_vwap_dip, stop=2.0, tp=None, maxbars=24,
              side="long", rule="UNPROVEN EXPERIMENT. Buy, after the first candle of each UTC day, when price is above its EMA100 but more than 1x ATR "
                                "below the daily VWAP (resets 00:00 UTC). Sell when price returns to VWAP, on a 2x ATR stop, or after 24 candles."),
    "F": dict(title="F: EMA 9/21 + EMA100 trend (1d, red-team survivor)", tf="1d", build=E.build_ema("long", "cross", 0), stop=2.0, tp=None,
              maxbars=R.NOEXIT, side="long",
              rule="Added 2026-11-02 after surviving the red-team run. Buy on daily candles when EMA9 crosses above EMA21 and price is above EMA100. "
                   "Sell on the reverse cross or a 2x ATR stop. Backtest final test: +7.1%, Sharpe 0.31, about one trade every three weeks."),
    "G": dict(title="G: EMA 9/21 SHORT + 2x/5x Lev (4h, experiment)", tf="4h", build=E.build_ema("short", "cross", 20), stop=2.0, tp=None,
              maxbars=R.NOEXIT, side="short", cap=2.0,
              rule="UNPROVEN EXPERIMENT ($1k capital model, 2x leverage in high risk / 5x in low risk). Short when EMA9 crosses below EMA21, "
                   "price is below EMA100, and ADX >= 20. Cover on reverse cross or 2x ATR stop."),
}


def main():
    os.makedirs(R.OUT, exist_ok=True)
    sp = os.path.join(R.OUT, "forward_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    cache, missing = {}, {}
    for tf in sorted({a["tf"] for a in ACC.values()}):
        data = {}
        for s in R.SYMBOLS:
            b = R.fetch(s, tf)
            if b and len(b) > R.WARM:
                data[s] = b
            time.sleep(0.2)
        missing[tf] = [s for s in R.SYMBOLS if s not in data]
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
        cs = L.curve_stats(st, a["tf"])
        eq = st["curve"][-1][1] if st["curve"] else R.START
        days = (st["curve"][-1][0] - st["start"]) / 86400000 if st["curve"] else 0
        accounts.append(dict(id=k, title=a["title"], rule=a["rule"], tf=a["tf"], equity=eq,
                             ret_pct=round((eq / R.START - 1) * 100, 2), trades=st["nc"],
                             win_rate=round(100 * st["nw"] / st["nc"], 1) if st["nc"] else 0, wr_ci=cs["wr_ci"],
                             expo=cs["expo"], sharpe=cs["sharpe"], missing=missing[a["tf"]],
                             days=round(days, 1), trades_per_day=round(st["nc"] / days, 2) if days >= 1 else None,
                             open=st["pos"], closed=st["closed"][-30:][::-1], curve=st["curve"]))
    if not accounts:
        raise SystemExit("No data")
    json.dump(dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), engine="v2", accounts=accounts),
              open(os.path.join(R.OUT, "forward.json"), "w"))
    json.dump(state, open(sp, "w"))
    for x in accounts:
        print(x["title"], x["equity"], x["trades"])


if __name__ == "__main__":
    main()
