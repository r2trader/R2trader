"""R2trader forward test: paper accounts on new candles only.
Updated 2026-11-17 after red-team v2 (docs/data/research_v2.json):
- F (EMA 9/21 + EMA100 long, 1d) is the only strategy that survived all red-team checks (+15.47% test, 0.59 Sharpe, 10.5% max DD).
- A, B, D were positive on the 25% test set (+14.28%, +7.42%, +8.69%) but failed the top-3-trades concentration check; kept for monitoring.
- C (RSI dip 1h, -24.24%), E (VWAP dip 4h, -0.68%), and G (broken 2x/5x short leverage) are retired.
Paper only. Writes docs/data/forward.json and forward_state.json."""
import json, os, time
from datetime import datetime, timezone
import research as R
import research_ema921 as E
import lab as L

ACC = {
    "F": dict(title="F: EMA 9/21 + EMA100 trend (1d, BEST / red-team survivor)", tf="1d", build=E.build_ema("long", "cross", 0), stop=2.0, tp=None,
              maxbars=R.NOEXIT, side="long",
              rule="Best available strategy (only red-team v2 survivor). Buy on daily candles when EMA9 crosses above EMA21 and price is above EMA100. "
                   "Sell on the reverse cross or a 2x ATR stop. Red-team v2 test: +15.47% (+0.72%/mo), Sharpe 0.59, max DD 10.5%, 33 trades."),
    "A": dict(title="A: EMA 20/50 trend (4h, secondary)", tf="4h", build=R.build_ema_cross, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="long", rule="Buy when EMA20 crosses above EMA50 and price is above EMA200. Sell on the reverse cross or a 3x ATR stop. "
                                "Red-team v2 test: +14.28% (+2.43%/mo), Sharpe 1.20, but failed top-3-trades concentration check."),
    "B": dict(title="B: Donchian 55 + BTC regime (4h, secondary)", tf="4h", build=R.build_donchian, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="long", rule="Buy a 55-bar high only when BTC is above its EMA200. Sell on a 20-bar low or a 3x ATR stop. "
                                "Red-team v2 test: +7.42% (+1.20%/mo), Sharpe 0.76, but failed top-3-trades concentration check."),
    "D": dict(title="D: EMA 20/50 SHORT (4h, secondary)", tf="4h", build=L.build_ema_cross_short, stop=3.0, tp=None, maxbars=R.NOEXIT,
              side="short", rule="Short when EMA20 crosses below EMA50 and price is below EMA200. Cover on the reverse cross or a 3x ATR stop. "
                                 "Red-team v2 test: +8.69% (+0.28%/mo), Sharpe 0.99, but failed top-3-trades concentration check."),
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
