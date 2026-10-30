"""R2trader research on ONE engine (lab v2): every strategy tested so far, plus a parity check against the legacy simulator.
All results are in-sample for selection purposes: rules were chosen after seeing them. Paper research only."""
import json, os, sys, time
from datetime import datetime, timezone
import research as R
import research_hf as H
import lab as L

SPECS = []
for name, (build, stop, tp, mb) in R.STRATS.items():
    SPECS.append((name, dict(build=build, stop=stop, tp=tp, maxbars=mb, side="long"), ("1h", "4h", "1d")))
for name, (build, stop, tp, mb) in H.HF.items():
    SPECS.append((name, dict(build=build, stop=stop, tp=tp, maxbars=mb, side="long"), ("1h",)))
SPECS += [
    ("ema20_50_short", dict(build=L.build_ema_cross_short, stop=3.0, tp=None, maxbars=R.NOEXIT, side="short"), ("1h", "4h", "1d")),
    ("vwap_reclaim_long", dict(build=L.build_vwap_reclaim, stop=1.5, tp=None, maxbars=24, side="long"), ("1h", "4h")),
    ("vwap_dip_long", dict(build=L.build_vwap_dip, stop=2.0, tp=None, maxbars=24, side="long"), ("1h", "4h")),
    ("vwap_reject_short", dict(build=L.build_vwap_reject, stop=1.5, tp=None, maxbars=24, side="short"), ("1h", "4h")),
]


def main():
    os.makedirs(R.OUT, exist_ok=True)
    rows, parity, legacy = [], [], dict(R.STRATS)
    for tf in ("1h", "4h", "1d"):
        data = {}
        for s in R.SYMBOLS:
            b = R.fetch(s, tf)
            if b and len(b) > R.WARM + 100:
                data[s] = b
            time.sleep(0.2)
        if "BTC" not in data:
            continue
        e200 = R.ema_s([x[4] for x in data["BTC"]], 200)
        ctx = dict(regime={x[0]: x[4] > e200[i] for i, x in enumerate(data["BTC"])})
        for name, cfg, tfs in SPECS:
            if tf in tfs:
                r = L.backtest(data, dict(cfg, tf=tf), ctx)
                if r:
                    rows.append(dict(tf=tf, strategy=name, coins=len(data), **r))
        if tf == "4h":
            R.STRATS.clear()
            R.STRATS.update(legacy)
            for name, cfg, tfs in SPECS:
                if name in legacy:
                    old = R.sim(data, name, ctx)
                    new = L.backtest(data, dict(cfg, tf=tf, fill="close", cap=None), ctx)
                    if old and new:
                        parity.append(dict(strategy=name, legacy_ret=old["ret"], new_ret=new["ret"],
                                           legacy_trades=old["trades"], new_trades=new["trades"],
                                           match=bool(abs(old["ret"] - new["ret"]) < 0.05 and old["trades"] == new["trades"])))
    if not rows:
        print("No data")
        sys.exit(1)
    rows.sort(key=lambda r: (not r["passes"], -(r["sharpe"] if r["sharpe"] is not None else -9)))
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), variants=len(SPECS), runs=len(rows),
               note="Engine v2: signals fill at the next candle open, exposure capped at 60% of equity, fees 0.1%/side, "
                    "slippage 0.05%, shorts pay an assumed 0.01% funding per 8h. Several rules were chosen after seeing "
                    "the final 30% of data, so 'test period' figures are NOT clean out-of-sample. Only forward results count.",
               rows=rows, parity=parity)
    json.dump(out, open(os.path.join(R.OUT, "research_all.json"), "w"), indent=1)
    for r in rows:
        print(r)
    for p in parity:
        print(p)


if __name__ == "__main__":
    main()
