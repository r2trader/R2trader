"""Red-team v2 for the MAIN dashboard strategy (1h 20-bar breakout with volume, long only).
Same engine and same checks as research_v2.py. The 3% daily-loss halt of the live account is not modelled here.
Writes docs/data/research_v2_main.json. Paper research only."""
import json, os
from datetime import datetime, timezone
import research as R
import research_v2 as V


def b_breakout(stop):
    def build(data):
        out = {}
        for sym, b in data.items():
            en, _, at = R.build_breakout(b, {})
            m = V.mk(len(b))
            for i in range(len(b)):
                if at[i]:
                    m["sd"][i] = stop * at[i]
                    m["sig"][i] = 1 if en[i] else 0
            out[sym] = m
        return out
    return build


S = dict(title="M: Main account, 20-bar volume breakout long (1h)", tf="1h", cfg=dict(V.BASE, tp=1.5, maxbars=48),
         mk=lambda p: b_breakout(p["stop"]), frozen=dict(stop=2.0), grid=[dict(stop=s) for s in (1.5, 2.0, 3.0)], keys=["stop"])


def main():
    os.makedirs(R.OUT, exist_ok=True)
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), engine="v2",
               note="Main account strategy through the same red-team checks as A to K. Target is 3 ATR (1.5R) with a 2 ATR stop and a 48-bar limit. Daily-loss halt not modelled.")
    try:
        data = V.load({}, S)
        if len(data) < 5:
            out["result"] = dict(title=S["title"], error="not enough data")
        else:
            rows = []
            for p in S["grid"]:
                rep = V.run(S, data, p)[0]
                rows.append(dict(params=p, **{a: b for a, b in rep.items() if not a.startswith("_")}))
            out["result"] = dict(title=S["title"], coins=len(data), grid=rows, frozen=V.redteam(S, data, S["frozen"]))
    except Exception as ex:
        out["result"] = dict(title=S["title"], error="%s: %s" % (type(ex).__name__, ex))
    json.dump(out, open(os.path.join(R.OUT, "research_v2_main.json"), "w"), indent=1)
    r = out["result"]
    print(r.get("error") or (r["frozen"]["base"]["test"], r["frozen"]["fails"]))


if __name__ == "__main__":
    main()
