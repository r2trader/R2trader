"""R2trader main account: EMA 9/21 + EMA100 trend, long only, daily candles (promoted 2026-11-26 after red-team v2).
Replaces the 1h 20-bar volume breakout, which failed every red-team check (see docs/data/research_v2_main.json).
The old account is archived in docs/data/state_breakout_archive.json and dashboard_breakout_archive.json. Paper only."""
import json, os, shutil, sys, time
from datetime import datetime, timezone
import requests
import research as R
import research_ema921 as E
import lab as L

TF = "1d"
A = dict(tf=TF, build=E.build_ema("long", "cross", 0), stop=2.0, tp=None, maxbars=R.NOEXIT, side="long")
TAG = "F"


def llm_note(payload):
    k, u, m = os.getenv("LLM_API_KEY"), os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
    if not (k and u and m):
        return None
    try:
        r = requests.post(u.rstrip("/") + "/chat/completions", timeout=40, headers={"Authorization": "Bearer " + k},
                          json={"model": m, "max_tokens": 300, "messages": [
                              {"role": "system", "content": "Summarise this paper-trading simulation in 4 short sentences. "
                               "Use only the numbers given. Give no trading advice."},
                              {"role": "user", "content": json.dumps(payload)}]})
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def snapshot(data):
    rows = []
    for s, b in data.items():
        c = b[-1]
        prev = b[-21:-1]
        va = sum(x[5] for x in prev) / 20
        ratio = c[5] / va if va else 0
        trend = c[4] > R.ema_s([x[4] for x in b[-200:]], 100)[-1]
        brk = c[4] > max(x[2] for x in prev)
        chg = (c[4] / b[-2][4] - 1) * 100 if len(b) >= 2 else 0
        en = A["build"](b, {})[0]
        rows.append(dict(sym=s, price=c[4], chg24=round(chg, 2), vol_ratio=round(ratio, 2), breakout=brk, trend=trend,
                         fire=bool(en[-1])))
    return sorted(rows, key=lambda x: -x["vol_ratio"])


def archive(sp, dp):
    base = os.path.dirname(sp)
    for src, name in ((sp, "state_breakout_archive.json"), (dp, "dashboard_breakout_archive.json")):
        if os.path.exists(src) and not os.path.exists(os.path.join(base, name)):
            shutil.copy(src, os.path.join(base, name))


def main():
    os.makedirs(R.OUT, exist_ok=True)
    sp, dp = os.path.join(R.OUT, "state.json"), os.path.join(R.OUT, "dashboard.json")
    st = json.load(open(sp)) if os.path.exists(sp) else None
    if st is not None and st.get("strategy") != TAG:
        archive(sp, dp)
        st = None
    if st is None:
        st = L.new_state()
        st["strategy"] = TAG
    data, missing = {}, []
    for s in R.SYMBOLS:
        b = R.fetch(s, TF)
        if b and len(b) > R.WARM:
            data[s] = b
        else:
            missing.append(s)
        time.sleep(0.2)
    if not data:
        print("No market data could be fetched.")
        sys.exit(1)
    L.advance(st, data, A, {})
    eq = st["curve"][-1][1]
    cs = L.curve_stats(st, TF)
    halted = eq < st["day"]["eq"] * 0.97
    live = dict(equity=round(eq, 2), return_pct=round((eq / R.START - 1) * 100, 2), trades=st["nc"],
                win_rate=round(100 * st["nw"] / st["nc"], 1) if st["nc"] else 0, open_positions=len(st["pos"]),
                halted_today=halted, expo=cs["expo"], sharpe=cs["sharpe"], wr_ci=cs["wr_ci"])
    bt = L.backtest(data, A, {})
    btm = dict(trades=bt["trades"], win_rate=bt["win_rate"], return_pct=bt["ret"], max_drawdown_pct=bt["max_dd"],
               buy_hold_pct=bt["bh"], days=bt["days"], sharpe=bt["sharpe"], bh_sharpe=bt["bh_sharpe"]) if bt else None
    sigs = snapshot(data)
    fired = [x["sym"] for x in sigs if x["fire"]]
    note = "Main account is now strategy F (EMA 9/21 + EMA100, daily, long only): %s equity, %s%% since 2026-11-26, %d closed trades. Signals firing now: %s." % (
        live["equity"], live["return_pct"], live["trades"], ", ".join(fired) or "none")
    if btm:
        note += " Backtest over %s days: %s%% vs %s%% buy-and-hold, max drawdown %s%%." % (
            btm["days"], btm["return_pct"], btm["buy_hold_pct"], btm["max_drawdown_pct"])
    note = llm_note(dict(live=live, backtest=btm, firing=fired)) or note
    opn = [dict(sym=p["s"], entry=p["e"], stop=p["stop"], tp=p["tp"], qty=p["q"], n=p["n"]) for p in st["pos"]]
    dash = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), engine="v2", missing=missing,
                live=live, backtest=btm, signals=sigs, open=opn, closed=st["closed"][-50:][::-1],
                curve=st["curve"][-2000:], note=note)
    st["closed"], st["curve"] = st["closed"][-500:], st["curve"][-5000:]
    json.dump(dash, open(dp, "w"))
    json.dump(st, open(sp, "w"))
    print(note)


if __name__ == "__main__":
    main()
