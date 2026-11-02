"""R2trader main account: 1h 20-bar breakout with volume confirmation, long only. Engine v2 (next-open fills). Paper only."""
import json, os, sys, time
from datetime import datetime, timezone
import requests
import research as R
import lab as L

A = dict(tf="1h", build=R.build_breakout, stop=2.0, tp=3.0, maxbars=48, side="long", daily_loss=0.03)


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
        trend = c[4] > R.ema_s([x[4] for x in b[-120:]], 50)[-1]
        brk = c[4] > max(x[2] for x in prev)
        chg = (c[4] / b[-25][4] - 1) * 100 if len(b) >= 25 else 0
        rows.append(dict(sym=s, price=c[4], chg24=round(chg, 2), vol_ratio=round(ratio, 2), breakout=brk, trend=trend,
                         fire=bool(brk and trend and ratio >= 2)))
    return sorted(rows, key=lambda x: -x["vol_ratio"])


def migrate(old):
    """Convert the first-generation state file (keys open/closed) into engine v2 format, keeping cash, trades and curve."""
    st = L.new_state()
    st["cash"] = old.get("cash", R.START)
    st["closed"] = [dict(sym=c["sym"], side="long", entry=c["entry"], exit=c["exit"], pnl=c["pnl"], why=c["why"],
                         t_out=c["t_out"]) for c in old.get("closed", [])]
    st["nc"] = len(st["closed"])
    st["nw"] = sum(1 for c in st["closed"] if c["pnl"] > 0)
    st["pos"] = [dict(s=p["sym"], side="long", e=p["entry"], q=p["qty"], fund=0.0, stop=p["stop"], tp=p.get("tp"),
                      n=p.get("n", 0), t=p.get("t", 0)) for p in old.get("open", [])]
    st["curve"] = old.get("curve", [])
    st["t"] = old.get("t", 0)
    st["start"] = st["curve"][0][0] if st["curve"] else 0
    return st


def main():
    os.makedirs(R.OUT, exist_ok=True)
    sp = os.path.join(R.OUT, "state.json")
    st = json.load(open(sp)) if os.path.exists(sp) else L.new_state()
    if "pos" not in st:
        st = migrate(st)
    data, missing = {}, []
    for s in R.SYMBOLS:
        b = R.fetch(s, "1h")
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
    cs = L.curve_stats(st, "1h")
    halted = eq < st["day"]["eq"] * 0.97
    live = dict(equity=round(eq, 2), return_pct=round((eq / R.START - 1) * 100, 2), trades=st["nc"],
                win_rate=round(100 * st["nw"] / st["nc"], 1) if st["nc"] else 0, open_positions=len(st["pos"]),
                halted_today=halted, expo=cs["expo"], sharpe=cs["sharpe"], wr_ci=cs["wr_ci"])
    bt = L.backtest(data, A, {})
    btm = dict(trades=bt["trades"], win_rate=bt["win_rate"], return_pct=bt["ret"], max_drawdown_pct=bt["max_dd"],
               buy_hold_pct=bt["bh"], days=bt["days"], sharpe=bt["sharpe"], bh_sharpe=bt["bh_sharpe"]) if bt else None
    sigs = snapshot(data)
    fired = [x["sym"] for x in sigs if x["fire"]]
    note = "Live paper account: %s equity, %s%% since start, %d closed trades. Signals firing now: %s." % (
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
    json.dump(dash, open(os.path.join(R.OUT, "dashboard.json"), "w"))
    json.dump(st, open(sp, "w"))
    print(note)


if __name__ == "__main__":
    main()
