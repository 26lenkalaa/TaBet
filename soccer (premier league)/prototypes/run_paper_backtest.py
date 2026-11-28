"""Paper-betting backtest: the live tracker's rule (tabet/paper_bets.py, settle) replayed on the frozen
walk-forward forecasts of V0, V1x and V2x (results/xg_strength/ledger.csv), 2022-23 -> 2026-27 GW 1-9.

    python prototypes/run_paper_backtest.py      # writes site/data/paper-bets-backtest.json

Descriptive only. Fixed before running:
  * each season starts at $100; every match bets 10% of the bankroll settled by the forecast's cutoff
    (the ledger's cutoff_utc) on the model's most likely result;
  * price: Pinnacle's pre-closing decimal odds from football-data (PSH/PSD/PSA, collected Friday or
    Tuesday afternoon), margin included; where football-data has no Pinnacle price (from mid-2025-26),
    the bookmaker average (AvgH/AvgD/AvgA), also with its margin. The counts of each are reported;
  * baseline: always back the market favourite (the shortest of the same prices), same staking;
  * value rule (added 2026-11-22, same prices and staking): bet only on a result whose model chance
    times the price exceeds 1 (positive expected value by the model), choosing the largest such edge;
    no such result, no bet;
  * 5% edge rule (added 2026-11-22): the value rule, but only when that edge is at least 0.05.
V2x's forecasts use the confirmed lineups, so they come later than the price they are paired with.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.ingest import load_all  # noqa: E402
from tabet.paper_bets import PICK, settle  # noqa: E402

OUT = ROOT.parent / "site" / "data" / "paper-bets-backtest.json"
MODELS = {"V0 Dixon-Coles": "V0: results only", "V1x adjusted xG strength (replaces raw)": "V1x: chances created",
          "V2x XI strength with adjusted xG": "V2x: lineups"}
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]


def prices(odds: pd.DataFrame) -> dict[str, tuple[list[float], str]]:
    out = {}
    for r in odds.itertuples():
        ps = [getattr(r, c) for c in ("PSH", "PSD", "PSA")]
        av = [getattr(r, c) for c in ("AvgH", "AvgD", "AvgA")]
        if all(pd.notna(x) and x > 1 for x in ps):
            out[r.match_id] = ([float(x) for x in ps], "pinnacle")
        elif all(pd.notna(x) and x > 1 for x in av):
            out[r.match_id] = ([float(x) for x in av], "average")
    return out


def pick(rule: str, p: list[float], odds: list[float]) -> int | None:
    if rule == "favourite":
        return int(np.argmax(p))
    ev = [p[k] * odds[k] - 1 for k in range(3)]
    k = int(np.argmax(ev))
    return k if ev[k] > (0.05 if rule == "value5" else 0.0) else None


def run() -> dict:
    matches, odds = load_all(ROOT / "res")
    price = prices(odds)
    led = pd.read_csv(ROOT / "results" / "xg_strength" / "ledger.csv")
    led = led[led["season"].isin(SEASONS)]
    out = {}
    for rule in ("favourite", "value", "value5"):
        out[rule] = table(led, price, rule)
    data = {"rule": "10% of the bankroll a bet, Pinnacle pre-closing price (else the bookmaker average), "
                    "$100 at the start of each season; favourite = the model's most likely result, value = "
                    "the result with the largest positive expected value by the model; value5 = the same with an edge of at least 5%", "seasons": SEASONS, **out}
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    return data


def table(led, price, rule) -> dict:
    rows = {}
    base = led[led["variant"] == "V0 Dixon-Coles"].drop_duplicates("match_id")
    extra = [("market", "Market favourite (baseline)")] if rule == "favourite" else []
    for name, label in list(MODELS.items()) + extra:
        src = base if name == "market" else led[led["variant"] == name].drop_duplicates("match_id")
        per = {}
        for season, g in src.groupby("season"):
            bets = []
            for r in g.itertuples():
                pr = price.get(r.match_id)
                if pr is None:
                    continue
                if name == "market":
                    i = int(np.argmin(pr[0]))
                else:
                    i = pick(rule, [r.p_home, r.p_draw, r.p_away], pr[0])
                    if i is None:
                        continue
                bets.append({"fixture": r.match_id, "logged": r.cutoff_utc, "kickoff": r.kickoff_utc, "pick": PICK[i],
                             "odds": pr[0], "result": r.result, "price": pr[1]})
            s = settle(bets)
            per[season] = {k: s[k] for k in ("bankroll", "n_settled", "n_won", "staked", "roi")}
            per[season]["n_pinnacle"] = sum(b["price"] == "pinnacle" for b in bets)
            per[season]["n_average"] = sum(b["price"] == "average" for b in bets)
            per[season]["lowest"] = round(min([100.0] + list(np.cumsum([b["pnl"] for b in s["bets"] if b["pnl"] is not None]) + 100.0)), 2)
            per[season]["picks"] = {k: sum(b["pick"] == k for b in bets) for k in PICK}
            flat = [(b["odds"][PICK.index(b["pick"])] - 1) if b["result"] == b["pick"] else -1.0 for b in bets]
            per[season]["flat_roi"] = round(float(np.mean(flat)), 4) if flat else None   # $1 on every bet
        rows[label] = per
    return rows


if __name__ == "__main__":
    d = run()
    for label, per in [(f"{r}: {k}", v) for r in ("favourite", "value", "value5") for k, v in d[r].items()]:
        print(label)
        for s, v in per.items():
            print(f"  {s}: ${v['bankroll']:>8.2f}  won {v['n_won']}/{v['n_settled']}  roi {v['roi']:+.3f} flat {v['flat_roi']:+.3f}  low ${v['lowest']}  "
                  f"(Pinnacle {v['n_pinnacle']}, average {v['n_average']}) picks {v['picks']}")
