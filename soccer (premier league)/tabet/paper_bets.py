"""Paper betting from gameweek 12 of 2026-27: imaginary money, descriptive only (not a test, and nothing
here feeds a forecast). Written to site/data/paper-bets.json for the notebook.

The rule, fixed before gameweek 12 (user's choice, 2026-11-22):
  * every match with a logged record bets on TaBet's most likely result (pE, the predicted-lineup
    forecast; p0 if a record has no pE);
  * at Pinnacle's actual decimal price, margin included, in The Odds API snapshot matched to that record
    (the same snapshot as "Market at our forecast time"). No matched Pinnacle price: no bet;
  * the stake is 10% of the bankroll at the moment the record was logged, counting only bets settled by
    then (kickoff + 2 h). The bankroll starts at $100. Bets logged before any of them settle share the
    same base, so a full gameweek stakes about 10 x 10%.
Pinnacle's own prices live in the private odds bucket (amendment A8); this file publishes only the one
price each bet was struck at, by the user's decision of 2026-11-22.
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pandas as pd

START, FRACTION, FIRST_GW = 100.0, 0.10, 12
SETTLE = pd.Timedelta(hours=2)
OUT = Path(__file__).resolve().parent.parent.parent / "site" / "data" / "paper-bets.json"
PICK = ("H", "D", "A")


def _private_get():
    from . import supabase_storage as ss
    from .odds_archive import RAW_BUCKET
    if not ss.configured():
        return None
    return lambda path: ss.get(RAW_BUCKET, path)


def pinnacle_price(fixture: int, retrieved: pd.Timestamp, get, cache: dict) -> list[float] | None:
    """Pinnacle's decimal odds (H, D, A) for one fixture in one private per-bookmaker snapshot."""
    stamp = pd.Timestamp(retrieved).strftime("%Y%m%dT%H%M%SZ")
    if stamp not in cache:
        try:
            cache[stamp] = pd.read_csv(io.BytesIO(gzip.decompress(get(f"oddsapi/books/{stamp}.csv.gz"))))
        except Exception:
            cache[stamp] = None
    b = cache[stamp]
    if b is None:
        return None
    row = b[(b["book"] == "pinnacle") & (pd.to_numeric(b["fpl_fixture"], errors="coerce") == fixture)]
    if row.empty:
        return None
    return [float(x) for x in row.iloc[0][["odds_h", "odds_d", "odds_a"]]]


def matched_snapshot(r: dict, odds_log: pd.DataFrame) -> pd.Timestamp | None:
    """The Odds API snapshot from the record's own run (as site_data's matched market), or None."""
    from .market_benchmark import select_snapshots
    from .odds_archive import API_SOURCE
    if odds_log is None or odds_log.empty:
        return None
    rec = pd.DataFrame([{k: r[k] for k in ("fpl_fixture", "logged_utc", "kickoff_utc")}])
    s = select_snapshots(rec, odds_log, source=API_SOURCE, with_last=False)
    if len(s) and not bool(s["later"].iloc[0]) and abs(float(s["hours_after_cutoff"].iloc[0])) <= 1.0:
        return pd.Timestamp(s["retrieved_utc"].iloc[0])
    return None


def build(logged: dict, odds_log: pd.DataFrame, results: dict, names: dict, get=None, out: Path = OUT) -> dict | None:
    """logged: fixture -> scoring record (site_data._logged). results: fixture -> "H"/"D"/"A" for
    finished matches. names: team id -> display name. get(path) reads the private bucket; without it,
    prices already in `out` are reused, and if a new bet would need one the file is left unchanged."""
    get = get if get is not None else _private_get()
    prev = {}
    if out.exists():
        prev = {b["fixture"]: b for b in json.loads(out.read_text()).get("bets", [])}
    cache: dict = {}
    bets = []
    for fx, r in logged.items():
        if (r.get("gameweek") or 0) < FIRST_GW:
            continue
        key = "pE" if "pE" in r["candidates"] else "p0"
        c = r["candidates"][key]
        p = [float(c["p_home"]), float(c["p_draw"]), float(c["p_away"])]
        i = max(range(3), key=lambda j: p[j])
        odds = (prev.get(fx) or {}).get("odds")
        if odds is None:
            snap = matched_snapshot(r, odds_log)
            if snap is not None:
                if get is None:
                    return None          # a price is needed and the private bucket is not reachable here
                odds = pinnacle_price(int(fx), snap, get, cache)
        bets.append({"fixture": int(fx), "gameweek": int(r["gameweek"]), "kickoff": r["kickoff_utc"],
                     "logged": r["logged_utc"], "home": names.get(r["home_id"], r["home_id"]),
                     "away": names.get(r["away_id"], r["away_id"]), "forecast": key, "pick": PICK[i],
                     "p_pick": round(p[i], 4), "odds": odds, "result": results.get(int(fx))})
    data = {"first_gameweek": FIRST_GW, **settle(bets)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return data


def settle(bets: list[dict], start: float = START, fraction: float = FRACTION) -> dict:
    """The staking rule, shared by the live tracker and the backtest. Each bet needs logged (when it is
    placed), kickoff, pick, odds (H, D, A decimal, or None for no bet) and result (or None). Sorts the
    bets by placement and fills stake, pnl and status; returns the summary with the bets."""
    bets.sort(key=lambda b: (b["logged"], b["kickoff"], b.get("fixture", 0)))
    placed = [pd.Timestamp(b["logged"]) for b in bets]
    due_at = [pd.Timestamp(b["kickoff"]) + SETTLE for b in bets]
    blocked = False       # an earlier bet that should have settled has no result yet
    for k, b in enumerate(bets):
        if b["odds"] is None:
            b.update(status="no price", stake=None, pnl=None)
            continue
        due = [x for j, x in enumerate(bets[:k]) if x["status"] in ("won", "lost", "pending") and due_at[j] <= placed[k]]
        if blocked or any(x["status"] == "pending" for x in due):
            b.update(status="pending", stake=None, pnl=None)
            blocked = True
            continue
        stake = round(fraction * (start + sum(x["pnl"] for x in due)), 2)
        b["stake"] = stake
        if b["result"] is None:
            b.update(status="pending", pnl=None)
            continue
        win = b["result"] == b["pick"]
        b.update(status="won" if win else "lost",
                 pnl=round(stake * (b["odds"][PICK.index(b["pick"])] - 1), 2) if win else -stake)
    done = [b for b in bets if b["status"] in ("won", "lost")]
    staked = sum(b["stake"] for b in done)
    pnl = sum(b["pnl"] for b in done)
    return {"start": start, "fraction": fraction, "bankroll": round(start + pnl, 2), "n_settled": len(done),
            "n_won": sum(b["status"] == "won" for b in done), "staked": round(staked, 2),
            "roi": round(pnl / staked, 4) if staked else None, "bets": bets}
