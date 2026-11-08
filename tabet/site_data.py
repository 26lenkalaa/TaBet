"""Data for the site's match explorer (site/matches.html): one JSON file, site/data/matches.json.

Every match from 2022-23 onward gets the forecasts that exist for it, each labelled by kind:
  backtest  frozen walk-forward forecasts (results/xg_strength/ledger.csv): V0, V1x, V2x. Made with
            only information available before each match, but computed afterwards.
  logged    the prospective record made 24-32 h before kickoff (results/prospective/log.jsonl; primary
            record if it exists, else the fallback), with p0 (lineup-free) and pE (predicted lineups).
  preview   for upcoming fixtures without a logged record yet: the lineup-free model run when this file
            was built. Not a record; it changes as results come in.
Market probabilities come from football-data average odds with the margin removed proportionally:
pre-closing (collected Friday/Tuesday afternoon) and closing. For upcoming fixtures, football-data's
fixtures.csv gives pre-closing odds once published. Odds are a benchmark only, never a model input.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .fpl import _ssl_context, fetch_live_season
from .ingest import implied_probabilities, load_all
from .teams import TEAMS, try_resolve_team

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data" / "matches.json"
LIVE_SEASON = "2026-27"
LEDGER = ROOT / "results" / "xg_strength" / "ledger.csv"
LOG = ROOT / "results" / "prospective" / "log.jsonl"
BACKTEST = {"V0 Dixon-Coles": ("V0", "Results only (V0)"),
            "V1x adjusted xG strength (replaces raw)": ("V1x", "Chances created (V1x)"),
            "V2x XI strength with adjusted xG": ("V2x", "With confirmed lineups (V2x)")}


def _p(row) -> list[float]:
    return [round(float(row["p_home"]), 4), round(float(row["p_draw"]), 4), round(float(row["p_away"]), 4)]


def _odds_upcoming() -> dict[tuple[str, str], list[float]]:
    """Pre-closing odds for upcoming PL fixtures from football-data's fixtures.csv (if reachable)."""
    try:
        req = urllib.request.Request("https://www.football-data.co.uk/fixtures.csv", headers={"User-Agent": "tabet-research/0.1"})
        raw = urllib.request.urlopen(req, timeout=60, context=_ssl_context()).read()
    except Exception:
        return {}
    f = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    f = f[f["Div"] == "E0"]
    out = {}
    for r in f.itertuples():
        h, a = try_resolve_team(r.HomeTeam), try_resolve_team(r.AwayTeam)
        if h and a and pd.notna(r.AvgH):
            q = 1 / np.array([r.AvgH, r.AvgD, r.AvgA], float)
            out[(h, a)] = [round(float(x), 4) for x in q / q.sum()]
    return out


def _logged() -> dict[int, dict]:
    """Per FPL fixture, the record scoring would use: primary (24 h) if present, else fallback."""
    if not LOG.exists():
        return {}
    best: dict[int, dict] = {}
    for r in map(json.loads, LOG.read_text().splitlines()):
        cur = best.get(r["fpl_fixture"])
        if cur is None or r["horizon_h"] < cur["horizon_h"]:
            best[r["fpl_fixture"]] = r
    return best


def build(preview: bool = True, out: Path = OUT) -> dict:
    matches, odds = load_all(ROOT / "res")
    pre = implied_probabilities(odds, "Avg").set_index("match_id")
    close = implied_probabilities(odds, "AvgC").set_index("match_id")
    led = pd.read_csv(LEDGER)
    led = {v: led[led["variant"] == v].drop_duplicates("match_id").set_index("match_id") for v in BACKTEST}
    logged = _logged()

    d = fetch_live_season(LIVE_SEASON, refresh=preview)
    boot = json.loads((d / "bootstrap.json").read_text())
    tid = {t["id"]: try_resolve_team(t["name"]) for t in boot["teams"]}
    fpl = pd.DataFrame(json.loads((d / "fixtures.json").read_text()))
    fpl["home_id"], fpl["away_id"] = fpl["team_h"].map(tid), fpl["team_a"].map(tid)
    gw_of = {(h, a): int(e) for h, a, e in zip(fpl["home_id"], fpl["away_id"], fpl["event"]) if pd.notna(e)}

    entries = []
    done = matches[(matches["status"] == "completed") & (matches["season"] >= "2022-23")]
    seen_live = set()
    for m in done.itertuples():
        fc = [{"key": key, "label": label, "kind": "backtest", "p": _p(led[v].loc[m.match_id])}
              for v, (key, label) in BACKTEST.items() if m.match_id in led[v].index]
        mk = []
        if m.match_id in pre.index and pre.loc[m.match_id].notna().all():
            mk.append({"key": "pre", "label": "Earlier odds", "p": _p(pre.loc[m.match_id])})
        if m.match_id in close.index and close.loc[m.match_id].notna().all():
            mk.append({"key": "close", "label": "Closing odds", "p": _p(close.loc[m.match_id])})
        e = {"id": m.match_id, "season": m.season, "kickoff": m.kickoff_utc.isoformat(), "home": TEAMS.get(m.home_id, m.home_team),
             "away": TEAMS.get(m.away_id, m.away_team), "score": f"{m.home_goals}-{m.away_goals}", "result": m.result,
             "status": "played", "forecasts": fc, "market": mk}
        if m.season == LIVE_SEASON:
            e["gameweek"] = gw_of.get((m.home_id, m.away_id))
            seen_live.add((m.home_id, m.away_id))
        entries.append(e)

    # current season: fixtures not (yet) in the football-data results file
    up_odds = _odds_upcoming()
    new = np.array([(h, a) not in seen_live for h, a in zip(fpl["home_id"], fpl["away_id"])], dtype=bool)
    pending = fpl[new & fpl["kickoff_time"].notna().to_numpy()]
    previews = {}
    if preview and len(pending[~pending["finished_provisional"].astype(bool)]):
        from .forecast import forecast
        try:
            fcst = forecast(cutoff=pd.Timestamp.now(tz="UTC").floor("min"), gameweeks=2, refresh=True, lineups=False)
            previews = {f["fpl_fixture"]: f for f in fcst["fixtures"]}
            preview_time = fcst["cutoff_utc"]
        except Exception as ex:  # the explorer must still build if a live fetch fails
            print(f"preview forecasts skipped ({ex})")
    for f in pending.sort_values("kickoff_time").itertuples():
        played = bool(f.finished_provisional)
        started = pd.Timestamp(f.kickoff_time) <= pd.Timestamp.now(tz="UTC")
        e = {"id": f"fpl-{f.id}", "season": LIVE_SEASON, "gameweek": int(f.event) if pd.notna(f.event) else None,
             "kickoff": pd.Timestamp(f.kickoff_time).isoformat(), "home": TEAMS.get(f.home_id, f.home_id),
             "away": TEAMS.get(f.away_id, f.away_id),
             "status": "played" if played else ("awaiting" if started else "upcoming"),
             "forecasts": [], "market": []}
        if played:
            h, a = int(f.team_h_score), int(f.team_a_score)
            e.update(score=f"{h}-{a}", result="H" if h > a else ("A" if a > h else "D"))
        r = logged.get(int(f.id))
        if r:
            role = "primary" if r["horizon_h"] == 24.0 else "fallback"
            note = f"logged {r['hours_before']:.1f} h before kickoff ({role} record)"
            for key, label in (("p0", "TaBet, no lineups (p0)"), ("pE", "TaBet, predicted lineups (pE)")):
                if key in r["candidates"]:
                    e["forecasts"].append({"key": key, "label": label, "kind": "logged", "note": note, "p": _p(r["candidates"][key])})
        elif int(f.id) in previews and not played:
            pv = previews[int(f.id)]
            e["forecasts"].append({"key": "preview", "label": "TaBet, no lineups (preview)", "kind": "preview",
                                   "note": f"built {preview_time[:16].replace('T', ' ')} UTC; the logged forecast comes 24-32 h before kickoff",
                                   "p": _p(pv)})
        elif played or started:
            e["missing"] = "No forecast was logged for this match (the logger missed its window)."
        if (f.home_id, f.away_id) in up_odds:
            e["market"].append({"key": "pre", "label": "Earlier odds", "p": up_odds[(f.home_id, f.away_id)]})
        entries.append(e)

    entries.sort(key=lambda x: x["kickoff"])
    n_log = len(LOG.read_text().splitlines()) if LOG.exists() else 0
    data = {"built_utc": pd.Timestamp.now(tz="UTC").floor("min").isoformat(), "n_log_records": n_log,
            "seasons": sorted({e["season"] for e in entries}), "matches": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    # one match per line, so the daily rebuild commits a small diff instead of rewriting the file
    head = {k: v for k, v in data.items() if k != "matches"}
    body = ",\n".join(json.dumps(e, separators=(",", ":")) for e in entries)
    out.write_text(json.dumps(head, separators=(",", ":"))[:-1] + ',"matches":[\n' + body + "\n]}\n")
    return data


def stale(max_age_h: float = 20.0, out: Path = OUT) -> bool:
    """Rebuild if the file is missing, older than max_age_h, or the log has records it has not seen."""
    if not out.exists():
        return True
    try:
        head = json.loads(out.read_text())
    except ValueError:
        return True
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(head["built_utc"])).total_seconds() / 3600
    n_log = len(LOG.read_text().splitlines()) if LOG.exists() else 0
    return age >= max_age_h or n_log != head.get("n_log_records")
