"""How accurate are the predicted lineups behind the day-before forecast (pE)? Descriptive only: it compares
predicted starting XIs with the actual ones and never looks at results or forecasts.

    python prototypes/lineup_accuracy.py      # writes results/lineup_accuracy/summary.json

Actual starters: FPL's gameweek live data (stats.starts per player; a player with starts in a single
fixture that gameweek). Predictions:
  * logged records (results/prospective/log.jsonl): the four XIs saved with each forecast: fpl_proj (FPL's
    own projection), fpl_proj_hist (our history-only projection), ffs (Fantasy Football Scout), rotowire;
  * timed snapshots (results/lineup_snapshots/): every FFS and RotoWire predicted XI retrieved since
    2026-11-05, matched to each team's next fixture, so accuracy can be read by hours before kickoff.
    RotoWire's "confirmed" rows (published after the official team sheets) are left out.
Accuracy = predicted starters who started, out of 11, per team and match.
"""
from __future__ import annotations

import glob
import json
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.teams import try_resolve_team  # noqa: E402

OUT = ROOT / "results" / "lineup_accuracy"
LOG = ROOT / "results" / "prospective" / "log.jsonl"
SNAPS = ROOT / "results" / "lineup_snapshots"
BUCKETS = [(0, 6, "under 6 h"), (6, 24, "6-24 h"), (24, 48, "24-48 h"), (48, 96, "48-96 h"), (96, 10_000, "over 96 h")]


def _get(url: str):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tabet-research/0.1"}),
                                            timeout=60, context=ctx))


def actual_xis(gameweeks) -> tuple[dict, pd.DataFrame]:
    """{(fixture, team_id): set of player codes who started}, and the FPL fixtures table."""
    boot = _get("https://fantasy.premierleague.com/api/bootstrap-static/")
    code = {e["id"]: e["code"] for e in boot["elements"]}
    team = {t["id"]: try_resolve_team(t["name"]) for t in boot["teams"]}
    fx = pd.DataFrame(_get("https://fantasy.premierleague.com/api/fixtures/"))
    fx["home_id"], fx["away_id"] = fx["team_h"].map(team), fx["team_a"].map(team)
    fx["kickoff"] = pd.to_datetime(fx["kickoff_time"], utc=True)
    side = {(f.id, t): tid for f in fx.itertuples() for t, tid in ((f.team_h, f.home_id), (f.team_a, f.away_id))}
    el_team = {e["id"]: e["team"] for e in boot["elements"]}
    out: dict = {}
    for gw in gameweeks:
        for e in _get(f"https://fantasy.premierleague.com/api/event/{gw}/live/")["elements"]:
            fixtures = [x["fixture"] for x in e["explain"]]
            if e["stats"].get("starts") and len(fixtures) == 1:
                key = (fixtures[0], side.get((fixtures[0], el_team[e["id"]])))
                out.setdefault(key, set()).add(code[e["id"]])
    return {k: v for k, v in out.items() if len(v) == 11}, fx


def logged(actual: dict) -> pd.DataFrame:
    rows = []
    for r in map(json.loads, LOG.read_text().splitlines()):
        for src, xi in (r.get("lineup_codes") or {}).items():
            for s, team in (("home", r["home_id"]), ("away", r["away_id"])):
                got = actual.get((r["fpl_fixture"], team))
                if got is None or not xi.get(s):
                    continue
                pred = {int(c) for c in xi[s]}
                rows.append({"gameweek": r["gameweek"], "fixture": r["fpl_fixture"], "team": team, "source": src,
                             "role": "primary" if r["horizon_h"] == 24.0 else "fallback", "hours_before": r["hours_before"],
                             "hits": len(pred & got), "n_pred": len(pred), "missed": sorted(got - pred), "wrong": sorted(pred - got)})
    return pd.DataFrame(rows)


def snapshots(actual: dict, fx: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob.glob(str(SNAPS / "lineups_*.csv")))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[(d["status"] == "predicted") & d["player_code"].notna()]
    d["t"] = pd.to_datetime(d["retrieved_utc"], format="%Y%m%dT%H%MZ", utc=True)
    long = pd.concat([fx[["id", "kickoff", "home_id"]].rename(columns={"home_id": "team"}),
                      fx[["id", "kickoff", "away_id"]].rename(columns={"away_id": "team"})]).dropna()
    rows = []
    for (t, team, src), g in d.groupby(["t", "team_id", "source"]):
        nxt = long[(long["team"] == team) & (long["kickoff"] > t)].sort_values("kickoff").head(1)
        if nxt.empty:
            continue
        fixture, ko = int(nxt["id"].iloc[0]), nxt["kickoff"].iloc[0]
        got = actual.get((fixture, team))
        if got is None:
            continue
        pred = set(g["player_code"].astype(int))
        rows.append({"retrieved": t, "team": team, "source": src, "fixture": fixture,
                     "hours_before": (ko - t).total_seconds() / 3600, "hits": len(pred & got), "n_pred": len(pred)})
    return pd.DataFrame(rows)


def bucket(h: float) -> str:
    return next(lab for lo, hi, lab in BUCKETS if lo <= h < hi)


def main() -> dict:
    recs = [json.loads(x) for x in LOG.read_text().splitlines()]
    gws = sorted({r["gameweek"] for r in recs})
    actual, fx = actual_xis(gws)
    lg, sn = logged(actual), snapshots(actual, fx)
    out = {"gameweeks": gws, "team_matches_with_actual_xi": len(actual)}
    if len(lg):
        p = lg[lg["role"] == "primary"]
        out["logged_primary_by_source"] = {s: {"mean_hits_of_11": round(float(g["hits"].mean()), 2),
                                               "all_11": int((g["hits"] == 11).sum()), "n": int(len(g))}
                                           for s, g in p.groupby("source")}
        out["logged_primary_by_gameweek"] = {f"GW{gw} {s}": round(float(g["hits"].mean()), 2) for (gw, s), g in p.groupby(["gameweek", "source"])}
        worst = p[p["source"].isin(["ffs", "rotowire"])].groupby(["fixture", "team"])["hits"].mean().sort_values().head(5)
        out["worst_team_matches_ffs_rotowire_mean"] = {f"{f} {t}": round(float(v), 1) for (f, t), v in worst.items()}
    if len(sn):
        sn["when"] = sn["hours_before"].map(bucket)
        out["snapshots_by_source_and_lead"] = {f"{s} {w}": {"mean_hits_of_11": round(float(g["hits"].mean()), 2), "n": int(len(g))}
                                               for (s, w), g in sn.groupby(["source", "when"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
