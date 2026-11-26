"""Kickoff records with confirmed lineups (docs/PROSPECTIVE_PROTOCOL.md, amendment A11). Descriptive only.

Called from every logger run. A fixture gets one record, in results/prospective/kickoff.jsonl, from the
first run that finds its kickoff 10-50 minutes ahead and RotoWire showing confirmed XIs for both teams:

    pK    V2x (the 24-hour pE model) with RotoWire's confirmed XIs, at this cutoff
    p0K   the lineup-free p0 at the same cutoff

The 24-hour log (log.jsonl), its candidates and the shadow log are never touched. Any failure is printed
and the logger carries on.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "prospective"
KICKOFF = LOG_DIR / "kickoff.jsonl"
MIN_BEFORE = pd.Timedelta(minutes=10)
MAX_BEFORE = pd.Timedelta(minutes=50)
SOURCE = "rotowire"


def _ko(x) -> str:
    return pd.Timestamp(x).tz_convert("UTC").isoformat()


def logged() -> set[tuple[int, str]]:
    if not KICKOFF.exists():
        return set()
    return {(int(r["fpl_fixture"]), _ko(r["kickoff_utc"])) for r in map(json.loads, KICKOFF.read_text().splitlines()) if r}


def due(now: pd.Timestamp, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Fixtures kicking off 10-50 minutes after `now` that have no kickoff record yet."""
    ahead = fixtures["kickoff_utc"] - now
    d = fixtures[(ahead >= MIN_BEFORE) & (ahead <= MAX_BEFORE)]
    done = logged()
    return d[[(int(f), _ko(k)) not in done for f, k in zip(d["fpl_fixture"], d["kickoff_utc"])]]


def records(fc: dict, now: pd.Timestamp) -> list[dict]:
    """Kickoff records from a forecast run, only for fixtures with both XIs confirmed on RotoWire."""
    status = fc.get("lineup_status", {}).get(SOURCE, {})
    commit = fc.get("git_commit")
    out = []
    for f in fc["fixtures"]:
        w = f.get("with_lineups") or {}
        both = status.get(f["home_id"]) == "confirmed" and status.get(f["away_id"]) == "confirmed"
        if not both or SOURCE not in w.get("sources", {}):
            continue
        p = w["sources"][SOURCE]
        out.append({
            "fpl_fixture": f["fpl_fixture"], "gameweek": f["gameweek"], "kickoff_utc": f["kickoff_utc"],
            "logged_utc": now.isoformat(), "minutes_before": (pd.Timestamp(f["kickoff_utc"]) - now).total_seconds() / 60,
            "home_id": f["home_id"], "away_id": f["away_id"], "git_commit": commit,
            "candidates": {"pK": {k: p[k] for k in ("p_home", "p_draw", "p_away")},
                           "p0K": {k: f[k] for k in ("p_home", "p_draw", "p_away")}},
            "xg": {"pK": [p["xg_home"], p["xg_away"]], "p0K": [f["xg_home"], f["xg_away"]]},
            "lineup_codes": w.get("lineup_codes", {}).get(SOURCE, {}),
            "xi_features": w.get("xi_features", {}).get(SOURCE, {}),
            "components": f["components"], "xi_stage2": fc["xi_stage2"],
            "lineup_retrieved_utc": fc["lineup_sources"]["retrieved_utc"],
        })
    return out


def log_kickoff(now: pd.Timestamp, fixtures: pd.DataFrame, forecast_fn=None) -> list[dict]:
    """Append kickoff records for fixtures now due (A11). Returns what was written."""
    d = due(now, fixtures)
    if d.empty:
        return []
    if forecast_fn is None:
        from .forecast import forecast as forecast_fn
    fc = forecast_fn(cutoff=now, fixture_ids=[int(i) for i in d["fpl_fixture"]], refresh=True, lineups=True)
    recs = records(fc, now)
    stamp = fc["lineup_sources"]["retrieved_utc"]
    page = ROOT / "data" / "raw" / "lineups" / SOURCE / f"{stamp}.html"
    if recs and page.exists():                          # keep the page the XIs came from
        raw = LOG_DIR / "raw" / f"{stamp}_{SOURCE}_kickoff.html.gz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        with page.open("rb") as a, gzip.open(raw, "wb") as b:
            shutil.copyfileobj(a, b)
        for r in recs:
            r["raw_snapshot"] = str(raw.relative_to(ROOT))
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "tabet"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    done = logged()                                     # re-read just before appending
    recs = [dict(r, code_dirty=dirty) for r in recs if (int(r["fpl_fixture"]), _ko(r["kickoff_utc"])) not in done]
    KICKOFF.parent.mkdir(parents=True, exist_ok=True)
    with KICKOFF.open("a") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=str) + "\n")
    return recs
