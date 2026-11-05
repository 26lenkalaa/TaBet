"""Other European top flights from football-data.co.uk, in the same match schema.

Used only to learn *shared* adjustment coefficients and hyperparameters. Team
ratings are always fitted within a league, and teams are never compared
across leagues. Team ids are league-prefixed slugs (e.g. "sp1_real_madrid"),
because these teams never meet Premier League sides in our data.

Files live in res/other_leagues/<DIV>_<yyyy>.csv (yyyy = e.g. 2122), so the
Premier League loader (res/*.csv) never picks them up.
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .fpl import _ssl_context
from .ingest import FLOAT_COLS, INT_COLS, MATCH_COLUMNS, TEXT_COLS, _parse_int, _parse_kickoff

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "res" / "other_leagues"
LEAGUES = {"SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
SEASONS = ("2122", "2223", "2324", "2425", "2526", "2627")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def season_label(code: str) -> str:
    return f"20{code[:2]}-{code[2:]}"


def download(divs=tuple(LEAGUES), seasons=SEASONS, refresh_current: bool = False) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for d in divs:
        for s in seasons:
            path = DIR / f"{d}_{s}.csv"
            if path.exists() and not (refresh_current and s == seasons[-1]):
                continue
            req = urllib.request.Request(f"https://football-data.co.uk/mmz4281/{s}/{d}.csv",
                                         headers={"User-Agent": "tabet-research/0.1"})
            with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
                path.write_bytes(r.read())
            time.sleep(0.5)


def load_league_season(path: Path) -> pd.DataFrame:
    div, code = path.stem.split("_")
    season = season_label(code)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[raw["HomeTeam"].str.strip() != ""].reset_index(drop=True)
    problems: list[str] = []
    out = pd.DataFrame(index=raw.index)
    out["season"] = season
    out["kickoff_utc"], out["kickoff_time_known"] = _parse_kickoff(raw, problems)
    for side, col in (("home", "HomeTeam"), ("away", "AwayTeam")):
        names = raw[col].str.strip()
        out[f"{side}_id"] = div.lower() + "_" + names.map(_slug)
        out[f"{side}_team"] = names
        out[f"{side}_team_raw"] = names
    for src, dst in TEXT_COLS.items():
        out[dst] = raw[src].str.strip() if src in raw.columns else ""
    for src, dst in INT_COLS.items():
        out[dst] = _parse_int(raw[src], src, problems) if src in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    for src, dst in FLOAT_COLS.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan
    if problems:
        raise ValueError(f"{path.name}: {problems}")
    done = out["home_goals"].notna() & out["away_goals"].notna()
    out["status"] = np.where(done, "completed", "scheduled")
    out["source_file"] = path.name
    out["match_id"] = (season + "-" + out["kickoff_utc"].dt.tz_convert("Europe/London").dt.strftime("%Y%m%d")
                       + "-" + out["home_id"] + "-" + out["away_id"])
    out["league"] = div
    return out[MATCH_COLUMNS + ["league"]]


def load_leagues(divs=tuple(LEAGUES)) -> pd.DataFrame:
    parts = [load_league_season(p) for d in divs for p in sorted(DIR.glob(f"{d}_*.csv"))]
    m = pd.concat(parts, ignore_index=True).sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    problems = validate_leagues(m)
    if problems:
        raise ValueError("other-league data failed validation:\n  " + "\n  ".join(problems))
    return m


def validate_leagues(m: pd.DataFrame) -> list[str]:
    """Double round robin per completed league-season (n teams -> n(n-1)
    matches, each ordered pair once), consistent results, no duplicates."""
    problems = []
    if m["match_id"].duplicated().any():
        problems.append(f"duplicate match ids: {m.loc[m['match_id'].duplicated(), 'match_id'].head(3).tolist()}")
    latest = m.groupby("league")["season"].transform("max")
    for (lg, season), s in m.groupby(["league", "season"]):
        teams = set(s["home_id"]) | set(s["away_id"])
        n = len(teams)
        if s.groupby(["home_id", "away_id"]).size().max() > 1:
            problems.append(f"{lg} {season}: repeated fixture")
        if season != latest[s.index].iloc[0] and len(s) != n * (n - 1):
            problems.append(f"{lg} {season}: {len(s)} matches for {n} teams (expected {n * (n - 1)})")
    c = m[m["status"] == "completed"]
    exp = np.select([c["home_goals"] > c["away_goals"], c["home_goals"] < c["away_goals"]], ["H", "A"], "D")
    bad = c["result"] != exp
    if bad.any():
        problems.append(f"{bad.sum()} results inconsistent with scores")
    return problems
