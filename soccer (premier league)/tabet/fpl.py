"""Fantasy Premier League player appearances, joined to our match table.

Sources:
  * Completed seasons: the public archive of FPL API snapshots at
    github.com/vaastav/Fantasy-Premier-League (merged_gw.csv, fixtures.csv,
    players_raw.csv per season). The repo declares no licence; the data comes
    from the public FPL API.
  * The current season: the FPL API itself (event/<gw>/live, fixtures,
    bootstrap-static), because the archive lags behind.

Output: one row per player per match with minutes > 0, keyed to our match_id.
`player_code` is FPL's `code`, which (unlike `element`) stays the same across
seasons. Per-match xG, xA, xGC and starts exist from 2022-23; 2021-22 rows
have them as NaN. Crowd fields (ownership, transfers, price) are not loaded:
like bookmaker odds they are market signals, and the spec keeps those out of
the model.

Downloads are cached under data/raw/fpl/ (gitignored).
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .teams import try_resolve_team

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fpl"
ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
API = "https://fantasy.premierleague.com/api"
ARCHIVE_FILES = ("gws/merged_gw.csv", "fixtures.csv", "players_raw.csv")
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STAT_COLS = {
    "goals_scored": "goals",
    "assists": "assists",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goals_conceded": "xgc",
    "threat": "threat",
    "creativity": "creativity",
    "influence": "influence",
    "defensive_contribution": "def_contrib",
}
APPEARANCE_COLUMNS = ["match_id", "season", "kickoff_utc", "team_id", "opponent_id", "venue",
                      "player_code", "player_name", "position", "minutes", "started", "started_inferred",
                      *STAT_COLS.values()]


class FplError(ValueError):
    pass


def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds ship without a CA bundle; use certifi's if present.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, dest: Path, refresh: bool = False, pause: float = 1.0) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tabet-research/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        body = r.read()
    dest.write_bytes(body)
    time.sleep(pause)  # be polite to both hosts
    return dest


def fetch_archive_season(season: str, raw_dir: Path = RAW_DIR) -> Path:
    d = raw_dir / season
    for f in ARCHIVE_FILES:
        _get(f"{ARCHIVE}/{season}/{f}", d / Path(f).name, pause=0.2)
    return d


def fetch_live_season(season: str, raw_dir: Path = RAW_DIR, refresh: bool = True) -> Path:
    d = raw_dir / f"{season}-live"
    boot = json.loads(_get(f"{API}/bootstrap-static/", d / "bootstrap.json", refresh).read_text())
    _get(f"{API}/fixtures/", d / "fixtures.json", refresh)
    for ev in boot["events"]:
        if ev["finished"]:
            # A finished gameweek doesn't change, so it's only fetched once.
            _get(f"{API}/event/{ev['id']}/live/", d / f"gw{ev['id']:02d}.json", refresh=False)
    return d


def _join_matches(fx: pd.DataFrame, matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """fx has fpl fixture id + team_h/team_a canonical ids -> add match_id."""
    ours = matches.loc[matches["season"] == season,
                       ["match_id", "home_id", "away_id", "kickoff_utc", "home_goals", "away_goals"]]
    j = fx.merge(ours, on=["home_id", "away_id"], how="left", validate="one_to_one")
    unmatched = j["match_id"].isna() & j["finished"]
    if unmatched.any():
        raise FplError(f"{season}: {unmatched.sum()} finished FPL fixtures not in match table")
    j = j[j["finished"] & j["match_id"].notna()]
    bad = (j["team_h_score"] != j["home_goals"]) | (j["team_a_score"] != j["away_goals"])
    if bad.any():
        raise FplError(f"{season}: FPL score disagrees for {list(j.loc[bad, 'match_id'][:5])}")
    return j


def _team_ids(names: pd.Series) -> pd.Series:
    ids = names.map(try_resolve_team)
    if ids.isna().any():
        raise FplError(f"unknown FPL team names: {sorted(set(names[ids.isna()]))}")
    return ids


def _finish(app: pd.DataFrame, fx: pd.DataFrame, season: str) -> pd.DataFrame:
    """Add match/team context from the joined fixtures and tidy columns."""
    app = app.merge(fx[["fixture", "match_id", "kickoff_utc", "home_id", "away_id"]], on="fixture")
    home = app["was_home"].astype(bool)
    app["team_id"] = np.where(home, app["home_id"], app["away_id"])
    app["opponent_id"] = np.where(home, app["away_id"], app["home_id"])
    app["venue"] = np.where(home, "H", "A")
    for c in STAT_COLS.values():
        if c not in app:
            app[c] = np.nan
        app[c] = pd.to_numeric(app[c], errors="coerce").astype(float)
    app = app[app["minutes"] > 0].assign(season=season)
    # The same player occasionally appears under two FPL element ids (identical rows).
    app = app.drop_duplicates(["match_id", "player_code"])
    app = _infer_missing_starts(app)
    app = _null_unpublished_xg(app)
    return app[APPEARANCE_COLUMNS].sort_values(["kickoff_utc", "match_id", "team_id"]).reset_index(drop=True)


def _null_unpublished_xg(app: pd.DataFrame) -> pd.DataFrame:
    """FPL only published xG/xA/xGC from 12 Nov 2022; the archive codes earlier
    2022-23 matches as 0. A real match always has some xG for one side, so a
    match whose players *all* show 0 xG and 0 xGC is treated as missing."""
    tot = app.groupby("match_id")[["xg", "xgc"]].transform("sum").sum(axis=1)
    unpublished = tot == 0
    app.loc[unpublished, ["xg", "xa", "xgc"]] = np.nan
    return app


def _infer_missing_starts(app: pd.DataFrame) -> pd.DataFrame:
    """FPL has no `starts` before part-way through 2022-23 (the archive codes it
    as 0 for every player, or omits it). For those match sides, treat the 11
    players with the most minutes as the starters and flag it: a starter
    subbed off early can be mistaken for a substitute."""
    app = app.copy()
    side_sum = app.groupby(["match_id", "team_id"])["started"].transform("sum")
    unknown = app["started"].isna() | (side_sum == 0)
    app["started_inferred"] = unknown
    rank = app[unknown].groupby(["match_id", "team_id"])["minutes"].rank(method="first", ascending=False)
    app.loc[unknown, "started"] = (rank <= 11).astype(float)
    return app


def load_archive_season(season: str, matches: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    d = fetch_archive_season(season, raw_dir)
    gw = pd.read_csv(d / "merged_gw.csv", encoding="latin-1", low_memory=False)
    fx = pd.read_csv(d / "fixtures.csv")
    pr = pd.read_csv(d / "players_raw.csv")
    # FPL team ids are per season: map through that season's teams file.
    teams = pd.read_csv(_get(f"{ARCHIVE}/{season}/teams.csv", d / "teams.csv", pause=0.2))
    tid = dict(zip(teams["id"], _team_ids(teams["name"])))
    fx = fx.rename(columns={"id": "fixture"})
    fx["home_id"], fx["away_id"] = fx["team_h"].map(tid), fx["team_a"].map(tid)
    fx["finished"] = fx["finished"].astype(str).str.lower() == "true"
    fx = _join_matches(fx, matches, season)

    gw = gw[gw["position"].isin(POSITIONS.values())]  # drops "AM" (assistant manager) rows
    meta = pr.set_index("id")
    gw["player_code"] = gw["element"].map(meta["code"])
    if gw["player_code"].isna().any():
        raise FplError(f"{season}: elements missing from players_raw")
    gw["player_name"] = gw["name"]
    gw["started"] = gw["starts"].astype(float) if "starts" in gw else np.nan
    gw = gw.rename(columns=STAT_COLS)
    return _finish(gw, fx, season)


def load_live_season(season: str, matches: pd.DataFrame, raw_dir: Path = RAW_DIR,
                     refresh: bool = True) -> pd.DataFrame:
    d = fetch_live_season(season, raw_dir, refresh)
    boot = json.loads((d / "bootstrap.json").read_text())
    teams = {t["id"]: t["name"] for t in boot["teams"]}
    tid = dict(zip(teams, _team_ids(pd.Series(teams.values()))))
    els = {e["id"]: e for e in boot["elements"]}
    fx = pd.DataFrame(json.loads((d / "fixtures.json").read_text())).rename(columns={"id": "fixture"})
    fx["home_id"], fx["away_id"] = fx["team_h"].map(tid), fx["team_a"].map(tid)
    fx["finished"] = fx["finished"].astype(bool)
    side = {r.fixture: (r.team_h, r.team_a) for r in fx.itertuples()}
    # Which side each player was on, from the fixture's own bonus-point list
    # (it lists everyone who played). The bootstrap's `team` is the player's
    # *current* club, which is wrong for anyone transferred since.
    played_for: dict[tuple[int, int], bool] = {}
    for r in fx.itertuples():
        for st in r.stats or []:
            if st["identifier"] == "bps":
                played_for.update({(r.fixture, x["element"]): True for x in st["h"]})
                played_for.update({(r.fixture, x["element"]): False for x in st["a"]})
    fx = _join_matches(fx, matches, season)

    rows, dropped = [], 0
    for f in sorted(d.glob("gw*.json")):
        for e in json.loads(f.read_text())["elements"]:
            if len(e["explain"]) != 1:
                dropped += int(e["stats"]["minutes"] > 0)  # double gameweek: stats can't be split
                continue
            meta = els.get(e["id"])
            fixture = e["explain"][0]["fixture"]
            if meta is None or meta["element_type"] not in POSITIONS or fixture not in side:
                continue
            h, a = side[fixture]
            was_home = played_for.get((fixture, e["id"]))
            if was_home is None:
                if meta["team"] not in (h, a):
                    dropped += int(e["stats"]["minutes"] > 0)  # side unknown
                    continue
                was_home = meta["team"] == h
            rows.append({"fixture": fixture, "was_home": was_home, "player_code": meta["code"],
                         "player_name": meta["web_name"], "position": POSITIONS[meta["element_type"]],
                         "started": float(e["stats"]["starts"]), **e["stats"]})
    app = pd.DataFrame(rows).rename(columns=STAT_COLS)
    out = _finish(app, fx, season)
    out.attrs["dropped_rows"] = dropped
    return out


def load_appearances(matches: pd.DataFrame, archive_seasons: list[str], live_season: str | None = None,
                     raw_dir: Path = RAW_DIR, refresh_live: bool = True) -> pd.DataFrame:
    parts = [load_archive_season(s, matches, raw_dir) for s in archive_seasons]
    dropped = 0
    if live_season:
        parts.append(load_live_season(live_season, matches, raw_dir, refresh_live))
        dropped = parts[-1].attrs.get("dropped_rows", 0)
    out = pd.concat(parts, ignore_index=True)
    out.attrs["live_dropped_rows"] = dropped
    return out


def validate_appearances(app: pd.DataFrame, matches: pd.DataFrame) -> list[str]:
    """Problems that would make player features wrong. Empty list = OK."""
    problems = []
    covered = matches[matches["season"].isin(app["season"].unique()) & (matches["status"] == "completed")]
    sides = app.groupby(["match_id", "team_id"]).agg(minutes=("minutes", "sum"), starters=("started", "sum"),
                                                     has_starts=("started", lambda s: s.notna().all()))
    expected = pd.concat([covered[["match_id", "home_id"]].set_axis(["match_id", "team_id"], axis=1),
                          covered[["match_id", "away_id"]].set_axis(["match_id", "team_id"], axis=1)])
    missing = expected.set_index(["match_id", "team_id"]).index.difference(sides.index)
    if len(missing):
        problems.append(f"{len(missing)} match sides have no appearances, e.g. {list(missing[:3])}")
    lo = sides[sides["minutes"] < 880]
    if len(lo):
        problems.append(f"{len(lo)} match sides with < 880 player-minutes, e.g. {list(lo.index[:3])}")
    st = sides[sides["has_starts"]]
    off = st[st["starters"] != 11]
    if len(off):
        problems.append(f"{len(off)} match sides without exactly 11 starters, e.g. {list(off.index[:3])}")
    if app.duplicated(["match_id", "player_code"]).any():  # pragma: no cover - deduped on load
        problems.append("duplicate player rows within a match")
    return problems
