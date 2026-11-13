"""Pre-xG seasons (2015-16 -> 2020-21) for the out-of-sample replication.

Kept apart from the live model: results live in res/history/ (never read by
load_all("res")), and nothing here is used by forecasts.

FPL archive (vaastav) for 2016-17 and 2017-18 has only gws/merged_gw.csv and
players_raw.csv: no fixtures or teams file. Each fixture's two FPL team ids are
recovered from the players' `opponent_team` and `was_home`, and FPL ids are
mapped to our team ids by voting on (local date, score) against the
football-data results. Every fixture must then join one-to-one with identical
scores, or loading fails. There is no xG and no `starts`: starters are inferred
as the 11 players with most minutes (flagged).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fpl import APPEARANCE_COLUMNS, POSITIONS, RAW_DIR, FplError, _get, _infer_missing_starts
from .ingest import load_all

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "res" / "history"
SEASONS = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21")
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def load_history_matches() -> pd.DataFrame:
    m, _ = load_all(HISTORY_DIR, fixes_path=None)
    return m


def _fixture_teams(gw: pd.DataFrame) -> pd.DataFrame:
    home = gw["was_home"].astype(str).str.lower() == "true"
    fx = gw.groupby("fixture").agg(kickoff=("kickoff_time", "first"), h=("team_h_score", "first"),
                                   a=("team_a_score", "first"))
    fx["fpl_away"] = gw[home].groupby("fixture")["opponent_team"].agg(lambda s: s.mode().iloc[0])
    fx["fpl_home"] = gw[~home].groupby("fixture")["opponent_team"].agg(lambda s: s.mode().iloc[0])
    fx["date"] = pd.to_datetime(fx["kickoff"], utc=True).dt.tz_convert("Europe/London").dt.date
    return fx.reset_index()


def _map_teams(fx: pd.DataFrame, ours: pd.DataFrame, season: str) -> dict[int, str]:
    o = ours.assign(date=ours["kickoff_utc"].dt.tz_convert("Europe/London").dt.date)
    j = fx.merge(o, left_on=["date", "h", "a"], right_on=["date", "home_goals", "away_goals"])
    votes = pd.concat([j[["fpl_home", "home_id"]].set_axis(["fpl", "ours"], axis=1),
                       j[["fpl_away", "away_id"]].set_axis(["fpl", "ours"], axis=1)])
    best = votes.value_counts().reset_index().drop_duplicates("fpl").drop_duplicates("ours")
    mp = dict(zip(best["fpl"].astype(int), best["ours"]))
    if len(mp) != 20:
        raise FplError(f"{season}: mapped {len(mp)} of 20 FPL team ids")
    return mp


def load_history_season(season: str, matches: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    d = raw_dir / season
    gw = pd.read_csv(_get(f"{BASE}/{season}/gws/merged_gw.csv", d / "merged_gw.csv"), encoding="latin-1",
                     low_memory=False)
    pr = pd.read_csv(_get(f"{BASE}/{season}/players_raw.csv", d / "players_raw.csv")).set_index("id")
    ours = matches[(matches["season"] == season) & (matches["status"] == "completed")]
    fx = _fixture_teams(gw)
    mp = _map_teams(fx, ours, season)
    fx["home_id"], fx["away_id"] = fx["fpl_home"].map(mp), fx["fpl_away"].map(mp)
    fx = fx.merge(ours[["match_id", "home_id", "away_id", "kickoff_utc", "home_goals", "away_goals"]],
                  on=["home_id", "away_id"], how="left", validate="one_to_one")
    bad = fx["match_id"].isna() | (fx["h"] != fx["home_goals"]) | (fx["a"] != fx["away_goals"])
    if bad.any() or len(fx) != len(ours):
        raise FplError(f"{season}: {int(bad.sum())} fixtures failed to join ({len(fx)} vs {len(ours)} matches)")

    gw = gw.merge(fx[["fixture", "match_id", "kickoff_utc", "home_id", "away_id"]], on="fixture")
    home = gw["was_home"].astype(str).str.lower() == "true"
    gw["team_id"] = np.where(home, gw["home_id"], gw["away_id"])
    gw["opponent_id"] = np.where(home, gw["away_id"], gw["home_id"])
    gw["venue"] = np.where(home, "H", "A")
    gw["player_code"] = gw["element"].map(pr["code"])
    if gw["player_code"].isna().any():
        raise FplError(f"{season}: elements missing from players_raw")
    gw["position"] = gw["element"].map(pr["element_type"]).map(POSITIONS)
    gw["player_name"] = gw["name"].astype(str).str.replace("_", " ").str.replace(r"\s\d+$", "", regex=True)
    gw["season"] = season
    gw["started"] = np.nan
    gw = gw.rename(columns={"goals_scored": "goals"})
    for c in ("xg", "xa", "xgc", "def_contrib"):
        gw[c] = np.nan
    gw["started_inferred"] = True
    app = gw[[c for c in APPEARANCE_COLUMNS]].copy()
    app["player_code"] = app["player_code"].astype(int)
    app["minutes"] = app["minutes"].astype(float)
    return app


def load_history_appearances(matches: pd.DataFrame, seasons=SEASONS) -> pd.DataFrame:
    app = pd.concat([load_history_season(s, matches) for s in seasons], ignore_index=True)
    # 2019-20: the archive repeats some players with 0 minutes in the renumbered post-restart
    # gameweeks. Keep the row with minutes; two rows that both have minutes is an error.
    played = app[app["minutes"] > 0]
    if played.duplicated(["match_id", "player_code"]).any():
        raise FplError("conflicting duplicate player-match rows with minutes")
    app = (app.sort_values("minutes", ascending=False).drop_duplicates(["match_id", "player_code"])
           .sort_values(["kickoff_utc", "match_id", "team_id", "player_code"]).reset_index(drop=True))
    return _infer_missing_starts(app.assign(started=np.nan))
