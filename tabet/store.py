"""Point-in-time access to match history.

Main rule: nothing returned for a cutoff may depend on information that
wasn't available before that cutoff. A result counts as available
RESULT_LAG after kickoff, which conservatively covers 90 minutes, half-time,
stoppage time and publication delay.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ingest import load_all
from .teams import resolve_team

RESULT_LAG = pd.Timedelta(hours=2, minutes=30)


def parse_cutoff(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"cutoff {value!r} must include a timezone, e.g. 2026-11-22T15:00Z")
    return ts.tz_convert("UTC")


class MatchStore:
    def __init__(self, matches: pd.DataFrame):
        self._m = matches.sort_values("kickoff_utc", kind="stable").reset_index(drop=True)

    @classmethod
    def from_dir(cls, res_dir: str | Path) -> "MatchStore":
        matches, _ = load_all(res_dir)
        return cls(matches)

    def completed_before(self, cutoff: str | pd.Timestamp) -> pd.DataFrame:
        """All results known before cutoff."""
        cutoff = parse_cutoff(cutoff)
        m = self._m
        known = (m["status"] == "completed") & (m["kickoff_utc"] + RESULT_LAG <= cutoff)
        return m[known].copy()

    def get_history(self, team: str, cutoff: str | pd.Timestamp, last: int | None = None) -> pd.DataFrame:
        """One team's completed matches before cutoff, from that team's perspective,
        oldest first. `last` keeps only the most recent N."""
        team_id = resolve_team(team)
        m = self.completed_before(cutoff)

        def side(df: pd.DataFrame, us: str, them: str, venue: str) -> pd.DataFrame:
            return pd.DataFrame({
                "match_id": df["match_id"],
                "season": df["season"],
                "kickoff_utc": df["kickoff_utc"],
                "venue": venue,
                "opponent_id": df[f"{them}_id"],
                "goals_for": df[f"{us}_goals"],
                "goals_against": df[f"{them}_goals"],
                "shots_for": df[f"{us}_shots"],
                "shots_against": df[f"{them}_shots"],
                "sot_for": df[f"{us}_shots_on_target"],
                "sot_against": df[f"{them}_shots_on_target"],
            })

        h = side(m[m["home_id"] == team_id], "home", "away", "H")
        a = side(m[m["away_id"] == team_id], "away", "home", "A")
        hist = pd.concat([h, a]).sort_values("kickoff_utc", kind="stable").reset_index(drop=True)
        diff = hist["goals_for"] - hist["goals_against"]
        hist["outcome"] = diff.map(lambda d: "W" if d > 0 else ("L" if d < 0 else "D")).astype(str)
        hist["points"] = hist["outcome"].map({"W": 3, "D": 1, "L": 0}).astype(int)
        hist.insert(0, "team_id", team_id)
        if last is not None:
            hist = hist.tail(last).reset_index(drop=True)
        return hist
