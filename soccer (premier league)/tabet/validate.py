"""Data-quality checks for the normalised match and odds tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TEAMS_PER_SEASON = 20
MATCHES_PER_SEASON = TEAMS_PER_SEASON * (TEAMS_PER_SEASON - 1)


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    check: str
    message: str
    season: str | None = None

    def __str__(self) -> str:
        where = f"[{self.season}] " if self.season else ""
        return f"{self.level.upper():7} {where}{self.check}: {self.message}"


def _ids(df: pd.DataFrame, mask: pd.Series, limit: int = 5) -> str:
    ids = df.loc[mask.fillna(False).astype(bool), "match_id"].tolist()
    return ", ".join(ids[:limit]) + (f" (+{len(ids) - limit} more)" if len(ids) > limit else "")


def _outcome(home: pd.Series, away: pd.Series) -> np.ndarray:
    return np.select([home > away, home < away], ["H", "A"], "D")


def _row_checks(m: pd.DataFrame, now: pd.Timestamp) -> list[Issue]:
    issues: list[Issue] = []

    def flag(check: str, mask: pd.Series, msg: str, level: str = "error") -> None:
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            issues.append(Issue(level, check, f"{mask.sum()} row(s) {msg}: {_ids(m, mask)}"))

    flag("duplicate_match", m["match_id"].duplicated(keep=False), "share a match_id")
    flag("self_match", m["home_id"] == m["away_id"], "have the same home and away team")
    flag("missing_kickoff", m["kickoff_utc"].isna(), "have no valid kickoff time")

    done = m["status"] == "completed"
    c = m[done]
    expected = pd.Series(_outcome(c["home_goals"].astype(int), c["away_goals"].astype(int)), index=c.index)
    flag("result_mismatch", done & (m["result"] != expected.reindex(m.index)), "have FTR inconsistent with the score")
    flag("bad_result_code", ~m["result"].isin(["H", "D", "A", ""]), "have an FTR other than H/D/A")

    ht = done & m["ht_home_goals"].notna() & m["ht_away_goals"].notna() & (m["ht_result"] != "")
    h = m[ht]
    ht_expected = pd.Series(_outcome(h["ht_home_goals"].astype(int), h["ht_away_goals"].astype(int)), index=h.index)
    flag("ht_result_mismatch", ht & (m["ht_result"] != ht_expected.reindex(m.index)), "have HTR inconsistent with the half-time score")
    flag("ht_exceeds_ft", (m["ht_home_goals"] > m["home_goals"]) | (m["ht_away_goals"] > m["away_goals"]), "have more half-time goals than full-time")

    count_cols = [c for c in m.columns if m[c].dtype == "Int64"]
    neg = pd.concat([m[c] < 0 for c in count_cols], axis=1).any(axis=1)
    flag("negative_count", neg, "contain negative counts")
    flag("sot_exceeds_shots", (m["home_shots_on_target"] > m["home_shots"]) | (m["away_shots_on_target"] > m["away_shots"]), "have more shots on target than shots")
    flag("goals_exceed_sot", (m["home_goals"] > m["home_shots_on_target"]) | (m["away_goals"] > m["away_shots_on_target"]),
         "have more goals than shots on target (possible with own goals; check)", level="warning")

    flag("future_result", done & (m["kickoff_utc"] > now), "are marked completed but kick off in the future")
    return issues


def _season_checks(m: pd.DataFrame, latest_season: str) -> list[Issue]:
    issues: list[Issue] = []
    for season, s in m.groupby("season", sort=True):
        def add(level: str, check: str, msg: str) -> None:
            issues.append(Issue(level, check, msg, season=season))

        start = int(season[:4])
        lo = pd.Timestamp(f"{start}-07-01", tz="UTC")
        hi = pd.Timestamp(f"{start + 1}-08-01", tz="UTC")
        outside = (s["kickoff_utc"] < lo) | (s["kickoff_utc"] >= hi)
        if outside.any():
            add("error", "date_outside_season", f"{outside.sum()} match(es) outside {lo.date()}..{hi.date()}: {_ids(s, outside)}")

        teams = set(s["home_id"]) | set(s["away_id"])
        pairs = s.groupby(["home_id", "away_id"]).size()
        repeated = pairs[pairs > 1]
        if len(repeated):
            add("error", "repeated_fixture", f"home/away pairing appears more than once: {list(repeated.index[:5])}")
        for side in ("home_id", "away_id"):
            counts = s[side].value_counts()
            over = counts[counts > TEAMS_PER_SEASON - 1]
            if len(over):
                add("error", "too_many_fixtures", f"teams with >{TEAMS_PER_SEASON - 1} {side[:4]} games: {over.to_dict()}")

        n_done = int((s["status"] == "completed").sum())
        complete = len(s) == MATCHES_PER_SEASON and n_done == MATCHES_PER_SEASON and len(teams) == TEAMS_PER_SEASON
        if complete:
            continue
        if season == latest_season and len(teams) <= TEAMS_PER_SEASON and len(s) <= MATCHES_PER_SEASON:
            add("warning", "partial_season", f"in progress: {n_done} completed, {len(s) - n_done} scheduled, of {MATCHES_PER_SEASON}")
        else:
            add("error", "incomplete_season", f"{len(teams)} teams, {len(s)} rows, {n_done} completed (expected {TEAMS_PER_SEASON}/{MATCHES_PER_SEASON}/{MATCHES_PER_SEASON})")
    return issues


def _odds_checks(matches: pd.DataFrame, odds: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    orphan = ~odds["match_id"].isin(matches["match_id"])
    if orphan.any():
        issues.append(Issue("error", "orphan_odds", f"{orphan.sum()} odds row(s) with no match"))
    # Decimal odds are > 1. Skip handicap lines (AHh, AHCh) and totals labels.
    price_cols = [c for c in odds.columns if c != "match_id" and not c.startswith("AH") and not c.endswith("AHh")]
    for c in price_cols:
        bad = odds[c].notna() & (odds[c] <= 1.0)
        if bad.any():
            issues.append(Issue("error", "invalid_odds", f"{c}: {bad.sum()} value(s) <= 1.0"))
    return issues


def validate(matches: pd.DataFrame, odds: pd.DataFrame | None = None, now: pd.Timestamp | None = None) -> list[Issue]:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    latest = max(matches["season"]) if len(matches) else ""
    issues = _row_checks(matches, now) + _season_checks(matches, latest)
    if odds is not None:
        issues += _odds_checks(matches, odds)
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)
