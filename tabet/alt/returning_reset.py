"""Option 2: returning-team reset.

The frozen Dixon-Coles model keeps one rating per club for its whole history, so a club that is
relegated and comes back (Burnley in 2023-24 and 2025-26) starts its new season from decayed
ratings earned by a different squad. Here every *spell* in the league is a separate team: a club
that was absent from the previous season in the data starts a new spell ("burnley#2"), with no
memory of the old one. Which clubs are in a season is known before it starts, so this uses no
future information.

Two variants:
  reset           new spells start at league average (rating 0), like any unseen team
  reset_offset    new spells also get the shared promoted-team offset learned from earlier
                  promoted sides (the D1 model of section 5.15)
"""

from __future__ import annotations

import pandas as pd

from ..ingest import LOCAL_TZ
from ..model import DixonColes, Hyperparams, promoted_team_seasons
from ..store import MatchStore


def spell_ids(matches: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(team, season) -> spell id. A new spell starts when the team was absent from the previous
    season in the data; the team's first season in the data is spell 1 (the plain team id)."""
    seasons = sorted(matches["season"].unique())
    members = {s: set(matches.loc[matches["season"] == s, "home_id"]) | set(matches.loc[matches["season"] == s, "away_id"])
               for s in seasons}
    out, spell = {}, {}
    for i, s in enumerate(seasons):
        for t in sorted(members[s]):
            if t not in spell:
                spell[t] = 1
            elif i > 0 and t not in members[seasons[i - 1]]:
                spell[t] += 1
            out[(t, s)] = t if spell[t] == 1 else f"{t}#{spell[t]}"
    return out


def relabel(matches: pd.DataFrame) -> pd.DataFrame:
    sp = spell_ids(matches)
    m = matches.copy()
    m["home_id"] = [sp[(t, s)] for t, s in zip(m["home_id"], m["season"])]
    m["away_id"] = [sp[(t, s)] for t, s in zip(m["away_id"], m["season"])]
    return m


def dc_walk_forward(matches: pd.DataFrame, eval_seasons: tuple[str, ...], hyper: Hyperparams) -> pd.DataFrame:
    """Walk-forward Dixon-Coles that also returns the pre-match rates (for stage 2)."""
    store = MatchStore(matches)
    promoted = promoted_team_seasons(matches)
    target = matches[matches["season"].isin(eval_seasons) & (matches["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    out = []
    for _, g in target.groupby(day, sort=True):
        cutoff = g["kickoff_utc"].min()
        dc = DixonColes(hyper).fit(store.completed_before(cutoff), cutoff, promoted)
        for f in g.itertuples():
            lh, la = dc.expected_goals(f.home_id, f.away_id, f.season)
            ph, pd_, pa = dc.predict_proba(f.home_id, f.away_id, f.season)
            out.append({"match_id": f.match_id, "season": f.season, "kickoff_utc": f.kickoff_utc, "result": f.result,
                        "lam_home": lh, "lam_away": la, "rho": dc.rho, "p_home": ph, "p_draw": pd_, "p_away": pa})
    return pd.DataFrame(out)


def walk_forward(matches: pd.DataFrame, eval_seasons: tuple[str, ...], offset: bool = False) -> pd.DataFrame:
    return dc_walk_forward(relabel(matches), eval_seasons, Hyperparams(promoted_prior=offset))


def returning_spells(matches: pd.DataFrame) -> list[str]:
    """Spell ids other than a club's first (i.e. the clubs this reset actually changes)."""
    return sorted({v for v in spell_ids(matches).values() if "#" in v})

