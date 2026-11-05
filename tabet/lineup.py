"""Lineup features as departures from a team's *expected* lineup (review item #3).

For team T at a cutoff, each player's start probability is the decayed share
of T's earlier matches he started (preceding selections only). With player
attack ratings r_p (decayed (xG+xA)/90, shrunk to the position prior):

    expected_role(T) = sum_p start_prob(T, p) * r_p      over T's players in role
    actual_role(T)   = sum over T's confirmed starters in role of r_p
    dep_role         = actual - expected                 (a difference, not a ratio)

for role in FWD / MID / DEF. Defensive proxies for the opponent O: the chance
O's starting keeper is *not* the usual one (1 - start_prob) and the mean start
probability of O's starting defenders. Defenders' xG+xA is not treated as a
defensive rating.

Continuity-only comparator: mean start probability of T's outfield starters,
plus the two opponent proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureParams, known, player_ratings

LN2 = np.log(2)
ROLES = ("FWD", "MID", "DEF")
DEPARTURE_COLS = [f"l_dep_{r.lower()}" for r in ROLES] + ["l_opp_gk_change", "l_opp_def_cont"]
CONTINUITY_COLS = ["l_own_cont", "l_opp_gk_change", "l_opp_def_cont"]


def start_probabilities(hist: pd.DataFrame, cutoff: pd.Timestamp, half_life: float) -> pd.DataFrame:
    """Index (team_id, player_code) -> start_prob, position."""
    age = (cutoff - hist["kickoff_utc"]).dt.total_seconds() / 86400
    h = hist.assign(w=np.exp(-LN2 * age.to_numpy() / half_life))
    team_w = h.groupby(["team_id", "match_id"])["w"].first().groupby("team_id").sum()
    sp = h.assign(ws=h["w"] * h["started"]).groupby(["team_id", "player_code"]).agg(
        ws=("ws", "sum"), position=("position", "last"))
    sp["start_prob"] = sp["ws"] / sp.index.get_level_values(0).map(team_w).to_numpy()
    return sp[["start_prob", "position"]]


def add_lineup_features(rows: pd.DataFrame, app: pd.DataFrame, p: FeatureParams = FeatureParams()) -> pd.DataFrame:
    out = rows.copy()
    for c in DEPARTURE_COLS + CONTINUITY_COLS:
        out[c] = np.nan
    for cutoff, grp in out.groupby("cutoff_utc"):
        hist = known(app, cutoff)
        rt = player_ratings(hist, cutoff, p, "position")
        fallback = rt["prior"].groupby(rt["position"]).first() if len(rt) else pd.Series(dtype=float)
        sp = start_probabilities(hist, cutoff, p.team_half_life)
        xi = app[app["match_id"].isin(grp["match_id"]) & (app["started"] == 1)]
        for i, r in grp.iterrows():
            mine = xi[(xi["match_id"] == r["match_id"]) & (xi["team_id"] == r["team_id"])]
            theirs = xi[(xi["match_id"] == r["match_id"]) & (xi["team_id"] == r["opponent_id"])]
            has_t = r["team_id"] in sp.index.get_level_values(0)
            has_o = r["opponent_id"] in sp.index.get_level_values(0)

            def rating(codes, pos):
                v = rt["rating"].reindex(codes)
                return v.fillna(fallback.get(pos, rt["prior"].median() if len(rt) else 0.0))

            if has_t:
                spt = sp.loc[r["team_id"]]
                for role in ROLES:
                    act = rating(mine.loc[mine["position"] == role, "player_code"], role).sum()
                    pool = spt[spt["position"] == role]
                    exp_ = (pool["start_prob"] * rating(pool.index, role).to_numpy()).sum()
                    out.at[i, f"l_dep_{role.lower()}"] = act - exp_
                own = spt["start_prob"].reindex(mine.loc[mine["position"] != "GK", "player_code"]).fillna(0)
                out.at[i, "l_own_cont"] = own.mean() if len(own) else np.nan
            else:
                for role in ROLES:
                    out.at[i, f"l_dep_{role.lower()}"] = 0.0
            if has_o:
                spo = sp.loc[r["opponent_id"]]["start_prob"]
                gk = theirs.loc[theirs["position"] == "GK", "player_code"]
                out.at[i, "l_opp_gk_change"] = 1 - spo.reindex(gk).fillna(0).max() if len(gk) else np.nan
                de = theirs.loc[theirs["position"] == "DEF", "player_code"]
                out.at[i, "l_opp_def_cont"] = spo.reindex(de).fillna(0).mean() if len(de) else np.nan
    return out
