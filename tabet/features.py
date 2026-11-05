"""Point-in-time match features from FPL player appearances and match stats.

Every function takes a `cutoff` and uses only rows whose result was known by
then (kickoff + RESULT_LAG <= cutoff). The one exception is the starting XI
of the match being predicted, which is treated as a *confirmed lineup*
(published about an hour before kickoff). Results here therefore describe
"confirmed-lineup mode"; projected lineups are not evaluated yet.

Feature groups, per scoring side (team T attacking opponent O):
  xg_form   log of T's decayed xG-for per match, and O's decayed xG-against,
            relative to the league average.
  xi        log(XI attack strength / T's usual attack strength), where a
            player's attack rating is decayed (xG + xA) per 90 shrunk toward a
            prior; plus O's back-line regularity (how often the starting GK/DEF
            usually play).
  style     z-scored team style (shot volume, xG per shot, corners, fouls),
            as main effects and attack-style x opponent-defence-style products.

Player priors ("prior" argument):
  position  shrink toward the minutes-weighted mean for the player's position.
  similar   shrink toward the mean of the K most similar established players,
            by per-90 FPL threat / creativity / influence profile within the
            same position. Players without a profile fall back to position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .store import RESULT_LAG

LN2 = np.log(2)


@dataclass(frozen=True)
class FeatureParams:
    player_half_life: float = 365.0  # days, for player ratings
    team_half_life: float = 120.0  # days, for usual XI, xG form and style
    shrink_minutes: float = 900.0  # prior weight, in minutes
    knn: int = 10
    min_neighbour_minutes: float = 900.0


def _w(age_days: pd.Series | np.ndarray, half_life: float) -> np.ndarray:
    return np.exp(-LN2 * np.asarray(age_days, dtype=float) / half_life)


def known(app: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return app[app["kickoff_utc"] + RESULT_LAG <= cutoff]


# ---- player ratings ---------------------------------------------------------

def player_ratings(hist: pd.DataFrame, cutoff: pd.Timestamp, p: FeatureParams,
                   prior: str = "position") -> pd.DataFrame:
    """One row per player_code: decayed minutes, raw and shrunk attack rate
    (xG+xA per 90), position and the prior used."""
    age = (cutoff - hist["kickoff_utc"]).dt.total_seconds() / 86400
    h = hist.assign(w=_w(age, p.player_half_life))
    h["wmin"] = h["w"] * h["minutes"]
    xg_era = h["xg"].notna()
    h["wmin_xg"] = h["wmin"].where(xg_era, 0.0)
    h["wout"] = (h["w"] * (h["xg"] + h["xa"])).where(xg_era, 0.0)
    for c in ("threat", "creativity", "influence"):
        h[f"w_{c}"] = h["w"] * h[c]
    g = h.groupby("player_code")
    pt = g[["wmin", "wmin_xg", "wout", "w_threat", "w_creativity", "w_influence"]].sum()
    pt["position"] = g["position"].last()
    pt["rate_raw"] = pt["wout"] / (pt["wmin_xg"] / 90)

    has = pt["wmin_xg"] > 0
    pos_prior = (pt[has].groupby("position")["wout"].sum()
                 / (pt[has].groupby("position")["wmin_xg"].sum() / 90))
    pt["prior"] = pt["position"].map(pos_prior).fillna(pos_prior.mean() if len(pos_prior) else 0.0)

    if prior == "similar":
        pt["prior"] = _similar_prior(pt, p)
    elif prior != "position":
        raise ValueError(f"unknown prior {prior!r}")

    k = p.shrink_minutes / 90
    pt["rating"] = (pt["wout"] + pt["prior"] * k) / (pt["wmin_xg"] / 90 + k)
    return pt


def _similar_prior(pt: pd.DataFrame, p: FeatureParams) -> pd.Series:
    """Mean raw rate of the K nearest established players in the same position,
    by per-90 ICT profile. ICT exists in every season (including 2021-22, which
    has no xG), so a player can have a profile before he has an xG history."""
    prof_cols = ["w_threat", "w_creativity", "w_influence"]
    prof = pt[prof_cols].div(pt["wmin"].clip(lower=1) / 90, axis=0)
    out = pt["prior"].copy()
    for pos, grp in pt.groupby("position"):
        pool = grp[grp["wmin_xg"] >= p.min_neighbour_minutes]
        target = grp[grp["wmin"] >= 90]  # needs at least some profile
        if len(pool) <= p.knn or target.empty:
            continue
        mu, sd = prof.loc[pool.index].mean(), prof.loc[pool.index].std().replace(0, 1)
        Z = ((prof.loc[pool.index] - mu) / sd).to_numpy()
        T = ((prof.loc[target.index] - mu) / sd).to_numpy()
        d = ((T[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
        # exclude the player himself from his own neighbours
        self_mask = target.index.to_numpy()[:, None] == pool.index.to_numpy()[None, :]
        d[self_mask] = np.inf
        nn = np.argsort(d, axis=1)[:, :p.knn]
        out.loc[target.index] = pool["rate_raw"].to_numpy()[nn].mean(axis=1)
    return out


# ---- team-level state -------------------------------------------------------

def team_state(hist_app: pd.DataFrame, hist_matches: pd.DataFrame, cutoff: pd.Timestamp,
               p: FeatureParams) -> dict[str, pd.DataFrame]:
    """Decayed team summaries: per-player usual share/regularity, xG form, style."""
    age = (cutoff - hist_app["kickoff_utc"]).dt.total_seconds() / 86400
    a = hist_app.assign(w=_w(age, p.team_half_life))
    a["wmin"] = a["w"] * a["minutes"]

    # usual minutes share per team-player (outfield), and regularity (GK/DEF)
    team_wm = a.groupby(["team_id", "match_id"])["w"].first().groupby("team_id").sum() * 90
    usage = a.groupby(["team_id", "player_code"]).agg(wmin=("wmin", "sum"), position=("position", "last"))
    usage["regularity"] = usage["wmin"] / usage.index.get_level_values(0).map(team_wm).to_numpy()
    of = usage[usage["position"] != "GK"]
    usage["out_share"] = (of["wmin"] / of.groupby(level=0)["wmin"].transform("sum") * 10).reindex(usage.index)

    # xG form from FPL sums (xG era only)
    xa = a[a["xg"].notna()]
    side = xa.groupby(["match_id", "team_id", "opponent_id"]).agg(xg=("xg", "sum"), w=("w", "first")).reset_index()
    opp = side.rename(columns={"team_id": "opponent_id", "opponent_id": "team_id", "xg": "xga"})[
        ["match_id", "team_id", "xga"]]
    side = side.merge(opp, on=["match_id", "team_id"])
    form = side.groupby("team_id").apply(
        lambda s: pd.Series({"xgf": np.average(s["xg"], weights=s["w"]),
                             "xga": np.average(s["xga"], weights=s["w"]), "wsum": s["w"].sum()}),
        include_groups=False) if len(side) else pd.DataFrame(columns=["xgf", "xga", "wsum"])
    league_xg = float(np.average(side["xg"], weights=side["w"])) if len(side) else np.nan

    # style from match stats (+ xG per shot from FPL)
    m = hist_matches
    mage = (cutoff - m["kickoff_utc"]).dt.total_seconds() / 86400
    mw = _w(mage, p.team_half_life)
    rows = []
    for us, them in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "match_id": m["match_id"], "team_id": m[f"{us}_id"], "w": mw,
            "shots_for": m[f"{us}_shots"], "shots_against": m[f"{them}_shots"],
            "corners_for": m[f"{us}_corners"], "corners_against": m[f"{them}_corners"],
            "fouls": m[f"{us}_fouls"]}))
    st = pd.concat(rows).merge(side[["match_id", "team_id", "xg", "xga"]], on=["match_id", "team_id"], how="left")
    num = ["shots_for", "shots_against", "corners_for", "corners_against", "fouls", "xg", "xga"]
    st[num] = st[num].astype(float)

    def wavg(s: pd.DataFrame) -> pd.Series:
        out = {}
        for c in num:
            ok = s[c].notna()
            out[c] = np.average(s.loc[ok, c], weights=s.loc[ok, "w"]) if ok.any() else np.nan
        return pd.Series(out)

    style = st.groupby("team_id").apply(wavg, include_groups=False)
    style["xg_per_shot_for"] = style["xg"] / style["shots_for"]
    style["xg_per_shot_against"] = style["xga"] / style["shots_against"]
    return {"usage": usage, "form": form, "league_xg": league_xg, "style": style}


STYLE_COLS = ["shots_for", "shots_against", "corners_for", "corners_against", "fouls",
              "xg_per_shot_for", "xg_per_shot_against"]


def match_features(fixtures: pd.DataFrame, lineups: pd.DataFrame, hist_app: pd.DataFrame,
                   hist_matches: pd.DataFrame, cutoff: pd.Timestamp, p: FeatureParams,
                   priors: tuple[str, ...] = ("position", "similar")) -> pd.DataFrame:
    """Two rows per fixture (home scoring, away scoring) with all feature groups.
    `lineups` = appearance rows of the fixtures themselves (for `started` only)."""
    state = team_state(hist_app, hist_matches, cutoff, p)
    rats = {pr: player_ratings(hist_app, cutoff, p, pr) for pr in priors}
    usage, form, style = state["usage"], state["form"], state["style"]
    cur = sorted(set(fixtures["home_id"]) | set(fixtures["away_id"]) | set(style.index))
    sz = style.reindex(cur)[STYLE_COLS]
    z = (sz - sz.mean()) / sz.std().replace(0, 1)
    starters = lineups[lineups["started"] == 1]

    rows = []
    for fx in fixtures.itertuples():
        for team, opp, venue in ((fx.home_id, fx.away_id, "H"), (fx.away_id, fx.home_id, "A")):
            r: dict[str, float | str] = {"match_id": fx.match_id, "side": venue, "team_id": team, "opponent_id": opp}
            # xG form
            lx = state["league_xg"]
            r["f_xgf"] = np.log(form.at[team, "xgf"] / lx) if team in form.index and lx > 0 else 0.0
            r["f_xga_opp"] = np.log(form.at[opp, "xga"] / lx) if opp in form.index and lx > 0 else 0.0
            # XI attack strength vs usual
            xi_t = starters[(starters["match_id"] == fx.match_id) & (starters["team_id"] == team)]
            xi_o = starters[(starters["match_id"] == fx.match_id) & (starters["team_id"] == opp)]
            u_t = usage.loc[team] if team in usage.index.get_level_values(0) else None
            for pr, rt in rats.items():
                fallback = float(rt["prior"].median()) if len(rt) else 0.0
                out_xi = xi_t[xi_t["position"] != "GK"]["player_code"]
                xi_att = rt["rating"].reindex(out_xi).fillna(fallback).sum()
                if u_t is not None and u_t["out_share"].notna().any():
                    sh = u_t["out_share"].dropna()
                    usual = (sh * rt["rating"].reindex(sh.index).fillna(fallback)).sum()
                    r[f"f_xi_{pr}"] = float(np.log(xi_att / usual)) if usual > 0 and xi_att > 0 else 0.0
                else:
                    r[f"f_xi_{pr}"] = 0.0
            # opponent back-line regularity (a weakened defence -> we score more)
            u_o = usage.loc[opp] if opp in usage.index.get_level_values(0) else None
            bl = xi_o[xi_o["position"].isin(["GK", "DEF"])]["player_code"]
            if u_o is not None and len(bl):
                r["f_backline_opp"] = float(u_o["regularity"].reindex(bl).fillna(0).clip(0, 1).mean())
            else:
                r["f_backline_opp"] = np.nan  # filled with the training mean by the model
            # style
            za, zd = z.loc[team] if team in z.index else None, z.loc[opp] if opp in z.index else None
            for c in STYLE_COLS:
                r[f"s_{c}"] = float(za[c]) if za is not None and pd.notna(za[c]) else 0.0
                r[f"so_{c}"] = float(zd[c]) if zd is not None and pd.notna(zd[c]) else 0.0
            r["x_volume"] = r["s_shots_for"] * r["so_shots_against"]
            r["x_quality"] = r["s_xg_per_shot_for"] * r["so_xg_per_shot_against"]
            r["x_corners"] = r["s_corners_for"] * r["so_corners_against"]
            r["x_vol_vs_quality"] = r["s_shots_for"] * r["so_xg_per_shot_against"]
            r["x_quality_vs_vol"] = r["s_xg_per_shot_for"] * r["so_shots_against"]
            rows.append(r)
    return pd.DataFrame(rows)
