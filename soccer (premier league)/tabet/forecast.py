"""Forecast upcoming fixtures with the lineup-free working model (V1x).

V1x = Dixon-Coles + opponent-adjusted xG strength (stage-2 Poisson adjustment).
The XI-strength model (V2) needs confirmed lineups, which don't exist before
team news, so pre-lineup forecasts use V1x.

Each run is fitted at `cutoff` (default: now) on results known by then and is
saved with its cutoff, model settings and git commit, so the saved files form
a log of forecasts made before kickoff.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import _fit_poisson, build_dataset
from .fpl import RAW_DIR, fetch_live_season, load_appearances
from .ingest import load_all
from .model import MAX_GOALS, DixonColes, Hyperparams, promoted_team_seasons
from .store import RESULT_LAG, MatchStore
from .teams import TEAMS, try_resolve_team
from .xg_strength import add_xg_strength, fit_xg_strength, side_xg
from .features import FeatureParams, known, match_features
from .lineup_sources import availability_field, fpl_players_now, project_xi, snapshot, unavailable_codes

ROOT = Path(__file__).resolve().parent.parent
XG_HALF_LIFE = 90.0  # chosen on dev (results/xg_strength)
STAGE2_L2 = 1.0  # chosen on dev (results/xg_strength)
ADJ = [f"f_xgadj_att_{int(XG_HALF_LIFE)}", f"f_xgadj_def_opp_{int(XG_HALF_LIFE)}"]
XI = ["f_xi_position", "f_backline_opp"]
STAGE2_L2_XI = 100.0  # chosen on dev for the XI model (results/fpl_v0_v5/manifest.json)
ARCHIVE_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
LIVE_SEASON = "2026-27"


def upcoming_fixtures(season: str = LIVE_SEASON, refresh: bool = True) -> pd.DataFrame:
    d = fetch_live_season(season, refresh=refresh)
    boot = json.loads((d / "bootstrap.json").read_text())
    tid = {t["id"]: try_resolve_team(t["name"]) for t in boot["teams"]}
    fx = pd.DataFrame(json.loads((d / "fixtures.json").read_text()))
    fx = fx[~fx["finished"].astype(bool) & fx["kickoff_time"].notna()]
    out = pd.DataFrame({"fpl_fixture": fx["id"], "gameweek": fx["event"].astype(int),
                        "kickoff_utc": pd.to_datetime(fx["kickoff_time"], utc=True),
                        "home_id": fx["team_h"].map(tid), "away_id": fx["team_a"].map(tid)})
    out["season"] = season
    deadlines = {e["id"]: e["deadline_time"] for e in boot["events"]}
    out["gw_deadline_utc"] = out["gameweek"].map(deadlines)
    return out.sort_values(["kickoff_utc", "home_id"]).reset_index(drop=True)


def training_rows(matches: pd.DataFrame, app: pd.DataFrame, cache: Path | None = None) -> pd.DataFrame:
    """Walk-forward rows with pre-match DC offsets and adjusted-xG features.
    The cache is keyed by the set of completed matches, so it is rebuilt
    whenever a new result arrives (a stale cache would leave new matches out
    of stage-2 training)."""
    done = matches[matches["status"] == "completed"]
    key = f"{len(done)}_{done['match_id'].max()}"
    path = None
    if cache is not None:
        path = cache.with_name(f"{cache.stem}_{key}.pkl")
        if path.exists():
            rows = pd.read_pickle(path)
            if set(ADJ) <= set(rows.columns):
                return rows
    rows, _ = build_dataset(matches, app)
    rows = add_xg_strength(rows, app, (XG_HALF_LIFE,))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in path.parent.glob(f"{cache.stem}_*.pkl"):
            old.unlink()
        rows.to_pickle(path)
    return rows


def _score_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    from scipy.stats import poisson
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def _hda(m: np.ndarray) -> tuple[float, float, float]:
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def proj_usual_codes(hist: pd.DataFrame, cutoff: pd.Timestamp, threshold: float = 0.5) -> set[int]:
    """Players who usually start (start probability >= threshold) for their latest team."""
    from .lineup import start_probabilities
    sp = start_probabilities(hist, cutoff, 120.0).reset_index()
    return set(sp.loc[sp["start_prob"] >= threshold, "player_code"].astype(int))


def _bp_l3(dc, train: pd.DataFrame, cutoff: pd.Timestamp) -> float | None:
    """Shared-goal rate of the bivariate Poisson (amendment A5), fitted at the cutoff on the training
    matches' Dixon-Coles rates with the addendum-M decay. None if the fit fails: pBP is then missing."""
    try:
        from .alt.goal_distributions import XI as BP_DECAY, fit_l3
        lam = np.array([dc.expected_goals(h, a, s) for h, a, s in zip(train["home_id"], train["away_id"], train["season"])])
        w = np.exp(-BP_DECAY * (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400)
        return fit_l3(lam[:, 0], lam[:, 1], train["home_goals"].to_numpy(int), train["away_goals"].to_numpy(int), w / w.mean())
    except Exception as e:  # never let the secondary contrast stop a forecast
        print(f"bivariate Poisson fit skipped ({e})")
        return None


def forecast(cutoff: pd.Timestamp | None = None, gameweeks: int | None = None, refresh: bool = True,
             lineups: bool = True, fixture_ids: list[int] | None = None,
             rows_cache: Path | None = ROOT / "data/processed/forecast_rows.pkl") -> dict:
    cutoff = cutoff or pd.Timestamp.now(tz="UTC").floor("min")
    matches, _ = load_all(ROOT / "res")
    app = load_appearances(matches, ARCHIVE_SEASONS, LIVE_SEASON, refresh_live=refresh)
    fx = upcoming_fixtures(refresh=refresh)
    fx = fx[fx["kickoff_utc"] > cutoff]
    if fixture_ids is not None:
        fx = fx[fx["fpl_fixture"].isin(fixture_ids)]
    elif gameweeks:
        fx = fx[fx["gameweek"] < fx["gameweek"].min() + gameweeks]

    store = MatchStore(matches)
    train = store.completed_before(cutoff)
    dc = DixonColes(Hyperparams()).fit(train, cutoff, promoted_team_seasons(matches))
    bp_l3 = _bp_l3(dc, train, cutoff)
    xg_r, xg_mu, xg_home = fit_xg_strength(known(side_xg(app), cutoff), cutoff, XG_HALF_LIFE)

    rows = training_rows(matches, app, rows_cache)
    tr = rows[rows["kickoff_utc"] + RESULT_LAG <= cutoff]
    Xtr = tr[ADJ].astype(float)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    b = _fit_poisson(((Xtr - mu) / sd).to_numpy(), tr["log_dc"].to_numpy(), tr["goals"].to_numpy(float), STAGE2_L2)

    def side_adj(team: str, opp: str) -> float:
        x = np.array([xg_r["att"].get(team, 0.0), -xg_r["def"].get(opp, 0.0)])
        return float(b[0] + ((x - mu.to_numpy()) / sd.to_numpy()) @ b[1:])

    # ---- XI model (V2x) with predicted lineups, for each team's next fixture ----
    trx = tr.copy()
    Xx = trx[ADJ + XI].astype(float)
    mux, sdx = Xx.mean(), Xx.std().replace(0, 1)
    bx = _fit_poisson(((Xx.fillna(mux) - mux) / sdx).to_numpy(), trx["log_dc"].to_numpy(),
                      trx["goals"].to_numpy(float), STAGE2_L2_XI)
    boot = json.loads((fetch_live_season(LIVE_SEASON, refresh=False) / "bootstrap.json").read_text())
    fp = fpl_players_now(boot)
    lineup_snap = snapshot(fp) if lineups else pd.DataFrame()
    if fixture_ids is not None:
        nxt = fx.assign(match_id=lambda d: d["fpl_fixture"].astype(str))
    else:  # the lineup sites only publish each team's next match
        nxt = fx[fx["gameweek"] == fx["gameweek"].min()].assign(match_id=lambda d: d["fpl_fixture"].astype(str))
    hist = known(app, cutoff)
    teams_next = sorted(set(nxt["home_id"]) | set(nxt["away_id"]))
    owner = dict(zip(fp["player_code"], fp["team_id"]))
    # availability is read from the field that refers to each fixture's own gameweek (amendment A2)
    team_field = {t: availability_field(boot, f.gameweek, cutoff) for f in nxt.itertuples() for t in (f.home_id, f.away_id)}
    fp["chance"] = [r[team_field.get(r["team_id"], "chance_next")] for _, r in fp.iterrows()]
    fp["chance_field"] = fp["team_id"].map(team_field).fillna("chance_next")
    unavailable = unavailable_codes(fp, "chance")
    proj = project_xi(hist, cutoff, teams_next, unavailable=unavailable, current_team=owner)
    proj_hist = project_xi(hist, cutoff, teams_next, current_team=owner)  # same rule, no availability filter
    pos_now = dict(zip(fp["player_code"], fp["position"]))
    xis = {"fpl_proj": proj, "fpl_proj_hist": proj_hist}
    unresolved: dict[str, dict[str, int]] = {}
    invalid: dict[str, dict[str, int]] = {}
    updated: dict[str, str] = {}
    for src in ("ffs", "rotowire"):
        if len(lineup_snap) and (lineup_snap["source"] == src).any():
            allx = lineup_snap[lineup_snap["source"] == src]
            unresolved[src] = allx[allx["player_code"].isna()].groupby("team_id").size().astype(int).to_dict()
            if "source_updated" in allx:
                updated.update({f"{src}:{t}": u for t, u in allx.groupby("team_id")["source_updated"].first().items()
                                if isinstance(u, str)})
            # amendment A3: a source's XI for a team is valid only with 11 listed players and no
            # duplicated resolved player; otherwise that source counts as missing for the team
            listed = allx.groupby("team_id").size()
            dup = allx.dropna(subset=["player_code"]).duplicated(["team_id", "player_code"])
            dup_teams = set(allx.dropna(subset=["player_code"]).loc[dup, "team_id"])
            bad = set(listed[listed != 11].index) | dup_teams
            if bad:
                invalid[src] = {t: int(listed.get(t, 0)) for t in sorted(bad)}
            x = allx[~allx["team_id"].isin(bad)].dropna(subset=["player_code"]).copy()
            x["player_code"] = x["player_code"].astype(int)
            x["position"] = x["player_code"].map(pos_now).fillna(x["position"])
            xis[src] = x[["team_id", "player_code", "position"]]
    names = dict(zip(fp["player_code"], fp["web_name"]))
    xi_preds: dict[int, dict] = {}
    for src, xi in xis.items():
        lu = pd.concat([xi[xi["team_id"].isin([f.home_id, f.away_id])].assign(match_id=f.match_id, started=1.0)
                        for f in nxt.itertuples()], ignore_index=True)
        feats = match_features(nxt, lu, hist, train, cutoff, FeatureParams(), priors=("position",))
        for f in nxt.itertuples():
            lam, xfeat = [], {}
            for side, (t, o) in (("H", (f.home_id, f.away_id)), ("A", (f.away_id, f.home_id))):
                sel = feats[(feats["match_id"] == f.match_id) & (feats["side"] == side)]
                if xi[xi["team_id"] == t].empty or xi[xi["team_id"] == o].empty:
                    lam = None  # this source has no XI for one of the teams
                    break
                row = sel.iloc[0]
                x = np.array([xg_r["att"].get(t, 0.0), -xg_r["def"].get(o, 0.0), row["f_xi_position"],
                              row["f_backline_opp"] if pd.notna(row["f_backline_opp"]) else mux["f_backline_opp"]])
                base = dc.expected_goals(f.home_id, f.away_id, f.season)[0 if side == "H" else 1]
                lam.append(base * np.exp(bx[0] + ((x - mux.to_numpy()) / sdx.to_numpy()) @ bx[1:]))
                xfeat[side] = {"f_xi_position": float(row["f_xi_position"]),
                               "f_backline_opp": None if pd.isna(row["f_backline_opp"]) else float(row["f_backline_opp"])}
            if lam is None:
                continue
            hda = _hda(_score_matrix(lam[0], lam[1], dc.rho))
            d = xi_preds.setdefault(int(f.fpl_fixture), {"sources": {}, "lineups": {}, "lineup_codes": {}, "xi_features": {}})
            d["sources"][src] = {"p_home": hda[0], "p_draw": hda[1], "p_away": hda[2], "xg_home": lam[0], "xg_away": lam[1]}
            d["xi_features"][src] = xfeat
            d["lineups"][src] = {side: [names.get(int(c), str(c)) for c in xi.loc[xi["team_id"] == t, "player_code"]]
                                 for side, t in (("home", f.home_id), ("away", f.away_id))}
            d["lineup_codes"][src] = {side: [int(c) for c in xi.loc[xi["team_id"] == t, "player_code"]]
                                      for side, t in (("home", f.home_id), ("away", f.away_id))}
    for d in xi_preds.values():
        # p_E: the plain average of the two scraped sources' outcome probabilities (benchmark ensemble)
        ps = [d["sources"][k] for k in ("ffs", "rotowire") if k in d["sources"]]
        if ps:
            d["source_average"] = {k: float(np.mean([p[k] for p in ps])) for k in ("p_home", "p_draw", "p_away", "xg_home", "xg_away")}
            d["source_average_of"] = [k for k in ("ffs", "rotowire") if k in d["sources"]]
        lu = d["lineups"]
        if "ffs" in lu and "rotowire" in lu:
            d["agreement"] = {s: len(set(lu["ffs"][s]) & set(lu["rotowire"][s])) for s in ("home", "away")}
    flagged = fp[fp["player_code"].isin(unavailable) & fp["player_code"].isin(proj_usual_codes(hist, cutoff))]
    absences = [{"team_id": r.team_id, "player": r.web_name, "news": r.news, "chance_next": r.chance_next,
                 "chance": r.chance}
                for r in flagged.itertuples()]

    preds = []
    for f in fx.itertuples():
        lh0, la0 = dc.expected_goals(f.home_id, f.away_id, f.season)
        ah, aa = side_adj(f.home_id, f.away_id), side_adj(f.away_id, f.home_id)
        lh, la = lh0 * np.exp(ah), la0 * np.exp(aa)
        base, final = _score_matrix(lh0, la0, dc.rho), _score_matrix(lh, la, dc.rho)
        bp = None
        if bp_l3 is not None:  # amendment A5: same expected goals, bivariate Poisson score grid
            from .alt.goal_distributions import _hda3, bp_grid
            pbp = _hda3(bp_grid(np.array([lh]), np.array([la]), bp_l3))[0]
            bp = {"p_home": float(pbp[0]), "p_draw": float(pbp[1]), "p_away": float(pbp[2])}
        top = np.dstack(np.unravel_index(np.argsort(final, axis=None)[::-1][:3], final.shape))[0]
        pb, pf = _hda(base), _hda(final)
        preds.append({
            "fpl_fixture": int(f.fpl_fixture), "gameweek": int(f.gameweek), "kickoff_utc": f.kickoff_utc.isoformat(),
            "home_id": f.home_id, "away_id": f.away_id, "home": TEAMS[f.home_id], "away": TEAMS[f.away_id],
            "p_home": pf[0], "p_draw": pf[1], "p_away": pf[2],
            "base_home": pb[0], "base_draw": pb[1], "base_away": pb[2],
            "xg_home": lh, "xg_away": la, "base_xg_home": lh0, "base_xg_away": la0,
            "top_scores": [{"score": f"{i}-{j}", "p": float(final[i, j])} for i, j in top],
            "p_btts": float(final[1:, 1:].sum()), "p_over_2_5": float(sum(final[i, j] for i in range(MAX_GOALS + 1)
                                                                          for j in range(MAX_GOALS + 1) if i + j > 2)),
            "home_promoted": [f.home_id, f.season] in dc.promoted, "away_promoted": [f.away_id, f.season] in dc.promoted,
            "with_lineups": xi_preds.get(int(f.fpl_fixture)), "bivariate_poisson": bp,
            "components": {"dc_rate_home": lh0, "dc_rate_away": la0, "rho": dc.rho, "bp_l3": bp_l3,
                           "xgadj_home": [float(xg_r["att"].get(f.home_id, 0.0)), -float(xg_r["def"].get(f.away_id, 0.0))],
                           "xgadj_away": [float(xg_r["att"].get(f.away_id, 0.0)), -float(xg_r["def"].get(f.home_id, 0.0))]},
        })

    current = sorted(set(fx["home_id"]) | set(fx["away_id"]))
    ratings = [{"team_id": t, "team": TEAMS[t], "dc_attack": dc.attack.get(t, 0.0), "dc_defence": dc.defence.get(t, 0.0),
                "xg_attack": float(xg_r["att"].get(t, 0.0)), "xg_defence": float(xg_r["def"].get(t, 0.0))}
               for t in current]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "model": "V1x: Dixon-Coles + opponent-adjusted xG strength (lineup-free)",
        "cutoff_utc": cutoff.isoformat(), "git_commit": commit,
        "results_through": train["kickoff_utc"].max().isoformat(), "n_train_matches": int(len(train)),
        "settings": {"dixon_coles": vars(Hyperparams()), "xg_half_life_days": XG_HALF_LIFE, "stage2_l2": STAGE2_L2},
        "params": {"home_adv": dc.home_adv, "rho": dc.rho, "mu": dc.mu, "stage2_coef": [float(x) for x in b]},
        "fixtures": preds, "ratings": ratings, "absences": absences,
        "xi_stage2": {"cols": ADJ + XI, "coef": [float(v) for v in bx], "mean": [float(v) for v in mux],
                      "sd": [float(v) for v in sdx], "l2": STAGE2_L2_XI},
        "lineup_unresolved": unresolved, "lineup_invalid": invalid, "source_updated": updated,
        "availability": fp[fp["team_id"].isin(teams_next)][["player_code", "team_id", "web_name", "status", "chance_next", "chance_this", "chance", "chance_field", "news"]]
                         .to_dict("records"),
        "lineup_sources": {"retrieved_utc": (lineup_snap["retrieved_utc"].iloc[0] if len(lineup_snap) else None),
                           "available": sorted(xis)},
        "xi_model": "V2x: V1x + XI strength (ratio, position prior); stage 2 trained on confirmed XIs",
    }


def save(fc: dict, out_dir: Path = ROOT / "results/forecasts") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(fc["cutoff_utc"]).strftime("%Y%m%dT%H%MZ")
    path = out_dir / f"forecast_{stamp}.json"
    path.write_text(json.dumps(fc, indent=2, default=str) + "\n")
    return path
