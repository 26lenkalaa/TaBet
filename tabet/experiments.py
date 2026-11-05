"""Does player / style information improve on the team-strength baseline?

Protocol (identical for every variant):
  1. For each matchday from 2022-23 on (the first season with FPL xG), refit
     Dixon-Coles on results known before that day's first kickoff, and record
     its out-of-sample goal rates for that day's matches.
  2. Build every feature at the same cutoff (tabet.features).
  3. Stage 2: log(rate) = log(DC rate) + b0 + b . x, a penalised Poisson
     regression fitted on *earlier* matchdays' rows only (expanding window).
     Until MIN_TRAIN_MATCHES matches are available, every variant falls back to
     plain Dixon-Coles, so all variants are scored on the same fixtures with
     the same rule.
  4. Probabilities come from the Dixon-Coles score matrix with the adjusted
     rates (rho from the stage-1 fit).

Splits: DEV_SEASONS pick each variant's penalty; TEST_SEASONS are scored once;
2026-27 is a forward check on the matches played so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .backtest import OUTCOMES, scores
from .features import FeatureParams, known, match_features, player_ratings
from .ingest import LOCAL_TZ
from .model import MAX_GOALS, DixonColes, Hyperparams, promoted_team_seasons
from .store import RESULT_LAG, MatchStore

DEV_SEASONS = ("2022-23", "2023-24")
TEST_SEASONS = ("2024-25", "2025-26")
FORWARD_SEASONS = ("2026-27",)
MIN_TRAIN_MATCHES = 300
L2_GRID = (1.0, 10.0, 100.0)

STYLE_MAIN = [f"{p}_{c}" for p in ("s", "so") for c in
              ("shots_for", "shots_against", "corners_for", "corners_against", "fouls",
               "xg_per_shot_for", "xg_per_shot_against")]
STYLE_X = ["x_volume", "x_quality", "x_corners", "x_vol_vs_quality", "x_quality_vs_vol"]
XG_FORM = ["f_xgf", "f_xga_opp"]
VARIANTS: dict[str, list[str]] = {
    "V0 Dixon-Coles": [],
    "V1 + team xG form": XG_FORM,
    "V1c = V1 + back-line continuity": XG_FORM + ["f_backline_opp"],
    "V1s = V1 + style main effects": XG_FORM + STYLE_MAIN,
    "V2 + XI strength (position prior)": XG_FORM + ["f_xi_position", "f_backline_opp"],
    "V3 + XI strength (similar-player prior)": XG_FORM + ["f_xi_similar", "f_backline_opp"],
    "V4 = V2 + style main effects": XG_FORM + ["f_xi_position", "f_backline_opp"] + STYLE_MAIN,
    "V5 = V4 + style interactions": XG_FORM + ["f_xi_position", "f_backline_opp"] + STYLE_MAIN + STYLE_X,
}


# Pre-specified primary contrasts (metric: outcome log-loss). Everything else is secondary.
PRIMARY_CONTRASTS = {
    "XI strength given xG form (V2 - V1)": ("V2 + XI strength (position prior)", "V1 + team xG form"),
    "similar-player vs position prior (V3 - V2)": ("V3 + XI strength (similar-player prior)",
                                                   "V2 + XI strength (position prior)"),
    "style main effects given xG form (V1s - V1)": ("V1s = V1 + style main effects", "V1 + team xG form"),
    "style interactions given main effects (V5 - V4)": ("V5 = V4 + style interactions",
                                                        "V4 = V2 + style main effects"),
}
SECONDARY_CONTRASTS = {
    "xG form (V1 - V0)": ("V1 + team xG form", "V0 Dixon-Coles"),
    "continuity only (V1c - V1)": ("V1c = V1 + back-line continuity", "V1 + team xG form"),
    "style main effects given XI (V4 - V2)": ("V4 = V2 + style main effects", "V2 + XI strength (position prior)"),
}


def build_dataset(matches: pd.DataFrame, app: pd.DataFrame, p: FeatureParams = FeatureParams(),
                  hyper: Hyperparams = Hyperparams(), first_season: str = "2022-23",
                  low_minutes: float = 600.0, lineup_provider=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (rows, player_checks). rows: two per match with DC offset, goals
    and features. player_checks: low-history starters' ratings under each prior
    vs their output in that match (for the player-level similarity check)."""
    store = MatchStore(matches)
    promoted = promoted_team_seasons(matches)
    target = matches[(matches["season"] >= first_season) & (matches["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    out, checks = [], []
    for _, fx in target.groupby(day, sort=True):
        cutoff = fx["kickoff_utc"].min()
        train = store.completed_before(cutoff)
        dc = DixonColes(hyper).fit(train, cutoff, promoted)
        hist = known(app, cutoff)
        lineups = app[app["match_id"].isin(fx["match_id"])]
        if lineup_provider is not None:
            # e.g. a projected XI built only from data known at the cutoff
            lineups = lineup_provider(fx, hist, cutoff)
        feats = match_features(fx, lineups, hist, train, cutoff, p)
        lam = {mid: dc.expected_goals(h, a, s) for mid, h, a, s in
               zip(fx["match_id"], fx["home_id"], fx["away_id"], fx["season"])}
        g = fx.set_index("match_id")
        feats["log_dc"] = [np.log(lam[m][0 if sd == "H" else 1]) for m, sd in zip(feats["match_id"], feats["side"])]
        feats["goals"] = [g.at[m, "home_goals" if sd == "H" else "away_goals"] for m, sd in
                          zip(feats["match_id"], feats["side"])]
        feats["rho"] = dc.rho
        feats["cutoff_utc"] = cutoff
        feats["kickoff_utc"] = feats["match_id"].map(g["kickoff_utc"])
        feats["season"] = feats["match_id"].map(g["season"])
        feats["result"] = feats["match_id"].map(g["result"])
        out.append(feats)

        if lineup_provider is not None:
            continue
        # player-level check: starters with little xG-era history
        rp = player_ratings(hist, cutoff, p, "position")
        rs = player_ratings(hist, cutoff, p, "similar")
        st = lineups[(lineups["started"] == 1) & (lineups["position"] != "GK") & (lineups["minutes"] >= 45)]
        if len(st):
            have = rp["wmin_xg"].reindex(st["player_code"]).fillna(0).to_numpy()
            low = st[have < low_minutes]
            if len(low):
                checks.append(pd.DataFrame({
                    "match_id": low["match_id"].to_numpy(), "season": low["season"].to_numpy(),
                    "player_code": low["player_code"].to_numpy(), "position": low["position"].to_numpy(),
                    "prior_minutes": rp["wmin_xg"].reindex(low["player_code"]).fillna(0).to_numpy(),
                    "minutes": low["minutes"].to_numpy(),
                    "observed": ((low["xg"] + low["xa"]) / (low["minutes"] / 90)).to_numpy(),
                    "pred_position": rp["rating"].reindex(low["player_code"]).fillna(rp["prior"].median()).to_numpy(),
                    "pred_similar": rs["rating"].reindex(low["player_code"]).fillna(rs["prior"].median()).to_numpy(),
                }))
    rows = pd.concat(out, ignore_index=True)
    return rows, (pd.concat(checks, ignore_index=True) if checks else pd.DataFrame())


# ---- stage 2 -----------------------------------------------------------------

def _fit_poisson(X: np.ndarray, off: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def f(b):
        eta = off + Xb @ b
        lam = np.exp(eta)
        pen = l2 * (b[1:] @ b[1:])
        grad = Xb.T @ (lam - y)
        grad[1:] += 2 * l2 * b[1:]
        return (lam - y * eta).sum() + pen, grad

    res = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B")
    return res.x


def _probs(lh: np.ndarray, la: np.ndarray, rho: np.ndarray) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    out = np.empty((len(lh), 3))
    for i in range(len(lh)):
        m = np.outer(poisson.pmf(g, lh[i]), poisson.pmf(g, la[i]))
        r = rho[i]
        m[0, 0] *= 1 - lh[i] * la[i] * r
        m[0, 1] *= 1 + lh[i] * r
        m[1, 0] *= 1 + la[i] * r
        m[1, 1] *= 1 - r
        m /= m.sum()
        out[i] = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
    return out


def run_variant(rows: pd.DataFrame, cols: list[str], l2: float, seasons: tuple[str, ...]) -> pd.DataFrame:
    """Walk forward over matchdays in `seasons`; returns one row per match with
    probabilities and adjusted rates."""
    rows = rows.sort_values(["cutoff_utc", "match_id", "side"]).reset_index(drop=True)
    preds = []
    eval_days = rows.loc[rows["season"].isin(seasons), "cutoff_utc"].unique()
    for cutoff in eval_days:
        te = rows[rows["cutoff_utc"] == cutoff]
        tr = rows[rows["kickoff_utc"] + RESULT_LAG <= cutoff]
        adj = np.zeros(len(te))
        n_train = tr["match_id"].nunique()
        active = bool(cols) and n_train >= MIN_TRAIN_MATCHES
        if active:
            Xtr = tr[cols].astype(float)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1).fillna(1)
            Xtr = ((Xtr.fillna(mu) - mu) / sd).to_numpy()
            b = _fit_poisson(Xtr, tr["log_dc"].to_numpy(), tr["goals"].to_numpy(float), l2)
            Xte = ((te[cols].astype(float).fillna(mu) - mu) / sd).to_numpy()
            adj = b[0] + Xte @ b[1:]
        lam = np.exp(te["log_dc"].to_numpy() + adj)
        t = te.assign(lam=lam)
        h = t[t["side"] == "H"].set_index("match_id")
        a = t[t["side"] == "A"].set_index("match_id").reindex(h.index)
        pr = _probs(h["lam"].to_numpy(), a["lam"].to_numpy(), h["rho"].to_numpy())
        preds.append(pd.DataFrame({
            "match_id": h.index, "season": h["season"].to_numpy(), "result": h["result"].to_numpy(),
            "kickoff_utc": h["kickoff_utc"].to_numpy(), "cutoff_utc": cutoff,
            "stage2_active": active, "stage2_train_matches": n_train,
            "p_home": pr[:, 0], "p_draw": pr[:, 1], "p_away": pr[:, 2],
            "lam_home": h["lam"].to_numpy(), "lam_away": a["lam"].to_numpy(),
            "goals_home": h["goals"].to_numpy(), "goals_away": a["goals"].to_numpy()}))
    return pd.concat(preds, ignore_index=True)


def per_match_logloss(pred: pd.DataFrame) -> np.ndarray:
    y = pred["result"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    p = pred[["p_home", "p_draw", "p_away"]].to_numpy()
    return -np.log(np.clip(p[np.arange(len(p)), y], 1e-15, 1))


def goal_deviance(pred: pd.DataFrame) -> float:
    """Mean Poisson deviance of predicted goals, per team-match (lower is better)."""
    d = 0.0
    for lam, y in (("lam_home", "goals_home"), ("lam_away", "goals_away")):
        mu, k = pred[lam].to_numpy(), pred[y].to_numpy(float)
        term = np.where(k > 0, k * np.log(np.where(k > 0, k, 1) / mu), 0.0)
        d += (2 * (term - (k - mu))).sum()
    return float(d / (2 * len(pred)))


def summary(pred: pd.DataFrame) -> dict[str, float]:
    s = scores(pred[["p_home", "p_draw", "p_away"]].to_numpy(), pred["result"])
    s["goal_deviance"] = goal_deviance(pred)
    return s


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) and a 95% percentile CI, resampling matches."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def block_bootstrap(a: np.ndarray, b: np.ndarray, kickoff: pd.Series, block_days: int,
                    reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Like paired_bootstrap, but resamples contiguous calendar blocks of
    `block_days`, so dependence between nearby matches (same matchday, same
    form spell) is kept inside a block."""
    d = a - b
    block = ((kickoff - kickoff.min()).dt.days // block_days).to_numpy()
    ids = np.unique(block)
    sums = np.array([d[block == i].sum() for i in ids])
    counts = np.array([(block == i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(reps, len(ids)))
    boots = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def tune_l2(rows: pd.DataFrame, variants: dict[str, list[str]] = VARIANTS,
            seasons: tuple[str, ...] = DEV_SEASONS, grid: tuple[float, ...] = L2_GRID) -> dict[str, float]:
    best = {}
    for name, cols in variants.items():
        if not cols:
            best[name] = 0.0
            continue
        ll = {l2: per_match_logloss(run_variant(rows, cols, l2, seasons)).mean() for l2 in grid}
        best[name] = min(ll, key=ll.get)
    return best
