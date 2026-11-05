"""Walk-forward evaluation and hyperparameter tuning (docs/SPEC.md sections 8-9).

For every matchday (local calendar date) in the evaluated seasons, the model is
refitted on results known before that day's first kickoff and then predicts all
of that day's matches. Using the day's *first* kickoff is conservative, because
an evening match doesn't get to see that afternoon's results.

Splits are by time, never random:
  * 2021-22 is warm-up only (training data, never scored).
  * TUNE_SEASONS choose the hyperparameters.
  * TEST_SEASONS are held out and scored once with the chosen hyperparameters.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .ingest import LOCAL_TZ, implied_probabilities
from .model import DixonColes, Hyperparams, promoted_team_seasons
from .store import MatchStore

TUNE_SEASONS = ("2022-23", "2023-24")
TEST_SEASONS = ("2024-25", "2025-26")
XI_GRID = (0.0, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006)
L2_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
OUTCOMES = ("H", "D", "A")
PROB_COLS = ["p_home", "p_draw", "p_away"]


def walk_forward(matches: pd.DataFrame, seasons: tuple[str, ...], hyper: Hyperparams) -> pd.DataFrame:
    """Out-of-sample predictions for every completed match in `seasons`.
    Also returns the naive baseline (H/D/A frequencies before the cutoff)."""
    store = MatchStore(matches)
    promoted = promoted_team_seasons(matches)
    target = matches[matches["season"].isin(seasons) & (matches["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    out = []
    for _, group in target.groupby(day, sort=True):
        cutoff = group["kickoff_utc"].min()
        train = store.completed_before(cutoff)
        model = DixonColes(hyper).fit(train, cutoff, promoted)
        pred = model.predict_frame(group)
        freq = train["result"].value_counts(normalize=True).reindex(list(OUTCOMES), fill_value=0)
        pred["naive_home"], pred["naive_draw"], pred["naive_away"] = freq["H"], freq["D"], freq["A"]
        pred["cutoff_utc"] = cutoff
        pred["n_train"] = len(train)
        out.append(pd.concat([group[["match_id", "season", "kickoff_utc", "home_id", "away_id",
                                     "home_goals", "away_goals", "result"]], pred], axis=1))
    return pd.concat(out).reset_index(drop=True)


def attach_market(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Add market-implied probabilities (benchmark only, never a model input)."""
    out = preds
    for prefix, name in (("Avg", "mkt"), ("AvgC", "close")):
        mp = implied_probabilities(odds, prefix).rename(
            columns={"p_home": f"{name}_home", "p_draw": f"{name}_draw", "p_away": f"{name}_away"})
        out = out.merge(mp, on="match_id", how="left")
    return out


def scores(probs: np.ndarray, result: pd.Series) -> dict[str, float]:
    """Mean log-loss, Brier (summed over the 3 outcomes) and ranked probability
    score (RPS, which treats H > D > A as ordered). Lower is better for all three."""
    y = np.stack([(result == o).to_numpy() for o in OUTCOMES], axis=1).astype(float)
    p = np.clip(probs, 1e-15, 1)
    cum_p, cum_y = np.cumsum(p, axis=1)[:, :2], np.cumsum(y, axis=1)[:, :2]
    return {
        "log_loss": float(-(y * np.log(p)).sum(axis=1).mean()),
        "brier": float(((p - y) ** 2).sum(axis=1).mean()),
        "rps": float(((cum_p - cum_y) ** 2).sum(axis=1).mean() / 2),
        "accuracy": float((p.argmax(axis=1) == y.argmax(axis=1)).mean()),
        "n": int(len(y)),
    }


def compare(preds: pd.DataFrame) -> pd.DataFrame:
    """Score table for model vs naive vs market, on rows where every source has
    a probability (so all are scored on exactly the same matches)."""
    sources = {"model": "p", "naive": "naive", "market (pre-match avg)": "mkt",
               "market (closing avg)": "close"}
    cols = {k: [f"{v}_home", f"{v}_draw", f"{v}_away"] if v != "p" else PROB_COLS
            for k, v in sources.items()}
    cols = {k: c for k, c in cols.items() if set(c) <= set(preds.columns)}
    ok = preds[[c for cs in cols.values() for c in cs]].notna().all(axis=1)
    p = preds[ok]
    return pd.DataFrame({k: scores(p[c].to_numpy(), p["result"]) for k, c in cols.items()}).T


def calibration(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, per outcome, in equal-width bins."""
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for o, col in zip(OUTCOMES, PROB_COLS):
        b = pd.cut(preds[col], edges, include_lowest=True)
        g = preds.assign(hit=(preds["result"] == o)).groupby(b, observed=True)
        for interval, grp in g:
            rows.append({"outcome": o, "bin": str(interval), "n": len(grp),
                         "mean_pred": grp[col].mean(), "observed": grp["hit"].mean()})
    return pd.DataFrame(rows)


def tune(matches: pd.DataFrame, seasons: tuple[str, ...] = TUNE_SEASONS,
         xi_grid=XI_GRID, l2_grid=L2_GRID, promoted_prior=(True, False)) -> pd.DataFrame:
    """Grid search on the tuning seasons. Returns one row per setting, best first."""
    rows = []
    for xi, l2, pp in itertools.product(xi_grid, l2_grid, promoted_prior):
        h = Hyperparams(xi=xi, l2=l2, promoted_prior=pp)
        preds = walk_forward(matches, seasons, h)
        s = scores(preds[PROB_COLS].to_numpy(), preds["result"])
        rows.append({"xi": xi, "half_life_days": h.half_life_days, "l2": l2,
                     "promoted_prior": pp, **s})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
