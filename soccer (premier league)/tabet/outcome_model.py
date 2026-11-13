"""Outcome-focused adjustment (review item #5).

The Poisson stage 2 optimises goal likelihood; the evaluation metric is H/D/A
log-loss. This fits a small penalised multinomial model directly on outcomes,
with the Dixon-Coles log-probabilities as offsets and the draw as reference:

    eta_H = log p_H + a_H + b_H . x
    eta_D = log p_D
    eta_A = log p_A + a_A + b_A . x

x is a few *match-level* features (standardised on training rows): the
home-minus-away difference of an attacking feature pair, and their sum (which
mainly moves the draw). All coefficients, intercepts included, get an L2
penalty toward 0, so the default is the Dixon-Coles forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .experiments import MIN_TRAIN_MATCHES, _probs
from .store import RESULT_LAG

OUT = {"H": 0, "D": 1, "A": 2}


def match_table(rows: pd.DataFrame, pair: tuple[str, str]) -> pd.DataFrame:
    """One row per match: Dixon-Coles probabilities and the match-level features
    built from the side-level `pair` (e.g. attack-for and opponent-defence)."""
    h = rows[rows["side"] == "H"].set_index("match_id")
    a = rows[rows["side"] == "A"].set_index("match_id").reindex(h.index)
    P = _probs(np.exp(h["log_dc"].to_numpy()), np.exp(a["log_dc"].to_numpy()), h["rho"].to_numpy())
    sh = h[list(pair)].sum(axis=1)
    sa = a[list(pair)].sum(axis=1)
    return pd.DataFrame({"match_id": h.index, "season": h["season"].to_numpy(), "result": h["result"].to_numpy(),
                         "kickoff_utc": h["kickoff_utc"].to_numpy(), "cutoff_utc": h["cutoff_utc"].to_numpy(),
                         "dc_home": P[:, 0], "dc_draw": P[:, 1], "dc_away": P[:, 2],
                         "x_diff": (sh - sa).to_numpy(), "x_sum": (sh + sa).to_numpy()})


def fit(logp: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    rows = np.arange(n)

    def f(theta):
        bH, bA = theta[:k + 1], theta[k + 1:]
        z = logp.copy()
        z[:, 0] += Xb @ bH
        z[:, 2] += Xb @ bA
        m = z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
        q = np.exp(z - lse[:, None])
        r = q.copy()
        r[rows, y] -= 1
        loss = (lse - z[rows, y]).sum() + l2 * theta @ theta
        g = np.concatenate([Xb.T @ r[:, 0], Xb.T @ r[:, 2]]) + 2 * l2 * theta
        return loss, g

    return minimize(f, np.zeros(2 * (k + 1)), jac=True, method="L-BFGS-B").x


def apply(logp: np.ndarray, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    z = logp.copy()
    z[:, 0] += Xb @ theta[:k + 1]
    z[:, 2] += Xb @ theta[k + 1:]
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def run(table: pd.DataFrame, cols: list[str], l2: float, seasons: tuple[str, ...]) -> pd.DataFrame:
    """Walk forward exactly like experiments.run_variant (same 300-match rule)."""
    t = table.sort_values(["cutoff_utc", "match_id"]).reset_index(drop=True)
    logp_all = np.log(np.clip(t[["dc_home", "dc_draw", "dc_away"]].to_numpy(), 1e-15, 1))
    y_all = t["result"].map(OUT).to_numpy()
    known_at = pd.to_datetime(t["kickoff_utc"], utc=True) + RESULT_LAG
    preds = []
    for cutoff in t.loc[t["season"].isin(seasons), "cutoff_utc"].unique():
        te = np.flatnonzero(t["cutoff_utc"].to_numpy() == cutoff)
        tr = np.flatnonzero((known_at <= cutoff).to_numpy())
        P = np.exp(logp_all[te])
        if cols and len(tr) >= MIN_TRAIN_MATCHES:
            Xtr = t.loc[tr, cols].to_numpy(float)
            mu, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0)
            sd[sd == 0] = 1
            Z = lambda X: (np.where(np.isnan(X), mu, X) - mu) / sd  # noqa: E731
            theta = fit(logp_all[tr], Z(Xtr), y_all[tr], l2)
            P = apply(logp_all[te], Z(t.loc[te, cols].to_numpy(float)), theta)
        preds.append(pd.DataFrame({"match_id": t.loc[te, "match_id"].to_numpy(), "season": t.loc[te, "season"].to_numpy(),
                                   "result": t.loc[te, "result"].to_numpy(), "kickoff_utc": t.loc[te, "kickoff_utc"].to_numpy(),
                                   "p_home": P[:, 0], "p_draw": P[:, 1], "p_away": P[:, 2]}))
    return pd.concat(preds, ignore_index=True)
