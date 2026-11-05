"""Restricted outcome calibration for H/D/A probabilities.

    q_k  is proportional to  exp( log(p_k) / T  +  delta * [k == D] ),   T > 0

T > 1 softens over-confident forecasts and T < 1 sharpens them; delta moves
probability into or out of the draw. Fitting is penalised toward the identity
(T = 1, delta = 0): the parameter is tau = log T, and the penalty is
lam * (tau^2 + delta^2) on a *summed* log-loss, so the identity is the default
and a few hundred matches are needed before the data move it far.

Walk-forward use: for each cutoff, (T, delta) are fitted only on earlier
predictions whose results were known by then. Those predictions must
themselves be out-of-sample (e.g. the walk-forward ledger), otherwise the
calibrator learns from in-sample over-fitting.

This only changes H/D/A probabilities. Any exact-score probabilities shown
alongside would need their own coherent adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .store import RESULT_LAG

OUTCOMES = ("H", "D", "A")
PROB = ["p_home", "p_draw", "p_away"]
MODES = ("identity", "temperature", "temperature+draw")
MIN_CAL_MATCHES = 200


def apply(p: np.ndarray, T: float = 1.0, delta: float = 0.0) -> np.ndarray:
    z = np.log(np.clip(p, 1e-15, 1)) / T
    z[:, 1] += delta
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit(p: np.ndarray, y: np.ndarray, mode: str, lam: float) -> tuple[float, float]:
    """y: integer outcome index (0=H, 1=D, 2=A). Returns (T, delta)."""
    if mode == "identity" or len(y) == 0:
        return 1.0, 0.0
    logp = np.log(np.clip(p, 1e-15, 1))
    use_delta = mode == "temperature+draw"
    rows = np.arange(len(y))

    def f(theta):
        tau = theta[0]
        delta = theta[1] if use_delta else 0.0
        z = logp * np.exp(-tau)
        z[:, 1] += delta
        m = z.max(axis=1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))
        q = np.exp(z - lse[:, None])
        nll = (lse - z[rows, y]).sum()
        # gradients
        dz_dtau = -logp * np.exp(-tau)  # d z_k / d tau (draw shift doesn't depend on tau)
        resid = q.copy()
        resid[rows, y] -= 1  # d nll / d z_k
        g_tau = (resid * dz_dtau).sum() + 2 * lam * tau
        grad = [g_tau]
        pen = lam * tau ** 2
        if use_delta:
            grad.append(resid[:, 1].sum() + 2 * lam * delta)
            pen += lam * delta ** 2
        return nll + pen, np.array(grad)

    x0 = np.zeros(2 if use_delta else 1)
    res = minimize(f, x0, jac=True, method="L-BFGS-B")
    tau = float(res.x[0])
    return float(np.exp(tau)), (float(res.x[1]) if use_delta else 0.0)


def walk_forward(preds: pd.DataFrame, mode: str, lam: float,
                 min_matches: int = MIN_CAL_MATCHES) -> pd.DataFrame:
    """Calibrate every row using only earlier rows with known results.
    preds needs match_id, kickoff_utc, cutoff_utc, result and p_home/p_draw/p_away.
    Until `min_matches` earlier results exist, rows are left unchanged (the same
    rule for every mode). Adds calib_T, calib_delta and calib_active."""
    d = preds.sort_values(["cutoff_utc", "kickoff_utc", "match_id"]).reset_index(drop=True)
    y_all = d["result"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    P = d[PROB].to_numpy(float)
    out = P.copy()
    Ts = np.ones(len(d))
    ds = np.zeros(len(d))
    active = np.zeros(len(d), dtype=bool)
    known_at = (d["kickoff_utc"] + RESULT_LAG).to_numpy()
    for cutoff in d["cutoff_utc"].unique():
        idx = np.flatnonzero(d["cutoff_utc"].to_numpy() == cutoff)
        tr = known_at <= cutoff
        if mode == "identity" or tr.sum() < min_matches:
            continue
        T, delta = fit(P[tr], y_all[tr], mode, lam)
        out[idx] = apply(P[idx], T, delta)
        Ts[idx], ds[idx], active[idx] = T, delta, True
    res = d.copy()
    res[PROB] = out
    res["calib_T"], res["calib_delta"], res["calib_active"] = Ts, ds, active
    return res
