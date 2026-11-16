"""Neural-network second stage (prototype; not used by the live forecast or the logger).

The current model's stage 2 is a penalised *linear* Poisson regression on top of Dixon-Coles:

    log(goal rate) = log(Dixon-Coles rate) + b0 + b . x

This replaces `b0 + b . x` with a small neural network f(x):

    log(goal rate) = log(Dixon-Coles rate) + f(x)

Design choices for a small dataset (~1,600 Premier League matches with these features):
  * Dixon-Coles stays as the offset, so the network only learns corrections. Its last layer starts
    at zero, so an untrained network is exactly Dixon-Coles.
  * Small (two hidden layers of `hidden` units), weight decay, and an average over several seeds.
  * Poisson loss on goals, as stage 2 uses; outcome probabilities then come from the same
    Dixon-Coles score grid (`experiments._probs`), so H/D/A stay consistent with the goal model.
  * Walk-forward exactly like `experiments.run_variant`: refit at every cutoff on matches whose
    result was known (kickoff + 2 h 30) before it. No odds are used anywhere.

PyTorch is an optional dependency (requirements-ml.txt); nothing on the logging path imports this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .experiments import MIN_TRAIN_MATCHES, _probs
from .store import RESULT_LAG


@dataclass(frozen=True)
class NNParams:
    hidden: int = 16
    weight_decay: float = 1e-2
    epochs: int = 300
    lr: float = 1e-2
    seeds: int = 5


def _fit_predict(Xtr: np.ndarray, off: np.ndarray, y: np.ndarray, Xte: np.ndarray, p: NNParams) -> np.ndarray:
    """Train `p.seeds` networks full-batch on (Xtr, offset, goals); return the mean f(Xte)."""
    import torch
    from torch import nn

    torch.set_num_threads(4)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    ot = torch.tensor(off, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    outs = []
    for seed in range(p.seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], p.hidden), nn.GELU(),
                            nn.Linear(p.hidden, p.hidden), nn.GELU(),
                            nn.Linear(p.hidden, 1))
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        for _ in range(p.epochs):
            opt.zero_grad()
            eta = ot + net(xt).squeeze(1)
            loss = (torch.exp(eta) - yt * eta).mean()   # Poisson negative log-likelihood (up to a constant)
            loss.backward()
            opt.step()
        with torch.no_grad():
            outs.append(net(xe).squeeze(1).numpy())
    return np.mean(outs, axis=0)


def run_nn_variant(rows: pd.DataFrame, cols: list[str], seasons: tuple[str, ...], p: NNParams = NNParams()) -> pd.DataFrame:
    """Same contract and output as experiments.run_variant, with the network as stage 2."""
    rows = rows.sort_values(["cutoff_utc", "match_id", "side"]).reset_index(drop=True)
    preds = []
    for cutoff in rows.loc[rows["season"].isin(seasons), "cutoff_utc"].unique():
        te = rows[rows["cutoff_utc"] == cutoff]
        tr = rows[rows["kickoff_utc"] + RESULT_LAG <= cutoff]
        adj = np.zeros(len(te))
        n_train = tr["match_id"].nunique()
        active = bool(cols) and n_train >= MIN_TRAIN_MATCHES
        if active:
            Xtr = tr[cols].astype(float)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1).fillna(1)
            Xtr = ((Xtr.fillna(mu) - mu) / sd).to_numpy()
            Xte = ((te[cols].astype(float).fillna(mu) - mu) / sd).to_numpy()
            adj = _fit_predict(Xtr, tr["log_dc"].to_numpy(), tr["goals"].to_numpy(float), Xte, p)
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
