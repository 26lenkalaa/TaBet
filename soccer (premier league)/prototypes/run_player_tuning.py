"""Exploratory: tune the player model's never-tuned settings (V2x lineup strength).

player_half_life (365 d), shrink_minutes (900) and team_half_life (120 d) were fixed in the original
pre-registration and never tuned. Here they are chosen on the dev seasons (2022-24) only, Dixon-Coles
offsets and xG features are reused from the cached rows, and only the two lineup features are
recomputed. Results on 2024-26 / 2026-27 GW1-9 are exploratory (those seasons were used before).
Nothing here changes the live forecast or the frozen protocol.

    python prototypes/run_player_tuning.py [check]
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import DEV_SEASONS, block_bootstrap, per_match_logloss, run_variant  # noqa: E402
from tabet.features import FeatureParams, known, match_features  # noqa: E402
from tabet.fpl import load_appearances  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.store import MatchStore  # noqa: E402

OUT = ROOT / "results" / "player_tuning"
ADJ, XI = ["f_xgadj_att_90", "f_xgadj_def_opp_90"], ["f_xi_position", "f_backline_opp"]


def load():
    m, _ = load_all(ROOT / "res")
    app = load_appearances(m, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27", refresh_live=False)
    rows = pd.read_pickle(ROOT / "data/processed/forecast_rows.pkl")
    for c in ("cutoff_utc", "kickoff_utc"):
        rows[c] = pd.to_datetime(rows[c], utc=True)
    return m, app, rows


def xi_features(m, app, rows, p: FeatureParams) -> pd.DataFrame:
    """Recompute f_xi_position and f_backline_opp for every cached row under settings p."""
    store = MatchStore(m)
    fx_all = m.set_index("match_id")
    out = []
    for cutoff, g in rows.groupby("cutoff_utc", sort=True):
        ids = g["match_id"].unique()
        fx = fx_all.loc[ids].reset_index()
        hist = known(app, cutoff)
        lu = app[app["match_id"].isin(ids)]
        f = match_features(fx, lu, hist, store.completed_before(cutoff), cutoff, p, priors=("position",))
        out.append(f[["match_id", "side"] + XI])
    return pd.concat(out, ignore_index=True)


def with_features(rows, feats):
    return rows.drop(columns=XI).merge(feats, on=["match_id", "side"], how="left")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m, app, rows = load()
    if sys.argv[1:] == ["check"]:
        t = time.time()
        f = xi_features(m, app, rows, FeatureParams())
        r = with_features(rows, f)
        for c in XI:
            print(c, "max |diff| vs cache:", float(np.nanmax(np.abs(r[c].to_numpy() - rows.sort_values(["match_id", "side"]).set_index(["match_id", "side"]).loc[list(zip(r.match_id, r.side)), c].to_numpy()))))
        print(f"one rebuild: {time.time() - t:.0f}s")
        sys.exit()

    # ---- grid search on the dev seasons only
    GRID = list(itertools.product((180.0, 365.0, 730.0), (300.0, 900.0, 2700.0), (60.0, 120.0, 240.0)))
    L2S = (30.0, 100.0, 300.0)
    feats, dev = {}, {}
    for php, shrink, thl in GRID:
        p = FeatureParams(player_half_life=php, shrink_minutes=shrink, team_half_life=thl)
        t = time.time()
        feats[(php, shrink, thl)] = f = xi_features(m, app, rows, p)
        r = with_features(rows, f)
        for l2 in L2S:
            dev[(php, shrink, thl, l2)] = float(per_match_logloss(run_variant(r, ADJ + XI, l2, DEV_SEASONS)).mean())
        b = min(L2S, key=lambda l: dev[(php, shrink, thl, l)])
        print(f"player_hl {php:4.0f}  shrink {shrink:5.0f}  team_hl {thl:4.0f}  best L2 {b:4.0f}  dev {dev[(php, shrink, thl, b)]:.4f}  ({time.time() - t:.0f}s)", flush=True)
    default = (365.0, 900.0, 120.0, 100.0)
    best = min(dev, key=dev.get)
    print(f"\ndefault {default} dev {dev[default]:.4f}\nbest    {best} dev {dev[best]:.4f}  ({dev[best] - dev[default]:+.4f})")

    # ---- check the dev winner against the current settings on later seasons
    res = {"note": "exploratory; chosen on 2022-24 only", "dev": {str(k): v for k, v in dev.items()},
           "default": default, "best": best, "periods": {}}
    for pname, seasons in {"2024-26": ("2024-25", "2025-26"), "2026-27 GW1-9": ("2026-27",)}.items():
        base = run_variant(with_features(rows, feats[default[:3]]), ADJ + XI, default[3], seasons)
        new = run_variant(with_features(rows, feats[best[:3]]), ADJ + XI, best[3], seasons)
        new = new.set_index("match_id").loc[base["match_id"]].reset_index()
        lb, ln = per_match_logloss(base), per_match_logloss(new)
        mean, lo, hi = block_bootstrap(ln, lb, pd.to_datetime(base["kickoff_utc"], utc=True).reset_index(drop=True), 14)
        res["periods"][pname] = {"n": len(base), "default": float(lb.mean()), "tuned": float(ln.mean()), "tuned_minus_default": [mean, lo, hi]}
        print(f"{pname} (n={len(base)}): default {lb.mean():.4f}  tuned {ln.mean():.4f}  diff {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
