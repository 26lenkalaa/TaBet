"""F14: results-only Elo, and its blend with Dixon-Coles / V1x (docs/PREREGISTRATION.md, addendum E).

Usage: python prototypes/run_elo_blend.py [OUT_DIR]   (default results/elo_blend)

Elo (tabet/elo.py): K from {10, 20, 30} and nothing else tuned; home edge 65 points; goal-difference
multiplier; newcomers start at the leavers' mean; ordered-logit outcome mapping refitted every
matchday on earlier matches. Elo uses the same data window as the model it is compared with.
Blend: p = w * p_model + (1 - w) * p_elo, one weight w on a 0.01 grid.
Choices on development seasons only: K by Elo-alone log-loss, then w by blend log-loss.
  clean   results 2015-16 -> 2020-21; burn-in 2015-17; dev 2017-18; scored once on 2018-21 (n 1,140)
          against the frozen Dixon-Coles walk-forward (results/dynamic_ratings, D0)
  recent  results 2021-22 -> 2026-27; dev 2022-24; reported on 2024-26 and 2026-27 GW1-9
          against frozen V0 (Dixon-Coles) and V1x (results/xg_strength ledger)
Contrasts (log-loss, 14-day block bootstrap, 2,000 resamples, seed 0):
  primary    blend(DC, Elo) - DC on clean 2018-21
  secondary  Elo - DC (clean); blend(V0, Elo) - V0 and blend(V1x, Elo) - V1x on 2024-26 and 2026-27
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import elo  # noqa: E402
from tabet.experiments import block_bootstrap, per_match_logloss  # noqa: E402
from tabet.history import load_history_matches  # noqa: E402
from tabet.ingest import load_all  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "elo_blend"
out.mkdir(parents=True, exist_ok=True)
COLS = ["p_home", "p_draw", "p_away"]
GRID_W = np.round(np.linspace(0, 1, 101), 2)
KS = (10.0, 20.0, 30.0)


def ll_vec(P, y):
    return -np.log(P[np.arange(len(y)), y])


def setup(matches, model_pred, eval_seasons, dev_seasons):
    """model_pred: DataFrame indexed by match_id with p_* columns. Returns aligned arrays and choices."""
    E = {k: elo.walk_forward(matches, eval_seasons, k).set_index("match_id") for k in KS}
    ids = E[KS[0]].index.intersection(model_pred.index)
    base = E[KS[0]].loc[ids]
    y = base["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    dev = base["season"].isin(dev_seasons).to_numpy()
    kdev = {k: float(ll_vec(E[k].loc[ids, COLS].to_numpy(), y)[dev].mean()) for k in KS}
    k = min(kdev, key=kdev.get)
    pe = E[k].loc[ids, COLS].to_numpy()
    return ids, base, y, dev, pe, k, kdev


def choose_w(pm, pe, y, dev):
    lls = [float(ll_vec(w * pm[dev] + (1 - w) * pe[dev], y[dev]).mean()) for w in GRID_W]
    return float(GRID_W[int(np.argmin(lls))])


def contrast(a, b, mask, ko):
    mean, lo, hi = block_bootstrap(a[mask], b[mask], ko[mask].reset_index(drop=True), 14)
    return {"n": int(mask.sum()), "mean": mean, "se": float(np.std(a[mask] - b[mask], ddof=1) / np.sqrt(mask.sum())), "ci": [lo, hi]}


res = {}
# ---- clean ----
mh = load_history_matches()
dc = pd.read_csv(ROOT / "results/dynamic_ratings/ledger_clean.csv")
dc = dc[dc["variant"] == "D0"].set_index("match_id")
ids, base, y, dev, pe, k, kdev = setup(mh, dc, ("2017-18", "2018-19", "2019-20", "2020-21"), ("2017-18",))
pm = dc.loc[ids, COLS].to_numpy()
w = choose_w(pm, pe, y, dev)
test = base["season"].isin(("2018-19", "2019-20", "2020-21")).to_numpy()
ko = pd.Series(pd.to_datetime(base["kickoff_utc"], utc=True).to_numpy())
pb = w * pm + (1 - w) * pe
L = {"DC": ll_vec(pm, y), "Elo": ll_vec(pe, y), "blend": ll_vec(pb, y)}
res["clean"] = {"K_dev": kdev, "K": k, "w_dc": w, "log_loss_2018_21": {n: float(v[test].mean()) for n, v in L.items()},
                "primary blend-DC": contrast(L["blend"], L["DC"], test, ko), "Elo-DC": contrast(L["Elo"], L["DC"], test, ko),
                "dev blend-DC (2017-18, used for w)": contrast(L["blend"], L["DC"], dev, ko)}
print(f"clean: K chosen {k} (dev {kdev}); weight on DC {w}")
print("  2018-21 log-loss:", {n: round(v, 4) for n, v in res['clean']['log_loss_2018_21'].items()})
for n in ("primary blend-DC", "Elo-DC"):
    r = res["clean"][n]
    print(f"  {n:18s} n={r['n']} {r['mean']:+.4f} (SE {r['se']:.4f}) [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")

# ---- recent ----
mr, _ = load_all(ROOT / "res")
led = pd.read_csv(ROOT / "results/xg_strength/ledger.csv")
V = {n: led[led["variant"] == v].drop_duplicates("match_id").set_index("match_id") for n, v in
     (("V0", "V0 Dixon-Coles"), ("V1x", "V1x adjusted xG strength (replaces raw)"))}
seasons = ("2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
ids, base, y, dev, pe, k, kdev = setup(mr, V["V0"], seasons, ("2022-23", "2023-24"))
ko = pd.Series(pd.to_datetime(base["kickoff_utc"], utc=True).to_numpy())
res["recent"] = {"K_dev": kdev, "K": k}
print(f"\nrecent: K chosen {k} (dev {kdev})")
for name in ("V0", "V1x"):
    pm = V[name].loc[ids, COLS].to_numpy()
    w = choose_w(pm, pe, y, dev)
    a, b = ll_vec(w * pm + (1 - w) * pe, y), ll_vec(pm, y)
    res["recent"][f"w_{name}"] = w
    for label, seas in (("2022-24 (dev)", ("2022-23", "2023-24")), ("2024-26", ("2024-25", "2025-26")), ("2026-27 GW1-9", ("2026-27",))):
        mk = base["season"].isin(seas).to_numpy()
        r = contrast(a, b, mk, ko)
        res["recent"][f"blend({name}, Elo) - {name} | {label}"] = r
        print(f"  blend({name}, Elo) - {name}, w={w:.2f}, {label:14s} n={r['n']:4d} {r['mean']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    if name == "V0":
        el = ll_vec(pe, y)
        for label, seas in (("2024-26", ("2024-25", "2025-26")),):
            mk = base["season"].isin(seas).to_numpy()
            r = contrast(el, b, mk, ko)
            res["recent"][f"Elo - V0 | {label}"] = r
            print(f"  Elo - V0, {label}: {r['mean']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
