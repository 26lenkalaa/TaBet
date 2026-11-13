"""Projected-lineup mode: does XI strength still help when the XI is *projected*
from preceding selections (no team news), instead of the confirmed lineup?

Usage: python prototypes/run_projected_xi.py [OUT_DIR]   (default results/projected_xi)

Primary: V2x_proj - V1x (identical fixtures; V2x settings frozen from round 2).
Reported alongside: V2x_confirmed - V1x, and how many projected starters were right.
Limitation: no historical injury/suspension news exists, so this projection is
weaker than a real pre-match prediction, which would remove known absentees.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import (DEV_SEASONS, FORWARD_SEASONS, TEST_SEASONS, block_bootstrap, build_dataset,  # noqa: E402
                               paired_bootstrap, per_match_logloss, run_variant)
from tabet.fpl import load_appearances  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.lineup_sources import project_xi  # noqa: E402
from tabet.xg_strength import add_xg_strength  # noqa: E402

SPLITS = {"dev": DEV_SEASONS, "test": TEST_SEASONS, "forward_2026_27": FORWARD_SEASONS}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "projected_xi"
out.mkdir(parents=True, exist_ok=True)
m, _ = load_all(ROOT / "res")
app = load_appearances(m, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27", refresh_live=False)
acc = []


def provider(fx, hist, cutoff):
    teams = sorted(set(fx["home_id"]) | set(fx["away_id"]))
    xi = project_xi(hist, cutoff, teams)
    rows = []
    for f in fx.itertuples():
        for t in (f.home_id, f.away_id):
            x = xi[xi["team_id"] == t]
            rows.append(x.assign(match_id=f.match_id, started=1.0))
            actual = set(app.loc[(app.match_id == f.match_id) & (app.team_id == t) & (app.started == 1), "player_code"])
            acc.append({"match_id": f.match_id, "team_id": t, "correct": len(actual & set(x["player_code"])),
                        "n_projected": len(x)})
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["match_id", "team_id", "player_code", "position", "started"])


proj, _ = build_dataset(m, app, lineup_provider=provider)
proj = add_xg_strength(proj, app, (90.0,))
conf = pd.read_pickle(ROOT / "results/review_round2/rows.pkl")
ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]
XI = ["f_xi_position", "f_backline_opp"]
L2_V1X = 1.0
L2_V2X = json.loads((ROOT / "results/fpl_v0_v5/manifest.json").read_text())["l2_chosen_on_dev"]["V2 + XI strength (position prior)"]
a = pd.DataFrame(acc)
a["season"] = a["match_id"].str[:7]
results = {"projection_accuracy": a.groupby("season")["correct"].mean().round(2).to_dict()}
print("projected starters correct (of 11), by season:", results["projection_accuracy"])
for split, seasons in SPLITS.items():
    base = run_variant(conf, ADJ, L2_V1X, seasons).sort_values("match_id").reset_index(drop=True)
    v2c = run_variant(conf, ADJ + XI, L2_V2X, seasons).sort_values("match_id").reset_index(drop=True)
    v2p = run_variant(proj, ADJ + XI, L2_V2X, seasons).sort_values("match_id").reset_index(drop=True)
    ll0, llc, llp = per_match_logloss(base), per_match_logloss(v2c), per_match_logloss(v2p)
    ko = pd.to_datetime(base["kickoff_utc"], utc=True)
    res = {}
    for name, (x, y) in {"primary: V2x_projected - V1x": (llp, ll0), "V2x_confirmed - V1x": (llc, ll0),
                         "V2x_projected - V2x_confirmed": (llp, llc)}.items():
        d, lo, hi = paired_bootstrap(x, y)
        res[name] = {"mean": d, "se": float(np.std(x - y, ddof=1) / np.sqrt(len(x))), "fixture_ci": [lo, hi],
                     "block14d_ci": list(block_bootstrap(x, y, ko, 14)[1:])}
    results[split] = res
    print(f"\n=== {split} ({len(base)} matches) ===")
    for k, e in res.items():
        print(f"  {k}: {e['mean']:+.4f} (SE {e['se']:.4f}) block14[{e['block14d_ci'][0]:+.4f},{e['block14d_ci'][1]:+.4f}]")
r = proj[proj.season >= "2023-24"]
results["f_xi_projected_sd"] = float(r["f_xi_position"].std())
results["f_xi_confirmed_sd"] = float(conf[conf.season >= "2023-24"]["f_xi_position"].std())
print("\nXI-feature spread: projected sd", round(results["f_xi_projected_sd"], 4), "| confirmed sd", round(results["f_xi_confirmed_sd"], 4))
(out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
print("wrote", out)
