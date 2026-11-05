"""Two cheap diagnostics from the third external review (no new model).

Usage: python prototypes/run_review_round3_diagnostics.py [OUT_DIR]  (default results/round3_diagnostics)

1. XI-deviation split. The XI gain (V2x - V1x) is null on dev and about -0.004 on test. Split every
   match by how far both XIs were from usual, |f_xi_H| + |f_xi_A|, using cut points fixed on all
   dev + test matches together. If the gain sits in high-deviation matches in both periods and dev
   just had fewer of them, that points to a regime difference; if dev shows nothing even there,
   noise or a real change is more likely. Also: the dev - test difference and its z.
2. Shorter xG half-lives. 90 days (the chosen value) was the shortest tried. Extend the V1x grid
   to 45 and 60 days on dev only. Test is reported for information; the live model is frozen by the
   prospective protocol and is not changed by this diagnostic.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import (DEV_SEASONS, L2_GRID, TEST_SEASONS, block_bootstrap, per_match_logloss,  # noqa: E402
                               run_variant)
from tabet.fpl import load_appearances  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.xg_strength import add_xg_strength  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "round3_diagnostics"
out.mkdir(parents=True, exist_ok=True)
rows = pd.read_pickle(ROOT / "results/review_round2/rows.pkl")
ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]
XI = ["f_xi_position", "f_backline_opp"]
L2_V1X, L2_V2X = 1.0, 100.0  # frozen (results/xg_strength, results/fpl_v0_v5 manifest)
res: dict = {}

# ---- 1. XI-deviation split ----
dev_abs = rows.assign(a=rows["f_xi_position"].abs()).groupby("match_id")["a"].sum()
per = {}
for split, seasons in (("dev", DEV_SEASONS), ("test", TEST_SEASONS)):
    v1 = run_variant(rows, ADJ, L2_V1X, seasons)
    v2 = run_variant(rows, ADJ + XI, L2_V2X, seasons).set_index("match_id").reindex(v1["match_id"]).reset_index()
    per[split] = pd.DataFrame({"match_id": v1["match_id"], "kickoff_utc": v1["kickoff_utc"],
                               "active": v2["stage2_active"].to_numpy(),
                               "d": per_match_logloss(v2) - per_match_logloss(v1)})
both = pd.concat(per.values())
both["dev_xi"] = both["match_id"].map(dev_abs)
cuts = np.quantile(both["dev_xi"].dropna(), [1 / 3, 2 / 3])
labels = ["low", "middle", "high"]
split_rows = []
for split, g in per.items():
    g = g.assign(dev_xi=g["match_id"].map(dev_abs))
    g["band"] = pd.cut(g["dev_xi"], [-np.inf, *cuts, np.inf], labels=labels)
    for band in labels:
        h = g[g["band"] == band]
        split_rows.append({"split": split, "band": band, "n": len(h), "n_stage2_active": int(h["active"].sum()),
                           "mean_dev_xi": h["dev_xi"].mean(), "v2x_minus_v1x": h["d"].mean(),
                           "se": h["d"].std(ddof=1) / np.sqrt(len(h))})
sp = pd.DataFrame(split_rows)
md = {s: (g["d"].mean(), g["d"].std(ddof=1) / np.sqrt(len(g))) for s, g in per.items()}
z = (md["dev"][0] - md["test"][0]) / np.hypot(md["dev"][1], md["test"][1])
# same contrast restricted to matches where stage 2 was active in both periods
act = {s: g[g["active"]] for s, g in per.items()}
res["xi_split"] = {"cuts": cuts.tolist(), "bands": sp.to_dict("records"),
                   "overall": {s: {"mean": m, "se": se} for s, (m, se) in md.items()},
                   "dev_minus_test": md["dev"][0] - md["test"][0], "z": z,
                   "active_only": {s: {"n": len(g), "mean": g["d"].mean(), "se": g["d"].std(ddof=1) / np.sqrt(len(g))}
                                   for s, g in act.items()}}
print("1. XI gain (V2x - V1x) by XI-deviation band (cut points from dev+test):")
print(sp.round(4).to_string(index=False))
print(f"   overall dev {md['dev'][0]:+.4f} (SE {md['dev'][1]:.4f}), test {md['test'][0]:+.4f} (SE {md['test'][1]:.4f}); "
      f"difference {md['dev'][0] - md['test'][0]:+.4f}, z = {z:.2f}")
for s, g in act.items():
    print(f"   stage 2 active only, {s}: n={len(g)} {g['d'].mean():+.4f} (SE {g['d'].std(ddof=1) / np.sqrt(len(g)):.4f})")

# ---- 2. half-life grid extended below 90 days ----
m, _ = load_all(ROOT / "res")
app = load_appearances(m, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27", refresh_live=False)
HL = (45.0, 60.0)
rows2 = add_xg_strength(rows, app, HL)
grid = []
for hl in (45.0, 60.0, 90.0):
    cols = [f"f_xgadj_att_{int(hl)}", f"f_xgadj_def_opp_{int(hl)}"]
    for l2 in L2_GRID:
        grid.append({"half_life": hl, "l2": l2,
                     "dev_log_loss": per_match_logloss(run_variant(rows2, cols, l2, DEV_SEASONS)).mean()})
grid = pd.DataFrame(grid).sort_values("dev_log_loss")
best = grid.iloc[0]
print("\n2. V1x dev grid with shorter half-lives:")
print(grid.round(5).to_string(index=False))
info = {}
if best["half_life"] != 90.0:
    cols = [f"f_xgadj_att_{int(best['half_life'])}", f"f_xgadj_def_opp_{int(best['half_life'])}"]
    a = run_variant(rows2, cols, best["l2"], TEST_SEASONS)
    b = run_variant(rows2, ADJ, L2_V1X, TEST_SEASONS)
    mean, lo, hi = block_bootstrap(per_match_logloss(a), per_match_logloss(b), a["kickoff_utc"], 14)
    info = {"half_life": best["half_life"], "l2": best["l2"], "test_minus_frozen": mean, "block14_ci": [lo, hi]}
    print(f"   dev picks {best['half_life']:.0f} d; on test vs frozen 90 d: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] (information only)")
else:
    print("   dev still picks 90 d: the edge-of-grid concern does not apply.")
res["half_life_grid"] = {"grid_dev": grid.to_dict("records"), "best": best.to_dict(), "test_info": info}
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
