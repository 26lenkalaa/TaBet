"""Do teams' *pre-match* player-action profiles (dribbles, duels, tackles, key passes, passing)
predict Premier League results beyond the lineup-free model V1x?

Usage: python prototypes/run_action_profile.py TEAM_MATCH_PKL ROWS_PKL [OUT_DIR]

Data: Hugging Face soccer-dataset player-match stats aggregated to team-matches (validated: team
fouls and shots on target match football-data 98.8% / 99.4% exactly). Features at each cutoff are
decayed (half-life 90 days) means over the team's earlier matches only.
  attacking team: dribbles completed, dribble %, duel %, key passes, passes, pass %
  opponent:       tackles + interceptions, duel %
Pre-specified: one contrast, (V1x + profile) - V1x, log-loss on identical fixtures; stage-2 L2
chosen on dev 2022-24 from {1, 10, 100}; test 2024-25 (the dataset covers 2025-26 only partially).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import L2_GRID, block_bootstrap, paired_bootstrap, per_match_logloss, run_variant  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.store import RESULT_LAG  # noqa: E402

TM, ROWS = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results" / "action_profile"
out.mkdir(parents=True, exist_ok=True)
pl, _ = load_all(ROOT / "res")
fd = pl.set_index("match_id")
t = pd.read_pickle(TM)
t["kickoff_utc"] = t["match_id"].map(fd["kickoff_utc"])
t["our_team"] = [fd.at[m, "home_id"] if v == "H" else fd.at[m, "away_id"] for m, v in zip(t["match_id"], t["venue"])]
t["tackles_int"] = t["tackles_total"] + t["tackles_interceptions"]
t["drib_pct"] = t["dribbles_success"] / t["dribbles_attempts"].replace(0, np.nan)
t["duel_pct"] = t["duels_won"] / t["duels_total"]
t["pass_pct"] = t["acc"] / t["passes_total"]
ATT = ["dribbles_success", "drib_pct", "duel_pct", "passes_key", "passes_total", "pass_pct"]
DEF = ["tackles_int", "duel_pct"]
rows = pd.read_pickle(ROWS)
feat_cols = [f"a_{c}" for c in ATT] + [f"o_{c}" for c in DEF]
for c in feat_cols:
    rows[c] = np.nan
for cutoff, grp in rows.groupby("cutoff_utc"):
    h = t[t["kickoff_utc"] + RESULT_LAG <= cutoff]
    if h.empty:
        continue
    w = np.exp(-np.log(2) * (cutoff - h["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400 / 90.0)
    hh = h.assign(w=w)
    prof = hh.groupby("our_team").apply(lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()), weights=g["w"])
                                                            for c in set(ATT + DEF)}), include_groups=False)
    for c in ATT:
        rows.loc[grp.index, f"a_{c}"] = grp["team_id"].map(prof[c]).to_numpy()
    for c in DEF:
        rows.loc[grp.index, f"o_{c}"] = grp["opponent_id"].map(prof[c]).to_numpy()
ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]
dev, test = ("2022-23", "2023-24"), ("2024-25",)
l2 = min(L2_GRID, key=lambda v: per_match_logloss(run_variant(rows, ADJ + feat_cols, v, dev)).mean())
res = {"l2": l2, "feature_coverage": float(rows[feat_cols].notna().all(axis=1).mean())}
for split, seasons in (("dev", dev), ("test", test)):
    a = run_variant(rows, ADJ + feat_cols, l2, seasons).sort_values("match_id").reset_index(drop=True)
    b = run_variant(rows, ADJ, 1.0, seasons).sort_values("match_id").reset_index(drop=True)
    la, lb = per_match_logloss(a), per_match_logloss(b)
    d, lo, hi = paired_bootstrap(la, lb)
    ko = pd.to_datetime(a["kickoff_utc"], utc=True)
    res[split] = {"n": len(a), "v1x_log_loss": float(lb.mean()), "with_profile_log_loss": float(la.mean()),
                  "diff": d, "se": float(np.std(la - lb, ddof=1) / np.sqrt(len(la))),
                  "block14d_ci": list(block_bootstrap(la, lb, ko, 14)[1:])}
# descriptive: full-data coefficients (standardised) for interpretation only
from tabet.experiments import _fit_poisson  # noqa: E402
tr = rows[rows["season"].isin(dev + test)].dropna(subset=feat_cols)
X = tr[ADJ + feat_cols].astype(float)
Z = ((X - X.mean()) / X.std()).to_numpy()
b = _fit_poisson(Z, tr["log_dc"].to_numpy(), tr["goals"].to_numpy(float), l2)
res["coef_per_sd_on_log_goals"] = dict(zip(ADJ + feat_cols, [float(x) for x in b[1:]]))
print(json.dumps(res, indent=2))
(out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
