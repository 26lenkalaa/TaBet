"""Descriptive backtest of the A10 shadow models on development data (2022-23 -> 2026-27 to date).

    python prototypes/run_shadow_backtest.py

Not a test and not a decision: amendment A10 is frozen, and nothing here may change its settings.
Same stage-2 walk-forward (tabet.experiments.run_variant) for both bases on identical rows, so the only
difference is the base rate:
  D0 base  the rows' own walk-forward Dixon-Coles rates (as the live model)
  S2 base  S2's walk-forward rates with the shadow's exact setup (tabet/shadow.py: filter from 2015-16,
           parameters fitted at each season's first kickoff, cached in shadow_s2_params.json)
Variants: p0-style (adjusted xG, L2 = 1) and pE-style with *confirmed* XIs (adjusted xG + XI, L2 = 100),
as in the V2x backtests; results-only (no stage 2) for reference.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import shadow as S  # noqa: E402
from tabet.experiments import block_bootstrap, per_match_logloss, run_variant  # noqa: E402
from tabet.forecast import ADJ, STAGE2_L2, STAGE2_L2_XI, XI  # noqa: E402
from tabet.ingest import load_all  # noqa: E402

rows = pd.read_pickle(sorted(glob.glob(str(ROOT / "data/processed/forecast_rows_*.pkl")))[-1])
matches, _ = load_all(ROOT / "res")
cutoff = rows["kickoff_utc"].max() + pd.Timedelta(days=1)
done = S._all_matches(matches, cutoff)
starts = S._season_starts(pd.concat([done, matches], ignore_index=True))
seasons = sorted(rows["season"].unique())
thetas = S.season_params(done, seasons, starts)
rates = S.past_rates(done, thetas)

s2 = rows[rows["match_id"].isin(rates.index)].copy()
lam = np.where(s2["side"] == "H", s2["match_id"].map(rates["lam_home"]), s2["match_id"].map(rates["lam_away"]))
s2["log_dc"] = np.log(lam.astype(float))
s2["rho"] = s2["season"].map({s: float(t[2]) for s, t in thetas.items()})
d0 = rows[rows["match_id"].isin(s2["match_id"])]

PERIODS = {"2022-24": ("2022-23", "2023-24"), "2024-26": ("2024-25", "2025-26"), "2026-27 to date": ("2026-27",),
           "all 2022-27": ("2022-23", "2023-24", "2024-25", "2025-26", "2026-27")}
VARIANTS = {"results only": ([], 0.0), "p0-style (xG)": (ADJ, STAGE2_L2), "pE-style (xG + confirmed XI)": (ADJ + XI, STAGE2_L2_XI)}
res = {}
for vname, (cols, l2) in VARIANTS.items():
    for pname, ss in PERIODS.items():
        a = run_variant(s2, cols, l2, ss)
        b = run_variant(d0, cols, l2, ss).set_index("match_id").reindex(a["match_id"]).reset_index()
        la, lb = per_match_logloss(a), per_match_logloss(b)
        mean, lo, hi = block_bootstrap(la, lb, a["kickoff_utc"], 14)
        res[f"{vname} | {pname}"] = {"n": int(len(a)), "S2 base": float(la.mean()), "D0 base": float(lb.mean()),
                                     "S2 - D0": mean, "ci_95": [lo, hi]}
        print(f"{vname:30s} {pname:16s} n={len(a):4d}  S2 {la.mean():.4f}  D0 {lb.mean():.4f}  "
              f"S2-D0 {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
out = ROOT / "results" / "shadow_backtest"
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps({"note": "descriptive, development data; A10 unchanged", "results": res}, indent=2))
