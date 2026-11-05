"""Frozen V0-V5 prediction ledger + paired incremental results.

Usage: python prototypes/run_fpl_experiments.py [OUT_DIR]   (default results/fpl_v0_v5)

Writes:
  ledger.csv      one row per (variant, match): probabilities, rates, cutoff,
                  stage-2 status, chosen L2. This is the frozen record.
  manifest.json   git commit, source-file hashes, parameters, L2 choices.
  results.json    scores, primary/secondary contrasts with fixture-level and
                  calendar-block bootstrap CIs, market benchmarks, draw totals.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import (DEV_SEASONS, FORWARD_SEASONS, MIN_TRAIN_MATCHES, PRIMARY_CONTRASTS,  # noqa: E402
                               SECONDARY_CONTRASTS, TEST_SEASONS, VARIANTS, block_bootstrap, build_dataset,
                               paired_bootstrap, per_match_logloss, run_variant, summary, tune_l2)
from tabet.features import FeatureParams  # noqa: E402
from tabet.fpl import RAW_DIR, load_appearances, validate_appearances  # noqa: E402
from tabet.ingest import implied_probabilities, load_all  # noqa: E402
from tabet.model import MAX_GOALS, Hyperparams  # noqa: E402

BLOCKS = (7, 14, 28)
SPLITS = {"dev": DEV_SEASONS, "test": TEST_SEASONS, "forward_2026_27": FORWARD_SEASONS}

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "fpl_v0_v5"
out.mkdir(parents=True, exist_ok=True)
m, odds = load_all(ROOT / "res")
app = load_appearances(m, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27", refresh_live=False)
problems = validate_appearances(app, m)
rows, checks = build_dataset(m, app)
best = tune_l2(rows)  # dev seasons only

ledger, results = [], {"l2_chosen_on_dev": best, "appearance_validation": problems}
markets = {
    "pre-closing avg (collected Fri/Tue afternoon; EARLIER info than a confirmed-XI forecast)":
        implied_probabilities(odds, "Avg").set_index("match_id"),
    "closing avg (LATER info than our forecast)": implied_probabilities(odds, "AvgC").set_index("match_id"),
}
for split, seasons in SPLITS.items():
    preds = {n: run_variant(rows, c, best[n], seasons) for n, c in VARIANTS.items()}
    for n, p in preds.items():
        ledger.append(p.assign(variant=n, split=split, l2=best[n]))
    ll = {n: per_match_logloss(p) for n, p in preds.items()}
    ko = pd.to_datetime(preds["V0 Dixon-Coles"]["kickoff_utc"], utc=True)
    base = preds["V0 Dixon-Coles"]
    res = {"n_matches": len(base), "n_stage2_active": int(base["stage2_active"].sum() if "stage2_active" in base else 0),
           "scores": {n: summary(p) for n, p in preds.items()}, "contrasts": {}}
    act = preds["V1 + team xG form"]["stage2_active"].to_numpy()
    res["n_stage2_active"] = int(act.sum())
    for group, cs in (("primary", PRIMARY_CONTRASTS), ("secondary", SECONDARY_CONTRASTS)):
        for name, (a, b) in cs.items():
            d, lo, hi = paired_bootstrap(ll[a], ll[b])
            entry = {"group": group, "mean": d, "fixture_ci": [lo, hi],
                     "sd_paired": float(np.std(ll[a] - ll[b], ddof=1))}
            for bd in BLOCKS:
                entry[f"block{bd}d_ci"] = list(block_bootstrap(ll[a], ll[b], ko, bd)[1:])
            res["contrasts"][name] = entry
    res["market"] = {}
    for label, mp in markets.items():
        j = base[["match_id", "result"]].join(mp, on="match_id")
        ok = j[["p_home", "p_draw", "p_away"]].notna().all(axis=1)
        mll = per_match_logloss(j[ok])
        d, lo, hi = paired_bootstrap(ll["V2 + XI strength (position prior)"][ok.to_numpy()], mll)
        res["market"][label] = {"n": int(ok.sum()), "log_loss": float(mll.mean()),
                                "V2_minus_market": d, "fixture_ci": [lo, hi]}
    res["draws"] = {n: {"predicted": float(p["p_draw"].sum()), "observed": int((p["result"] == "D").sum()),
                        "sd": float(np.sqrt((p["p_draw"] * (1 - p["p_draw"])).sum()))} for n, p in preds.items()}
    results[split] = res

led = pd.concat(ledger, ignore_index=True)
led.to_csv(out / "ledger.csv", index=False)

c = checks.dropna(subset=["observed"])
for split in ("dev", "test"):
    cc = c[c["season"].isin(SPLITS[split])]
    w = cc["minutes"].to_numpy(float)
    e_pos = (cc["pred_position"] - cc["observed"]).to_numpy() ** 2
    e_sim = (cc["pred_similar"] - cc["observed"]).to_numpy() ** 2
    idx = np.random.default_rng(0).integers(0, len(cc), size=(2000, len(cc)))
    boots = [np.average(e_sim[i], weights=w[i]) - np.average(e_pos[i], weights=w[i]) for i in idx]
    results[f"player_surrogate_{split}"] = {"n": len(cc), "wmse_position": float(np.average(e_pos, weights=w)),
                                            "wmse_similar": float(np.average(e_sim, weights=w)),
                                            "diff_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
manifest = {
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "results_sources": {f.name: sha(f) for f in sorted((ROOT / "res").glob("*.csv"))},
    "fpl_sources": {str(f.relative_to(RAW_DIR)): sha(f) for f in sorted(RAW_DIR.rglob("*")) if f.is_file()},
    "dixon_coles": asdict(Hyperparams()), "features": asdict(FeatureParams()), "max_goals": MAX_GOALS,
    "min_train_matches": MIN_TRAIN_MATCHES, "l2_chosen_on_dev": best, "splits": SPLITS,
    "ledger_sha256": sha(out / "ledger.csv"),
    "lineup_mode": "confirmed (actual starters of the predicted match)",
    "loss_definitions": "Dixon-Coles: summed decay-weighted NLL (weights normalised to mean 1) + l2*sum(r^2); "
                        "stage 2: summed Poisson NLL over training team-rows + l2*||b||^2 on standardised "
                        "features, intercept unpenalised. Both penalties are fixed while data grow, so their "
                        "relative strength weakens as the training window expands.",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
(out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")

for split in SPLITS:
    r = results[split]
    print(f"\n=== {split}: {r['n_matches']} matches, stage 2 active for {r['n_stage2_active']} ===")
    print(pd.DataFrame(r["scores"]).T[["log_loss", "brier", "rps", "goal_deviance"]].round(4).to_string())
    for name, e in r["contrasts"].items():
        cis = "  ".join(f"{k.replace('_ci', '')}[{v[0]:+.4f},{v[1]:+.4f}]" for k, v in e.items() if k.endswith("_ci"))
        print(f"  [{e['group'][:3]}] {name}: {e['mean']:+.4f} (sd {e['sd_paired']:.4f})  {cis}")
    for label, e in r["market"].items():
        print(f"  market {label[:22]}...: LL {e['log_loss']:.4f}, V2-market {e['V2_minus_market']:+.4f} "
              f"[{e['fixture_ci'][0]:+.4f},{e['fixture_ci'][1]:+.4f}]")
    d0 = r["draws"]["V0 Dixon-Coles"]; d2 = r["draws"]["V2 + XI strength (position prior)"]
    print(f"  draws: observed {d0['observed']}, V0 predicted {d0['predicted']:.1f}, V2 {d2['predicted']:.1f} (sd ~{d0['sd']:.1f})")
print("\nplayer surrogate:", {k: v for k, v in results.items() if k.startswith("player")})
print("\nwrote", out)
