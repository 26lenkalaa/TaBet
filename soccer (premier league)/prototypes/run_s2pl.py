"""Addendum S2PL of docs/PREREGISTRATION.md (dynamic ratings on the Premier League), merged in PR #100
before this file existed and before any of its test seasons were loaded.

    python prototypes/run_s2pl.py

One phase (nothing is tuned): PL 1993-94 -> 2014-15 as one continuous series; scored 1995-96 -> 1999-00
and 2008-09 -> 2014-15; D0 (Dixon-Coles refitted every matchday) against S2 (tabet/dynamic.py).

Implementation details the addendum leaves open, fixed here before the run:
  * Data: football-data E0 season files via tabet.fd_cache (codes 9394 -> 1415). A match is a row with
    both scores.
  * D0 is addendum TS's d0_forecasts (a matchday whose fit does not converge gives no D0 forecast).
  * "At least 5 earlier PL matches": earlier kickoffs in the run, any season (addendum TS's rule).
  * Pooled evidence (secondary): inverse-variance weighting of this mean (SE = CI width / 3.92) with
    addendum D's 2018-21 result (+0.0009, SE 0.0034).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_ts as TS  # noqa: E402
from tabet import dynamic as D  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet.experiments import block_bootstrap, per_match_logloss  # noqa: E402

OUT = ROOT / "results" / "s2pl"
FIRST, LAST = 1993, 2014
BLOCKS = {"1995-2000": [f"{y}-{(y + 1) % 100:02d}" for y in range(1995, 2000)],
          "2008-15": [f"{y}-{(y + 1) % 100:02d}" for y in range(2008, 2015)]}
SCORED = tuple(s for b in BLOCKS.values() for s in b)
MARGIN = 0.0020
D_PRIOR = (0.0009, 0.0034)          # addendum D, 2018-21: mean, SE


def claim():
    r = ledger.season_range
    ledger.claim("S2PL", "E0", r("1993-94", "1994-95"), "training")
    ledger.claim("S2PL", "E0", r("2000-01", "2007-08"), "training")
    ledger.claim("S2PL", "E0", r("1995-96", "1999-00"), "test")
    ledger.claim("S2PL", "E0", r("2008-09", "2014-15"), "test")


def ci(a: np.ndarray, b: np.ndarray, kickoff: pd.Series) -> dict:
    mean, lo, hi = block_bootstrap(a, b, kickoff.reset_index(drop=True), 14)
    return {"n": int(len(a)), "mean": mean, "ci_95": [lo, hi]}


def main():
    claim()
    fd_cache._load_env()
    m = TS.league_matches("E0", FIRST, LAST)
    got = sorted(m["season"].unique())
    assert got[0] == "1993-94" and got[-1] == "2014-15" and len(got) == 22, got
    d0, failed = TS.d0_forecasts(m, SCORED)
    print(f"D0 done ({len(failed)} matchdays without a converged fit)", flush=True)
    s2 = D.walk_forward(m, SCORED, summer=True)
    print("S2 done", flush=True)

    n = TS.earlier_counts(m)
    base = m[m["season"].isin(SCORED)][["match_id", "season", "kickoff_utc", "result"]].copy()
    base["eligible"] = base["match_id"].map((n["h"] >= TS.MIN_EARLIER) & (n["a"] >= TS.MIN_EARLIER)).fillna(False)
    for k, p in (("D0", d0), ("S2", s2)):
        q = p.set_index("match_id").reindex(base["match_id"]).reset_index()
        q["result"] = base["result"].to_numpy()
        ok = q["p_home"].notna().to_numpy()
        ll = np.full(len(q), np.nan)
        ll[ok] = per_match_logloss(q[ok])
        base[k] = ll
    w = base[base["eligible"] & base["D0"].notna() & base["S2"].notna()].reset_index(drop=True)
    w["kickoff_utc"] = pd.to_datetime(w["kickoff_utc"], utc=True)
    OUT.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT / "ledger.csv.gz", index=False)

    prim = ci(w["S2"].to_numpy(), w["D0"].to_numpy(), w["kickoff_utc"])
    mean, (lo, hi) = prim["mean"], prim["ci_95"]
    if hi < 0:
        decision = "better"
    elif hi < MARGIN:
        decision = "not worse"
    elif lo > 0:
        decision = "worse"
    else:
        decision = "inconclusive"
    se = (hi - lo) / 3.92
    wts = np.array([1 / se ** 2, 1 / D_PRIOR[1] ** 2])
    pooled = float((wts * np.array([mean, D_PRIOR[0]])).sum() / wts.sum())
    first = m.groupby("season")["kickoff_utc"].min()
    early = ((w["kickoff_utc"] - w["season"].map(first)).dt.days < 56).to_numpy()
    th = s2.groupby("season")[[f"theta_{k}" for k in D.NAMES]].first()

    res = {"addendum": "S2PL", "n_scored": int(len(w)), "d0_not_converged": failed,
           "excluded_ineligible": int((~base["eligible"]).sum()),
           "log_loss": {"D0": float(w["D0"].mean()), "S2": float(w["S2"].mean())},
           "primary": {"S2 - D0": prim, "decision": decision, "margin": MARGIN},
           "secondary": {
               "by block": {b: ci(g["S2"].to_numpy(), g["D0"].to_numpy(), g["kickoff_utc"])
                            for b, g in ((b, w[w["season"].isin(ss)]) for b, ss in BLOCKS.items())},
               "by season (mean, n)": {s: [float((g["S2"] - g["D0"]).mean()), int(len(g))] for s, g in w.groupby("season")},
               "early season": ci(w["S2"].to_numpy()[early], w["D0"].to_numpy()[early], w["kickoff_utc"][early]),
               "S2 parameters by season": th.round(5).reset_index().to_dict("records"),
               "pooled with addendum D (inverse variance)": {"mean": pooled, "se": float(np.sqrt(1 / wts.sum()))},
           }}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("n_scored", "log_loss", "primary")}, indent=2, default=float))
    print(json.dumps({k: v for k, v in res["secondary"].items() if k != "S2 parameters by season"}, indent=2, default=float))


if __name__ == "__main__":
    main()
