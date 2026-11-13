"""Experiment F4: a promoted-team prior learned from Championship data.

Usage: python prototypes/run_promoted_prior.py [OUT_DIR]   (default results/promoted_prior)

Promoted team-season = a team in the PL season that was in the Championship (football-data E1)
the season before; all 18 promotions 2021-22 -> 2026-27 map to our team ids.
Championship features (known before the PL season): opponent-adjusted shots-on-target attack and
defence over the team's Championship season, from the same quasi-Poisson model as section 5.11
(half-life 90 days, cutoff one day after the last Championship match). Not tuned.

Variants (walk-forward per matchday, identical fixtures, Dixon-Coles settings frozen at xi=0.004, L2=10):
  D0  Dixon-Coles, no promoted prior (the frozen V0)
  D1  shared promoted offset (promo_att, promo_def) learned from earlier promoted sides
  D2  D1 + slopes on the Championship features (team-specific offset), same L2 on the slopes
Pre-specified contrasts (log-loss, paired; CI = 14-day calendar-block bootstrap):
  primary    D2 - D0 on "early promoted" matches: a promoted side within its first 10 PL matches
             of the season; 2022-23 -> 2025-26 pooled (nothing is tuned, so dev/test need no split)
  secondary  D2 - D1 on the same subset; D2 - D0 on all matches; D1 - D0 on the subset;
             each also reported for 2026-27 gameweeks 1-9 (completed matches) separately
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import block_bootstrap  # noqa: E402
from tabet.ingest import LOCAL_TZ, load_all  # noqa: E402
from tabet.leagues import DIR, load_league_season  # noqa: E402
from tabet.model import DixonColes, Hyperparams  # noqa: E402
from tabet.multileague import _sot_sides  # noqa: E402
from tabet.store import MatchStore  # noqa: E402
from tabet.teams import try_resolve_team  # noqa: E402
from tabet.xg_strength import fit_xg_strength  # noqa: E402

FIRST_N = 10
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "promoted_prior"
out.mkdir(parents=True, exist_ok=True)

m, _ = load_all(ROOT / "res")


def next_season(s: str) -> str:
    return f"{int(s[:4]) + 1}-{int(s[5:]) + 1:02d}"


# ---- promoted sets and Championship features ----
promoted, feats, info = set(), {}, []
for f in sorted(DIR.glob("E1_*.csv")):
    e = load_league_season(f)
    s_pl = next_season(e["season"].iloc[0])
    pl_teams = set(m.loc[m["season"] == s_pl, "home_id"])
    done = e[e["status"] == "completed"]
    cutoff = done["kickoff_utc"].max() + pd.Timedelta(days=1)
    r, _, _ = fit_xg_strength(_sot_sides(done), cutoff, 90.0)
    gd = {}
    for side, o in (("home", "away"), ("away", "home")):
        for t, g, a in zip(done[f"{side}_team"], done[f"{side}_goals"], done[f"{o}_goals"]):
            gd[t] = gd.get(t, 0) + g - a
    for name in sorted(set(e["home_team"])):
        tid = try_resolve_team(name)
        if tid in pl_teams:
            key = (tid, s_pl)
            promoted.add(key)
            feats[key] = (float(r.loc[e.loc[e["home_team"] == name, "home_id"].iloc[0], "att"]),
                          float(r.loc[e.loc[e["home_team"] == name, "home_id"].iloc[0], "def"]))
            info.append({"team": tid, "pl_season": s_pl, "champ_sot_att": feats[key][0], "champ_sot_def": feats[key][1],
                         "champ_goal_diff": gd[name]})
info = pd.DataFrame(info)
print(info.round(3).to_string(index=False))

# ---- walk-forward ----
store = MatchStore(m)
target = m[(m["season"] >= "2022-23") & (m["status"] == "completed")].copy()
day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
H0, H1 = Hyperparams(), Hyperparams(promoted_prior=True)
rows = []
for _, fx in target.groupby(day, sort=True):
    cutoff = fx["kickoff_utc"].min()
    train = store.completed_before(cutoff)
    models = {"D0": DixonColes(H0).fit(train, cutoff), "D1": DixonColes(H1).fit(train, cutoff, promoted),
              "D2": DixonColes(H1).fit(train, cutoff, promoted, feats)}
    for f in fx.itertuples():
        row = {"match_id": f.match_id, "season": f.season, "kickoff_utc": f.kickoff_utc, "result": f.result,
               "home_id": f.home_id, "away_id": f.away_id}
        for k, md in models.items():
            row.update(dict(zip([f"{k}_h", f"{k}_d", f"{k}_a"], md.predict_proba(f.home_id, f.away_id, f.season))))
        row["D2_b_att"], row["D2_b_def"] = models["D2"].promo_b_att, models["D2"].promo_b_def
        row["D1_off"] = (models["D1"].promo_att, models["D1"].promo_def)
        rows.append(row)
P = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)

# frozen-ledger check: D0 must reproduce V0
led = pd.read_csv(ROOT / "results/fpl_v0_v5/ledger.csv")
v0 = led[led["variant"] == "V0 Dixon-Coles"].drop_duplicates("match_id").set_index("match_id")["p_home"]
common = P["match_id"].isin(v0.index)
assert np.allclose(P.loc[common, "D0_h"], v0.reindex(P.loc[common, "match_id"]).to_numpy(), atol=1e-6), "D0 != frozen V0"

# early-promoted flag: a promoted side within its first FIRST_N PL matches of the season
order = pd.concat([P[["match_id", "season", "kickoff_utc", "home_id"]].rename(columns={"home_id": "team"}),
                   P[["match_id", "season", "kickoff_utc", "away_id"]].rename(columns={"away_id": "team"})])
order = order.sort_values("kickoff_utc")
order["n"] = order.groupby(["team", "season"]).cumcount() + 1
order["early_prom"] = [(t, s) in promoted and n <= FIRST_N for t, s, n in zip(order["team"], order["season"], order["n"])]
P["early_promoted"] = P["match_id"].map(order.groupby("match_id")["early_prom"].any())
P["any_promoted"] = [(h, s) in promoted or (a, s) in promoted for h, a, s in zip(P.home_id, P.away_id, P.season)]
for k in ("D0", "D1", "D2"):
    p = np.where(P["result"] == "H", P[f"{k}_h"], np.where(P["result"] == "D", P[f"{k}_d"], P[f"{k}_a"]))
    P[f"{k}_ll"] = -np.log(p)

res = {"n_promoted_team_seasons": len(promoted), "features": info.to_dict("records"), "contrasts": {}}
periods = {"2022-26": P["season"] < "2026-27", "2026-27 GW1-9": P["season"] == "2026-27"}
subsets = {"early promoted": P["early_promoted"], "all matches": pd.Series(True, index=P.index),
           "any promoted side": P["any_promoted"]}
print()
for pname, pm in periods.items():
    for sname, sm in subsets.items():
        g = P[pm & sm]
        for a, b in (("D2", "D0"), ("D2", "D1"), ("D1", "D0")):
            mean, lo, hi = block_bootstrap(g[f"{a}_ll"].to_numpy(), g[f"{b}_ll"].to_numpy(), g["kickoff_utc"], 14)
            se = (g[f"{a}_ll"] - g[f"{b}_ll"]).std(ddof=1) / np.sqrt(len(g))
            tag = "primary" if (pname, sname, a, b) == ("2022-26", "early promoted", "D2", "D0") else ""
            res["contrasts"][f"{pname} | {sname} | {a}-{b}"] = {"n": len(g), "mean": mean, "se": se, "ci": [lo, hi]}
            print(f"{tag:8s}{pname:14s} {sname:18s} {a}-{b}: n={len(g):4d} {mean:+.4f} (SE {se:.4f}) [{lo:+.4f}, {hi:+.4f}]")
last = P.iloc[-1]
res["final_fit"] = {"b_att": last["D2_b_att"], "b_def": last["D2_b_def"], "shared_offset_D1": list(last["D1_off"])}
res["log_loss"] = {pname: {k: float(P.loc[pm, f"{k}_ll"].mean()) for k in ("D0", "D1", "D2")} for pname, pm in periods.items()}
print("\nfinal-fit slopes (D2):", round(last["D2_b_att"], 3), round(last["D2_b_def"], 3),
      "| D1 shared offset:", [round(x, 3) for x in last["D1_off"]])
P.drop(columns=["D1_off"]).to_csv(out / "ledger.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
