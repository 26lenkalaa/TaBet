"""Table stakes (F6) on the clean 2015-21 results (docs/PREREGISTRATION.md, addendum S).

Usage: python prototypes/run_table_stakes.py [OUT_DIR]   (default results/table_stakes)

V0: frozen Dixon-Coles walk-forward by matchday (2015-16 burn-in). F6: V0 rates times a stage-2
penalised Poisson (offset log V0 rate, L2 fixed at 10) on two per-side features, own `settled` and
opponent `settled`, computed from standings before the cutoff. Refitted every matchday on earlier
side-rows from 2016-17 on. Primary: F6 - V0 on 2018-19 -> 2020-21 matches where at least one side has
played 29+ matches; confirmed if mean < 0 and the 95% 14-day block bootstrap interval excludes 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import _fit_poisson, _probs, block_bootstrap  # noqa: E402
from tabet.history import load_history_matches  # noqa: E402
from tabet.ingest import LOCAL_TZ  # noqa: E402
from tabet.model import DixonColes, Hyperparams, promoted_team_seasons  # noqa: E402
from tabet.store import MatchStore  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "table_stakes"
out.mkdir(parents=True, exist_ok=True)
FORECAST = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21")
PRIMARY = ("2018-19", "2019-20", "2020-21")
BOUNDARIES = (1, 4, 6, 17)
ACTIVE_FROM = 29
L2 = 10.0
IDX = {"H": 0, "D": 1, "A": 2}


def standings(done: pd.DataFrame) -> tuple[dict, dict]:
    pts, pl = {}, {}
    for h, a, x, y in zip(done["home_id"], done["away_id"], done["home_goals"], done["away_goals"]):
        pts[h] = pts.get(h, 0) + (3 if x > y else 1 if x == y else 0)
        pts[a] = pts.get(a, 0) + (3 if y > x else 1 if x == y else 0)
        pl[h], pl[a] = pl.get(h, 0) + 1, pl.get(a, 0) + 1
    return pts, pl


def settled(team: str, teams: list[str], pts: dict, pl: dict) -> int:
    """1 if the team has played ACTIVE_FROM+ matches and can no longer cross any boundary on points."""
    if pl.get(team, 0) < ACTIVE_FROM:
        return 0
    order = sorted(teams, key=lambda t: (-pts.get(t, 0), t))
    pos = order.index(team) + 1
    mx = {t: pts.get(t, 0) + 3 * (2 * (len(teams) - 1) - pl.get(t, 0)) for t in teams}
    for k in BOUNDARIES:
        if pos <= k:
            if not all(pts.get(team, 0) > mx[u] for u in order[k:]):
                return 0
        elif not mx[team] < pts.get(order[k - 1], 0):
            return 0
    return 1


m = load_history_matches()
m = m[m["status"] == "completed"].reset_index(drop=True)
store, promoted = MatchStore(m), promoted_team_seasons(m)
teams_of = {s: sorted(set(g["home_id"]) | set(g["away_id"])) for s, g in m.groupby("season")}
target = m[m["season"].isin(FORECAST)]
rows = []
for _, g in target.groupby(target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date, sort=True):
    cutoff = g["kickoff_utc"].min()
    dc = DixonColes(Hyperparams()).fit(store.completed_before(cutoff), cutoff, promoted)
    s = g["season"].iloc[0]
    pts, pl = standings(m[(m["season"] == s) & (m["kickoff_utc"] < cutoff)])
    for f in g.itertuples():
        lh, la = dc.expected_goals(f.home_id, f.away_id, f.season)
        sh, sa = settled(f.home_id, teams_of[s], pts, pl), settled(f.away_id, teams_of[s], pts, pl)
        rows.append({"match_id": f.match_id, "season": s, "kickoff_utc": f.kickoff_utc, "cutoff_utc": cutoff,
                     "result": f.result, "home_goals": f.home_goals, "away_goals": f.away_goals,
                     "lam_home": lh, "lam_away": la, "rho": dc.rho, "settled_home": sh, "settled_away": sa,
                     "active": int(pl.get(f.home_id, 0) >= ACTIVE_FROM or pl.get(f.away_id, 0) >= ACTIVE_FROM)})
p = pd.DataFrame(rows)
print(f"V0 walk-forward done: {len(p)} matches", flush=True)

# side rows: goals, offset log V0 rate, own settled, opponent settled
side = pd.concat([
    pd.DataFrame({"match_id": p["match_id"], "cutoff_utc": p["cutoff_utc"], "goals": p["home_goals"],
                  "off": np.log(p["lam_home"]), "own": p["settled_home"], "opp": p["settled_away"], "side": "H"}),
    pd.DataFrame({"match_id": p["match_id"], "cutoff_utc": p["cutoff_utc"], "goals": p["away_goals"],
                  "off": np.log(p["lam_away"]), "own": p["settled_away"], "opp": p["settled_home"], "side": "A"}),
], ignore_index=True)

lam_f6 = {}
coefs = []
for cutoff, g in p[p["season"] != FORECAST[0]].groupby("cutoff_utc", sort=True):
    tr = side[side["cutoff_utc"] < cutoff]
    b = _fit_poisson(tr[["own", "opp"]].to_numpy(float), tr["off"].to_numpy(), tr["goals"].to_numpy(float), L2)
    coefs.append({"cutoff_utc": str(cutoff), "b0": b[0], "b_own": b[1], "b_opp": b[2]})
    for f in g.itertuples():
        lam_f6[f.match_id] = (f.lam_home * np.exp(b[0] + b[1] * f.settled_home + b[2] * f.settled_away),
                              f.lam_away * np.exp(b[0] + b[1] * f.settled_away + b[2] * f.settled_home))
p = p[p["match_id"].isin(lam_f6)].reset_index(drop=True)
P0 = _probs(p["lam_home"].to_numpy(), p["lam_away"].to_numpy(), p["rho"].to_numpy())
L6 = np.array([lam_f6[i] for i in p["match_id"]])
P6 = _probs(L6[:, 0], L6[:, 1], p["rho"].to_numpy())
y = p["result"].map(IDX).to_numpy()
p["ll_v0"] = -np.log(P0[np.arange(len(p)), y])
p["ll_f6"] = -np.log(P6[np.arange(len(p)), y])
p[["p0_home", "p0_draw", "p0_away"]] = P0
p[["f6_home", "f6_draw", "f6_away"]] = P6
p.to_csv(out / "preds.csv", index=False)
pd.DataFrame(coefs).to_csv(out / "coefficients.csv", index=False)


def contrast(df):
    mean, lo, hi = block_bootstrap(df["ll_f6"].to_numpy(), df["ll_v0"].to_numpy(), df["kickoff_utc"], 14)
    d = df["ll_f6"] - df["ll_v0"]
    return {"n": int(len(df)), "mean": mean, "se": float(d.std(ddof=1) / np.sqrt(len(d))), "ci95": [lo, hi]}


act = p[p["active"] == 1]
prim = act[act["season"].isin(PRIMARY)]
res = {"primary_f6_minus_v0_2018_21_active": contrast(prim)}
r = res["primary_f6_minus_v0_2018_21_active"]
res["primary_confirmed"] = bool(r["mean"] < 0 and r["ci95"][1] < 0)
closed = pd.Timestamp("2020-06-01", tz="UTC")
res["secondary"] = {
    "2017_18_active": contrast(act[act["season"] == "2017-18"]),
    "2018_21_active_excluding_behind_closed_doors": contrast(prim[prim["kickoff_utc"] < closed]),
    "share_settled_sides_active_2018_21": float((prim["settled_home"].sum() + prim["settled_away"].sum()) / (2 * len(prim))),
    "matches_with_a_settled_side_2018_21": int(((prim["settled_home"] + prim["settled_away"]) > 0).sum()),
    "final_coefficients": coefs[-1],
    "mean_log_loss_2018_21_active": {"v0": float(prim["ll_v0"].mean()), "f6": float(prim["ll_f6"].mean())},
}
(out / "summary.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
print(json.dumps(res, indent=2, default=str))
