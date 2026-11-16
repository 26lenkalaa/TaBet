"""R6, pre-registered as addendum P of docs/PREREGISTRATION.md (committed before this ran).

Championship -> first-Premier-League-season mapping learned from every earlier promotion (ridge),
used as the promoted prior mean in Dixon-Coles (D3), against no prior (D0) and the shared offset
(D1). Development: PL 2000-01 -> 2007-08 (ridge lambda only). Scored once: PL 2008-09 -> 2014-15.

    python prototypes/run_r6_promoted_mapping.py [check]   # check = data and timing only, no scoring
Reads football-data's E0/E1 files through tabet.fd_cache (cached in Supabase, cleaned as addendum P
describes), so nothing is kept on disk.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.ingest import LOCAL_TZ  # noqa: E402
from tabet import fd_cache  # noqa: E402
from tabet.model import DixonColes, Hyperparams  # noqa: E402
from tabet.multileague import _sot_sides  # noqa: E402
from tabet.store import MatchStore  # noqa: E402
from tabet.xg_strength import fit_xg_strength  # noqa: E402

CODES = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(1999, 2015)]   # 1999-00 .. 2014-15
OUT = ROOT / "results" / "r6_promoted_mapping"
DEV_COHORTS = [f"{y}-{(y + 1) % 100:02d}" for y in range(2001, 2008)]      # 2001-02 .. 2007-08
TEST_SEASONS = [f"{y}-{(y + 1) % 100:02d}" for y in range(2008, 2015)]     # 2008-09 .. 2014-15
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
FIRST_N = 10
ALPHA = 0.05 / 3                                                             # Bonferroni over three primaries
H0, H1 = Hyperparams(), Hyperparams(promoted_prior=True)


def prev(s: str) -> str:
    """'2008-09' -> '2007-08'."""
    return f"{int(s[:4]) - 1}-{s[2:4]}"


def load():
    e0 = fd_cache.seasons(["E0"], CODES)
    e1 = {s: g for s, g in fd_cache.seasons(["E1"], CODES).groupby("season")}
    e0 = e0.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    return e0, e1


def promotions(e0, e1):
    """{(e0 team id, PL season): (x1 sot att, x2 sot def, x3 goal diff per match)} for 2001-02 on."""
    out = {}
    for s in sorted(e0["season"].unique()):
        p = prev(s)
        if p not in e1 or s < "2001-02":
            continue
        champ = e1[p][e1[p]["status"] == "completed"]
        cutoff = champ["kickoff_utc"].max() + pd.Timedelta(days=1)
        r, _, _ = fit_xg_strength(_sot_sides(champ), cutoff, 90.0)
        gd, n = {}, {}
        for side, o in (("home", "away"), ("away", "home")):
            for t, g, a in zip(champ[f"{side}_id"], champ[f"{side}_goals"], champ[f"{o}_goals"]):
                gd[t] = gd.get(t, 0) + int(g) - int(a)
                n[t] = n.get(t, 0) + 1
        champ_by_slug = {t.split("_", 1)[1]: t for t in set(champ["home_id"])}
        for t in sorted(set(e0.loc[e0["season"] == s, "home_id"])):
            c = champ_by_slug.get(t.split("_", 1)[1])
            if c is not None:
                out[(t, s)] = (float(r.loc[c, "att"]), float(r.loc[c, "def"]), gd[c] / n[c])
    return out


def season_end_ratings(e0, store, seasons):
    """Dixon-Coles ratings (no promoted prior) one day after each season's last PL match."""
    out = {}
    for s in seasons:
        end = e0.loc[e0["season"] == s, "kickoff_utc"].max() + pd.Timedelta(days=1)
        dc = DixonColes(H0).fit(store.completed_before(end), end)
        out[s] = dc
    return out


def fit_mapping(keys, feats, targets, lam):
    """Two ridge regressions (standardised features, unpenalised intercept). Returns predict(x)."""
    X = np.array([feats[k] for k in keys])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    coefs = {}
    for name, cols in (("att", [0, 2]), ("def", [1, 2])):
        y = np.array([targets[k][0 if name == "att" else 1] for k in keys])
        A = np.column_stack([np.ones(len(Z)), Z[:, cols]])
        P = lam * np.eye(A.shape[1])
        P[0, 0] = 0
        coefs[name] = np.linalg.solve(A.T @ A + P, A.T @ y)

    def predict(x):
        z = (np.asarray(x) - mu) / sd
        return (float(coefs["att"] @ np.r_[1, z[[0, 2]]]), float(coefs["def"] @ np.r_[1, z[[1, 2]]]))
    predict.coefs = {k: v.tolist() for k, v in coefs.items()}
    predict.scale = (mu.tolist(), sd.tolist())
    return predict


def block_ci(d, kickoff, alpha, block_days=14, reps=2000, seed=0):
    block = ((kickoff - kickoff.min()).dt.days // block_days).to_numpy()
    ids = np.unique(block)
    sums = np.array([d[block == i].sum() for i in ids])
    counts = np.array([(block == i).sum() for i in ids])
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(reps, len(ids)))
    boots = sums[pick].sum(1) / counts[pick].sum(1)
    return float(d.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def main(check: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    e0, e1 = load()
    store = MatchStore(e0)
    feats = promotions(e0, e1)
    per = pd.Series([s for _, s in feats]).value_counts().sort_index()
    print("promoted team-seasons per PL season:", per.to_dict())
    assert len(feats) == 42 and (per == 3).all(), "expected 3 promotions a season, 2001-02 -> 2014-15"
    promoted = set(feats)
    seasons_with_targets = sorted({s for _, s in feats if s <= "2013-14"})

    t = time.time()
    ends = season_end_ratings(e0, store, seasons_with_targets)
    targets = {k: (ends[k[1]].attack.get(k[0], 0.0), ends[k[1]].defence.get(k[0], 0.0))
               for k in feats if k[1] in ends}
    print(f"season-end targets: {len(targets)} team-seasons ({time.time() - t:.0f}s)")

    # lambda: leave-one-cohort-out on the development cohorts only
    loco = {}
    for lam in LAMBDAS:
        err = 0.0
        for c in DEV_COHORTS:
            train = [k for k in targets if k[1] in DEV_COHORTS and k[1] != c]
            f = fit_mapping(train, feats, targets, lam)
            for k in [k for k in targets if k[1] == c]:
                pa, pd_ = f(feats[k])
                err += (pa - targets[k][0]) ** 2 + (pd_ - targets[k][1]) ** 2
        loco[lam] = err
    lam = min(loco, key=loco.get)
    print("leave-one-cohort-out error by lambda:", {k: round(v, 3) for k, v in loco.items()}, "-> lambda", lam)

    # leakage-safe mapped offsets: each promoted team-season mapped from cohorts of earlier seasons only
    mapped = {}
    for k in sorted(feats, key=lambda k: k[1]):
        earlier = [j for j in targets if j[1] < k[1]]
        mapped[k] = fit_mapping(earlier, feats, targets, lam)(feats[k]) if len({j[1] for j in earlier}) >= 2 else (0.0, 0.0)

    day = e0["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    groups = [(d, g) for d, g in e0[e0["season"].isin(TEST_SEASONS) & (e0["status"] == "completed")].groupby(day)]
    if check:
        d0, fx = groups[0]
        cutoff = fx["kickoff_utc"].min()
        tr = store.completed_before(cutoff)
        tm = time.time()
        DixonColes(H0).fit(tr, cutoff)
        DixonColes(H1).fit(tr, cutoff, promoted)
        DixonColes(H1).fit(tr, cutoff, promoted, mapped, fixed_promo=True)
        print(f"check only: {len(groups)} scoring dates; one date's three fits take {time.time() - tm:.1f}s "
              f"on {len(tr)} training matches. No scoring done.")
        return

    rows = []
    for i, (d, fx) in enumerate(groups):
        cutoff = fx["kickoff_utc"].min()
        tr = store.completed_before(cutoff)
        models = {"D0": DixonColes(H0).fit(tr, cutoff),
                  "D1": DixonColes(H1).fit(tr, cutoff, promoted),
                  "D3": DixonColes(H1).fit(tr, cutoff, promoted, mapped, fixed_promo=True)}
        for f in fx.itertuples():
            row = {"match_id": f.match_id, "season": f.season, "kickoff_utc": f.kickoff_utc, "result": f.result,
                   "home_id": f.home_id, "away_id": f.away_id}
            for k, md in models.items():
                row.update(dict(zip([f"{k}_h", f"{k}_d", f"{k}_a"], md.predict_proba(f.home_id, f.away_id, f.season))))
            rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(groups)} dates", flush=True)
    P = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)
    order = pd.concat([P[["match_id", "season", "kickoff_utc", "home_id"]].rename(columns={"home_id": "team"}),
                       P[["match_id", "season", "kickoff_utc", "away_id"]].rename(columns={"away_id": "team"})])
    order = order.sort_values("kickoff_utc")
    order["n"] = order.groupby(["team", "season"]).cumcount() + 1
    order["early"] = [(t, s) in promoted and n <= FIRST_N for t, s, n in zip(order.team, order.season, order.n)]
    P["early_promoted"] = P["match_id"].map(order.groupby("match_id")["early"].any())
    for k in ("D0", "D1", "D3"):
        p = np.where(P.result == "H", P[f"{k}_h"], np.where(P.result == "D", P[f"{k}_d"], P[f"{k}_a"]))
        P[f"{k}_ll"] = -np.log(p)
    ko = pd.to_datetime(P["kickoff_utc"], utc=True)

    res = {"addendum": "P", "lambda": lam, "loco_error": loco, "n_matches": len(P),
           "n_early_promoted": int(P.early_promoted.sum()), "primaries": {}, "secondary": {}}
    E = P[P.early_promoted]
    for name, sub, a, b in (("P1 D3-D0 early promoted", E, "D3", "D0"), ("P2 D3-D1 early promoted", E, "D3", "D1"),
                            ("P3 D3-D0 all matches", P, "D3", "D0")):
        m, lo, hi = block_ci((sub[f"{a}_ll"] - sub[f"{b}_ll"]).to_numpy(), pd.to_datetime(sub.kickoff_utc, utc=True), ALPHA)
        res["primaries"][name] = {"n": len(sub), "mean": m, "ci_98.33": [lo, hi]}
        print(f"{name}: n={len(sub)} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (98.33%)")
    for s in TEST_SEASONS:
        g = E[E.season == s]
        res["secondary"][f"P1 {s}"] = {"n": len(g), "mean": float((g.D3_ll - g.D0_ll).mean()) if len(g) else None}
    m, lo, hi = block_ci((E.D1_ll - E.D0_ll).to_numpy(), pd.to_datetime(E.kickoff_utc, utc=True), 0.05)
    res["secondary"]["D1-D0 early promoted"] = {"mean": m, "ci_95": [lo, hi]}

    # final mapping's slopes, bootstrapped over team-seasons (stop rule)
    keys = [k for k in targets if k[1] <= "2013-14"]
    final = fit_mapping(keys, feats, targets, lam)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        sample = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        f = fit_mapping(sample, feats, targets, lam)
        boots.append(f.coefs["att"][1:] + f.coefs["def"][1:])
    boots = np.array(boots)
    names = ["att~sot_att", "att~gd", "def~sot_def", "def~gd"]
    point = final.coefs["att"][1:] + final.coefs["def"][1:]
    res["final_slopes"] = {n: {"coef": float(c), "ci_95": [float(np.percentile(boots[:, i], 2.5)),
                                                           float(np.percentile(boots[:, i], 97.5))]}
                           for i, (n, c) in enumerate(zip(names, point))}
    for n, v in res["final_slopes"].items():
        print(f"  slope {n}: {v['coef']:+.3f} [{v['ci_95'][0]:+.3f}, {v['ci_95'][1]:+.3f}]")

    p1, p2, p3 = (res["primaries"][k] for k in res["primaries"])
    confirmed = p1["ci_98.33"][1] < 0 and p2["ci_98.33"][1] < 0 and p3["mean"] <= 0
    slopes_zero = all(v["ci_95"][0] <= 0 <= v["ci_95"][1] for v in res["final_slopes"].values())
    res["decision"] = {"confirmed": confirmed,
                       "stop_rule_met": bool(p1["ci_98.33"][0] <= 0 <= p1["ci_98.33"][1] and slopes_zero)}
    print("decision:", res["decision"])
    P.to_csv(OUT / "ledger.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(check=sys.argv[1:] == ["check"])
