"""Addenda XC and CAP script on synthetic data. The test runs once, so it must not be the first time the
code runs: LO must reproduce the current Dixon-Coles, cross matches must carry information, variant B
must drop unrated opponents, the cap must bite, and no later result may reach a forecast."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent.parent / "prototypes"
sys.path.insert(0, str(PROTO))
spec = importlib.util.spec_from_file_location("run_xc", PROTO / "run_xc.py")
xc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xc)
from tabet.model import DixonColes, Hyperparams  # noqa: E402


def league(lg, teams, strength, start, rounds, rng, fid):
    rows, t = [], pd.Timestamp(start, tz="UTC")
    for _ in range(rounds):
        order = rng.permutation(teams)
        for i in range(0, len(order) - 1, 2):
            h, a = order[i], order[i + 1]
            lh, la = np.exp(0.35 + strength[h] - strength[a]), np.exp(0.1 + strength[a] - strength[h])
            rows.append({"match_id": str(fid), "league": lg, "season": f"{t.year}-{(t.year + 1) % 100:02d}",
                         "kickoff_utc": t, "status_short": "FT", "home_id": h, "away_id": a,
                         "x": float(rng.poisson(lh)), "y": float(rng.poisson(la)), "round": "Regular Season",
                         "kind": "league", "neutral": False})
            fid += 1
        t += pd.Timedelta(days=7)
    return rows, fid


def synthetic(seed=0, cross=True):
    rng = np.random.default_rng(seed)
    A = [f"a{i}" for i in range(10)]
    B = [f"b{i}" for i in range(10)]
    strength = {**{t: rng.normal(0.2, 0.25) for t in A}, **{t: rng.normal(-0.3, 0.25) for t in B}, "u0": -0.8}
    ra, fid = league(39, A, strength, "2020-08-01", 120, rng, 1)
    rb, fid = league(140, B, strength, "2020-08-02", 120, rng, fid)
    rows = ra + rb
    if cross:
        t = pd.Timestamp("2020-08-05", tz="UTC")
        for k in range(160):
            h, a = (rng.choice(A), rng.choice(B)) if k % 2 else (rng.choice(B), rng.choice(A))
            if k % 10 == 0:
                a = "u0"                   # a club from a league we don't hold
            lh, la = np.exp(0.3 + strength[h] - strength[a]), np.exp(0.1 + strength[a] - strength[h])
            rows.append({"match_id": str(fid), "league": 2, "season": "x", "kickoff_utc": t, "status_short": "FT",
                         "home_id": h, "away_id": a, "x": float(rng.poisson(lh)), "y": float(rng.poisson(la)),
                         "round": "Final" if k == 5 else "Group", "kind": "europe", "neutral": k == 5})
            fid += 1
            t += pd.Timestamp("2020-08-12", tz="UTC") - pd.Timestamp("2020-08-05", tz="UTC")
    m = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)
    m["div"] = m["league"].map({39: "E0", 140: "SP1"})
    return xc.current_league(m), strength


def test_lo_reproduces_the_current_dixon_coles():
    m, _ = synthetic(cross=False)
    cut = pd.Timestamp("2022-06-01", tz="UTC")
    tr = m[(m["kickoff_utc"] < cut) & (m["kickoff_utc"] >= cut - xc.WINDOW)]
    par = xc.fit(tr, cut, 0.0, 0.0, "A", None)
    lg = tr[tr["league"] == 39].assign(status="completed", home_goals=lambda d: d["x"], away_goals=lambda d: d["y"])
    dc = DixonColes(Hyperparams()).fit(lg, cut)
    for t in ("a0", "a5"):
        assert abs(par["att"][(t, 39)] - dc.attack[t]) < 2e-3 and abs(par["def"][(t, 39)] - dc.defence[t]) < 2e-3
    assert abs(par["home"][39] - dc.home_adv) < 2e-3 and abs(par["rho"][39] - dc.rho) < 2e-3


def test_cross_matches_carry_information_and_variant_b_drops_unrated():
    m, strength = synthetic()
    cut = pd.Timestamp("2022-06-01", tz="UTC")
    tr = m[(m["kickoff_utc"] < cut) & (m["kickoff_utc"] >= cut - xc.WINDOW)]
    lo = xc.fit(tr, cut, 0.0, 0.0, "A", None)
    a = xc.fit(tr, cut, 1.0, 0.0, "A", None)
    b = xc.fit(tr, cut, 1.0, 0.0, "B", None)
    assert any(e[1] == "U" for e in a["att"]) and not any(e[1] == "U" for e in b["att"])
    assert a["converged"] and b["converged"]
    # ratings move once cross matches enter, and stay ordered like the truth within a league
    moved = np.mean([abs(a["att"][(t, 39)] - lo["att"][(t, 39)]) for t in [f"a{i}" for i in range(10)]])
    assert moved > 1e-3
    true = [strength[f"a{i}"] for i in range(10)]
    est = [a["att"][(f"a{i}", 39)] + a["def"][(f"a{i}", 39)] for i in range(10)]
    assert np.corrcoef(true, est)[0, 1] > 0.6


def test_cap_changes_the_fit():
    m, _ = synthetic(cross=False)
    m.loc[m.index[100], "x"] = 9.0
    cut = pd.Timestamp("2022-06-01", tz="UTC")
    tr = m[(m["kickoff_utc"] < cut) & (m["kickoff_utc"] >= cut - xc.WINDOW)]
    t = m.loc[m.index[100], "home_id"]
    lg = int(m.loc[m.index[100], "league"])
    full, capped = xc.fit(tr, cut, 0, 0, "A", None), xc.fit(tr, cut, 0, 0, "A", 3)
    assert capped["att"][(t, lg)] < full["att"][(t, lg)]


def test_walk_forward_uses_only_earlier_matches(monkeypatch):
    m, _ = synthetic()
    cfg = {"LO": (0.0, 0.0, "A", None), "XC": (1.0, 0.0, "B", None)}
    r1 = xc.walk_forward(m, ["E0"], ["2021-22"], cfg, workers=1)
    later = m["kickoff_utc"] >= pd.Timestamp("2022-03-01", tz="UTC")
    m2 = m.copy()
    m2.loc[later, "x"] = m2.loc[later, "x"] + 3           # rewrite every later result
    r2 = xc.walk_forward(m2, ["E0"], ["2021-22"], cfg, workers=1)
    early = set(m.loc[(m["kickoff_utc"] < pd.Timestamp("2022-02-21", tz="UTC")) & (m["div"] == "E0"), "match_id"])
    a = r1[r1["match_id"].isin(early)].sort_values(["model", "match_id"]).reset_index(drop=True)
    b = r2[r2["match_id"].isin(early)].sort_values(["model", "match_id"]).reset_index(drop=True)
    assert len(a) and np.allclose(a[["lh", "la", "rho"]], b[["lh", "la", "rho"]])
    w = xc.scored(m, r1)
    assert {"LO", "XC", "div", "season", "date", "early"} <= set(w.columns) and len(w) > 100


def test_ledger_claims_are_registered():
    xc._claim("XC", include_test=True)
    xc._claim("CAP", include_test=False)
