"""Addendum MGR script on synthetic data. The test runs once, so it must not be the first time the code
runs: with no appointments MGR is exactly LO; a team that changes after an appointment is tracked faster;
only matches after the appointment move; affected matches are the next 10; names match; nothing leaks."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent.parent / "prototypes"
sys.path.insert(0, str(PROTO))
spec = importlib.util.spec_from_file_location("run_mgr", PROTO / "run_mgr.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)


def synthetic(seed=0, jump=0.6, when="2021-09-01"):
    """One league, 12 teams, weekly rounds; team t0 gets much stronger from `when`."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(12)]
    base = {t: rng.normal(0, 0.2) for t in teams}
    rows, fid, t = [], 1, pd.Timestamp("2019-08-03", tz="UTC")
    T = pd.Timestamp(when, tz="UTC")
    for _ in range(140):
        order = rng.permutation(teams)
        for i in range(0, 12, 2):
            h, a = order[i], order[i + 1]
            sh = base[h] + (jump if (h == "t0" and t >= T) else 0)
            sa = base[a] + (jump if (a == "t0" and t >= T) else 0)
            rows.append({"match_id": str(fid), "league": 39, "season": f"{t.year}-{(t.year + 1) % 100:02d}",
                         "kickoff_utc": t, "status_short": "FT", "home_id": h, "away_id": a,
                         "x": float(rng.poisson(np.exp(0.35 + sh - sa))), "y": float(rng.poisson(np.exp(0.1 + sa - sh))),
                         "round": "", "kind": "league", "neutral": False, "div": "E0"})
            fid += 1
        t += pd.Timedelta(days=7)
    m = pd.DataFrame(rows)
    m["lg_h"] = m["lg_a"] = m["league"]
    ev = pd.DataFrame({"league_id": [39], "api_team": ["t0"], "date": [T]})
    return m, ev


def test_no_appointments_is_exactly_lo():
    m, _ = synthetic()
    cut = pd.Timestamp("2021-06-01", tz="UTC")
    tr = m[(m["kickoff_utc"] < cut) & (m["kickoff_utc"] >= cut - mg.X.WINDOW)]
    a, b = mg.fit(tr, cut, None, 10.0), mg.X.fit(tr, cut, 0.0, 0.0, "A", None)
    for e in (("t0", 39), ("t5", 39)):
        assert abs(a["att"][e] - b["att"][e]) < 1e-4 and abs(a["def"][e] - b["def"][e]) < 1e-4
    assert abs(a["home"][39] - b["home"][39]) < 1e-4


def test_shift_tracks_the_change_and_only_after_the_date():
    m, ev = synthetic()
    cut = pd.Timestamp("2021-11-15", tz="UTC")        # about 10 rounds after the jump
    tr = m[(m["kickoff_utc"] < cut) & (m["kickoff_utc"] >= cut - mg.X.WINDOW)]
    lo, mgr = mg.fit(tr, cut, None, 10.0), mg.fit(tr, cut, ev, 3.0)
    (dt, sa, sd), = mgr["shifts"][("t0", 39)]
    assert sa + sd > 0.15                               # stronger after the appointment
    nxt = m[(m["kickoff_utc"] >= cut)].head(6)
    before = m[(m["kickoff_utc"] < ev["date"].iloc[0]) & ((m["home_id"] == "t0") | (m["away_id"] == "t0"))].tail(1)
    r_lo, r_mg = mg.rates(lo, nxt), mg.rates(mgr, nxt)
    t0 = nxt[(nxt["home_id"] == "t0") | (nxt["away_id"] == "t0")]["match_id"]
    other = nxt[~nxt["match_id"].isin(t0)]["match_id"]
    assert not np.allclose(r_lo.set_index("match_id").loc[t0, "lh"], r_mg.set_index("match_id").loc[t0, "lh"])
    # a match before the appointment date would not get the shift
    pre = mg.rates(mgr, before).iloc[0]
    base = mg.rates(dict(mgr, shifts={}), before).iloc[0]
    assert np.isclose(pre["lh"], base["lh"]) and np.isclose(pre["la"], base["la"])
    assert len(other)


def test_affected_is_the_next_ten_and_walk_forward_has_no_leak():
    m, ev = synthetic()
    aff = mg.affected(m, ev)
    t0 = m[((m["home_id"] == "t0") | (m["away_id"] == "t0")) & (m["kickoff_utc"] >= ev["date"].iloc[0])]
    assert aff == set(t0.sort_values("kickoff_utc")["match_id"].head(10))
    cfg = {"LO": (None, 0.0), "MGR": (ev, 3.0)}
    r1 = mg.walk_forward(m, ["E0"], ["2021-22"], aff, cfg, workers=1)
    m2 = m.copy()
    late = m2["kickoff_utc"] >= pd.Timestamp("2021-11-01", tz="UTC")
    m2.loc[late, "y"] += 2
    r2 = mg.walk_forward(m2, ["E0"], ["2021-22"], aff, cfg, workers=1)
    early = set(m.loc[m["kickoff_utc"] < pd.Timestamp("2021-10-25", tz="UTC"), "match_id"]) & aff
    a = r1[r1["match_id"].isin(early)].sort_values(["model", "match_id"]).reset_index(drop=True)
    b = r2[r2["match_id"].isin(early)].sort_values(["model", "match_id"]).reset_index(drop=True)
    assert len(a) and np.allclose(a[["lh", "la"]], b[["lh", "la"]])
    w = mg.X.scored(m, r1)
    assert {"LO", "MGR"} <= set(w.columns) and set(w["match_id"]) <= aff


def test_names_match_by_normalised_and_closest_name():
    names = pd.DataFrame({"league": [39] * 4, "season": ["2023-24"] * 4, "team_id": ["50", "47", "39", "33"],
                          "name": ["Manchester City", "Tottenham", "Wolves", "Manchester United"]})
    ch = pd.DataFrame({"league": ["E0"] * 4, "season": ["2023-24"] * 4,
                       "team_name": ["Manchester City", "Tottenham Hotspur", "Wolverhampton Wanderers", "Nowhere FC"]})
    got = mg.match_names(ch, names, {"E0": 39})["api_team"].tolist()
    assert got[:2] == ["50", "47"] and pd.isna(got[3])
    mg.ALIASES["Wolverhampton Wanderers"] = "Wolves"
    assert mg.match_names(ch, names, {"E0": 39})["api_team"].tolist()[2] == "39"
    mg.ALIASES.clear()


def test_ledger_claims_are_registered():
    mg._claim(include_test=False)      # MGR's test was withdrawn after development
