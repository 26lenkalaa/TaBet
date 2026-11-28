"""Addendum GK script on synthetic data: keeper ratings from earlier matches only, shrinkage toward the
league rate, the usual-keeper flag, the starting keeper, the run condition, and an end-to-end
development-style run (the test runs once, so it must not be the first time the code runs)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent.parent / "prototypes"
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("run_gk", PROTO / "run_gk.py")
gk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gk)


def league(n_rounds=60, seed=0, weak="k_t1"):
    """Six teams; each team's keeper k_<team> plays every match except team t2 in rounds >= 40 (stand-in)."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(6)]
    m, k, fid, d = [], [], 1, pd.Timestamp("2021-08-07", tz="UTC")
    for rnd in range(n_rounds):
        order = rng.permutation(teams)
        for i in range(0, 6, 2):
            h, a = order[i], order[i + 1]
            mid = str(fid)
            m.append({"match_id": mid, "league": 39, "season": "2021-22", "kickoff_utc": d, "date": d.normalize(),
                      "home_id": h, "away_id": a})
            for t in (h, a):
                keeper = "stand_in" if (t == "t2" and rnd >= 40) else f"k_{t}"
                sot = int(rng.integers(3, 7))
                conc = int(rng.binomial(sot, 0.5 if keeper == weak else 0.3))
                k.append({"match_id": mid, "team_id": t, "player_id": keeper, "minutes": 90, "substitute": False,
                          "saves": sot - conc, "conceded": conc})
                k.append({"match_id": mid, "team_id": t, "player_id": f"bench_{t}", "minutes": 0, "substitute": True,
                          "saves": None, "conceded": None})
            fid += 1
        d += pd.Timedelta(days=7)
    return pd.DataFrame(m), pd.DataFrame(k)


def test_starting_keeper_and_ratings_from_earlier_matches_only():
    m, k = league()
    st = gk.starters(k)
    assert not st["player_id"].str.startswith("bench").any() and len(st) == 2 * len(m)
    f = gk.gk_features(m, k)
    first = f[f["match_id"] == "1"]
    assert first.filter(like="g_").isna().all().all()             # no league rate yet in the first round
    late = f.dropna(subset=["g_20"])
    weak = late[late["team_id"] == "t1"]["g_20"].iloc[-1]
    good = late[late["team_id"] == "t3"]["g_20"].iloc[-1]
    assert weak > 0.1 > good                                       # the weak keeper concedes more per shot
    # more shrinkage (larger k) pulls the weak keeper toward 0
    row = late[late["team_id"] == "t1"].iloc[-1]
    assert row["g_20"] > row["g_50"] > row["g_100"] > 0
    # changing a later match's numbers leaves an earlier rating unchanged
    k2 = k.copy()
    k2.loc[k2["match_id"].astype(int) > 150, "conceded"] = 0
    f2 = gk.gk_features(m, k2)
    early = f["match_id"].astype(int) <= 150
    a = f[early].sort_values(["match_id", "team_id"])["g_50"].to_numpy()
    b = f2[f2["match_id"].astype(int) <= 150].sort_values(["match_id", "team_id"])["g_50"].to_numpy()
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_usual_keeper_flag():
    m, k = league()
    f = gk.gk_features(m, k).merge(m[["match_id", "kickoff_utc"]], on="match_id")
    t2 = f[f["team_id"] == "t2"].sort_values("kickoff_utc")
    assert t2["u"].iloc[42] == 1.0                                 # the stand-in, early in his run
    assert t2["u"].iloc[45] == 0.0                                 # 5-5 tie: the more recent starter is usual
    assert t2["u"].iloc[-1] == 0.0                                 # after 10 starts he is the usual keeper
    assert (f[f["team_id"] == "t0"]["u"].dropna() == 0).all()


def test_run_condition():
    m, k = league(n_rounds=4)
    k.loc[k["match_id"] == "1", ["saves", "conceded"]] = None
    tb = {"matches": m.assign(status_short="FT")}
    k = k.assign(league=39)
    c = gk.check(tb, k, {39: "E0"}, ["2021-22"])
    assert c["E0 2021-22"]["keepers_identified_share"] == 1.0           # identities are all there
    assert c["E0 2021-22"]["league_lines_complete"] == round(22 / 24, 4)  # 2 of 24 lines lack numbers
    assert not gk.failing(c)


def test_ledger_claims_are_registered():
    gk._claim(include_test=True)


def test_end_to_end_dev_style(monkeypatch):
    from test_nlx import synthetic
    tb = synthetic()
    rng = np.random.default_rng(1)
    s = tb["starters"]
    g = s[s["pos"] == "G"]
    k = pd.DataFrame({"match_id": g["match_id"], "team_id": g["team_id"], "player_id": g["player_id"], "minutes": 90,
                      "substitute": False, "saves": rng.integers(1, 6, len(g)), "conceded": rng.integers(0, 3, len(g))})
    monkeypatch.setattr(gk.F, "dc_all", lambda m: {i: v for _, lm in m.groupby("league") for i, v in gk.D.dc_rates(lm).items()})
    leagues = {140: "SP1", 135: "I1"}
    rows = gk.side_rows(tb, k, leagues)
    assert len(rows) and {"g_opp_50", "u_opp"} <= set(rows.columns)
    w, _ = gk.N.walk_forward(rows, ["SP1", "I1"], ["2020-21", "2021-22"], gk.specs(leagues))
    assert {"LIN_10.0", "GK50_10.0"} <= set(w.columns)
    choice = {"k": 50, "l2": {"LIN": 10.0, "GK": 10.0}}
    w2, _ = gk.N.walk_forward(rows, ["SP1", "I1"], ["2021-22"], gk.specs(leagues, choice))
    assert {"LIN", "GK", "G_ONLY", "U_ONLY"} <= set(w2.columns)
