"""Addendum DXI script: parsing, no look-ahead in ratings, and an end-to-end run on synthetic data
(the test phase runs once, so it must not be the first time the code runs)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("run_dxi", Path(__file__).resolve().parent.parent / "prototypes" / "run_dxi.py")
dxi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dxi)

POS = ["G"] + ["D"] * 4 + ["M"] * 4 + ["F"] * 2


def fixture(fid, date, h, a, gh, ga, poss_h=55, status="FT", rng=None):
    rng = rng or np.random.default_rng(fid)
    def players(t):
        return [{"player": {"id": t * 100 + i}, "statistics": [{
            "games": {"position": POS[i], "minutes": 90, "substitute": False},
            "shots": {"total": int(rng.poisson(1)) if POS[i] != "G" else None}, "passes": {"key": int(rng.poisson(0.5))},
            "tackles": {"total": int(rng.poisson(1.5)), "interceptions": int(rng.poisson(1))},
            "duels": {"total": (d := int(rng.integers(4, 13))), "won": int(rng.integers(0, d + 1))}}]} for i in range(11)]
    return {"fixture": {"id": fid, "date": date, "status": {"short": status}},
            "teams": {"home": {"id": h}, "away": {"id": a}}, "goals": {"home": gh, "away": ga},
            "statistics": [{"team": {"id": h}, "statistics": [{"type": "Ball Possession", "value": f"{poss_h}%"}]},
                           {"team": {"id": a}, "statistics": [{"type": "Ball Possession", "value": f"{100 - poss_h}%"}]}],
            "lineups": [{"team": {"id": t}, "startXI": [{"player": {"id": t * 100 + i, "pos": POS[i]}} for i in range(11)]}
                        for t in (h, a)],
            "players": [{"team": {"id": t}, "players": players(t)} for t in (h, a)]}


def test_parse_fixture_reads_starters_possession_and_counts():
    m, t, s = dxi.parse_fixture(fixture(7, "2023-01-07T15:00:00+00:00", 1, 2, 2, 0, poss_h=60), 88, 2022)
    assert m["season"] == "2022-23" and m["home_goals"] == 2 and m["match_id"] == "7"
    assert {r["team_id"]: r["possession"] for r in t} == {"1": 60.0, "2": 40.0}
    assert len(s) == 22 and all(r["has_stats"] for r in s)
    gk = [r for r in s if r["pos"] == "G"][0]
    assert gk["shots"] == 0.0                     # a missing count counts as 0


def synthetic(n_seasons=3, teams=10, league=88, seed=0):
    rng = np.random.default_rng(seed)
    m, t, s, fid = [], [], [], 1
    for k in range(n_seasons):
        y = 2022 + k
        day = pd.Timestamp(f"{y}-08-06", tz="UTC")
        pairs = [(h, a) for h in range(1, teams + 1) for a in range(1, teams + 1) if h != a]
        rng.shuffle(pairs)
        for i in range(0, len(pairs), teams // 2):
            for h, a in pairs[i:i + teams // 2]:
                f = fixture(fid, day.isoformat(), h, a, int(rng.poisson(1.5)), int(rng.poisson(1.1)),
                            poss_h=int(rng.integers(35, 66)), rng=rng)
                mm, tt, st = dxi.parse_fixture(f, league, y)
                mm["has_details"] = True
                m.append(mm), t.extend(tt), s.extend(st)
                fid += 1
            day += pd.Timedelta(days=7)
    tb = {"matches": pd.DataFrame(m), "teams": pd.DataFrame(t), "starters": pd.DataFrame(s)}
    tb["starters"]["has_stats"] = tb["starters"]["has_stats"].astype("boolean")
    tb["matches"]["has_details"] = tb["matches"]["has_details"].astype("boolean")
    return tb


@pytest.fixture(scope="module")
def prepared():
    tb = synthetic()
    return tb, *dxi.prepare(tb, {88: "N1"})


def test_completeness_and_field_checks(prepared):
    tb = prepared[0]
    comp = dxi.completeness(tb, {88: "N1"})
    assert all(v["share"] == 1.0 for v in comp.values())
    f = dxi.field_checks(tb)
    assert f["won_gt_total_rows"] == 0 and f["positions"]["D"] > 0


def test_ratings_use_earlier_dates_only(prepared):
    _, tb, st, _ = prepared
    first = st["date"].min()
    assert (st.loc[st["date"] == first, ["minutes_prev", "att_prev"]] == 0).all().all()
    p = st[st["player_id"] == "105"].sort_values("date")
    earlier = np.array([(p["date"] < d).sum() for d in p["date"]])   # matches on strictly earlier dates
    assert (p["minutes_prev"].to_numpy() == 90 * earlier).all()      # nothing from the same date or later


def test_end_to_end_walk_forward(prepared, monkeypatch):
    monkeypatch.setattr(dxi, "MIN_STAGE2", 100)      # the synthetic league is smaller than a real one
    _, tb, st, dc = prepared
    rows = dxi.side_rows(tb, st, dc, 900, 900, 20)
    assert len(rows) and set(rows.side) == {"H", "A"} and rows[["att", "opp_bw", "opp_duel"]].notna().all().all()
    specs = {n: (f, 10) for n, f in dxi.MODELS.items()}
    w = dxi.walk_forward(rows, ["N1"], specs)
    assert len(w) and np.isfinite(w[["DC", "BASE", "DEF", "BW", "DUEL"]].to_numpy()).all()
    mean, lo, hi = dxi.block_ci(w.assign(primary=w.DEF - w.BASE), "primary", reps=200)
    assert lo <= mean <= hi
