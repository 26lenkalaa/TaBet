"""Addendum FAT script on synthetic data: the fatigue inputs by hand, no use of the match itself or
later matches, the run condition, and an end-to-end development-style run (the test runs once, so it
must not be the first time the code runs)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent.parent / "prototypes"
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("run_fat", PROTO / "run_fat.py")
fat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fat)
from test_dxi import fixture  # noqa: E402
from test_nlx import synthetic  # noqa: E402


def _tables(fixtures, league):
    m, t, s = [], [], []
    for f, season in fixtures:
        mm, tt, st = fat.D.parse_fixture(f, league, season)
        mm["has_details"] = True
        for r in tt:
            r["sot"] = 4.0
        m.append(mm), t.extend(tt), s.extend(st)
    return {"matches": pd.DataFrame(m), "teams": pd.DataFrame(t), "starters": pd.DataFrame(s)}


def test_fatigue_by_hand_and_no_future():
    # team 1: league Sat 1 Mar, Europe Thu 6 Mar, league Sun 9 Mar (the target); and a later cup match
    lg = _tables([(fixture(1, "2025-03-01T15:00:00+00:00", 1, 2, 1, 0), 2024),
                  (fixture(3, "2025-03-09T15:00:00+00:00", 1, 4, 0, 0), 2024)], 140)
    co = _tables([(fixture(2, "2025-03-06T20:00:00+00:00", 1, 3, 2, 2), 2024)], 3)
    later = _tables([(fixture(4, "2025-03-12T20:00:00+00:00", 1, 5, 1, 0), 2024)], 143)
    co = {k: pd.concat([co[k], later[k]], ignore_index=True) for k in co}
    cm, cs = fat.counted(lg, co)
    tg = pd.DataFrame({"match_id": ["3"], "team_id": ["1"], "kickoff_utc": [pd.Timestamp("2025-03-09T15:00Z")]})
    r = fat.fatigue(tg, cm, cs).iloc[0]
    assert r.rest == 3 and r.load14 == 2          # Thu -> Sun; two matches in the previous 14 days
    assert r.xi7 == 1.0                           # each starter started on Thursday only (Sat 1 Mar is 8 days back)
    assert bool(r.euro4)                          # Europe 3 days earlier
    tg2 = tg.assign(match_id="1", kickoff_utc=pd.Timestamp("2025-03-01T15:00Z"))
    r2 = fat.fatigue(tg2, cm, cs).iloc[0]
    assert r2.rest == fat.REST_CAP and r2.load14 == 0 and r2.xi7 == 0 and not r2.euro4


def test_check_and_failing():
    lg = _tables([(fixture(1, "2025-03-01T15:00:00+00:00", 1, 2, 1, 0), 2024),
                  (fixture(3, "2025-03-09T15:00:00+00:00", 1, 4, 0, 0), 2024)], 140)
    f = fixture(2, "2025-03-06T20:00:00+00:00", 1, 3, 2, 2)
    f["lineups"] = []                             # a counted cup match without lineups
    co = _tables([(f, 2024)], 143)
    comp = fat.check(lg, co, {140: "SP1"}, ["2024-25"])
    assert comp["SP1 2024-25"]["league_share"] == 1.0 and comp["SP1 2024-25"]["other_share"] == 0.0
    assert "SP1 2024-25" in fat.failing(comp)


def test_end_to_end_dev_style(monkeypatch):
    tb = synthetic()
    for k in ("starters",):
        tb[k]["has_stats"] = tb[k]["has_stats"].astype("boolean")
    rng = np.random.default_rng(3)
    teams = sorted(set(tb["matches"]["home_id"]))
    cups = []
    for i, d in enumerate(pd.date_range("2018-09-01", "2022-04-01", freq="21D", tz="UTC")):
        h, a = rng.choice(teams, 2, replace=False)
        cups.append((fixture(900000 + i, (d + pd.Timedelta(days=3, hours=20)).isoformat(), int(h), int(a), 1, 1), d.year))
    ct = _tables(cups, 143)
    monkeypatch.setattr(fat, "dc_all", lambda m: {k: v for _, lm in m.groupby("league") for k, v in fat.D.dc_rates(lm).items()})
    leagues = {140: "SP1", 135: "I1"}
    rows = fat.side_rows(tb, ct, leagues)
    assert set(fat.FAT) <= set(rows.columns) and rows[fat.FAT].notna().all().all()
    assert (rows["load14_own"] >= 1).any() and (rows["xi7_own"] > 0).any()
    w, _ = fat.N.walk_forward(rows, ["SP1", "I1"], ["2020-21", "2021-22"], fat.specs(leagues))
    ch = fat.choose(w)
    assert set(ch["l2"]) == {"LIN", "FAT", "F1", "F2"}
    w2, _ = fat.N.walk_forward(rows, ["SP1", "I1"], ["2021-22"], fat.specs(leagues, ch))
    assert {"DC", "LIN", "FAT", "F1", "F2"} <= set(w2.columns)
    m, lo, hi = fat.D.block_ci(w2.assign(d=w2["FAT"] - w2["LIN"]), "d")
    assert lo <= m <= hi


def test_ledger_claims_are_registered():
    fat._claim(include_test=True)                 # raises if any role is not registered
