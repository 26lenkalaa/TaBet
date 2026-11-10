import numpy as np
import pandas as pd
import pytest

from tabet.prospective import fit_alpha, summarise


def _scores(n=400, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p0 = rng.dirichlet([5, 3, 3])
        truth = rng.dirichlet([5, 3, 3]) if informative else p0
        pe = 0.5 * p0 + 0.5 * truth  # pE carries half of the true signal
        y = "HDA"[rng.choice(3, p=truth)]
        for name, p in (("p0", p0), ("pE", pe)):
            rows.append({"fpl_fixture": i, "gameweek": 10 + i // 40, "horizon_h": 24.0, "candidate": name,
                         "result": y, "log_loss": -np.log(p["HDA".index(y)]),
                         "p_home": p[0], "p_draw": p[1], "p_away": p[2]})
    return pd.DataFrame(rows)


def test_alpha_moves_toward_informative_source_and_stays_bounded():
    s = _scores()
    a = fit_alpha(s, through_gameweek=19)
    assert 0.5 < a <= 1.0
    assert fit_alpha(_scores(informative=False), through_gameweek=19) == pytest.approx(0.0, abs=0.35)


def test_alpha_uses_only_earlier_gameweeks():
    s = _scores()
    a1 = fit_alpha(s, through_gameweek=12)
    later = s[s.gameweek > 12].copy()
    later["result"] = "D"  # rewrite the future
    s2 = pd.concat([s[s.gameweek <= 12], later])
    assert fit_alpha(s2, through_gameweek=12) == a1


def test_summary_counts_each_fixture_once_and_pairs_with_p0():
    s = _scores(n=50)
    t = summarise(s).set_index("candidate")
    assert t.loc["p0", "n"] == 50 and t.loc["pE", "n"] == 50
    assert t.loc["p0", "minus_p0"] == pytest.approx(0)


# ---- amendment A2 rules ----
from tabet.lineup_sources import availability_field, unavailable_codes  # noqa: E402
from tabet.prospective import ALPHA_MIN_FIXTURES, _ko  # noqa: E402


def _boot():
    return {"events": [{"id": 10, "deadline_time": "2026-11-06T18:30:00Z"},
                       {"id": 11, "deadline_time": "2026-11-21T11:00:00Z"}]}


def test_availability_field_follows_the_fixtures_own_gameweek():
    before = pd.Timestamp("2026-11-06T12:00Z")
    after = pd.Timestamp("2026-11-07T14:00Z")  # a Sunday GW10 fixture logged after the GW10 deadline
    assert availability_field(_boot(), 10, before) == "chance_next"
    assert availability_field(_boot(), 10, after) == "chance_this"
    assert availability_field(_boot(), 11, after) == "chance_next"


def test_missing_chance_counts_as_zero_unless_available():
    fp = pd.DataFrame({"player_code": [1, 2, 3, 4, 5], "status": ["i", "s", "a", "d", "d"],
                       "chance": [None, None, None, 75, 25]})
    assert unavailable_codes(fp, "chance") == {1, 2, 5}


def test_moved_kickoff_is_a_new_logging_key():
    assert _ko("2026-11-08T14:00:00Z") == _ko(pd.Timestamp("2026-11-08 14:00", tz="UTC"))
    assert _ko("2026-11-08T14:00:00Z") != _ko("2026-11-09T20:00:00Z")


def test_alpha_defaults_to_zero_when_degenerate():
    s = _scores(n=ALPHA_MIN_FIXTURES - 1)
    assert fit_alpha(s, through_gameweek=19) == 0.0
    flat = _scores(n=100, informative=False)
    flat.loc[flat.candidate == "pE", ["p_home", "p_draw", "p_away"]] = \
        flat.loc[flat.candidate == "p0", ["p_home", "p_draw", "p_away"]].to_numpy()
    assert fit_alpha(flat, through_gameweek=19) == 0.0


def test_superseded_records_are_left_out_of_the_summary():
    s = _scores(n=50).assign(superseded=False)
    s.loc[s.fpl_fixture < 5, "superseded"] = True
    t = summarise(s).set_index("candidate")
    assert t.loc["p0", "n"] == 45


def test_alpha_ignores_fitting_fixtures_played_after_the_evaluation_cutoff():
    s = _scores()
    s["kickoff_utc"] = pd.Timestamp("2026-12-01", tz="UTC") + pd.to_timedelta(s["fpl_fixture"], unit="h")
    late = s["fpl_fixture"] >= 150  # pretend these GW10-19 fixtures were postponed past the cutoff
    s.loc[late, "kickoff_utc"] = pd.Timestamp("2027-03-01", tz="UTC")
    cut = pd.Timestamp("2027-01-15", tz="UTC")
    a = fit_alpha(s, through_gameweek=19, known_before=cut)
    s2 = s.copy()
    s2.loc[late, "result"] = "D"
    assert fit_alpha(s2, through_gameweek=19, known_before=cut) == a


def test_robustness_reports_complete_case_and_leave_one_gameweek_out():
    from tabet.prospective import robustness
    s = _scores(n=80).assign(superseded=False)
    s["pE_basis"] = np.where(s["candidate"] == "pE", np.where(s["fpl_fixture"] % 4 == 0, "p0", "ffs+rotowire"), None)
    r = robustness(s)
    assert r["n"] == 80 and r["n_complete"] == 60
    assert set(r["leave_one_gameweek_out"]) == set(s["gameweek"].unique())


# ---- amendment A4: window [24, 24 + W) h, separate from the run interval ----
import json as _json  # noqa: E402

import tabet.prospective as PR  # noqa: E402


def _fx(kickoffs):
    return pd.DataFrame({"fpl_fixture": list(range(1, len(kickoffs) + 1)),
                         "kickoff_utc": pd.to_datetime(kickoffs, utc=True),
                         "home_id": "h", "away_id": "a", "gameweek": 11})


def _simulate(tmp_path, monkeypatch, fixtures, run_times, window_h):
    """Run due_fixtures at each time, appending a minimal record for each due fixture, as log_due does."""
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr(PR, "LOG", log)
    logged = {}
    for t in run_times:
        t = pd.Timestamp(t, tz="UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t)
        due = PR.due_fixtures(t, window_h=window_h, fixtures=fixtures)
        with log.open("a") as fh:
            for f in due.itertuples():
                fh.write(_json.dumps({"fpl_fixture": int(f.fpl_fixture), "horizon_h": PR.HORIZON_H,
                                      "kickoff_utc": f.kickoff_utc.isoformat(), "logged_utc": t.isoformat()}) + "\n")
                logged.setdefault(int(f.fpl_fixture), []).append(t)
    return logged, log


def test_fixture_is_logged_by_the_first_run_inside_the_window(tmp_path, monkeypatch):
    fx = _fx(["2026-11-21 12:30"])
    runs = pd.date_range("2026-11-20 00:00", "2026-11-21 12:00", freq="30min", tz="UTC")
    logged, _ = _simulate(tmp_path, monkeypatch, fx, runs, window_h=8.0)
    # the window is half-open, kickoff in [now + 24, now + 32): a run at exactly kickoff - 32 h is outside,
    # so with 30-minute runs the first one inside is 05:00
    assert logged[1] == [pd.Timestamp("2026-11-20 05:00", tz="UTC")]


def test_fixture_is_never_logged_twice(tmp_path, monkeypatch):
    fx = _fx(["2026-11-21 12:30", "2026-11-21 15:00", "2026-11-22 14:00"])
    runs = pd.date_range("2026-11-19 00:00", "2026-11-22 14:00", freq="30min", tz="UTC")
    logged, log = _simulate(tmp_path, monkeypatch, fx, runs, window_h=8.0)
    assert all(len(v) == 1 for v in logged.values()) and len(logged) == 3
    assert PR.duplicate_keys(log) == []


def test_a_7_5_hour_gap_still_catches_every_fixture_when_the_window_is_8_hours(tmp_path, monkeypatch):
    # kickoffs spread across a day; runs only every 7.5 h (worse than GitHub's gameweek-10 gaps)
    fx = _fx(list(pd.date_range("2026-11-21 00:00", periods=48, freq="30min")))
    runs = pd.date_range("2026-11-19 00:17", "2026-11-21 00:00", freq="450min", tz="UTC")
    logged, _ = _simulate(tmp_path, monkeypatch, fx, runs, window_h=8.0)
    assert sorted(logged) == list(range(1, 49))
    for i, ts in logged.items():
        ko = fx.set_index("fpl_fixture").loc[i, "kickoff_utc"]
        assert pd.Timedelta(hours=24) < ko - ts[0] <= pd.Timedelta(hours=32)


def test_a_two_hour_window_misses_fixtures_with_the_same_gaps(tmp_path, monkeypatch):
    fx = _fx(list(pd.date_range("2026-11-21 00:00", periods=48, freq="30min")))
    runs = pd.date_range("2026-11-19 00:17", "2026-11-21 00:00", freq="450min", tz="UTC")
    logged, _ = _simulate(tmp_path, monkeypatch, fx, runs, window_h=2.0)
    assert len(logged) < 48  # the gameweek-10 failure mode, reproduced


def test_moved_kickoff_is_logged_again_once_and_the_original_is_kept(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr(PR, "LOG", log)
    first = _fx(["2026-11-21 12:30"])
    for t in pd.date_range("2026-11-20 04:00", "2026-11-20 06:00", freq="30min", tz="UTC"):
        for f in PR.due_fixtures(t, window_h=8.0, fixtures=first).itertuples():
            log.open("a").write(_json.dumps({"fpl_fixture": 1, "horizon_h": 24.0, "kickoff_utc": f.kickoff_utc.isoformat()}) + "\n")
    moved = _fx(["2026-11-22 16:30"])  # broadcaster moves the match after it was logged
    for t in pd.date_range("2026-11-20 06:30", "2026-11-21 16:30", freq="30min", tz="UTC"):
        for f in PR.due_fixtures(t, window_h=8.0, fixtures=moved).itertuples():
            log.open("a").write(_json.dumps({"fpl_fixture": 1, "horizon_h": 24.0, "kickoff_utc": f.kickoff_utc.isoformat()}) + "\n")
    recs = [_json.loads(x) for x in log.read_text().splitlines()]
    assert [r["kickoff_utc"][:16] for r in recs] == ["2026-11-21T12:30", "2026-11-22T16:30"]
    assert PR.duplicate_keys(log) == []


def _simulate_both(tmp_path, monkeypatch, fixtures, run_times):
    """Both A4 windows, as log_due runs them: one record per window a fixture is due in."""
    log = tmp_path / "log2.jsonl"
    monkeypatch.setattr(PR, "LOG", log)
    for t in run_times:
        for h, w in PR.LOG_HORIZONS:
            for f in PR.due_fixtures(t, h, w, fixtures=fixtures).itertuples():
                with log.open("a") as fh:
                    fh.write(_json.dumps({"fpl_fixture": int(f.fpl_fixture), "horizon_h": h,
                                          "kickoff_utc": f.kickoff_utc.isoformat(),
                                          "hours_before": (f.kickoff_utc - t).total_seconds() / 3600}) + "\n")
    return [_json.loads(x) for x in log.read_text().splitlines()], log


def test_every_30_minutes_gives_a_fresh_primary_and_an_early_fallback(tmp_path, monkeypatch):
    fx = _fx(["2026-11-21 12:30", "2026-11-21 15:00", "2026-11-23 20:00"])
    runs = pd.date_range("2026-11-19 00:00", "2026-11-23 20:00", freq="30min", tz="UTC")
    recs, log = _simulate_both(tmp_path, monkeypatch, fx, runs)
    assert PR.duplicate_keys(log) == []
    for i in (1, 2, 3):
        hb = sorted(r["hours_before"] for r in recs if r["fpl_fixture"] == i)
        assert len(hb) == 2 and 24 < hb[0] <= 26 and 26 < hb[1] <= 32


def test_gameweek_10_gaps_still_leave_a_record_for_every_fixture(tmp_path, monkeypatch):
    runs = pd.to_datetime(["2026-11-05 21:06", "2026-11-06 02:34", "2026-11-06 11:10", "2026-11-06 18:06",
                           "2026-11-06 23:17", "2026-11-07 02:55", "2026-11-07 10:27", "2026-11-07 16:54",
                           "2026-11-07 20:58", "2026-11-08 00:46"], utc=True)
    # the nine gameweek-10 kickoffs after the first scheduled run (fixture 94's window came before it)
    fx = _fx(["2026-11-07 12:30", "2026-11-07 15:00", "2026-11-07 15:00", "2026-11-07 17:30", "2026-11-08 14:00",
              "2026-11-08 14:00", "2026-11-08 14:00", "2026-11-08 14:00", "2026-11-08 16:30"])
    recs, _ = _simulate_both(tmp_path, monkeypatch, fx, runs)
    assert {r["fpl_fixture"] for r in recs} == set(range(1, 10))


def test_scoring_uses_the_primary_record_and_falls_back_otherwise():
    rows = []
    for fix, horizons in ((1, (24.0, 26.0)), (2, (26.0,))):
        for h in horizons:
            for c in ("p0", "pE"):
                rows.append({"fpl_fixture": fix, "horizon_h": h, "candidate": c, "superseded": False})
    sel = PR.select_records(pd.DataFrame(rows))
    assert set(sel.loc[sel.fpl_fixture == 1, "horizon_h"]) == {24.0}
    assert set(sel.loc[sel.fpl_fixture == 2, "horizon_h"]) == {26.0}
    assert len(sel) == 4


def test_heartbeat_reports_next_window_misses_and_fallback_only():
    fx = _fx(["2026-11-21 12:30", "2026-11-21 15:00", "2026-11-21 17:30"])
    now = pd.Timestamp("2026-11-20 13:00", tz="UTC")
    fb = {(2, 26.0, PR._ko("2026-11-21 15:00Z"))}  # fixture 2 got only a fallback record
    hb = PR.heartbeat(now, fixtures=fx, logged=fb)
    assert hb["missed"] == [1]  # fixture 1: primary window closed at 12:30 with no record at all
    assert hb["fallback_only"] == []  # fixture 2's primary window is still open
    assert hb["next"]["fpl_fixture"] == 3 and hb["window_open_now"]
    later = PR.heartbeat(pd.Timestamp("2026-11-20 15:30", tz="UTC"), fixtures=fx, logged=fb)
    assert later["fallback_only"] == [2]
    old = _fx(["2026-11-08 14:00"])  # before A4: already in the operations log, not re-reported
    assert PR.heartbeat(pd.Timestamp("2026-11-08 05:00", tz="UTC"), fixtures=old, logged=set())["missed"] == []


def test_results_split_by_hours_before_kickoff():
    s = _scores(n=40).assign(superseded=False)
    s["hours_before"] = np.where(s["fpl_fixture"] < 15, 25.0, 30.0)
    t = PR.by_hours_before(s)
    assert set(t["hours_before"]) == {"24-26", "26-32"}
    assert t[(t.candidate == "p0")]["minus_p0"].abs().max() == 0
    assert t.groupby("hours_before")["n"].max().to_dict() == {"24-26": 15, "26-32": 25}


def test_nothing_due_keeps_columns_and_a_real_run_exits_cleanly(tmp_path, monkeypatch):
    """Regression: with no fixture in any window, due_fixtures returned a frame with no columns and
    log_due crashed on d["fpl_fixture"] (every non-dry run between gameweeks failed)."""
    monkeypatch.setattr(PR, "LOG", tmp_path / "log.jsonl")
    fx = _fx(["2026-11-21 12:30"])
    now = pd.Timestamp("2026-11-08 13:37", tz="UTC")  # nothing due for 12 days
    due = PR.due_fixtures(now, fixtures=fx)
    assert due.empty and "fpl_fixture" in due.columns
    monkeypatch.setattr(PR, "upcoming_fixtures", lambda refresh=True: fx)
    monkeypatch.setattr(PR, "snapshot_live_xg", lambda now: None)
    monkeypatch.setattr(PR, "daily_refresh", lambda now: None)  # amendment A6; tested in test_odds_archive.py
    assert PR.log_due(now=now) == []          # real run, nothing due: no crash, nothing written
    assert PR.log_due(now=now, dry_run=True) == []
    assert not (tmp_path / "log.jsonl").exists()


# ---- amendment A5 ---------------------------------------------------------
from tabet.alt.goal_distributions import _hda3, bp_grid  # noqa: E402
from tabet.prospective import alpha_report, by_record_role, robustness, _updated_days  # noqa: E402


def test_source_update_date_is_read_as_days_before_the_record():
    upd = {"ffs:leeds": "Tue 3rd Nov", "ffs:tottenham": "Wed 4th Nov", "rotowire:leeds": "no date"}
    assert _updated_days(upd, "ffs", "2026-11-04T15:00:00+00:00") == 1.0
    assert _updated_days(upd, "rotowire", "2026-11-04T15:00:00+00:00") is None
    assert _updated_days({"ffs:x": "Thu 31st Dec"}, "ffs", "2027-01-01T10:00:00+00:00") == 1.0


def test_results_are_split_by_record_role():
    s = _scores(80)
    s.loc[s["fpl_fixture"] % 2 == 1, "horizon_h"] = 26.0
    t = by_record_role(s, candidates=("pE",)).set_index("split")
    assert t.loc["record primary", "n"] == 40 and t.loc["record fallback", "n"] == 40


def test_alpha_report_gives_an_interval_containing_alpha_and_evaluation_sensitivity():
    s = _scores(800)
    r = alpha_report(s, through_gameweek=19, eval_gameweeks=(20, 29))
    assert r["interval"][0] <= r["alpha"] <= r["interval"][1]
    assert r["eval_minus_p0"]["0.0"] == pytest.approx(0.0)
    assert r["eval_minus_p0"]["1.0"] < 0  # pE carries real signal in this simulation


def test_robustness_reports_a_gameweek_sign_flip_p_value_and_works_for_pBP():
    s = _scores(200)
    s.loc[s["candidate"] == "pE", "candidate"] = "pBP"
    r = robustness(s, "pBP")
    assert 0 < r["sign_flip_p_gameweek"] <= 1 and r["n_gameweeks"] == 5


def test_bivariate_poisson_grid_sums_to_one_and_more_draws_with_shared_goals():
    lh, la = np.array([1.5]), np.array([1.1])
    p_ind = _hda3(bp_grid(lh, la, 1e-9))[0]
    p_bp = _hda3(bp_grid(lh, la, 0.15))[0]
    assert p_bp.sum() == pytest.approx(1.0) and p_bp[1] > p_ind[1]
