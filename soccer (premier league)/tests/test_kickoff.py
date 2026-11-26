import json

import pandas as pd

from tabet import kickoff as K

NOW = pd.Timestamp("2026-11-28 14:30", tz="UTC")


def _fx():
    ko = [NOW + pd.Timedelta(minutes=m) for m in (5, 30, 45, 90)]
    return pd.DataFrame({"fpl_fixture": [1, 2, 3, 4], "kickoff_utc": ko, "home_id": ["a", "c", "e", "g"],
                         "away_id": ["b", "d", "f", "h"], "gameweek": 13})


def _fc(status):
    probs = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
    fixtures = []
    for f, h, a in ((2, "c", "d"), (3, "e", "f")):
        fixtures.append({"fpl_fixture": f, "gameweek": 13, "kickoff_utc": str(NOW + pd.Timedelta(minutes=30)),
                         "home_id": h, "away_id": a, **probs, "xg_home": 1.5, "xg_away": 1.0, "components": {},
                         "with_lineups": {"sources": {"rotowire": {**probs, "p_home": 0.55, "p_away": 0.15,
                                                                    "xg_home": 1.6, "xg_away": 0.9}},
                                          "lineup_codes": {"rotowire": {"home": [1], "away": [2]}}}})
    return {"fixtures": fixtures, "lineup_status": {"rotowire": status}, "git_commit": "x",
            "xi_stage2": {}, "lineup_sources": {"retrieved_utc": "20261128T1430Z"}}


def test_due_window_is_10_to_50_minutes():
    assert list(K.due(NOW, _fx())["fpl_fixture"]) == [2, 3]


def test_a_record_needs_both_xis_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "KICKOFF", tmp_path / "kickoff.jsonl")
    status = {"c": "confirmed", "d": "confirmed", "e": "confirmed", "f": "predicted"}
    recs = K.log_kickoff(NOW, _fx(), forecast_fn=lambda **kw: _fc(status))
    assert [r["fpl_fixture"] for r in recs] == [2]
    r = json.loads((tmp_path / "kickoff.jsonl").read_text())
    assert r["candidates"]["pK"]["p_home"] == 0.55 and r["candidates"]["p0K"]["p_home"] == 0.5
    # the next run: fixture 2 is done; fixture 3 gets its record once its XI is confirmed
    status["f"] = "confirmed"
    recs = K.log_kickoff(NOW + pd.Timedelta(minutes=5), _fx(), forecast_fn=lambda **kw: _fc(status))
    assert [r["fpl_fixture"] for r in recs] == [3]
    assert len((tmp_path / "kickoff.jsonl").read_text().splitlines()) == 2


def test_nothing_due_means_no_forecast_call(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "KICKOFF", tmp_path / "kickoff.jsonl")
    called = []
    far = _fx().assign(kickoff_utc=NOW + pd.Timedelta(hours=5))
    assert K.log_kickoff(NOW, far, forecast_fn=lambda **kw: called.append(1)) == [] and not called
