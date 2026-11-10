"""Amendment A7: The Odds API snapshot at each forecast record's cutoff (benchmark only)."""

import gzip
import json

import pandas as pd
import pytest

from tabet import market_benchmark as MB
from tabet import odds_archive as OA

KEY = "test-key-0123456789abcdef"
NOW = pd.Timestamp("2026-11-20 10:31", tz="UTC")


def _event(eid, home, away, ko, books):
    return {"id": eid, "sport_key": "soccer_epl", "commence_time": ko, "home_team": home, "away_team": away,
            "bookmakers": [{"key": k, "title": k, "last_update": "2026-11-20T10:25:00Z",
                            "markets": [{"key": "h2h", "last_update": "2026-11-20T10:25:00Z",
                                         "outcomes": [{"name": home, "price": h}, {"name": away, "price": a},
                                                      {"name": "Draw", "price": d}]}]} for k, (h, d, a) in books.items()]}


BODY = json.dumps([
    _event("e1", "Manchester City", "Fulham", "2026-11-21T12:30:00Z",
           {"pinnacle": (1.30, 6.0, 10.0), "williamhill": (1.28, 5.8, 9.5), "betfair_ex_uk": (1.32, 6.2, 11.0)}),
    _event("e2", "Aston Villa", "Sunderland", "2026-11-21T15:00:00Z", {"skybet": (1.8, 3.8, 4.5)}),
    _event("e3", "Atlantis FC", "Fulham", "2026-11-22T15:00:00Z", {"skybet": (2.0, 3.4, 3.6)}),
    _event("e4", "Chelsea", "Arsenal", "2026-11-20T09:00:00Z", {"skybet": (2.5, 3.4, 2.8)}),  # already kicked off
]).encode()
HEADERS = {"x-requests-remaining": "496", "x-requests-used": "4", "x-requests-last": "2"}


def _fx():
    return pd.DataFrame({"fpl_fixture": [108, 101, 200], "gameweek": 11,
                         "home_id": ["man_city", "aston_villa", "chelsea"], "away_id": ["fulham", "sunderland", "arsenal"],
                         "kickoff_utc": pd.to_datetime(["2026-11-21 12:30", "2026-11-21 15:00", "2026-11-20 09:00"], utc=True)})


def _rec(f=108, logged="2026-11-20T10:30:00+00:00", ko="2026-11-21T12:30:00+00:00"):
    return {"fpl_fixture": f, "logged_utc": logged, "kickoff_utc": ko}


def _fetch(calls, body=BODY, headers=HEADERS):
    def f(key):
        calls.append(key)
        return body, headers
    return f


def test_parse_builds_a_bookmaker_average_and_keeps_unmatched():
    rows, unmatched = OA.api_parse(BODY, NOW, _fx())
    avg = rows[rows.book == "Avg"].set_index("event_id")
    assert avg.loc["e1", "fpl_fixture"] == 108 and avg.loc["e1", "n_books"] == 3
    assert avg.loc["e1", ["odds_h", "odds_d", "odds_a"]].tolist() == pytest.approx([1.30, 6.0, (10 + 9.5 + 11) / 3], abs=1e-4)
    assert set(rows.loc[rows.event_id == "e1", "book"]) == {"pinnacle", "williamhill", "betfair_ex_uk", "Avg"}
    assert "e4" not in set(rows.event_id)  # kicked off before the call
    assert unmatched["home_team_src"].tolist() == ["Atlantis FC"]
    assert rows.loc[rows.event_id == "e3", "fpl_fixture"].isna().all()


def test_disabled_without_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr(OA, "api_key", lambda: None)
    assert OA.archive_api(NOW, _fx(), [_rec()], out_dir=tmp_path)["status"] == "disabled"
    assert not any(tmp_path.iterdir())


def test_calls_only_when_a_record_was_just_logged(tmp_path):
    calls = []
    assert OA.archive_api(NOW, _fx(), [], out_dir=tmp_path, key=KEY, fetch=_fetch(calls))["status"] == "not needed"
    r = OA.archive_api(NOW, _fx(), [_rec()], out_dir=tmp_path, key=KEY, fetch=_fetch(calls))
    assert r["status"] == "ok" and r["credits_remaining"] == "496" and r["missing"] == []
    # the next run finds the record covered and makes no call
    assert OA.archive_api(NOW + pd.Timedelta(minutes=30), _fx(), [_rec()], out_dir=tmp_path, key=KEY,
                          fetch=_fetch(calls))["status"] == "not needed"
    assert len(calls) == 1
    raw = list((tmp_path / "raw").glob("*_oddsapi.json.gz"))
    assert len(raw) == 1 and gzip.decompress(raw[0].read_bytes()) == BODY


def test_a_failed_call_is_retried_next_run_and_the_key_never_reaches_disk(tmp_path):
    def boom(key):
        raise RuntimeError(f"HTTP 401 Unauthorized for apiKey={key}")
    r = OA.archive_api(NOW, _fx(), [_rec()], out_dir=tmp_path, key=KEY, fetch=boom)
    assert r["status"] == "error" and KEY not in r["error"]
    assert OA.archive_api(NOW + pd.Timedelta(minutes=10), _fx(), [_rec()], out_dir=tmp_path, key=KEY,
                          fetch=_fetch([]))["status"] == "skipped"
    assert OA.archive_api(NOW + pd.Timedelta(minutes=30), _fx(), [_rec()], out_dir=tmp_path, key=KEY,
                          fetch=_fetch([]))["status"] == "ok"
    for p in tmp_path.rglob("*"):
        if p.is_file():
            data = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            assert KEY.encode() not in data


def test_stops_below_the_credit_floor_until_the_month_changes(tmp_path):
    calls = []
    low = dict(HEADERS, **{"x-requests-remaining": "40"})
    OA.archive_api(NOW, _fx(), [_rec()], out_dir=tmp_path, key=KEY, fetch=_fetch(calls, headers=low))
    later = _rec(101, "2026-11-21T12:00:00+00:00", "2026-11-22T15:00:00+00:00")
    r = OA.archive_api(pd.Timestamp("2026-11-21 12:01", tz="UTC"), _fx(), [later], out_dir=tmp_path, key=KEY,
                       fetch=_fetch(calls))
    assert r["status"] == "credit floor" and len(calls) == 1
    december = _rec(101, "2026-12-01T09:00:00+00:00", "2026-12-02T15:00:00+00:00")
    r = OA.archive_api(pd.Timestamp("2026-12-01 09:01", tz="UTC"), _fx(), [december], out_dir=tmp_path, key=KEY,
                       fetch=_fetch(calls))
    assert r["status"] == "ok" and len(calls) == 2


def test_api_key_reads_the_environment_first(tmp_path, monkeypatch):
    monkeypatch.setattr(OA, "API_KEY_FILE", tmp_path / "missing")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert OA.api_key() is None
    (tmp_path / "k").write_text(" file-key \n")
    monkeypatch.setattr(OA, "API_KEY_FILE", tmp_path / "k")
    assert OA.api_key() == "file-key"
    monkeypatch.setenv("ODDS_API_KEY", "env-key")
    assert OA.api_key() == "env-key"


def test_matched_snapshot_from_the_same_run_counts_and_sources_stay_separate(tmp_path):
    OA.archive_api(NOW, _fx(), [_rec()], out_dir=tmp_path, key=KEY, fetch=_fetch([]))
    log = OA.load(tmp_path)
    assert set(log.source) == {"the-odds-api"}
    recs = pd.DataFrame([_rec()])
    s = MB.select_snapshots(recs, log, source="the-odds-api", with_last=False)
    assert len(s) == 1 and not s["later"].iloc[0]  # retrieved 1 minute after the cutoff, same run
    assert s["hours_after_cutoff"].iloc[0] == pytest.approx(1 / 60)
    assert MB.select_snapshots(recs, log, source="football-data").empty
    late = pd.DataFrame([_rec(logged="2026-11-20T10:00:00+00:00")])  # 31 min before the call: a different run
    assert MB.select_snapshots(late, log, source="the-odds-api", with_last=False)["later"].iloc[0]


def test_status_reports_credits_and_low_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(OA, "api_key", lambda: KEY)
    assert OA.api_status(tmp_path)["odds_api_last"] is None
    OA.archive_api(pd.Timestamp.now(tz="UTC").floor("min"), _fx(), [], out_dir=tmp_path, key=KEY,
                   fetch=_fetch([], headers=dict(HEADERS, **{"x-requests-remaining": "12"})), force=True)
    st = OA.api_status(tmp_path)
    assert st["odds_api_enabled"] and st["odds_api_credits"]["remaining"] == 12 and st["odds_api_low"]
