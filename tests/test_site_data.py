import json

import pandas as pd

import tabet.site_data as S


def test_logged_prefers_the_primary_record(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in (
        {"fpl_fixture": 108, "horizon_h": 26.0, "tag": "fallback"},
        {"fpl_fixture": 108, "horizon_h": 24.0, "tag": "primary"},
        {"fpl_fixture": 101, "horizon_h": 26.0, "tag": "fallback only"})) + "\n")
    monkeypatch.setattr(S, "LOG", log)
    got = S._logged()
    assert got[108]["tag"] == "primary" and got[101]["tag"] == "fallback only"


def test_rebuild_when_old_or_when_the_log_grows(tmp_path, monkeypatch):
    log, out = tmp_path / "log.jsonl", tmp_path / "matches.json"
    monkeypatch.setattr(S, "LOG", log)
    assert S.stale(out=out)  # missing file
    log.write_text('{"x": 1}\n')
    now = pd.Timestamp.now(tz="UTC")
    out.write_text(json.dumps({"built_utc": now.isoformat(), "n_log_records": 1, "matches": []}))
    assert not S.stale(20, out=out)
    log.write_text('{"x": 1}\n{"x": 2}\n')  # a forecast was logged since the build
    assert S.stale(20, out=out)
    out.write_text(json.dumps({"built_utc": (now - pd.Timedelta(hours=21)).isoformat(), "n_log_records": 2, "matches": []}))
    assert S.stale(20, out=out)


def test_a_logged_forecast_stays_after_the_match_reaches_the_results_file(tmp_path):
    """Regression: once gameweek 10 was in the results file, the site dropped Leeds v Tottenham's logged
    p0/pE, because logged records were attached only to fixtures not yet in that file."""
    out = tmp_path / "matches.json"
    S.build(preview=False, out=out)
    d = json.loads(out.read_text())
    m = [e for e in d["matches"] if e["id"] == "2026-27-20261107-leeds-tottenham"]
    assert len(m) == 1 and m[0]["status"] == "played"
    assert [f["key"] for f in m[0]["forecasts"] if f["kind"] == "logged"] == ["p0", "pE"]
    missed = [e for e in d["matches"] if e["season"] == "2026-27" and e.get("gameweek") == 10
              and e["id"] != "2026-27-20261107-leeds-tottenham"]
    assert missed and all(e.get("missing") and not e["forecasts"] for e in missed)


def test_market_at_our_forecast_time_comes_from_the_same_run_only(tmp_path, monkeypatch):
    # Leeds v Tottenham was logged 2026-11-06 11:10 UTC (FPL fixture 97)
    def log(t):
        return pd.DataFrame([{"fpl_fixture": 97, "book": "Avg", "source": "the-odds-api", "n_books": 12,
                              "retrieved_utc": pd.Timestamp(t, tz="UTC"), "odds_h": 2.7, "odds_d": 3.4, "odds_a": 2.6}])
    out = tmp_path / "m.json"
    monkeypatch.setattr(S, "_odds_log", lambda: log("2026-11-06 11:11"))
    S.build(preview=False, out=out)
    m = next(e for e in json.loads(out.read_text())["matches"] if e["id"] == "2026-27-20261107-leeds-tottenham")
    mk = m["market"][0]
    assert mk["key"] == "matched" and "12 bookmakers" in mk["note"] and abs(sum(mk["p"]) - 1) < 1e-3
    monkeypatch.setattr(S, "_odds_log", lambda: log("2026-11-06 13:00"))  # a later run: not time-matched
    S.build(preview=False, out=out)
    m = next(e for e in json.loads(out.read_text())["matches"] if e["id"] == "2026-27-20261107-leeds-tottenham")
    assert all(x["key"] != "matched" for x in m["market"])
