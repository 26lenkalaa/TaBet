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
    (tmp_path / "index.json").write_text("{}")  # the per-season index must exist too
    monkeypatch.setattr(S, "_newer_odds", lambda built: False)
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
    # missed by the logger: flagged, and shown with the forecast run saved before kickoff (5 Nov), never a logged one
    assert missed and all(e.get("missing") for e in missed)
    assert all(e["forecasts"] and {f["kind"] for f in e["forecasts"]} == {"saved"} for e in missed)


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


def test_build_writes_one_file_per_season_and_lineups_for_logged_matches(tmp_path):
    out = tmp_path / "matches.json"
    S.build(preview=False, out=out)
    ix = json.loads((tmp_path / "index.json").read_text())
    assert ix["seasons"][-1] == "2026-27" and set(ix["files"]) == set(ix["seasons"])
    full = json.loads(out.read_text())["matches"]
    per = sum(len(json.loads((tmp_path / f).read_text())["matches"]) for f in ix["files"].values())
    assert per == len(full)  # the split loses nothing
    lu = json.loads((tmp_path / ix["lineup_files"]["2026-27"]).read_text())["lineups"]
    lt = lu["2026-27-20261107-leeds-tottenham"]
    assert set(lt["pred"]) <= {"ffs", "rotowire"} and lt["pred"]
    for xi in [lt["actual"], *lt["pred"].values()]:  # [code, name, position] per player
        for side in ("home", "away"):
            assert len(xi[side]) == 11 and all(p[2] in ("GK", "DEF", "MID", "FWD") for p in xi[side])
            assert sum(p[2] == "GK" for p in xi[side]) == 1
    # actual XIs exist for completed matches of every season
    for season, f in ix["lineup_files"].items():
        if season < "2026-27":
            assert len(json.loads((tmp_path / f).read_text())["lineups"]) >= 370


def test_mis_decoded_player_names_are_repaired_for_display():
    assert S._fix_text("A. DoucourÃ©") == "A. Doucouré"
    assert S._fix_text("A. BegoviÄ\x87") == "A. Begović"
    assert S._fix_text("A. GrÃ¸nbÃ¦k") == "A. Grønbæk"
    assert S._fix_text("Ødegaard") == "Ødegaard"  # already correct: unchanged
    assert S._fix_text("Saka") == "Saka"
    assert S._short("Abdoulaye DoucourÃ©") == "A. Doucouré"
