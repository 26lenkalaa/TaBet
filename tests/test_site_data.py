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
