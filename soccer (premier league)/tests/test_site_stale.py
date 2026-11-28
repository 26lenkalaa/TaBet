"""The site data rebuilds when a logged match kicks off or its result should be in, not only daily."""
import json

import pandas as pd

from tabet import site_data as sd


def test_rebuilds_after_kickoff_and_full_time(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"kickoff_utc": "2026-11-22T14:00:00+00:00"}) + "\n")
    monkeypatch.setattr(sd, "LOG", log)
    built = pd.Timestamp("2026-11-22T12:31Z")
    assert not sd._match_moment_passed(built, pd.Timestamp("2026-11-22T13:59Z"))
    assert sd._match_moment_passed(built, pd.Timestamp("2026-11-22T14:05Z"))          # kicked off
    after_ko = pd.Timestamp("2026-11-22T14:05Z")
    assert not sd._match_moment_passed(after_ko, pd.Timestamp("2026-11-22T16:00Z"))
    assert sd._match_moment_passed(after_ko, pd.Timestamp("2026-11-22T16:20Z"))       # result due
    assert not sd._match_moment_passed(pd.Timestamp("2026-11-22T18:05Z"), pd.Timestamp("2026-11-23T09:00Z"))
