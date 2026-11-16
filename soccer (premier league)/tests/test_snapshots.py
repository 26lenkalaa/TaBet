"""Lineup and live-xG snapshots are written only when their content changed (retrieved_utc ignored)."""
import json

import pandas as pd

import tabet.lineup_sources as LS
import tabet.prospective as PR


# ---- lineup snapshots ------------------------------------------------------------------------

def _patch_sources(monkeypatch, tmp_path, rows):
    monkeypatch.setattr(LS, "SNAP", tmp_path / "lineup_snapshots")
    monkeypatch.setattr(LS, "fetch", lambda src, stamp: src)
    monkeypatch.setattr(LS, "parse_ffs", lambda page: pd.DataFrame(rows["ffs"]))
    monkeypatch.setattr(LS, "parse_rotowire", lambda page, fpl: pd.DataFrame(rows["rotowire"]))


ROWS = {"ffs": [{"team_id": "arsenal", "player_code": 1, "updated": "Tue 3rd Nov"},
                {"team_id": "arsenal", "player_code": 2, "updated": "Tue 3rd Nov"}],
        "rotowire": [{"team_id": "arsenal", "player_code": 1, "updated": None}]}


def test_first_lineup_snapshot_is_written(tmp_path, monkeypatch):
    _patch_sources(monkeypatch, tmp_path, ROWS)
    out = LS.snapshot(pd.DataFrame(), stamp="20261120T0430Z")
    assert [p.name for p in (tmp_path / "lineup_snapshots").iterdir()] == ["lineups_20261120T0430Z.csv"]
    assert len(out) == 3


def test_unchanged_lineups_write_no_new_file_but_still_return_the_data(tmp_path, monkeypatch):
    _patch_sources(monkeypatch, tmp_path, ROWS)
    first = LS.snapshot(pd.DataFrame(), stamp="20261120T0430Z")
    again = LS.snapshot(pd.DataFrame(), stamp="20261120T0500Z")
    assert sorted(p.name for p in (tmp_path / "lineup_snapshots").iterdir()) == ["lineups_20261120T0430Z.csv"]
    assert len(again) == len(first) and set(again["retrieved_utc"]) == {"20261120T0500Z"}   # callers get today's frame
    # the same rows in a different order are also unchanged
    _patch_sources(monkeypatch, tmp_path, {"ffs": ROWS["ffs"][::-1], "rotowire": ROWS["rotowire"]})
    LS.snapshot(pd.DataFrame(), stamp="20261120T0530Z")
    assert len(list((tmp_path / "lineup_snapshots").iterdir())) == 1


def test_changed_lineups_write_a_new_file(tmp_path, monkeypatch):
    _patch_sources(monkeypatch, tmp_path, ROWS)
    LS.snapshot(pd.DataFrame(), stamp="20261120T0430Z")
    changed = {"ffs": ROWS["ffs"][:1] + [{"team_id": "arsenal", "player_code": 9, "updated": "Wed 4th Nov"}],
               "rotowire": ROWS["rotowire"]}
    _patch_sources(monkeypatch, tmp_path, changed)
    LS.snapshot(pd.DataFrame(), stamp="20261120T0500Z")
    assert sorted(p.name for p in (tmp_path / "lineup_snapshots").iterdir()) == [
        "lineups_20261120T0430Z.csv", "lineups_20261120T0500Z.csv"]


# ---- live xG snapshots -----------------------------------------------------------------------

def _patch_fpl(monkeypatch, tmp_path, xg):
    """A fake FPL API with one finished gameweek and two players; `xg` is player 1's expected goals."""
    calls = []

    def get(url, path, refresh=False, **kw):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        if "bootstrap" in url:
            body = {"elements": [{"id": 1, "code": 101}, {"id": 2, "code": 102}],
                    "events": [{"id": 10, "finished": True}, {"id": 11, "finished": False}]}
        else:
            stat = lambda g: {"minutes": 90, "expected_goals": g, "expected_assists": "0.1",  # noqa: E731
                              "expected_goals_conceded": "1.0", "goals_scored": 0, "assists": 0}
            body = {"elements": [{"id": 1, "stats": stat(xg), "explain": [{"fixture": 97}]},
                                 {"id": 2, "stats": stat("0.2"), "explain": [{"fixture": 97}]}]}
        path.write_text(json.dumps(body))
        return path
    monkeypatch.setattr(PR, "_get", get)
    monkeypatch.setattr(PR, "XG_SNAP_DIR", tmp_path / "xg_snapshots")
    monkeypatch.setattr(PR, "RAW_DIR", tmp_path / "raw")
    return calls


def test_first_xg_snapshot_is_written(tmp_path, monkeypatch):
    _patch_fpl(monkeypatch, tmp_path, "0.5")
    dest = PR.snapshot_live_xg(pd.Timestamp("2026-11-20 05:00", tz="UTC"))
    assert dest is not None and dest.name == "2026-11-20.csv.gz"
    assert len(pd.read_csv(dest, compression="gzip")) == 2


def test_unchanged_xg_writes_nothing_and_is_not_redownloaded_the_same_day(tmp_path, monkeypatch):
    _patch_fpl(monkeypatch, tmp_path, "0.5")
    PR.snapshot_live_xg(pd.Timestamp("2026-11-20 05:00", tz="UTC"))
    calls = _patch_fpl(monkeypatch, tmp_path, "0.5")
    assert PR.snapshot_live_xg(pd.Timestamp("2026-11-21 05:00", tz="UTC")) is None   # same content next day
    assert [p.name for p in (tmp_path / "xg_snapshots").iterdir()] == ["2026-11-20.csv.gz"]
    n = len(calls)
    assert PR.snapshot_live_xg(pd.Timestamp("2026-11-21 05:30", tz="UTC")) is None
    assert len(calls) == n   # already checked today: no new download


def test_changed_xg_writes_a_new_file(tmp_path, monkeypatch):
    _patch_fpl(monkeypatch, tmp_path, "0.5")
    PR.snapshot_live_xg(pd.Timestamp("2026-11-20 05:00", tz="UTC"))
    _patch_fpl(monkeypatch, tmp_path, "0.7")   # FPL revised player 1's xG
    dest = PR.snapshot_live_xg(pd.Timestamp("2026-11-21 05:00", tz="UTC"))
    assert dest is not None and sorted(p.name for p in (tmp_path / "xg_snapshots").iterdir()) == [
        "2026-11-20.csv.gz", "2026-11-21.csv.gz"]
