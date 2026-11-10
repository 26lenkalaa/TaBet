"""Amendment A6: pre-match odds archive (benchmark only), daily results refresh, market benchmark."""

import gzip
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tabet.prospective as PR
from tabet import market_benchmark as MB
from tabet import odds_archive as OA

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
SAMPLE = (DATA / "fixtures_sample.csv").read_bytes()  # football-data fixtures.csv, gameweek 10 (6 Nov 2026)
HEADER_ONLY = (DATA / "fixtures_header_only.csv").read_bytes()


# ---- the hard rule: odds are a benchmark only --------------------------------

# Modules allowed to refer to the odds archive at all: the archiver, the benchmark, and the CLI
# that wires them into the logging run and score-forecasts.
ARCHIVE_ALLOWED = {"odds_archive.py", "market_benchmark.py", "cli.py"}
# Modules allowed to name bookmaker odds columns: the above, plus the existing benchmark plumbing
# for the results file (ingest splits odds into a separate table; validate checks it; backtest and
# site_data attach it to finished predictions for display and comparison).
COLUMNS_ALLOWED = ARCHIVE_ALLOWED | {"ingest.py", "validate.py", "backtest.py", "site_data.py"}
ARCHIVE_TOKENS = re.compile(r"odds_archive|market_benchmark|odds_log|fetch_log|prospective/odds|ODDS_DIR"
                            r"|[\"']odds[\"']\s*\)|/\s*[\"']odds[\"']")
ODDS_COLUMN = re.compile(r"\b(?:B365|BFDC?|BVC?|BWC?|PPC?|SKBC?|BFEC?|PSC?|WHC?|IWC?|VCC?|LB|Max|Avg)C?[HDA]\b"
                         r"|implied_probabilities")
CROWD = re.compile(r"selected_by_percent|transfers_(?:in|out)|now_cost|cost_change")


def _modules():
    return sorted((ROOT / "tabet").rglob("*.py"))


def test_no_model_code_refers_to_the_odds_archive():
    bad = [p.relative_to(ROOT).as_posix() for p in _modules()
           if p.name not in ARCHIVE_ALLOWED and ARCHIVE_TOKENS.search(p.read_text())]
    assert bad == [], f"only {sorted(ARCHIVE_ALLOWED)} may refer to the odds archive: {bad}"


def test_no_model_code_reads_odds_columns():
    bad = [p.relative_to(ROOT).as_posix() for p in _modules()
           if p.name not in COLUMNS_ALLOWED and ODDS_COLUMN.search(p.read_text())]
    assert bad == [], f"odds columns referenced outside the benchmark code: {bad}"


def test_no_crowd_signals_in_the_package():
    bad = [p.relative_to(ROOT).as_posix() for p in _modules() if CROWD.search(p.read_text())]
    assert bad == []


def test_the_guard_patterns_catch_what_they_should():
    assert ARCHIVE_TOKENS.search('pd.read_csv(ROOT / "results" / "prospective" / "odds" / "odds_log.csv")')
    assert ARCHIVE_TOKENS.search("from .odds_archive import load")
    assert ODDS_COLUMN.search('x = row["AvgH"]') and ODDS_COLUMN.search("B365CA")
    assert not ODDS_COLUMN.search("p_home, AvgHomeGoals, MaxAge")


def test_ingest_keeps_odds_out_of_the_match_table(real):
    matches, odds = real
    assert not [c for c in matches.columns if ODDS_COLUMN.fullmatch(c)]
    assert "AvgH" in odds.columns


# ---- archiver ----------------------------------------------------------------------

def _fx():
    rows = [(94, "everton", "coventry", "2026-11-06 20:00"), (97, "leeds", "tottenham", "2026-11-07 12:30"),
            (91, "arsenal", "hull", "2026-11-07 15:00"), (95, "fulham", "newcastle", "2026-11-07 15:00"),
            (99, "nottm_forest", "man_city", "2026-11-07 17:30"), (92, "brighton", "brentford", "2026-11-08 14:00"),
            (93, "crystal_palace", "liverpool", "2026-11-08 14:00"), (96, "ipswich", "bournemouth", "2026-11-08 14:00"),
            (100, "sunderland", "chelsea", "2026-11-08 14:00"), (98, "man_united", "aston_villa", "2026-11-08 16:30")]
    return pd.DataFrame({"fpl_fixture": [r[0] for r in rows], "home_id": [r[1] for r in rows],
                         "away_id": [r[2] for r in rows], "kickoff_utc": pd.to_datetime([r[3] for r in rows], utc=True),
                         "gameweek": 10})


def _fetcher(*bodies):
    it = iter(bodies)
    return lambda: (next(it), "Fri, 06 Nov 2026 11:14:29 GMT")


def test_parse_matches_fpl_fixtures_and_converts_uk_time():
    rows, unmatched = OA.parse(SAMPLE, pd.Timestamp("2026-11-06 12:00", tz="UTC"), _fx())
    avg = rows[(rows.book == "Avg") & rows.fpl_fixture.notna()].set_index("fpl_fixture")
    assert sorted(avg.index) == [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    # 06/11/2026 20:00 UK time is 20:00 UTC (GMT in November)
    assert avg.loc[94, "kickoff_utc"].startswith("2026-11-06T20:00")
    assert avg.loc[97, ["odds_h", "odds_d", "odds_a"]].tolist() == [2.69, 3.33, 2.54]
    assert avg.loc[97, "hours_before"] == pytest.approx(24.5)
    assert {"B365", "Max", "Avg", "BFE"} <= set(rows.book) and "AvgC" not in set(rows.book)  # closing columns empty
    assert set(rows.home_team_fd) >= {"Everton", "Atlantis FC"}  # other divisions are dropped, unmatched rows kept
    assert unmatched["home_team_fd"].tolist() == ["Atlantis FC"]
    assert rows.loc[rows.home_team_fd == "Atlantis FC", "fpl_fixture"].isna().all()


def test_parse_keeps_only_fixtures_that_have_not_kicked_off():
    rows, _ = OA.parse(SAMPLE, pd.Timestamp("2026-11-07 16:00", tz="UTC"), _fx())
    left = set(rows.loc[rows.book == "Avg", "fpl_fixture"].dropna().astype(int))
    assert left == {99, 92, 93, 96, 100, 98}  # Friday and Saturday 12:30 / 15:00 kickoffs are gone


@pytest.mark.parametrize("body", [b"", HEADER_ONLY])
def test_empty_and_header_only_files_are_handled(tmp_path, body):
    r = OA.archive(pd.Timestamp("2026-11-10 09:00", tz="UTC"), _fx(), out_dir=tmp_path, fetch=_fetcher(body))
    assert r["status"] == "ok" and r["n_e0_upcoming"] == 0 and r["changed"]
    assert not (tmp_path / "odds_log.csv").exists()
    assert len(list((tmp_path / "raw").glob("*.csv.gz"))) == 1  # the empty file is still evidence of what was there


def test_archiver_saves_only_on_change_and_fetches_at_most_hourly(tmp_path):
    t0 = pd.Timestamp("2026-11-06 06:00", tz="UTC")
    fetch = _fetcher(SAMPLE, SAMPLE, SAMPLE.replace(b",2.69,3.33,2.54,", b",2.75,3.30,2.50,"))
    r1 = OA.archive(t0, _fx(), out_dir=tmp_path, fetch=fetch)
    assert r1["changed"] and r1["n_e0_upcoming"] == 11
    assert OA.archive(t0 + pd.Timedelta(minutes=30), _fx(), out_dir=tmp_path, fetch=fetch)["status"] == "skipped"
    r2 = OA.archive(t0 + pd.Timedelta(hours=1), _fx(), out_dir=tmp_path, fetch=fetch)
    assert r2["status"] == "ok" and not r2["changed"] and r2["raw_file"] == ""
    r3 = OA.archive(t0 + pd.Timedelta(hours=2), _fx(), out_dir=tmp_path, fetch=fetch)
    assert r3["changed"]
    raws = sorted((tmp_path / "raw").glob("*_fixtures.csv.gz"))
    assert len(raws) == 2 and gzip.decompress(raws[0].read_bytes()) == SAMPLE
    log = OA.load(tmp_path)
    assert log["retrieved_utc"].nunique() == 2  # the unchanged fetch added no rows
    fetches = pd.read_csv(tmp_path / "fetch_log.csv")
    assert len(fetches) == 3 and fetches["changed"].tolist() == [True, False, True]


def test_a_failed_download_is_recorded_and_does_not_raise(tmp_path):
    def boom():
        raise OSError("HTTP 503")
    r = OA.archive(pd.Timestamp("2026-11-06 06:00", tz="UTC"), _fx(), out_dir=tmp_path, fetch=boom)
    assert r["status"] == "error"
    assert pd.read_csv(tmp_path / "fetch_log.csv")["status"].tolist() == ["error"]
    # an error does not count as a fetch for the hourly limit
    assert OA.archive(pd.Timestamp("2026-11-06 06:30", tz="UTC"), _fx(), out_dir=tmp_path,
                      fetch=_fetcher(SAMPLE))["status"] == "ok"


def test_status_reports_missing_snapshots_and_coverage(tmp_path):
    fx = _fx()
    OA.archive(pd.Timestamp("2026-11-06 06:00", tz="UTC"), fx[fx.fpl_fixture != 98], out_dir=tmp_path,
               fetch=_fetcher(SAMPLE))
    st = OA.status(pd.Timestamp("2026-11-07 17:00", tz="UTC"), fx, out_dir=tmp_path)
    assert st["odds_missing"] == [98]  # within 24 h of kickoff, never matched
    assert st["odds_coverage"] == {"gameweek": 10, "fixtures": 10, "with_snapshot": 9}


# ---- snapshot selection (amendment A6, item 2) -----------------------------------------

def _odds(rows):
    return pd.DataFrame([{"fpl_fixture": f, "book": "Avg", "retrieved_utc": pd.Timestamp(t, tz="UTC"),
                          "odds_h": h, "odds_d": 3.5, "odds_a": 4.0} for f, t, h in rows])


def test_matched_snapshot_is_the_last_at_or_before_the_cutoff_else_the_first_after_flagged_later():
    recs = pd.DataFrame({"fpl_fixture": [1, 2, 3], "logged_utc": ["2026-11-20T11:00:00+00:00"] * 3,
                         "kickoff_utc": ["2026-11-21T12:30:00+00:00"] * 3})
    log = _odds([(1, "2026-11-20 09:00", 2.0), (1, "2026-11-20 11:00", 2.1), (1, "2026-11-21 10:00", 2.2),
                 (2, "2026-11-20 14:00", 2.4), (2, "2026-11-21 09:00", 2.5),
                 (3, "2026-11-21 13:00", 9.0)])  # fixture 3: only after kickoff, never used
    s = MB.select_snapshots(recs, log).set_index(["fpl_fixture", "snapshot"])
    m1 = s.loc[(1, "matched")]
    assert m1["retrieved_utc"] == pd.Timestamp("2026-11-20 11:00", tz="UTC") and not m1["later"]
    assert m1["hours_after_cutoff"] == 0
    m2 = s.loc[(2, "matched")]
    assert m2["retrieved_utc"] == pd.Timestamp("2026-11-20 14:00", tz="UTC") and m2["later"]
    assert m2["hours_after_cutoff"] == pytest.approx(3.0)
    assert s.loc[(1, "last_pre_kickoff"), "retrieved_utc"] == pd.Timestamp("2026-11-21 10:00", tz="UTC")
    assert s.loc[(2, "last_pre_kickoff"), "retrieved_utc"] == pd.Timestamp("2026-11-21 09:00", tz="UTC")
    assert 3 not in s.index.get_level_values(0)
    p = s.loc[(1, "matched"), ["p_home", "p_draw", "p_away"]].to_numpy(float)
    inv = 1 / np.array([2.1, 3.5, 4.0])
    assert p == pytest.approx(inv / inv.sum()) and p.sum() == pytest.approx(1.0)


def _scores(n=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p0 = rng.dirichlet([5, 3, 3])
        pe = 0.7 * p0 + 0.3 * rng.dirichlet([5, 3, 3])
        y = "HDA"[rng.choice(3, p=p0)]
        for c, p in (("p0", p0), ("pE", pe)):
            rows.append({"fpl_fixture": i, "gameweek": 11 + i // 10, "horizon_h": 24.0, "candidate": c, "result": y,
                         "logged_utc": "2026-11-20T11:00:00+00:00", "kickoff_utc": "2026-11-21T12:30:00+00:00",
                         "home_id": f"h{i}", "away_id": f"a{i}", "superseded": False,
                         "log_loss": -np.log(p["HDA".index(y)]), "p_home": p[0], "p_draw": p[1], "p_away": p[2]})
    return pd.DataFrame(rows)


def test_benchmark_pairs_identical_fixtures_and_counts_disagreements():
    s = _scores()
    ids = range(0, 60, 2)  # odds for half the fixtures only
    log = _odds([(i, "2026-11-20 10:00", 1.5 + i / 20) for i in ids])
    b = MB.benchmark(s, odds_log=log, market=pd.DataFrame(columns=["fpl_fixture", "snapshot"])).set_index(
        ["snapshot", "candidate"])
    r = b.loc[("matched", "p0")]
    assert r["n"] == 30 and r["coverage"] == pytest.approx(0.5) and r["n_later"] == 0
    assert r["tabet_only_right"] + r["market_only_right"] + r["both_wrong"] == r["disagree"]
    assert r["tabet_minus_market"] == pytest.approx(r["log_loss_tabet"] - r["log_loss_market"])
    assert r["se_gameweek_clustered"] > 0 and r["n_gameweeks"] == 6
    assert b.loc[("fd_closing", "p0"), "n"] == 0
    assert b.loc[("matched", "p_alpha"), "n"] == 30


def test_benchmark_does_not_change_alpha():
    s = _scores(300)
    a = PR.fit_alpha(s, through_gameweek=19)
    MB.benchmark(s, odds_log=_odds([(i, "2026-11-20 10:00", 2.0) for i in range(300)]),
                 market=pd.DataFrame(columns=["fpl_fixture", "snapshot"]))
    assert PR.fit_alpha(s, through_gameweek=19) == a


# ---- daily results refresh -----------------------------------------------------------

class _Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def res_copy(tmp_path, monkeypatch):
    """A private copy of res/ so refresh_results can replace and restore the current season file."""
    root = tmp_path / "root"
    shutil.copytree(ROOT / "res", root / "res", ignore=shutil.ignore_patterns("history", "other_leagues"))
    monkeypatch.setattr(PR, "ROOT", root)
    monkeypatch.setattr(PR, "RESULTS_FILE", root / "res" / "Premier League 26-27.csv")
    monkeypatch.setattr(PR, "RESULTS_REFRESH_LOG", tmp_path / "results_refresh.csv")
    return root / "res" / "Premier League 26-27.csv"


def _serve(monkeypatch, bodies, calls):
    it = iter(bodies)

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        return _Resp(next(it))
    monkeypatch.setattr(PR.urllib.request, "urlopen", urlopen)


def test_daily_refresh_runs_once_per_utc_day(res_copy, monkeypatch):
    good = res_copy.read_bytes()
    calls = []
    _serve(monkeypatch, [good, good], calls)
    day = pd.Timestamp("2026-11-10 00:15", tz="UTC")
    assert PR.daily_refresh(day)["ok"] is True
    for h in (0.5, 6, 23.5):
        assert PR.daily_refresh(day + pd.Timedelta(hours=h)) is None
    assert PR.daily_refresh(pd.Timestamp("2026-11-11 00:10", tz="UTC"))["ok"] is True
    assert len(calls) == 2
    log = pd.read_csv(PR.RESULTS_REFRESH_LOG)
    assert log["local_rows"].tolist() == [90, 90] and log["remote_rows"].tolist() == [90, 90]


def test_bad_download_restores_the_old_file_and_retries_hourly(res_copy, monkeypatch):
    before = res_copy.read_bytes()
    calls = []
    _serve(monkeypatch, [b"<html>maintenance</html>", before], calls)
    t = pd.Timestamp("2026-11-10 00:15", tz="UTC")
    r = PR.daily_refresh(t)
    assert r["ok"] is False and res_copy.read_bytes() == before
    assert not res_copy.with_suffix(".bak").exists() and not res_copy.with_suffix(".download").exists()
    assert PR.daily_refresh(t + pd.Timedelta(minutes=30)) is None  # no retry within the hour
    assert PR.daily_refresh(t + pd.Timedelta(hours=1))["ok"] is True
    assert len(calls) == 2


def test_results_behind_needs_two_days_in_a_row():
    def log(rows):
        return pd.DataFrame([{"attempted_utc": pd.Timestamp(t, tz="UTC"), "ok": ok, "remote_rows": rem, "local_rows": loc}
                             for t, ok, rem, loc in rows])
    now = pd.Timestamp("2026-11-12 08:00", tz="UTC")
    one_day = log([("2026-11-11 00:10", True, 90, 90), ("2026-11-12 00:10", False, 100, 90)])
    assert not PR.results_behind(now, one_day)["results_behind"]
    two_days = log([("2026-11-11 00:10", False, 100, 90), ("2026-11-12 00:10", False, 100, 90)])
    assert PR.results_behind(now, two_days)["results_behind"]
    failed = log([("2026-11-11 00:10", False, None, 90), ("2026-11-12 00:10", False, None, 90)])
    assert PR.results_behind(now, failed)["results_behind"]  # a download that keeps failing also alerts
    recovered = log([("2026-11-11 00:10", False, 100, 90), ("2026-11-12 00:10", False, 100, 90),
                     ("2026-11-12 01:10", True, 100, 100)])
    assert not PR.results_behind(now, recovered)["results_behind"]
