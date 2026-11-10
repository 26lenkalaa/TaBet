"""Pre-match odds archive (docs/PROSPECTIVE_PROTOCOL.md, amendment A6). BENCHMARK ONLY.

No model, feature, tuning or forecast code may read anything written here. Only this module and
tabet/market_benchmark.py (scoring) may refer to the archive; tests/test_odds_archive.py enforces it.

archive(): called on every logger run (not only when a forecast is due). At most once an hour it
    downloads football-data's fixtures.csv and records the fetch in fetch_log.csv. If the file's
    content changed since the last saved copy (SHA-256), it keeps the raw file gzipped as
    raw/<UTC stamp>_fixtures.csv.gz and appends the parsed Premier League (E0) rows for fixtures that
    have not kicked off to odds_log.csv. Unchanged content saves nothing new: a snapshot is valid from
    its retrieval time until the next changed file. Everything is append-only.

odds_log.csv is long: one row per fixture, snapshot and bookmaker column group (e.g. "Avg", "Max",
"B365"; a closing group like "AvgC" appears only if the file ever fills it before kickoff). The
file's columns change over time, so a long table never needs a header change. Rows whose teams or
date do not match an FPL fixture are kept with an empty fpl_fixture and listed in the run output.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

from .fpl import _ssl_context
from .ingest import LOCAL_TZ
from .teams import try_resolve_team

ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = ROOT / "results" / "prospective" / "odds"
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
USER_AGENT = "tabet-research/0.1"  # the same client name as refresh_results()
MIN_FETCH_GAP = pd.Timedelta(minutes=55)  # "at most once an hour" with runs every 30 min
MATCH_TOLERANCE = pd.Timedelta(hours=36)  # football-data date vs FPL kickoff, for the same home/away pair
LOG_COLUMNS = ["fpl_fixture", "gameweek", "home_id", "away_id", "home_team_fd", "away_team_fd", "kickoff_utc",
               "fd_kickoff_utc", "retrieved_utc", "source_last_modified", "hours_before", "book",
               "odds_h", "odds_d", "odds_a", "raw_sha256"]
FETCH_COLUMNS = ["retrieved_utc", "status", "sha256", "changed", "raw_file", "n_rows", "n_e0", "n_e0_upcoming",
                 "n_unmatched", "source_last_modified", "error"]
_TRIPLE = re.compile(r"^(?P<book>[A-Za-z0-9]+?)(?P<side>[HDA])$")
_NOT_BOOKS = {"FT", "HT", "AH", "B365AH", "MaxAH", "AvgAH", "BFEAH", "B365CAH", "MaxCAH", "AvgCAH", "BFECAH"}


def _paths(d: Path) -> tuple[Path, Path, Path]:
    return d / "raw", d / "odds_log.csv", d / "fetch_log.csv"


def _append(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _fetch_log(d: Path) -> pd.DataFrame:
    p = _paths(d)[2]
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=FETCH_COLUMNS)
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def books(columns) -> list[str]:
    """Column groups with all three of H, D and A, e.g. B365H/B365D/B365A -> "B365"."""
    seen: dict[str, set] = {}
    for c in columns:
        m = _TRIPLE.match(str(c).strip())
        if m and m["book"] not in _NOT_BOOKS:
            seen.setdefault(m["book"], set()).add(m["side"])
    return [b for b, s in seen.items() if s == {"H", "D", "A"}]


def parse(raw: bytes, retrieved: pd.Timestamp, fixtures: pd.DataFrame | None, last_modified: str = "",
          sha: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parsed E0 odds rows for fixtures not yet kicked off at `retrieved` (long format), and the E0 rows
    that could not be matched to an FPL fixture. An empty or header-only file gives empty frames."""
    empty = pd.DataFrame(columns=LOG_COLUMNS)
    text = raw.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return empty, empty
    f = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    f.columns = [c.strip() for c in f.columns]
    if f.empty or "Div" not in f:
        return empty, empty
    f = f[f["Div"].str.strip() == "E0"]
    if f.empty:
        return empty, empty
    date = pd.to_datetime(f["Date"].str.strip(), format="%d/%m/%Y", errors="coerce")
    t = f["Time"].str.strip() if "Time" in f else pd.Series("", index=f.index)
    t = t.where(t.str.fullmatch(r"\d{1,2}:\d{2}"), "15:00")  # football-data times are UK local time
    local = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + t, format="%Y-%m-%d %H:%M", errors="coerce")
    fd_ko = local.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    base = pd.DataFrame({"home_team_fd": f["HomeTeam"].str.strip(), "away_team_fd": f["AwayTeam"].str.strip(),
                         "fd_kickoff_utc": fd_ko})
    base["home_id"] = base["home_team_fd"].map(try_resolve_team)
    base["away_id"] = base["away_team_fd"].map(try_resolve_team)
    base["fpl_fixture"], base["gameweek"], base["kickoff_utc"] = pd.NA, pd.NA, base["fd_kickoff_utc"]
    if fixtures is not None and len(fixtures):
        for i, r in base.iterrows():
            c = fixtures[(fixtures["home_id"] == r["home_id"]) & (fixtures["away_id"] == r["away_id"])]
            if pd.isna(r["fd_kickoff_utc"]) or c.empty:
                continue
            gap = (c["kickoff_utc"] - r["fd_kickoff_utc"]).abs()
            if gap.min() <= MATCH_TOLERANCE:
                best = c.loc[gap.idxmin()]
                base.loc[i, ["fpl_fixture", "gameweek", "kickoff_utc"]] = [int(best["fpl_fixture"]), int(best["gameweek"]),
                                                                           best["kickoff_utc"]]
    ko = pd.to_datetime(base["kickoff_utc"], utc=True)
    keep = ko.isna() | (ko > retrieved)  # only fixtures that have not kicked off (an unparseable date is kept)
    base, f, ko = base[keep], f[keep], ko[keep]
    rows = []
    for b in books(f.columns):
        o = f[[f"{b}H", f"{b}D", f"{b}A"]].apply(pd.to_numeric, errors="coerce")
        ok = o.notna().all(axis=1) & (o > 1).all(axis=1)
        if not ok.any():
            continue
        part = base[ok].copy()
        part["book"] = b
        part["odds_h"], part["odds_d"], part["odds_a"] = o.loc[ok].to_numpy().T
        rows.append(part)
    out = pd.concat(rows, ignore_index=False) if rows else empty.copy()
    if len(out):
        out["retrieved_utc"] = retrieved.isoformat()
        out["source_last_modified"] = last_modified
        out["raw_sha256"] = sha
        out["hours_before"] = ((pd.to_datetime(out["kickoff_utc"], utc=True) - retrieved).dt.total_seconds() / 3600).round(3)
        for c in ("kickoff_utc", "fd_kickoff_utc"):
            out[c] = pd.to_datetime(out[c], utc=True).map(lambda x: x.isoformat() if pd.notna(x) else "")
        out = out.reindex(columns=LOG_COLUMNS).sort_values(["kickoff_utc", "home_team_fd", "book"])
    unmatched = base[base["fpl_fixture"].isna()][["home_team_fd", "away_team_fd", "fd_kickoff_utc"]]
    return out, unmatched


def download(url: str = FIXTURES_URL) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        return r.read(), r.headers.get("Last-Modified", "") or ""


def archive(now: pd.Timestamp, fixtures: pd.DataFrame | None, out_dir: Path = ODDS_DIR, fetch=download,
            force: bool = False) -> dict:
    """One archiver step (see module docstring). `fetch` is for tests. Returns what happened."""
    now = pd.Timestamp(now).tz_convert("UTC").floor("s")
    raw_dir, odds_log, fetch_log = _paths(out_dir)
    log = _fetch_log(out_dir)
    ok = log[log["status"] == "ok"]
    if not force and len(ok) and now - pd.Timestamp(ok["retrieved_utc"].iloc[-1]) < MIN_FETCH_GAP:
        return {"status": "skipped", "reason": "fetched less than an hour ago"}
    try:
        raw, last_mod = fetch()
    except Exception as e:  # a failed download is recorded and never stops the forecast run
        _append(fetch_log, [{"retrieved_utc": now.isoformat(), "status": "error", "error": str(e)[:200]}], FETCH_COLUMNS)
        return {"status": "error", "error": str(e)}
    sha = hashlib.sha256(raw).hexdigest()
    saved = log[(log["status"] == "ok") & (log["changed"] == "True")]
    changed = saved.empty or saved["sha256"].iloc[-1] != sha
    rows, unmatched = parse(raw, now, fixtures, last_mod, sha)
    text = raw.decode("utf-8-sig", errors="replace")
    n_rows = max(len([ln for ln in text.splitlines() if ln.strip()]) - 1, 0)
    n_e0 = sum(1 for ln in text.splitlines() if ln.startswith("E0,"))
    raw_file = ""
    if changed:
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / f"{now:%Y%m%dT%H%M%SZ}_fixtures.csv.gz"
        with gzip.open(dest, "wb") as fh:
            fh.write(raw)
        raw_file = str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest)
        if len(rows):
            _append(odds_log, rows.to_dict("records"), LOG_COLUMNS)
    info = {"retrieved_utc": now.isoformat(), "status": "ok", "sha256": sha, "changed": changed, "raw_file": raw_file,
            "n_rows": n_rows, "n_e0": n_e0,
            "n_e0_upcoming": len(rows[["home_team_fd", "away_team_fd"]].drop_duplicates()) if len(rows) else 0,
            "n_unmatched": len(unmatched), "source_last_modified": last_mod, "error": ""}
    _append(fetch_log, [info], FETCH_COLUMNS)
    info["unmatched"] = unmatched.to_dict("records")
    return info


def load(out_dir: Path = ODDS_DIR) -> pd.DataFrame:
    p = _paths(out_dir)[1]
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    d = pd.read_csv(p)
    for c in ("kickoff_utc", "retrieved_utc"):
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


def status(now: pd.Timestamp, fixtures: pd.DataFrame, out_dir: Path = ODDS_DIR, book: str = "Avg",
           gameweek_fixtures: pd.DataFrame | None = None) -> dict:
    """Monitoring for the heartbeat: fixtures kicking off within 24 h with no `book` snapshot yet, and the
    share of the current gameweek's fixtures that have one. `fixtures` = not yet kicked off;
    `gameweek_fixtures` = every fixture of the season (for coverage), defaults to `fixtures`."""
    d = load(out_dir)
    have = set(pd.to_numeric(d.loc[d["book"] == book, "fpl_fixture"], errors="coerce").dropna().astype(int))
    up = fixtures[fixtures["kickoff_utc"] > now]
    soon = up[up["kickoff_utc"] <= now + pd.Timedelta(hours=24)]
    out = {"odds_missing": [int(i) for i in soon["fpl_fixture"] if int(i) not in have], "odds_coverage": None}
    if len(up):
        gw = int(up.sort_values("kickoff_utc")["gameweek"].iloc[0])
        allfx = fixtures if gameweek_fixtures is None else gameweek_fixtures
        ids = [int(i) for i in allfx.loc[allfx["gameweek"] == gw, "fpl_fixture"]]
        out["odds_coverage"] = {"gameweek": gw, "fixtures": len(ids), "with_snapshot": sum(i in have for i in ids)}
    log = _fetch_log(out_dir)
    ok = log[log["status"] == "ok"]
    out["odds_last_fetch_utc"] = ok["retrieved_utc"].iloc[-1] if len(ok) else None
    return out
