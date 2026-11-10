"""Pre-match odds archive (docs/PROSPECTIVE_PROTOCOL.md, amendments A6 and A7). BENCHMARK ONLY.

Two sources, kept in separate files: football-data's fixtures.csv (archive(), below) and The Odds API
(archive_api(), amendment A7: one call whenever a forecast record is logged, so the market can be
compared at the record's own cutoff).

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
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
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


def _match(base: pd.DataFrame, fixtures: pd.DataFrame | None, ko_col: str) -> pd.DataFrame:
    """Add fpl_fixture, gameweek and kickoff_utc (FPL's) by home team, away team and a kickoff within
    MATCH_TOLERANCE of the source's; unmatched rows keep the source kickoff and an empty fixture."""
    base = base.copy()
    base["fpl_fixture"], base["gameweek"], base["kickoff_utc"] = pd.NA, pd.NA, base[ko_col]
    if fixtures is None or not len(fixtures):
        return base
    for i, r in base.iterrows():
        c = fixtures[(fixtures["home_id"] == r["home_id"]) & (fixtures["away_id"] == r["away_id"])]
        if pd.isna(r[ko_col]) or c.empty:
            continue
        gap = (c["kickoff_utc"] - r[ko_col]).abs()
        if gap.min() <= MATCH_TOLERANCE:
            best = c.loc[gap.idxmin()]
            base.loc[i, ["fpl_fixture", "gameweek", "kickoff_utc"]] = [int(best["fpl_fixture"]), int(best["gameweek"]),
                                                                       best["kickoff_utc"]]
    return base


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
    base = _match(base, fixtures, "fd_kickoff_utc")
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
    """Every archived snapshot row, both sources, with a `source` column ("football-data" or
    "the-odds-api"). The two sources stay in separate files so neither file's header ever changes."""
    parts = []
    for path, source in ((_paths(out_dir)[1], "football-data"), (out_dir / API_LOG, API_SOURCE)):
        if path.exists() and path.stat().st_size:
            parts.append(pd.read_csv(path).assign(source=source))
    if not parts:
        return pd.DataFrame(columns=LOG_COLUMNS + ["source"])
    d = pd.concat(parts, ignore_index=True)
    for c in ("kickoff_utc", "retrieved_utc"):
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


# ---- The Odds API (amendment A7): a snapshot at each forecast's cutoff ------------------------------

API_SOURCE = "the-odds-api"
API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
API_QUERY = {"regions": "uk,eu", "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"}
API_LOG = "oddsapi_log.csv"
API_FETCH_LOG = "oddsapi_fetch_log.csv"
API_KEY_FILE = Path.home() / ".config" / "tabet" / "odds_api_key"  # local runs only; Actions uses the secret
CREDIT_FLOOR = 50              # stop calling when fewer credits than this are left this month
API_MIN_GAP = pd.Timedelta(minutes=25)  # at most one call per 30-minute run
API_LOOKBACK = pd.Timedelta(hours=6)    # retry for a record logged this recently that has no snapshot yet
SAME_RUN = pd.Timedelta(minutes=15)     # a snapshot this soon after a record's cutoff came from the same run
API_COLUMNS = ["fpl_fixture", "gameweek", "home_id", "away_id", "home_team_src", "away_team_src", "kickoff_utc",
               "src_kickoff_utc", "retrieved_utc", "hours_before", "book", "book_last_update", "n_books",
               "odds_h", "odds_d", "odds_a", "event_id", "raw_file"]
API_FETCH_COLUMNS = ["retrieved_utc", "status", "credits_remaining", "credits_used", "last_cost", "n_events",
                     "n_upcoming", "n_unmatched", "needed", "raw_file", "error"]


def api_key() -> str | None:
    """ODDS_API_KEY from the environment (the GitHub secret), else the local key file. Never logged."""
    k = os.environ.get("ODDS_API_KEY", "").strip()
    if not k and API_KEY_FILE.exists():
        k = API_KEY_FILE.read_text().strip()
    return k or None


def _scrub(text: str, key: str | None) -> str:
    return text.replace(key, "***") if key else text


def api_download(key: str) -> tuple[bytes, dict]:
    """One call: every upcoming Premier League match, 1X2 odds, UK and EU bookmakers (2 credits)."""
    url = API_URL + "?" + urllib.parse.urlencode({"apiKey": key, **API_QUERY})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            h = {k: r.headers.get(k, "") for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
            return r.read(), h
    except urllib.error.HTTPError as e:  # the message never includes the URL, but scrub anyway
        raise RuntimeError(_scrub(f"HTTP {e.code} {e.reason}", key)) from None
    except urllib.error.URLError as e:
        raise RuntimeError(_scrub(f"network error: {e.reason}", key)) from None


def api_parse(body: bytes, retrieved: pd.Timestamp, fixtures: pd.DataFrame | None,
              raw_file: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long rows (one per match and bookmaker, plus an "Avg" row: the mean decimal odds over the
    bookmakers quoting all three outcomes, as football-data's Avg columns are built) for matches not yet
    kicked off, and the matches that did not match an FPL fixture."""
    empty = pd.DataFrame(columns=API_COLUMNS)
    events = json.loads(body.decode("utf-8") or "[]")
    if not events:
        return empty, empty
    base = pd.DataFrame([{"event_id": e.get("id"), "home_team_src": e.get("home_team"), "away_team_src": e.get("away_team"),
                          "src_kickoff_utc": pd.Timestamp(e["commence_time"]).tz_convert("UTC") if e.get("commence_time") else pd.NaT,
                          "bookmakers": e.get("bookmakers") or []} for e in events])
    base["home_id"] = base["home_team_src"].map(lambda n: try_resolve_team(n) if isinstance(n, str) else None)
    base["away_id"] = base["away_team_src"].map(lambda n: try_resolve_team(n) if isinstance(n, str) else None)
    base = _match(base, fixtures, "src_kickoff_utc")
    ko = pd.to_datetime(base["kickoff_utc"], utc=True)
    base = base[ko.isna() | (ko > retrieved)]
    rows = []
    for _, ev in base.iterrows():
        books = []
        for bm in ev["bookmakers"]:
            m = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
            if not m:
                continue
            price = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
            trio = [price.get(ev["home_team_src"]), price.get("Draw"), price.get(ev["away_team_src"])]
            if all(isinstance(x, (int, float)) and x > 1 for x in trio):
                books.append((bm.get("key"), m.get("last_update") or bm.get("last_update"), trio))
        common = {k: ev[k] for k in ("fpl_fixture", "gameweek", "home_id", "away_id", "home_team_src", "away_team_src",
                                     "kickoff_utc", "src_kickoff_utc", "event_id")}
        for key, upd, (h, d, a) in books:
            rows.append({**common, "book": key, "book_last_update": upd, "odds_h": h, "odds_d": d, "odds_a": a})
        if books:
            o = np.array([b[2] for b in books], float).mean(axis=0)
            rows.append({**common, "book": "Avg", "book_last_update": max(str(b[1]) for b in books),
                         "n_books": len(books), "odds_h": round(o[0], 4), "odds_d": round(o[1], 4), "odds_a": round(o[2], 4)})
    out = pd.DataFrame(rows, columns=API_COLUMNS) if rows else empty.copy()
    if len(out):
        out["retrieved_utc"] = retrieved.isoformat()
        out["raw_file"] = raw_file
        out["hours_before"] = ((pd.to_datetime(out["kickoff_utc"], utc=True) - retrieved).dt.total_seconds() / 3600).round(3)
        for c in ("kickoff_utc", "src_kickoff_utc"):
            out[c] = pd.to_datetime(out[c], utc=True).map(lambda x: x.isoformat() if pd.notna(x) else "")
    unmatched = base[base["fpl_fixture"].isna()][["home_team_src", "away_team_src", "src_kickoff_utc"]]
    return out, unmatched


def _api_fetch_log(d: Path) -> pd.DataFrame:
    p = d / API_FETCH_LOG
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=API_FETCH_COLUMNS)
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def api_needed(now: pd.Timestamp, records: list[dict], out_dir: Path = ODDS_DIR) -> list[int]:
    """Fixtures with a forecast record logged in the last API_LOOKBACK, not yet kicked off, that have no
    Odds API snapshot retrieved at or after that record's cutoff (minus a minute)."""
    d = load(out_dir)
    d = d[(d.get("source") == API_SOURCE) & (d["book"] == "Avg")] if len(d) else d
    need = []
    for r in records:
        cut, ko = pd.Timestamp(r["logged_utc"]), pd.Timestamp(r["kickoff_utc"])
        if ko <= now or now - cut > API_LOOKBACK:
            continue
        f = int(r["fpl_fixture"])
        have = d[(pd.to_numeric(d["fpl_fixture"], errors="coerce") == f)
                 & (d["retrieved_utc"] >= cut - pd.Timedelta(minutes=1))] if len(d) else d
        if not len(have) and f not in need:
            need.append(f)
    return need


def archive_api(now: pd.Timestamp, fixtures: pd.DataFrame | None, records: list[dict], out_dir: Path = ODDS_DIR,
                key: str | None = None, fetch=None, force: bool = False) -> dict:
    """Amendment A7: one Odds API call when a record was just logged (or one logged in the last 6 h still
    lacks a snapshot), at most once per run, never below CREDIT_FLOOR credits. `force` makes one call
    regardless (manual test). The raw response is always kept (each call costs credits)."""
    now = pd.Timestamp(now).tz_convert("UTC").floor("s")
    key = key if key is not None else api_key()
    if not key:
        return {"status": "disabled", "reason": "no ODDS_API_KEY"}
    needed = api_needed(now, records, out_dir)
    if not needed and not force:
        return {"status": "not needed"}
    log = _api_fetch_log(out_dir)
    tried = log[log["status"].isin(["ok", "error"])]
    if not force and len(tried) and now - pd.Timestamp(tried["retrieved_utc"].iloc[-1]) < API_MIN_GAP:
        return {"status": "skipped", "reason": "called less than 25 minutes ago"}
    ok = log[log["status"] == "ok"]
    if len(ok):
        last = ok.iloc[-1]
        same_month = pd.Timestamp(last["retrieved_utc"]).strftime("%Y-%m") == now.strftime("%Y-%m")
        rem = pd.to_numeric(last["credits_remaining"], errors="coerce")
        if same_month and pd.notna(rem) and rem < CREDIT_FLOOR:
            row = {"retrieved_utc": now.isoformat(), "status": "credit floor", "credits_remaining": int(rem),
                   "needed": ";".join(map(str, needed))}
            _append(out_dir / API_FETCH_LOG, [row], API_FETCH_COLUMNS)
            return {"status": "credit floor", "credits_remaining": int(rem), "needed": needed}
    try:
        body, headers = (fetch or api_download)(key)
    except Exception as e:
        err = _scrub(str(e), key)[:200]
        _append(out_dir / API_FETCH_LOG, [{"retrieved_utc": now.isoformat(), "status": "error", "error": err,
                                            "needed": ";".join(map(str, needed))}], API_FETCH_COLUMNS)
        return {"status": "error", "error": err, "needed": needed}
    raw_dir = _paths(out_dir)[0]
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{now:%Y%m%dT%H%M%SZ}_oddsapi.json.gz"
    with gzip.open(dest, "wb") as fh:
        fh.write(body)
    raw_file = str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest)
    try:
        rows, unmatched = api_parse(body, now, fixtures, raw_file)
        n_events = len(json.loads(body.decode("utf-8") or "[]"))
    except Exception as e:  # keep the raw response and record the problem
        _append(out_dir / API_FETCH_LOG, [{"retrieved_utc": now.isoformat(), "status": "parse error", "raw_file": raw_file,
                                            "error": str(e)[:200]}], API_FETCH_COLUMNS)
        return {"status": "parse error", "error": str(e), "raw_file": raw_file}
    if len(rows):
        _append(out_dir / API_LOG, rows.to_dict("records"), API_COLUMNS)
    info = {"retrieved_utc": now.isoformat(), "status": "ok", "credits_remaining": headers.get("x-requests-remaining", ""),
            "credits_used": headers.get("x-requests-used", ""), "last_cost": headers.get("x-requests-last", ""),
            "n_events": n_events, "n_upcoming": int((rows["book"] == "Avg").sum()) if len(rows) else 0,
            "n_unmatched": len(unmatched), "needed": ";".join(map(str, needed)), "raw_file": raw_file, "error": ""}
    _append(out_dir / API_FETCH_LOG, [info], API_FETCH_COLUMNS)
    covered = set(pd.to_numeric(rows.loc[rows["book"] == "Avg", "fpl_fixture"], errors="coerce").dropna().astype(int)) if len(rows) else set()
    info.update(unmatched=unmatched.to_dict("records"), missing=[f for f in needed if f not in covered])
    return info


def api_status(out_dir: Path = ODDS_DIR) -> dict:
    """For the heartbeat: whether a key is configured, the last call and credits left this month."""
    log = _api_fetch_log(out_dir)
    out = {"odds_api_enabled": api_key() is not None, "odds_api_last": None, "odds_api_credits": None, "odds_api_low": False}
    if len(log):
        last = log.iloc[-1]
        out["odds_api_last"] = {"utc": last["retrieved_utc"], "status": last["status"], "error": last.get("error", "")}
        ok = log[log["status"] == "ok"]
        if len(ok):
            rem = pd.to_numeric(ok["credits_remaining"].iloc[-1], errors="coerce")
            month = pd.Timestamp(ok["retrieved_utc"].iloc[-1]).strftime("%Y-%m")
            if pd.notna(rem):
                out["odds_api_credits"] = {"remaining": int(rem), "as_of": ok["retrieved_utc"].iloc[-1]}
                out["odds_api_low"] = bool(rem < CREDIT_FLOOR and month == pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))
    return out


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
