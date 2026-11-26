"""Transfers involving English clubs, from Wikipedia's "List of English football transfers" pages.

    python -m tabet transfers                 # fetch/refresh, parse, store; prints a summary
    from tabet.transfers import load          # the parsed table (DataFrame)

One page per window: "summer YYYY" and "winter YYYY–YY", 2005 onwards. Pages come from Wikipedia's
official API (action=parse) with a plain User-Agent, never by scraping around anything. Text is CC BY-SA
4.0: anything derived and published must credit Wikipedia.

Caching: a window's raw page is fetched live while the window is open or recent (CURRENT_DAYS after it
closes), and stored once settled in the private bucket football-data (wikipedia/transfers/<slug>.json.gz),
with the page revision id, so backtests see a fixed snapshot. Pages are still edited after a window
closes (fees revised, moves added); the stored revision and its timestamp are kept with each row.

Parsed columns: window, date, player, from_club, to_club, from_id, to_id (our team ids when the club is
one we know, else None), kind (fee | undisclosed | free | loan | other), fee_gbp_m (reported fee in £m
when one is given), fee_text (as written), revision_id, revision_utc (when that revision was fetched).

Descriptive input for research only: no model or logged forecast reads it (a pre-registered test would
come first).
"""
from __future__ import annotations

import gzip
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

import pandas as pd

from . import supabase_storage as ss
from .fpl import _ssl_context
from .teams import try_resolve_team

API = "https://en.wikipedia.org/w/api.php"
UA = "TaBet-research/0.1 (student project; transfers list reader)"
BUCKET = "football-data"
PREFIX = "wikipedia/transfers"
FIRST_YEAR = 2005
CURRENT_DAYS = 60          # a window is re-fetched until this many days after it closes
PAUSE_S = 3.0              # between API requests
RETRIES = 6                # on HTTP 429 / 503, backing off (honours Retry-After)


# ---------------------------------------------------------------- windows

def windows(today: pd.Timestamp | None = None) -> list[dict]:
    """Every window page from summer 2005 to the latest one that has opened. A summer window runs
    June-August, a winter one January."""
    today = (today or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    out = []
    for y in range(FIRST_YEAR, today.year + 1):
        out.append({"slug": f"summer_{y}", "title": f"List of English football transfers summer {y}",
                    "opens": pd.Timestamp(f"{y}-05-01", tz="UTC"), "closes": pd.Timestamp(f"{y}-09-02", tz="UTC")})
        out.append({"slug": f"winter_{y}_{(y + 1) % 100:02d}",
                    "title": f"List of English football transfers winter {y}–{(y + 1) % 100:02d}",
                    "opens": pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), "closes": pd.Timestamp(f"{y + 1}-02-03", tz="UTC")})
    return [w for w in out if w["opens"] <= today]


def _settled(w: dict, today: pd.Timestamp) -> bool:
    return today > w["closes"] + pd.Timedelta(days=CURRENT_DAYS)


# ---------------------------------------------------------------- fetching

def _api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == RETRIES - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or 10 * 2 ** attempt
            print(f"transfers: Wikipedia asked us to slow down ({e.code}); waiting {wait:.0f} s", flush=True)
            time.sleep(min(wait, 300))


def fetch_page(title: str) -> dict | None:
    """{"title", "revision_id", "revision_utc", "html"} for the page's current revision, or None if missing.
    revision_utc is when we fetched it (one request per page; the revision id pins the exact text)."""
    d = _api({"action": "parse", "page": title, "prop": "text|revid", "redirects": "1"})
    if "error" in d:
        return None
    return {"title": d["parse"]["title"], "revision_id": d["parse"].get("revid"),
            "revision_utc": pd.Timestamp.now(tz="UTC").isoformat(), "html": d["parse"]["text"]}


def page(w: dict, today: pd.Timestamp, refresh: bool = True) -> dict | None:
    """The window's page: from storage when settled and stored, else live (and stored once settled)."""
    path = f"{PREFIX}/{w['slug']}.json.gz"
    settled = _settled(w, today)
    if settled or not refresh:
        try:
            return json.loads(gzip.decompress(ss.get(BUCKET, path)))
        except Exception:
            pass
    p = fetch_page(w["title"])
    time.sleep(PAUSE_S)
    if p is not None and settled:
        _put(path, gzip.compress(json.dumps(p).encode()), "application/gzip")
    return p


def _put(path: str, data: bytes, content_type: str, tries: int = 3) -> None:
    for i in range(tries):
        try:
            ss.put(BUCKET, path, data, content_type)
            return
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


# ---------------------------------------------------------------- parsing

class _Tables(HTMLParser):
    """Rows of every "wikitable", with rowspan cells carried down."""

    def __init__(self):
        super().__init__()
        self.tables, self._t, self._row, self._cell, self._depth = [], None, None, None, 0
        self._span: dict[int, list] = {}     # column -> [text, rows left]
        self._col = 0
        self._skip = 0                        # inside <style>/<script>: their text is not cell content

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "script"):
            self._skip += 1
            return
        if tag == "table" and "wikitable" in (a.get("class") or "") and self._t is None:
            self._t, self._depth, self._span = [], 1, {}
            return
        if self._t is None:
            return
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth == 1:
            self._row, self._col = [], 0
            self._fill()
        elif tag in ("td", "th") and self._row is not None and self._depth == 1:
            self._cell = []
            self._rowspan = int(re.sub(r"\D", "", a.get("rowspan", "1")) or 1)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def _fill(self):
        while self._col in self._span:
            text, left = self._span[self._col]
            self._row.append(text)
            if left <= 1:
                del self._span[self._col]
            else:
                self._span[self._col] = [text, left - 1]
            self._col += 1

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._skip = max(0, self._skip - 1)
            return
        if self._t is None:
            return
        if tag in ("td", "th") and self._cell is not None and self._depth == 1:
            text = re.sub(r"\[[^\]]*\]|\s+", " ", "".join(self._cell)).strip()
            self._row.append(text)
            if self._rowspan > 1:
                self._span[self._col] = [text, self._rowspan - 1]
            self._col += 1
            self._cell = None
            self._fill()
        elif tag == "tr" and self._row is not None and self._depth == 1:
            if self._row:
                self._t.append(self._row)
            self._row = None
        elif tag == "table":
            self._depth -= 1
            if self._depth == 0:
                self.tables.append(self._t)
                self._t = None

    def handle_data(self, data):
        if self._cell is not None and not self._skip:
            self._cell.append(data)


_MONEY = re.compile(r"£\s*([\d.,]+)\s*(m|million|k|thousand|bn)?", re.I)


def parse_fee(text: str) -> tuple[str, float | None]:
    t = (text or "").lower()
    m = _MONEY.search(text or "")
    if m:
        v = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit in ("k", "thousand"):
            v /= 1000
        elif unit == "bn":
            v *= 1000
        elif not unit and v >= 1000:      # "£500,000"
            v /= 1e6
        return "fee", v
    for k in ("undisclosed", "free", "loan"):
        if k in t:
            return k, None
    return "other", None


def _date(s: str, window: str) -> pd.Timestamp | None:
    d = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(d):
        y = int(re.search(r"\d{4}", window).group())
        d = pd.to_datetime(f"{s} {y}", errors="coerce", dayfirst=True)
    return None if pd.isna(d) else d.normalize().tz_localize("UTC")      # the calendar day of the move


def parse_page(p: dict, window: str) -> pd.DataFrame:
    t = _Tables()
    t.feed(p["html"])
    rows = []
    for tb in t.tables:
        head = [h.lower().strip() for h in tb[0]]
        col = {}
        for i, h in enumerate(head):
            key = ("date" if h in ("date", "start date", "date from") else "player" if h in ("name", "player")
                   else "from" if h == "moving from" else "to" if h == "moving to" else "fee" if h == "fee" else None)
            if key and key not in col:
                col[key] = i
        if not {"date", "player", "from", "to"} <= set(col):
            continue                      # e.g. released players ("New club"): no move between clubs
        loans = "fee" not in col           # loan tables have start/end dates and no fee column
        for r in tb[1:]:
            if len(r) < len(head):
                continue
            when, player, frm, to = (r[col[k]] for k in ("date", "player", "from", "to"))
            if loans:
                kind, fee, fee_text = "loan", None, "Loan"
            else:
                fee_text = r[col["fee"]]
                kind, fee = parse_fee(fee_text)
            rows.append({"window": window, "date": _date(when, window), "player": player, "from_club": frm,
                         "to_club": to, "kind": kind, "fee_gbp_m": fee, "fee_text": fee_text})
    df = pd.DataFrame(rows, columns=["window", "date", "player", "from_club", "to_club", "kind", "fee_gbp_m", "fee_text"])
    df["from_id"] = df["from_club"].map(try_resolve_team)
    df["to_id"] = df["to_club"].map(try_resolve_team)
    df["revision_id"], df["revision_utc"] = p["revision_id"], p["revision_utc"]
    return df


# ---------------------------------------------------------------- build and load

def build(refresh: bool = True, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fetch (or read) every window, parse, store the table as parquet in the bucket; returns it."""
    from .fd_cache import _load_env
    _load_env()
    today = (today or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    parts, missing = [], []
    for w in windows(today):
        p = page(w, today, refresh)
        if p is None:
            missing.append(w["slug"])
            continue
        parts.append(parse_page(p, w["slug"]))
    df = pd.concat(parts, ignore_index=True)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    _put(f"{PREFIX}/transfers.parquet", buf.getvalue(), "application/octet-stream")
    if missing:
        print(f"transfers: no page for {missing}")
    return df


def load() -> pd.DataFrame:
    from .fd_cache import _load_env
    _load_env()
    return pd.read_parquet(io.BytesIO(ss.get(BUCKET, f"{PREFIX}/transfers.parquet")))


def summary(df: pd.DataFrame) -> str:
    by = df.groupby("window").agg(moves=("player", "size"), fees=("kind", lambda k: int((k == "fee").sum())),
                                  undisclosed=("kind", lambda k: int((k == "undisclosed").sum())),
                                  loans=("kind", lambda k: int((k == "loan").sum())),
                                  no_date=("date", lambda d: int(d.isna().sum())))
    return by.to_string()


# ---------------------------------------------------------------- automatic refresh (from the logger)

from pathlib import Path  # noqa: E402

STATE = Path(__file__).resolve().parent.parent / "results" / "prospective" / "transfers_state.json"
DAILY = pd.Timedelta(hours=24)
NEW_PLAYER_GAP = pd.Timedelta(hours=1)


def _active(now: pd.Timestamp) -> bool:
    """A window is open, or closed less than CURRENT_DAYS ago (its page is still being filled in)."""
    return any(w["opens"] <= now <= w["closes"] + pd.Timedelta(days=CURRENT_DAYS) for w in windows(now))


def auto_refresh(now: pd.Timestamp, fpl_codes: set[int], state_path: Path = STATE, build_fn=None) -> dict | None:
    """Called by every logger run. Rebuilds the transfer table when
      * it has never been built;
      * FPL lists players it did not list at the last build (new signings), at most hourly; or
      * a window is open or recent and the last build is a day old.
    Settled windows come from storage, so a rebuild fetches only the current pages. Returns what it did,
    or None when nothing was due. Collected for research; no forecast reads it."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    last = pd.Timestamp(state["built_utc"]) if state.get("built_utc") else None
    known = set(state.get("fpl_codes", []))
    new = sorted(fpl_codes - known) if known else []
    reason = None
    if last is None:
        reason = "first build"
    elif new and now - last >= NEW_PLAYER_GAP:
        reason = f"{len(new)} new FPL player(s)"
    elif _active(now) and now - last >= DAILY:
        reason = "daily refresh during a window"
    if reason is None:
        return None
    df = (build_fn or build)(refresh=True, today=now)
    out = {"built_utc": now.isoformat(), "reason": reason, "rows": int(len(df)),
           "windows": int(df["window"].nunique()), "fpl_codes": sorted(int(c) for c in fpl_codes)}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(out) + "\n")
    return out


# ---------------------------------------------------------------- point-in-time club features

SIGNING_KINDS = ("fee", "undisclosed", "free")


def club_features(moves: pd.DataFrame, team: str, season_start_year: int, cutoff: pd.Timestamp) -> dict:
    """A club's transfer activity for a season, using only moves dated before the cutoff's day: from 1 May
    of the season's start year (the summer window) up to the cutoff, so a mid-season cutoff also counts
    the winter window so far. Fees are reported fees in £m; undisclosed fees count as signings only."""
    start = pd.Timestamp(f"{season_start_year}-05-01", tz="UTC")
    day = pd.Timestamp(cutoff).tz_convert("UTC").normalize()
    m = moves[(moves["date"] >= start) & (moves["date"] < day)]
    inn, out = m[m["to_id"] == team], m[m["from_id"] == team]
    return {"fees_in": float(inn["fee_gbp_m"].sum()), "fees_out": float(out["fee_gbp_m"].sum()),
            "net_spend": float(inn["fee_gbp_m"].sum() - out["fee_gbp_m"].sum()),
            "signings": int(inn["kind"].isin(SIGNING_KINDS).sum()),
            "undisclosed_in": int((inn["kind"] == "undisclosed").sum()),
            "loans_in": int((inn["kind"] == "loan").sum()),
            "departures": int(out["kind"].isin(SIGNING_KINDS).sum())}


# ---------------------------------------------------------------- squad cost (Tomkins-style)

PERMANENT = ("fee", "undisclosed", "free")


def fee_index(moves: pd.DataFrame) -> pd.Series:
    """Fee inflation by window year: the mean disclosed fee paid by clubs we know (Premier League ids), relative
    to the latest year. A fee from year y counts as fee * index[latest] / index[y] in today's money."""
    paid = moves[(moves["kind"] == "fee") & moves["to_id"].notna() & moves["date"].notna()]
    by = paid.groupby(paid["date"].dt.year)["fee_gbp_m"].mean()
    return by.rolling(3, min_periods=1, center=True).mean()          # smooth thin years


def squad_cost(moves: pd.DataFrame, team: str, cutoff: pd.Timestamp, index: pd.Series | None = None) -> dict:
    """Inflation-adjusted fees (£m, latest-year money) of the players at `team` on the cutoff's day: each
    player whose last move before that day was a permanent move to `team`. Players who joined before 2005
    or came through the academy have no recorded fee and count 0; a player out on loan counts as gone
    (loan returns are not listed). Only moves dated before the cutoff's day are used."""
    day = pd.Timestamp(cutoff).tz_convert("UTC").normalize()
    m = moves[(moves["date"] < day) & ((moves["to_id"] == team) | (moves["from_id"] == team))]
    players = moves[(moves["date"] < day) & moves["player"].isin(m["player"])].sort_values("date")
    last = players.groupby("player").tail(1)
    here = last[(last["to_id"] == team) & last["kind"].isin(PERMANENT)]
    idx = index if index is not None else fee_index(moves)
    ref = float(idx.iloc[-1])
    yr = here["date"].dt.year.map(lambda y: float(idx.get(y, idx.iloc[0] if y < idx.index[0] else ref)))
    cost = (here["fee_gbp_m"].fillna(0.0) * ref / yr).sum()
    return {"squad_cost": float(cost), "squad_players_with_fee": int(here["fee_gbp_m"].notna().sum()),
            "squad_players_recorded": int(len(here))}
