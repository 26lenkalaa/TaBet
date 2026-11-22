"""Managerial changes for the leagues we model, from Wikipedia's season articles (e.g. "2023–24 Premier
League"), whose "Managerial changes" table lists each team's outgoing manager, date of vacancy, incoming
manager and date of appointment. Text from Wikipedia is CC BY-SA 4.0, credited in docs/DATA_SOURCES.md.

    python -m tabet.managers                 # build and upsert into Supabase table manager_changes
    python -m tabet.managers --csv out.csv   # also write a local copy (outside the repo)

Not a model input yet: feature F5 (brief §7) would let a team's rating move faster after a new manager
arrives, once pre-registered. Caretaker or interim appointments are kept and flagged.
"""
from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from .teams import try_resolve_team

ROOT = Path(__file__).resolve().parent.parent
TABLE = "manager_changes"      # backend/supabase/migrations/0006_manager_changes.sql
API = "https://en.wikipedia.org/w/api.php"
AGENT = "TaBet-research/0.1 (student project; managerial changes)"
# league code -> Wikipedia article name after the season, e.g. "2023–24 Premier League"
# (names tried in order: articles were renamed over the years)
LEAGUES = {"E0": ("Premier League",), "E1": ("EFL Championship", "Football League Championship"),
           "SP1": ("La Liga",), "I1": ("Serie A",), "D1": ("Bundesliga",), "F1": ("Ligue 1",), "N1": ("Eredivisie",),
           "P1": ("Primeira Liga",), "B1": ("Belgian Pro League", "Belgian First Division A"),
           "SC0": ("Scottish Premiership",)}
SEASONS = [f"{y}–{(y + 1) % 100:02d}" for y in range(2015, 2027)]       # en dash, as Wikipedia titles
# The sealed holdout (addendum MGR): collected without "position in table" or "manner of departure"
# (they reflect results), and nothing but counts is printed.
SEALED_LEAGUES = {"AUT": ("Austrian Football Bundesliga",), "SWZ": ("Swiss Super League",),
                  "DNK": ("Danish Superliga",), "G1": ("Super League Greece",), "T1": ("Süper Lig",),
                  "I2": ("Serie B",), "D2": ("2. Bundesliga",), "F2": ("Ligue 2",), "SP2": ("Segunda División",)}
SEALED_SEASONS = SEASONS[:-1]                                             # 2015-16 .. 2025-26
INTERIM = re.compile(r"caretaker|interim|\(c\)|\(i\)", re.I)


class _Tables(HTMLParser):
    """Every table as rows of cell texts, with rowspan and colspan expanded; footnote markers dropped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._stack: list[dict] = []
        self._cell: list[str] | None = None
        self._skip: list[str] = []     # tags whose text is dropped (footnotes, sort keys, styles)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._stack.append({"rows": [], "row": None, "spans": {}})
        elif not self._stack:
            return
        elif tag == "tr":
            self._stack[-1]["row"] = []
        elif tag in ("td", "th"):
            self._cell = []
            self._attrs = a
        elif tag == "sup" or (tag == "span" and "sortkey" in (a.get("class") or "")) or tag == "style":
            self._skip.append(tag)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if not self._stack:
            return
        t = self._stack[-1]
        if self._skip and tag == self._skip[-1]:
            self._skip.pop()
        elif tag in ("td", "th") and self._cell is not None and t["row"] is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            col = len(t["row"])
            while col in t["spans"]:                        # cells carried down by an earlier rowspan
                t["row"].append(t["spans"][col][0])
                left = t["spans"][col][1] - 1
                t["spans"].pop(col) if left == 0 else t["spans"].__setitem__(col, (t["spans"][col][0], left))
                col += 1
            def _n(v):
                try:
                    return max(1, int(float(re.sub(r"[^\d.]", "", str(v)) or 1)))
                except ValueError:
                    return 1
            for k in range(_n(self._attrs.get("colspan", 1))):
                rs = _n(self._attrs.get("rowspan", 1))
                if rs > 1:
                    t["spans"][col + k] = (text, rs - 1)
                t["row"].append(text)
            self._cell = None
        elif tag == "tr" and t["row"] is not None:
            col = len(t["row"])
            while col in t["spans"]:
                t["row"].append(t["spans"][col][0])
                left = t["spans"][col][1] - 1
                t["spans"].pop(col) if left == 0 else t["spans"].__setitem__(col, (t["spans"][col][0], left))
                col += 1
            if t["row"]:
                t["rows"].append(t["row"])
            t["row"] = None
        elif tag == "table":
            self.tables.append(self._stack.pop()["rows"])

    def handle_data(self, data):
        if self._cell is not None and not self._skip:
            self._cell.append(data)


def tables(html: str) -> list[list[list[str]]]:
    p = _Tables()
    p.feed(html)
    return p.tables


def _date(s: str) -> pd.Timestamp | None:
    s = re.sub(r"\[.*?\]|\(.*?\)", "", s).strip()
    for fmt in ("%d %B %Y", "%B %d, %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return None


def changes_from_html(html: str, league: str, season: str) -> list[dict]:
    """Rows of the managerial-changes table(s) on one season article."""
    out = []
    for t in tables(html):
        if not t:
            continue
        head = [h.lower() for h in t[0]]
        if not any("outgoing" in h for h in head) or not any(("incoming" in h or "replaced" in h) for h in head):
            continue
        col = lambda *keys: next((i for i, h in enumerate(head) if all(k in h for k in keys)), None)  # noqa: E731
        c_team, c_out, c_why = col("team"), col("outgoing"), col("manner")
        c_vac, c_app = col("vacancy"), col("appointment")
        c_in = col("incoming") if col("incoming") is not None else col("replaced")
        c_pos = col("position") if col("position") is not None else col("pos")
        for r in t[1:]:
            need = max(i for i in (c_team, c_out, c_vac, c_in, c_app) if i is not None)
            if c_team is None or c_in is None or len(r) <= need:
                continue
            team = r[c_team]
            inc = r[c_in]
            out.append({"league": league, "season": season, "team_name": team, "team_id": try_resolve_team(team),
                        "outgoing": r[c_out] if c_out is not None else "", "manner": r[c_why] if c_why is not None else "",
                        "vacancy_date": _date(r[c_vac]) if c_vac is not None else None,
                        "incoming": INTERIM.sub("", inc).strip(" ()"), "interim": bool(INTERIM.search(inc)),
                        "appointment_date": _date(r[c_app]) if c_app is not None else None,
                        "position": r[c_pos] if c_pos is not None else "", "source": f"{season} {({**LEAGUES, **SEALED_LEAGUES}[league])[0]}"})
    return out


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_html(title: str) -> str | None:
    q = urllib.parse.urlencode({"action": "parse", "page": title, "prop": "text", "format": "json",
                                "formatversion": 2, "redirects": 1})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": AGENT})
    for attempt in range(4):
        try:
            body = json.load(urllib.request.urlopen(req, timeout=60, context=_ctx()))
            return (body.get("parse") or {}).get("text")     # None when the article does not exist
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)
    return None


def build(leagues=LEAGUES, seasons=SEASONS, fetch=fetch_html, pause: float = 1.0) -> pd.DataFrame:
    rows, missing = [], []
    for code, names in leagues.items():
        for s in seasons:
            got = []
            for name in names:
                try:
                    html = fetch(f"{s} {name}")
                except Exception:
                    html = None
                time.sleep(pause)
                got = changes_from_html(html, code, s) if html else []
                if got:
                    break
            if not got:
                missing.append(f"{s} {names[0]}")
            rows += got
    df = pd.DataFrame(rows)
    df.attrs["missing"] = missing
    return df


def to_rows(df: pd.DataFrame, sealed: bool = False) -> list[dict]:
    """Rows for the Supabase table: seasons as "2023-24", dates as ISO strings, one row per key. Sealed
    leagues keep no "position in table" or "manner of departure"."""
    out = {}
    for r in df.itertuples():
        key = (r.league, r.season.replace("–", "-"), r.team_name, r.outgoing, r.incoming)
        out[key] = {"league": r.league, "season": key[1], "team_name": r.team_name,
                    "team_id": r.team_id if isinstance(r.team_id, str) else None, "outgoing": r.outgoing,
                    "manner": r.manner, "incoming": r.incoming, "interim": bool(r.interim),
                    "vacancy_date": r.vacancy_date.date().isoformat() if pd.notna(r.vacancy_date) else None,
                    "appointment_date": r.appointment_date.date().isoformat() if pd.notna(r.appointment_date) else None,
                    "table_position": r.position, "source": r.source.replace("–", "-")}
        if sealed:
            out[key].update(manner="", table_position="")
    return list(out.values())


def upload(rows: list[dict], store=None) -> int:
    """Upsert into manager_changes with the secret key (SupabaseStore from tabet.api_football)."""
    import os
    from .api_football import SupabaseStore
    from .fd_cache import _load_env
    if store is None:
        _load_env()
        store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    for i in range(0, len(rows), 500):
        store._insert(TABLE, rows[i:i + 500], upsert=True)
    return len(rows)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Managerial changes from Wikipedia into Supabase")
    ap.add_argument("--csv", default=None, help="also write a local CSV here (keep it outside the repo)")
    ap.add_argument("--sealed", action="store_true", help="the sealed holdout's leagues (addendum MGR): counts only")
    a = ap.parse_args(argv)
    if a.sealed:
        df = build(SEALED_LEAGUES, SEALED_SEASONS)
        n = upload(to_rows(df, sealed=True))
        print(json.dumps({"rows_upserted": n, "by_league": df.groupby("league").size().to_dict() if len(df) else {},
                          "articles_without_a_table": len(df.attrs.get("missing", []))}, indent=1))
        return 0
    df = build()
    rows = to_rows(df)
    n = upload(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    un = df[df["team_id"].isna() & df["league"].isin(["E0", "E1"])]["team_name"].unique() if len(df) else []
    print(json.dumps({"rows_upserted": n, "by_league": df.groupby("league").size().to_dict() if len(df) else {},
                      "no_table": df.attrs.get("missing", []), "unresolved_english_teams": sorted(un)[:40],
                      "missing_appointment_dates": int(df["appointment_date"].isna().sum()) if len(df) else 0},
                     indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
