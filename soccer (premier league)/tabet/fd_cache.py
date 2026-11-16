"""football-data.co.uk season files, cached in Supabase Storage instead of on disk.

    from tabet.fd_cache import season
    df = season("E1", "0809")      # one division-season, in the match schema of tabet.leagues

Each file is looked up in the private bucket `football-data` (history/<DIV>_<yyyy>.csv, already
cleaned). If it is not there, it is downloaded once from football-data, cleaned (each line cut to its
header's width, re-encoded as UTF-8: several old files have stray fields or Latin-1 bytes), uploaded,
and returned. Nothing is kept on disk: parsing uses a temporary file that is deleted at once, and an
in-memory cache avoids repeat downloads within one process.

Credentials: SUPABASE_URL / SUPABASE_SECRET_KEY from the environment, or from backend/.env (local).
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import supabase_storage as ss
from .fpl import _ssl_context
from .leagues import load_league_season

BUCKET = "football-data"
PREFIX = "history"
SOURCE = "https://football-data.co.uk/mmz4281/{code}/{div}.csv"
ENV_FILE = Path(__file__).resolve().parents[2] / "backend" / ".env"


def _load_env() -> None:
    """Use backend/.env for local runs when the variables are not already set."""
    if ss.configured() or not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def clean(raw: bytes) -> bytes:
    """Cut every line to the header's width and re-encode as UTF-8; drop blank lines."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return b""
    head = [h for h in rows[0] if h.strip()] or rows[0]
    k = len(head)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(head)
    for r in rows[1:]:
        if any(x.strip() for x in r):
            w.writerow((r + [""] * k)[:k])
    return out.getvalue().encode("utf-8")


def _download(div: str, code: str) -> bytes:
    req = urllib.request.Request(SOURCE.format(code=code, div=div), headers={"User-Agent": "tabet-research/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        data = r.read()
    time.sleep(0.4)   # polite to football-data
    return data


@lru_cache(maxsize=None)
def raw_csv(div: str, code: str) -> bytes:
    """The cleaned CSV bytes for one division-season: Supabase first, else football-data (then cached)."""
    _load_env()
    if not ss.configured():
        raise RuntimeError("Supabase is not configured (SUPABASE_URL / SUPABASE_SECRET_KEY or backend/.env)")
    path = f"{PREFIX}/{div}_{code}.csv"
    try:
        return ss.get(BUCKET, path)
    except RuntimeError as e:
        if "404" not in str(e) and "400" not in str(e) and "not_found" not in str(e).lower():
            raise
    data = clean(_download(div, code))
    ss.ensure_bucket(BUCKET)
    ss.put(BUCKET, path, data, "text/csv")
    return data


def season(div: str, code: str) -> pd.DataFrame:
    """One division-season in the match schema (tabet.leagues), parsed from the cache."""
    data = raw_csv(div, code)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"{div}_{code}.csv"   # load_league_season reads the division and season from the name
        p.write_bytes(data)
        return load_league_season(p)


def seasons(divs, codes) -> pd.DataFrame:
    """Several division-seasons concatenated; files football-data does not have are skipped."""
    parts = []
    for d in divs:
        for c in codes:
            try:
                parts.append(season(d, c))
            except Exception as e:  # e.g. a division that did not exist that season
                print(f"fd_cache: skipped {d}_{c} ({str(e)[:80]})")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def upload_local(folder: Path) -> int:
    """One-off: upload already-downloaded <DIV>_<yyyy>.csv files (cleaned) to the cache."""
    _load_env()
    ss.ensure_bucket(BUCKET)
    n = 0
    for f in sorted(folder.glob("*_*.csv")):
        ss.put(BUCKET, f"{PREFIX}/{f.name}", clean(f.read_bytes()), "text/csv")
        n += 1
    return n
