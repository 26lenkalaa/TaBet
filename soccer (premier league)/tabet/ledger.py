"""Data ledger: which league-seasons each experiment family has used, enforced in scripts.

docs/data_ledger.csv has one row per (division, season range, family) with a role:
  training     inputs only, never scored
  development  scored while choosing settings or ideas
  test         scored once for a pre-registered decision
and a status: reserved (pre-registered, not yet run) or spent (run).

An experiment registers its rows in the ledger (in the same commit as its pre-registration), then its
script calls `claim(family, division, seasons, role)` before using the data. claim() refuses if:
  * the family has not registered that role for those cells;
  * role "test": any *other* family has used the cells in any way, unless the cells form a closed test
    set whose quota (number of test families) still has room;
  * role "training" or "development": another family holds an unrun (reserved) test on the cells,
    because using a test set before its test runs would contaminate it.
A spent test set may be reused for development; it is never again "untouched".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEDGER = Path(__file__).resolve().parent.parent / "docs" / "data_ledger.csv"
ROLES = {"training", "development", "test"}


class LedgerError(RuntimeError):
    """A claim that would break the data ledger."""


def season_range(first: str, last: str) -> list[str]:
    """['2018-19', '2019-20', '2020-21'] for ('2018-19', '2020-21')."""
    y0, y1 = int(first[:4]), int(last[:4])
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(y0, y1 + 1)]


def load(path: Path = LEDGER) -> pd.DataFrame:
    """One row per (division, season, family, role, status, quota)."""
    raw = pd.read_csv(path, dtype=str)
    rows = []
    for r in raw.itertuples():
        for s in season_range(r.season_from, r.season_to):
            rows.append({"division": r.division, "season": s, "family": r.family, "role": r.role,
                         "status": r.status, "quota": int(r.quota)})
    return pd.DataFrame(rows)


def claim(family: str, division: str, seasons, role: str, path: Path = LEDGER) -> None:
    """Raise LedgerError unless `family` may use these league-seasons in this role."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    seasons = list(seasons)
    led = load(path)
    for s in seasons:
        here = led[(led.division == division) & (led.season == s)]
        mine = here[(here.family == family) & (here.role == role)]
        if mine.empty:
            raise LedgerError(f"{family} has not registered {division} {s} as {role} in {path.name}")
        others = here[here.family != family]
        if role == "test":
            closed = others[(others.role == "test") & (others.quota > 1)]
            if len(others) and not (len(others) == len(closed) and
                                    here[here.role == "test"].family.nunique() <= int(closed.quota.min())):
                raise LedgerError(f"{division} {s} is not untouched: already used by "
                                  f"{sorted(set(others.family))} ({sorted(set(others.role))})")
        else:
            pending = others[(others.role == "test") & (others.status == "reserved")]
            if len(pending):
                raise LedgerError(f"{division} {s} is reserved as a test set by {sorted(set(pending.family))}; "
                                  f"it cannot be used for {role} before that test runs")
