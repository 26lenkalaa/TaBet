"""Load football-data.co.uk season CSVs into a normalised match table plus a
separate bookmaker-odds benchmark table.

Odds are kept apart on purpose: they're a benchmark for judging predictions,
not a model input (see docs/SPEC.md section 8).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .teams import TEAMS, try_resolve_team

LOCAL_TZ = "Europe/London"
# Used when a source row has no kickoff time. A late kickoff is the safe
# assumption, because it can only make a result look *later*, never earlier.
UNKNOWN_KICKOFF_LOCAL = "22:00"

IDENTITY_COLS = ("Date", "Time", "HomeTeam", "AwayTeam")
TEXT_COLS = {"Div": "division", "FTR": "result", "HTR": "ht_result", "Referee": "referee"}
INT_COLS = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}
# Team xG, added to football-data.co.uk files from 2026-27 onwards.
FLOAT_COLS = {"HxG": "home_xg", "AxG": "away_xg"}
HANDICAP_LINE_COLS = {"AHh", "AHCh"}
FIXES_FILE = Path(__file__).resolve().parent.parent / "data" / "source_fixes.csv"
REQUIRED_RAW = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")

MATCH_COLUMNS = [
    "match_id",
    "season",
    "division",
    "kickoff_utc",
    "kickoff_time_known",
    "home_id",
    "away_id",
    "home_team",
    "away_team",
    "home_team_raw",
    "away_team_raw",
    "status",
    *INT_COLS.values(),
    *FLOAT_COLS.values(),
    "result",
    "ht_result",
    "referee",
    "source_file",
]


class IngestError(ValueError):
    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = problems
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        super().__init__(f"{source}: {len(problems)} problem(s)\n  {shown}{more}")


def season_from_filename(path: str | Path) -> str:
    """'Premier League 21-22.csv' -> '2021-22'."""
    m = re.search(r"(\d{2})-(\d{2})(?!.*\d{2}-\d{2})", Path(path).stem)
    if not m:
        raise ValueError(f"cannot infer season from file name {Path(path).name!r}")
    start, end = int(m.group(1)), int(m.group(2))
    if (start + 1) % 100 != end:
        raise ValueError(f"season in {Path(path).name!r} is not consecutive years")
    # football-data.co.uk starts at 1993-94, so 90-99 means the 1900s.
    century = 1900 if start >= 90 else 2000
    return f"{century + start}-{end:02d}"


def _row_labels(mask: pd.Series) -> str:
    # +2: one for the header row, one for 1-based line numbers.
    lines = [str(i + 2) for i in mask[mask].index[:10]]
    return ", ".join(lines) + (" ..." if mask.sum() > 10 else "")


def _parse_int(raw: pd.Series, col: str, problems: list[str]) -> pd.Series:
    s = raw.str.strip()
    num = pd.to_numeric(s.where(s != ""), errors="coerce")
    bad = num.isna() & (s != "")
    if bad.any():
        problems.append(f"column {col}: non-numeric values on lines {_row_labels(bad)}")
    frac = num.notna() & (num % 1 != 0)
    if frac.any():
        problems.append(f"column {col}: non-integer values on lines {_row_labels(frac)}")
    return num.where(~frac).astype("Int64")


def _parse_kickoff(raw: pd.DataFrame, problems: list[str]) -> tuple[pd.Series, pd.Series]:
    d = raw["Date"].str.strip()
    dates = pd.to_datetime(d, format="%d/%m/%Y", errors="coerce")
    short = dates.isna()
    dates[short] = pd.to_datetime(d[short], format="%d/%m/%y", errors="coerce")
    if dates.isna().any():
        problems.append(f"unparseable Date on lines {_row_labels(dates.isna())}")

    if "Time" in raw.columns:
        t = raw["Time"].str.strip()
    else:
        t = pd.Series("", index=raw.index)
    time_known = t.str.fullmatch(r"\d{1,2}:\d{2}")
    malformed = ~time_known & (t != "")
    if malformed.any():
        problems.append(f"unparseable Time on lines {_row_labels(malformed)}")
    t = t.where(time_known, UNKNOWN_KICKOFF_LOCAL)

    local = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + t, format="%Y-%m-%d %H:%M", errors="coerce"
    )
    kickoff = local.dt.tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    return kickoff, time_known.astype(bool)


def load_season(path: str | Path, season: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one season file. Returns (matches, odds). Raises IngestError on bad input."""
    path = Path(path)
    season = season or season_from_filename(path)
    problems: list[str] = []

    # utf-8-sig strips the BOM some of these files start with.
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    missing = [c for c in REQUIRED_RAW if c not in raw.columns]
    if missing:
        raise IngestError(path.name, [f"missing required columns: {missing}"])
    # football-data files sometimes end with blank rows of commas.
    raw = raw[raw["HomeTeam"].str.strip() != ""].reset_index(drop=True)

    out = pd.DataFrame(index=raw.index)
    out["season"] = season
    kickoff, time_known = _parse_kickoff(raw, problems)
    out["kickoff_utc"] = kickoff
    out["kickoff_time_known"] = time_known

    for side, col in (("home", "HomeTeam"), ("away", "AwayTeam")):
        names = raw[col].str.strip()
        ids = names.map(try_resolve_team)
        unknown = sorted(set(names[ids.isna()]))
        if unknown:
            problems.append(f"unknown team names (add to tabet/teams.py): {unknown}")
        out[f"{side}_id"] = ids
        out[f"{side}_team"] = ids.map(TEAMS)
        out[f"{side}_team_raw"] = names

    for src, dst in TEXT_COLS.items():
        out[dst] = raw[src].str.strip() if src in raw.columns else ""
    for src, dst in INT_COLS.items():
        if src in raw.columns:
            out[dst] = _parse_int(raw[src], src, problems)
        else:
            out[dst] = pd.Series(pd.NA, index=raw.index, dtype="Int64")

    for src, dst in FLOAT_COLS.items():
        if src in raw.columns:
            v = raw[src].str.strip()
            num = pd.to_numeric(v.where(v != ""), errors="coerce")
            bad = num.isna() & (v != "")
            if bad.any():
                problems.append(f"column {src}: non-numeric values on lines {_row_labels(bad)}")
            out[dst] = num
        else:
            out[dst] = pd.Series(float("nan"), index=raw.index, dtype="float64")

    has_score = out["home_goals"].notna() & out["away_goals"].notna() & (out["result"] != "")
    no_score = out["home_goals"].isna() & out["away_goals"].isna() & (out["result"] == "")
    partial = ~has_score & ~no_score
    if partial.any():
        problems.append(f"incomplete result fields (FTHG/FTAG/FTR) on lines {_row_labels(partial)}")
    out["status"] = has_score.map({True: "completed", False: "scheduled"})
    out["source_file"] = path.name

    if problems:
        raise IngestError(path.name, problems)

    local_date = out["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y%m%d")
    out["match_id"] = season + "-" + local_date + "-" + out["home_id"] + "-" + out["away_id"]
    matches = out[MATCH_COLUMNS]

    known = set(IDENTITY_COLS) | set(TEXT_COLS) | set(INT_COLS) | set(FLOAT_COLS)
    odds_cols = [c for c in raw.columns if c not in known and c != ""]
    cols = {"match_id": matches["match_id"]}
    for c in odds_cols:
        v = raw[c].str.strip()
        num = pd.to_numeric(v.where(v != ""), errors="coerce")
        # A price of exactly 0 is the source's placeholder for "missing". The
        # handicap line columns (AHh/AHCh) legitimately contain 0, so skip them.
        if c not in HANDICAP_LINE_COLS:
            num = num.mask(num == 0)
        cols[c] = num
    odds = pd.DataFrame(cols)
    return matches, odds


def apply_fixes(matches: pd.DataFrame, fixes_path: str | Path | None = FIXES_FILE) -> pd.DataFrame:
    """Apply the documented corrections in data/source_fixes.csv. The only action
    supported is set_null: we blank values we know are wrong rather than guess
    replacements. A fix that no longer matches a row is an error, so the file
    can't go stale without anyone noticing."""
    if fixes_path is None or not Path(fixes_path).exists():
        return matches
    fixes = pd.read_csv(fixes_path, dtype=str, keep_default_na=False)
    problems = []
    out = matches.copy()
    for _, f in fixes.iterrows():
        if f["action"] != "set_null":
            problems.append(f"unsupported action {f['action']!r} for {f['match_id']}")
            continue
        if f["field"] not in out.columns or f["field"] in ("match_id", "season", "home_id", "away_id"):
            problems.append(f"cannot null field {f['field']!r}")
            continue
        hit = out["match_id"] == f["match_id"]
        if hit.sum() != 1:
            problems.append(f"fix target {f['match_id']} matched {hit.sum()} rows")
            continue
        out.loc[hit, f["field"]] = pd.NA
    if problems:
        raise IngestError(Path(fixes_path).name, problems)
    return out


def load_all(res_dir: str | Path, fixes_path: str | Path | None = FIXES_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every *.csv in res_dir, then apply the source fixes. Problems from
    all files are reported together."""
    files = sorted(Path(res_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {res_dir}")
    matches, odds, errors = [], [], []
    for f in files:
        try:
            m, o = load_season(f)
        except (IngestError, ValueError) as e:
            errors.append(str(e))
            continue
        matches.append(m)
        odds.append(o)
    if errors:
        raise IngestError(str(res_dir), errors)
    all_matches = (
        pd.concat(matches, ignore_index=True)
        .sort_values(["kickoff_utc", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    all_odds = pd.concat(odds, ignore_index=True)
    return apply_fixes(all_matches, fixes_path), all_odds


def implied_probabilities(odds: pd.DataFrame, prefix: str = "Avg") -> pd.DataFrame:
    """Turn decimal H/D/A odds into probabilities by dividing out the overround.
    Benchmark use only. prefix='Avg' for pre-match market average, 'AvgC' for closing."""
    cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
    inv = 1.0 / odds[cols]
    probs = inv.div(inv.sum(axis=1), axis=0)
    probs.columns = ["p_home", "p_draw", "p_away"]
    return pd.concat([odds[["match_id"]], probs], axis=1)
