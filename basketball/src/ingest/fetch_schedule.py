"""Download this season's NBA schedule into basketball/data/raw/schedule.csv.

fetch_games.py only has games that were played. The site's "Upcoming games"
view needs the games still to come, which NBA.com serves as one schedule table
(read through nba_api, like the game logs).

Run from the repo root:

    python3 basketball/src/ingest/fetch_schedule.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2  # type: ignore[import-untyped]

BASKETBALL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASKETBALL_DIR))

from src.ingest.fetch_games import MAX_ATTEMPTS, PAUSE_SECONDS, TIMEOUT_SECONDS, current_season  # noqa: E402

SCHEDULE_FILE = BASKETBALL_DIR / "data" / "raw" / "schedule.csv"

# First three digits of a game id: regular season, playoffs, play-in (see clean.py).
KEPT_PREFIXES = {"002": "regular", "004": "playoffs", "005": "play_in"}
NEEDED_COLUMNS = ["gameId", "gameStatus", "gameDateEst", "homeTeam_teamId", "homeTeam_teamTricode",
                  "awayTeam_teamId", "awayTeam_teamTricode"]
COLUMNS = ["game_id", "date", "game_type", "home_team_id", "home", "away_team_id", "away", "played"]


def download(season: str | None = None) -> pd.DataFrame:
    """NBA.com's schedule for one season (default: the current one), retrying on errors."""
    season = season or current_season()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            schedule = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=TIMEOUT_SECONDS)
            frame: pd.DataFrame = schedule.season_games.get_data_frame()
            return frame
        except Exception as error:  # nba_api raises plain requests errors
            if attempt == MAX_ATTEMPTS:
                raise OSError(f"{error.__class__.__name__}: {error}") from None
            wait = PAUSE_SECONDS * 2 ** attempt
            print(f"  attempt {attempt} failed ({error.__class__.__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """NBA.com's schedule table -> one row per game: id, date, type, teams, and whether it was played.

    Preseason and All-Star games are left out, and so are playoff slots whose teams are not known yet.
    """
    missing = [column for column in NEEDED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"The schedule is missing columns: {', '.join(missing)}. "
                         "NBA.com may have changed its layout.")

    game_id = raw["gameId"].astype(str).str.strip().str.zfill(10)
    schedule = pd.DataFrame({
        "game_id": game_id,
        # the U.S. Eastern date, as in the game logs
        "date": pd.to_datetime(raw["gameDateEst"], format="mixed", utc=True).dt.strftime("%Y-%m-%d"),
        "game_type": game_id.str[:3].map(KEPT_PREFIXES),
        "home_team_id": pd.to_numeric(raw["homeTeam_teamId"], errors="coerce").fillna(0).astype("int64"),
        "home": raw["homeTeam_teamTricode"].fillna("").astype(str).str.strip(),
        "away_team_id": pd.to_numeric(raw["awayTeam_teamId"], errors="coerce").fillna(0).astype("int64"),
        "away": raw["awayTeam_teamTricode"].fillna("").astype(str).str.strip(),
        "played": pd.to_numeric(raw["gameStatus"], errors="coerce") == 3,   # 1 scheduled, 2 in progress, 3 final
    })
    known = (schedule["game_type"].notna() & (schedule["home"] != "") & (schedule["away"] != "")
             & (schedule["home_team_id"] > 0) & (schedule["away_team_id"] > 0))
    schedule = schedule[known].drop_duplicates("game_id")
    return schedule.sort_values(["date", "game_id"]).reset_index(drop=True)[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download this season's NBA schedule.")
    parser.add_argument("--season", help="season to download, e.g. 2026-27 (default: the current one)")
    parser.add_argument("--output", type=Path, default=SCHEDULE_FILE, help="where to write schedule.csv")
    args = parser.parse_args(argv)

    try:
        schedule = parse_schedule(download(args.season))
    except (OSError, ValueError) as error:
        print(f"Error: could not get the schedule: {error}", file=sys.stderr)
        return 1
    if schedule.empty:
        print("Error: the schedule has no regular-season or playoff games.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(args.output, index=False)
    to_play = int((~schedule["played"]).sum())
    print(f"Wrote {len(schedule)} games ({to_play} still to play, "
          f"{schedule['date'].min()} to {schedule['date'].max()}) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
