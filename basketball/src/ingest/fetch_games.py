"""Download NBA team game logs from NBA.com (via nba_api) into data/raw/.

One CSV per season and season type, saved exactly as NBA.com returns it.
Each game appears twice (one row per team); clean.py combines them later.

Run from the repo root:
    python3 basketball/src/ingest/fetch_games.py
    python3 basketball/src/ingest/fetch_games.py --seasons 2021-22 2022-23
    python3 basketball/src/ingest/fetch_games.py --force   # re-download existing files
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog  # type: ignore[import-untyped]

# basketball/src/ingest/fetch_games.py -> basketball/data/raw
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Seasons with free historical odds (Princeton database ends 2022-23).
DEFAULT_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]

# NBA.com's names for each part of the season, and the short name used in filenames.
SEASON_TYPES = {
    "regular": "Regular Season",
    "playin": "PlayIn",
    "playoffs": "Playoffs",
}

PAUSE_SECONDS = 2.0   # NBA.com blocks clients that send requests too fast
MAX_ATTEMPTS = 3      # NBA.com often times out; retry before giving up
TIMEOUT_SECONDS = 60


def raw_path(season: str, season_type: str) -> Path:
    return RAW_DIR / f"games_{season}_{season_type}.csv"


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    """Ask NBA.com for every team game log in one season, retrying on errors."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=SEASON_TYPES[season_type],
                player_or_team_abbreviation="T",   # T = one row per team per game
                timeout=TIMEOUT_SECONDS,
            )
            frame: pd.DataFrame = log.get_data_frames()[0]
            return frame
        except Exception as error:  # nba_api raises plain requests errors
            if attempt == MAX_ATTEMPTS:
                raise
            wait = PAUSE_SECONDS * 2 ** attempt
            print(f"  attempt {attempt} failed ({error.__class__.__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def check(frame: pd.DataFrame, season: str, season_type: str) -> None:
    """Catch obviously broken downloads before they reach clean.py."""
    if frame.empty:
        if season_type == "playin":
            return  # no play-in tournament before 2020-21
        raise ValueError(f"No games returned for {season} {season_type}")

    per_game = frame.groupby("GAME_ID").size()
    if not (per_game == 2).all():
        bad = per_game[per_game != 2].index.tolist()[:5]
        raise ValueError(f"Games without exactly 2 team rows in {season}: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                        help="seasons in NBA.com format, e.g. 2022-23")
    parser.add_argument("--types", nargs="+", choices=SEASON_TYPES, default=["regular", "playoffs"],
                        help="parts of the season to download")
    parser.add_argument("--force", action="store_true",
                        help="download again even if the file already exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    first_request = True

    for season in args.seasons:
        for season_type in args.types:
            path = raw_path(season, season_type)
            if path.exists() and not args.force:
                print(f"{path.name}: already downloaded, skipping")
                continue

            if not first_request:
                time.sleep(PAUSE_SECONDS)
            first_request = False

            print(f"{season} {season_type}: downloading...")
            try:
                frame = fetch_season(season, season_type)
                check(frame, season, season_type)
            except Exception as error:
                sys.exit(f"Failed on {season} {season_type}: {error}")

            frame.to_csv(path, index=False)
            print(f"  saved {len(frame) // 2} games to {path.name}")


if __name__ == "__main__":
    main()
