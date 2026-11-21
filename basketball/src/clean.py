"""Turn the raw NBA.com team game logs into one row per game.

NBA.com lists every game twice, once for each team. This script reads every
CSV in basketball/data/raw/, pairs the two team rows of each game and writes
basketball/data/processed/games.csv with one row per game: date, season,
home and away teams, scores, box-score columns and is_playoffs.

Run it from the repo root:

    python basketball/src/clean.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASKETBALL_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASKETBALL_DIR / "data" / "raw"
PROCESSED_DIR = BASKETBALL_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "games.csv"

# Columns every raw file must have.
REQUIRED_COLUMNS = ["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]

# Box-score columns copied for each team when the raw files have them.
BOX_SCORE_COLUMNS = [
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
]

# The first three digits of an NBA game id say what kind of game it is.
GAME_TYPES = {
    "001": "preseason",
    "002": "regular",
    "003": "all_star",
    "004": "playoffs",
    "005": "play_in",
    "006": "cup_final",
}
KEPT_GAME_TYPES = ["regular", "play_in", "playoffs"]


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read every CSV in raw_dir into one table of team rows."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. Run fetch_games.py first.")
    frames = []
    for file in files:
        frame = pd.read_csv(file, dtype={"GAME_ID": str})
        if frame.empty:
            continue
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{file.name} is missing columns: {', '.join(missing)}")
        frames.append(frame)
    if not frames:
        raise ValueError(f"Every CSV in {raw_dir} is empty.")
    return pd.concat(frames, ignore_index=True)


def season_from_game_id(game_id: str) -> str:
    """'0021800001' -> '2018-19'. Digits 4 and 5 are the year the season started."""
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def clean_games(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the two team rows of each game into one row.

    Returns (games, dropped). `dropped` lists the game ids left out because
    they did not have exactly one home row and one away row with a score.
    """
    rows = raw.copy()
    # Game ids are 10 digits with leading zeros, which are lost if the id was read as a number.
    rows["GAME_ID"] = rows["GAME_ID"].astype(str).str.strip().str.zfill(10)
    rows["game_type"] = rows["GAME_ID"].str[:3].map(GAME_TYPES)
    rows = rows[rows["game_type"].isin(KEPT_GAME_TYPES)]

    # A game can appear in more than one raw file; keep the row from the last file read.
    rows = rows.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")

    # "BOS vs. NYK" is Boston at home; "NYK @ BOS" is New York away.
    rows = rows.assign(
        is_home=rows["MATCHUP"].str.contains("vs.", regex=False),
        is_away=rows["MATCHUP"].str.contains("@", regex=False),
    )

    per_game = rows.groupby("GAME_ID").agg(
        team_rows=("TEAM_ID", "size"),
        home_rows=("is_home", "sum"),
        away_rows=("is_away", "sum"),
        scores=("PTS", "count"),
    )
    reasons = pd.Series("", index=per_game.index)
    reasons[per_game["scores"] != per_game["team_rows"]] = "missing score"
    reasons[(per_game["home_rows"] != 1) | (per_game["away_rows"] != 1)] = "no clear home and away team"
    reasons[per_game["team_rows"] != 2] = "does not have exactly two team rows"
    dropped = reasons[reasons != ""].rename("reason").reset_index().rename(columns={"GAME_ID": "game_id"})
    rows = rows[~rows["GAME_ID"].isin(dropped["game_id"])]

    box_columns = [column for column in BOX_SCORE_COLUMNS if column in rows.columns]
    team_columns = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"] + box_columns

    def side(mask: pd.Series, prefix: str) -> pd.DataFrame:
        names = {"TEAM_ID": f"{prefix}_team_id", "TEAM_ABBREVIATION": prefix, "PTS": f"{prefix}_pts"}
        names.update({column: f"{prefix}_{column.lower()}" for column in box_columns})
        return rows.loc[mask, team_columns].rename(columns=names)

    home_rows = rows["is_home"]
    game_info = rows.loc[home_rows, ["GAME_ID", "GAME_DATE", "game_type"]]
    games = game_info.merge(side(home_rows, "home"), on="GAME_ID").merge(side(~home_rows, "away"), on="GAME_ID")

    games = games.rename(columns={"GAME_ID": "game_id"})
    games["date"] = pd.to_datetime(games["GAME_DATE"], format="mixed").dt.strftime("%Y-%m-%d")
    games["season"] = games["game_id"].map(season_from_game_id)
    games["is_playoffs"] = games["game_type"] == "playoffs"

    number_columns = [column for column in games.columns if column.endswith("_pts") or column.endswith("_team_id")]
    number_columns += [f"{prefix}_{column.lower()}" for prefix in ("home", "away") for column in box_columns]
    complete = games[number_columns].notna().all()
    games = games.astype({column: "int64" for column in number_columns if complete[column]})

    first_columns = [
        "game_id", "season", "date", "game_type", "is_playoffs",
        "home_team_id", "home", "away_team_id", "away", "home_pts", "away_pts",
    ]
    box_score = [f"{prefix}_{column.lower()}" for prefix in ("home", "away") for column in box_columns]
    games = games[first_columns + box_score]
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    return games, dropped


def check_games(games: pd.DataFrame) -> None:
    """Stop with an error if the cleaned table breaks a rule the model relies on."""
    problems = []
    if games["game_id"].duplicated().any():
        problems.append("duplicate game ids")
    if games.duplicated(subset=["date", "home_team_id", "away_team_id"]).any():
        problems.append("the same matchup twice on one date")
    team_columns = ["home_team_id", "away_team_id", "home", "away", "home_pts", "away_pts"]
    if games[team_columns].isna().any().any():
        problems.append("games missing a team or a score")
    if (games["home_team_id"] == games["away_team_id"]).any():
        problems.append("a team playing itself")
    if (games["home_pts"] == games["away_pts"]).any():
        problems.append("tied games (NBA games cannot end in a tie)")
    if problems:
        raise ValueError("games table failed its checks: " + "; ".join(problems))


def season_summary(games: pd.DataFrame) -> pd.DataFrame:
    """Games per season, split by game type."""
    counts = games.groupby(["season", "game_type"]).size().unstack(fill_value=0)
    return counts.reindex(columns=[column for column in KEPT_GAME_TYPES if column in counts.columns])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine raw NBA team game logs into one row per game.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder with the raw CSVs")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="where to write games.csv")
    args = parser.parse_args(argv)

    games, dropped = clean_games(load_raw(args.raw_dir))
    check_games(games)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.output, index=False)

    print(f"Wrote {len(games)} games to {args.output}")
    print(season_summary(games).to_string())
    if len(dropped):
        print(f"\nLeft out {len(dropped)} games:")
        print(dropped.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
