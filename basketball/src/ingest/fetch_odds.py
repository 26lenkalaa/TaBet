"""Read the Kaggle NBA betting file and turn it into the odds table market.py uses.

Kaggle needs a login, so this script does not download anything. Download
"NBA Betting Data | October 2007 to June 2026" (one file, nba_2008-2026.csv)
in your browser and put it in basketball/data/raw/odds/. Then, from the repo root
(after fetch_games.py and clean.py):

    python3 basketball/src/ingest/fetch_odds.py

It writes basketball/data/processed/odds.csv (one row per game: date, home, away,
spread_close, total_close, home_ml, away_ml) and prints how many games in
games.csv now have a closing spread.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BASKETBALL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASKETBALL_DIR))

from src import market  # noqa: E402

ODDS_RAW_DIR = BASKETBALL_DIR / "data" / "raw" / "odds"
ODDS_FILE = BASKETBALL_DIR / "data" / "processed" / "odds.csv"
GAMES_FILE = BASKETBALL_DIR / "data" / "processed" / "games.csv"

REQUIRED_COLUMNS = ["date", "home", "away", "whos_favored", "spread"]
OPTIONAL_COLUMNS = ["total", "moneyline_home", "moneyline_away", "score_home", "score_away"]


def find_raw_file(raw_dir: Path = ODDS_RAW_DIR) -> Path:
    """The Kaggle CSV in raw_dir (the newest one if there are several)."""
    files = sorted(Path(raw_dir).glob("*.csv"), key=lambda file: file.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            f"No CSV in {raw_dir}. Download nba_2008-2026.csv from Kaggle and put it there.")
    return files[-1]


def read_kaggle(file: Path) -> pd.DataFrame:
    """Kaggle's layout -> the standard odds table (the spread becomes the home team's line)."""
    raw = pd.read_csv(file)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"{Path(file).name} is missing columns: {', '.join(missing)}.\n"
            f"It has: {', '.join(raw.columns)}")

    # Kaggle gives the size of the spread and says who is favored.
    # market.py wants the home team's line: negative when the home team is favored.
    size = pd.to_numeric(raw["spread"], errors="coerce").abs()
    favored = raw["whos_favored"].astype(str).str.strip().str.lower()
    spread = size.where(favored == "away", -size)
    spread = spread.where(favored.isin(["home", "away"]) | (size == 0))   # unknown favorite -> no spread
    spread = spread.where(size != 0, 0.0)                                 # pick'em

    odds = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], format="mixed").dt.strftime("%Y-%m-%d"),
        "home": raw["home"].map(market.team_abbreviation),
        "away": raw["away"].map(market.team_abbreviation),
        "spread_close": spread,
    })
    if "total" in raw.columns:
        total = pd.to_numeric(raw["total"], errors="coerce")
        odds["total_close"] = total.where(total > 0)
    for ours, theirs in (("home_ml", "moneyline_home"), ("away_ml", "moneyline_away")):
        if theirs in raw.columns:
            moneyline = pd.to_numeric(raw[theirs], errors="coerce")
            odds[ours] = moneyline.where(moneyline.abs() >= 100)   # anything smaller is not a real price
    for ours, theirs in (("home_pts", "score_home"), ("away_pts", "score_away")):
        if theirs in raw.columns:
            odds[ours] = pd.to_numeric(raw[theirs], errors="coerce")
    return odds.sort_values(["date", "home"]).reset_index(drop=True)


def score_mismatches(games_with_odds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Matched games whose final score in the odds file differs from NBA.com's.

    A mismatch means the odds were attached to the wrong game (or the odds file is wrong).
    """
    if not {"home_pts", "away_pts"} <= set(odds.columns):
        return games_with_odds.iloc[0:0]
    scores = odds.rename(columns={"home_pts": "odds_home_pts", "away_pts": "odds_away_pts"})
    scores = scores[["home", "away", "odds_home_pts", "odds_away_pts"]]
    checked = games_with_odds.dropna(subset=["spread_close"]).merge(
        scores, left_on=["home", "away", "home_pts", "away_pts"],
        right_on=["home", "away", "odds_home_pts", "odds_away_pts"], how="left")
    wrong = checked[checked["odds_home_pts"].isna()]
    return wrong[["game_id", "date", "home", "away", "home_pts", "away_pts"]].drop_duplicates("game_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn the Kaggle NBA betting file into odds.csv.")
    parser.add_argument("--file", type=Path, help="the Kaggle CSV (default: newest CSV in data/raw/odds/)")
    parser.add_argument("--games", type=Path, default=GAMES_FILE, help="cleaned games table")
    parser.add_argument("--output", type=Path, default=ODDS_FILE, help="where to write odds.csv")
    args = parser.parse_args(argv)

    try:
        odds = read_kaggle(args.file or find_raw_file())
        standard = market.standardize_odds(odds)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    standard.to_csv(args.output, index=False)
    print(f"Wrote odds for {len(standard)} games ({standard['date'].min()} to {standard['date'].max()}) "
          f"to {args.output}")

    if not args.games.exists():
        print(f"{args.games} not found, so coverage was not checked. Run clean.py, then run this again.")
        return 0

    games = pd.read_csv(args.games, dtype={"game_id": str})
    matched = market.match_odds(games, standard)
    print("\nGames with a closing spread, per season:")
    print(market.coverage(matched).to_string())

    missing = market.games_without_spread(matched)
    if len(missing):
        print(f"\n{len(missing)} games have no closing spread. First 20:")
        print(missing.head(20).to_string(index=False))
    wrong = score_mismatches(matched, odds)
    if len(wrong):
        print(f"\n{len(wrong)} matched games have a different final score in the odds file. First 20:")
        print(wrong.head(20).to_string(index=False))
    if not len(missing) and not len(wrong):
        print("\nEvery game has a closing spread and every score agrees.")
    print(f"\nSpread of real margins around the closing spread: {market.estimate_margin_sd(matched):.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
