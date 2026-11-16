"""Read the Kaggle odds files and attach them to the games in games.csv.

Kaggle needs a login, so this script does not download anything. Download the
files in your browser and put them in basketball/data/raw/odds/. It reads every
CSV in that folder and understands two layouts:

  * the betting file (nba_2008-2026.csv): spread, total and moneylines per game,
    matched to our games by date and teams;
  * the moneyline file (game_odds.csv): moneylines per game, matched by NBA game id.

One of them is enough; with both, spreads and totals come from the betting file
and moneylines from whichever file has them (the betting file first).

Run from the repo root, after fetch_games.py and clean.py:

    python3 basketball/src/ingest/fetch_odds.py

It writes basketball/data/processed/odds.csv (one row per game: game_id, date,
home, away, spread_close, total_close, home_ml, away_ml) and prints how many
games have a closing spread and a moneyline.
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

BETTING_COLUMNS = ["date", "home", "away", "whos_favored", "spread"]
ODDS_OUTPUT_COLUMNS = ["spread_close", "total_close", "home_ml", "away_ml"]


def read_columns(file: Path) -> pd.DataFrame:
    table = pd.read_csv(file, dtype={"game_id": str})
    table.columns = [str(column).strip().lower() for column in table.columns]
    return table


def file_kind(table: pd.DataFrame) -> str:
    """'betting', 'moneyline', or 'unknown', judged by the column names."""
    columns = set(table.columns)
    if set(BETTING_COLUMNS) <= columns:
        return "betting"
    if "game_id" in columns and ({"moneyline_home", "moneyline_away"} <= columns
                                 or {"decimal_home", "decimal_away"} <= columns):
        return "moneyline"
    return "unknown"


def clean_moneyline(values: pd.Series) -> pd.Series:
    """Numbers only; anything between -100 and +100 is not a real American price."""
    moneyline = pd.to_numeric(values, errors="coerce")
    return moneyline.where(moneyline.abs() >= 100)


def read_betting(table: pd.DataFrame) -> pd.DataFrame:
    """The betting file -> date, home, away, spread_close (the home team's line), total, moneylines, scores."""
    # The file gives the size of the spread and says who is favored.
    # market.py wants the home team's line: negative when the home team is favored.
    size = pd.to_numeric(table["spread"], errors="coerce").abs()
    favored = table["whos_favored"].astype(str).str.strip().str.lower()
    spread = size.where(favored == "away", -size)
    spread = spread.where(favored.isin(["home", "away"]) | (size == 0))   # unknown favorite -> no spread
    spread = spread.where(size != 0, 0.0)                                 # pick'em

    odds = pd.DataFrame({
        "date": pd.to_datetime(table["date"], format="mixed").dt.strftime("%Y-%m-%d"),
        "home": table["home"].map(market.team_abbreviation),
        "away": table["away"].map(market.team_abbreviation),
        "spread_close": spread,
    })
    if "total" in table.columns:
        total = pd.to_numeric(table["total"], errors="coerce")
        odds["total_close"] = total.where(total > 0)
    for ours, theirs in (("home_ml", "moneyline_home"), ("away_ml", "moneyline_away")):
        if theirs in table.columns:
            odds[ours] = clean_moneyline(table[theirs])
    for ours, theirs in (("home_pts", "score_home"), ("away_pts", "score_away")):
        if theirs in table.columns:
            odds[ours] = pd.to_numeric(table[theirs], errors="coerce")
    return odds.sort_values(["date", "home"]).reset_index(drop=True)


def read_moneylines(table: pd.DataFrame) -> pd.DataFrame:
    """The moneyline file -> game_id, home_ml, away_ml: one row per game, the latest odds."""
    table = table.copy()
    table["game_id"] = table["game_id"].astype(str).str.strip().str.zfill(10)   # NBA ids are 10 digits

    for side in ("home", "away"):
        if f"moneyline_{side}" in table.columns:
            moneyline = clean_moneyline(table[f"moneyline_{side}"])
        else:
            moneyline = pd.Series(float("nan"), index=table.index)
        if f"decimal_{side}" in table.columns:   # fill gaps from the decimal price
            decimal = pd.to_numeric(table[f"decimal_{side}"], errors="coerce")
            decimal = decimal.where(decimal > 1)
            moneyline = moneyline.fillna(decimal.map(market.decimal_to_moneyline, na_action="ignore"))
        table[f"{side}_ml"] = moneyline

    # A game can be listed several times as the odds moved; the last listing is closest to tip-off.
    order = [column for column in ("odds_date", "last_fetched") if column in table.columns]
    if order:
        table = table.sort_values(order, kind="stable")
    table = table.dropna(subset=["home_ml", "away_ml"]).drop_duplicates("game_id", keep="last")
    return table[["game_id", "home_ml", "away_ml"]].reset_index(drop=True)


def load_raw(raw_dir: Path = ODDS_RAW_DIR, files: list[Path] | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(betting odds, moneylines) from the CSVs in raw_dir. Either one is None if no file has it."""
    files = files or sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {raw_dir}. Download the Kaggle odds files and put them there.")
    betting, moneylines = [], []
    for file in files:
        table = read_columns(file)
        kind = file_kind(table)
        if kind == "betting":
            betting.append(read_betting(table))
        elif kind == "moneyline":
            moneylines.append(read_moneylines(table))
        else:
            raise ValueError(
                f"{Path(file).name} is not a layout this script knows.\n"
                f"It has: {', '.join(table.columns)}\n"
                f"A betting file needs: {', '.join(BETTING_COLUMNS)}\n"
                "A moneyline file needs: game_id, moneyline_home, moneyline_away")
        print(f"Read {Path(file).name}: {kind} file")
    return (pd.concat(betting, ignore_index=True) if betting else None,
            pd.concat(moneylines, ignore_index=True).drop_duplicates("game_id", keep="last") if moneylines else None)


def attach_odds(games: pd.DataFrame, betting: pd.DataFrame | None, moneylines: pd.DataFrame | None) -> pd.DataFrame:
    """`games` with spread_close, total_close, home_ml, away_ml and the market's predictions added."""
    if betting is not None:
        with_odds = market.match_odds(games, market.standardize_odds(betting))
    else:
        with_odds = games.copy()
    for column in ODDS_OUTPUT_COLUMNS:
        if column not in with_odds.columns:
            with_odds[column] = float("nan")

    if moneylines is not None:
        by_game = moneylines.set_index("game_id")
        # Keep a game's two prices together: only use the moneyline file where the betting file has no pair.
        new_home = with_odds["game_id"].map(by_game["home_ml"])
        new_away = with_odds["game_id"].map(by_game["away_ml"])
        use = (with_odds["home_ml"].isna() | with_odds["away_ml"].isna()) & new_home.notna() & new_away.notna()
        with_odds.loc[use, "home_ml"] = new_home[use]
        with_odds.loc[use, "away_ml"] = new_away[use]
    return market.add_market_predictions(with_odds)


def score_mismatches(games_with_odds: pd.DataFrame, betting: pd.DataFrame | None) -> pd.DataFrame:
    """Games with a spread whose final score in the betting file differs from NBA.com's.

    A mismatch means the odds were attached to the wrong game (or the betting file is wrong).
    """
    if betting is None or not {"home_pts", "away_pts"} <= set(betting.columns):
        return games_with_odds.iloc[0:0]
    scores = betting.rename(columns={"home_pts": "odds_home_pts", "away_pts": "odds_away_pts"})
    scores = scores[["home", "away", "odds_home_pts", "odds_away_pts"]]
    checked = games_with_odds.dropna(subset=["spread_close"]).merge(
        scores, left_on=["home", "away", "home_pts", "away_pts"],
        right_on=["home", "away", "odds_home_pts", "odds_away_pts"], how="left")
    wrong = checked[checked["odds_home_pts"].isna()]
    return wrong[["game_id", "date", "home", "away", "home_pts", "away_pts"]].drop_duplicates("game_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach the Kaggle odds files to games.csv and write odds.csv.")
    parser.add_argument("--raw-dir", type=Path, default=ODDS_RAW_DIR, help="folder with the Kaggle CSVs")
    parser.add_argument("--games", type=Path, default=GAMES_FILE, help="cleaned games table")
    parser.add_argument("--output", type=Path, default=ODDS_FILE, help="where to write odds.csv")
    args = parser.parse_args(argv)

    try:
        if not args.games.exists():
            raise FileNotFoundError(f"{args.games} not found. Run basketball/src/clean.py first.")
        betting, moneylines = load_raw(args.raw_dir)
        games = pd.read_csv(args.games, dtype={"game_id": str})
        with_odds = attach_odds(games, betting, moneylines)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if moneylines is not None:
        shared = moneylines["game_id"].isin(games["game_id"]).sum()
        print(f"Moneyline file: {shared} of its {len(moneylines)} games are in games.csv")
        if not shared:
            print(f"  None match. Its ids look like {moneylines['game_id'].iloc[0]}; "
                  f"ours look like {games['game_id'].iloc[0]}.")

    has_odds = with_odds[ODDS_OUTPUT_COLUMNS].notna().any(axis=1)
    output = with_odds.loc[has_odds, ["game_id", "date", "home", "away"] + ODDS_OUTPUT_COLUMNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote odds for {len(output)} of {len(games)} games to {args.output}")

    print("\nGames with odds, per season:")
    print(market.coverage(with_odds).to_string())

    missing = market.games_without_spread(with_odds)
    if len(missing):
        print(f"\n{len(missing)} games have no closing spread. First 20:")
        print(missing.head(20).to_string(index=False))
    wrong = score_mismatches(with_odds, betting)
    if len(wrong):
        print(f"\n{len(wrong)} games have a different final score in the betting file. First 20:")
        print(wrong.head(20).to_string(index=False))
    if not len(missing) and not len(wrong):
        print("\nEvery game has a closing spread and every score agrees.")
    if with_odds["spread_close"].notna().sum() >= 2:
        print(f"\nSpread of real margins around the closing spread: "
              f"{market.estimate_margin_sd(with_odds):.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
