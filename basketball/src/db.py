"""The SQLite database: one file holding teams, games, odds, team stats and predictions.

db.py loads the CSVs that clean.py and fetch_odds.py write into the database;
after that, predict.py and backtest.py read from the database, not the CSVs.

Run from the repo root, after clean.py and fetch_odds.py:

    python3 basketball/src/db.py

It rebuilds basketball/data/basketball.db from scratch every time, so it is safe to rerun.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (Boolean, Column, Engine, Float, ForeignKey, Integer, MetaData, String, Table,
                        create_engine, delete, func, select)
from sqlalchemy.pool import NullPool

BASKETBALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASKETBALL_DIR))

from src import market  # noqa: E402
from src.models import elo, ratings  # noqa: E402

DB_FILE = BASKETBALL_DIR / "data" / "basketball.db"
GAMES_FILE = BASKETBALL_DIR / "data" / "processed" / "games.csv"
ODDS_FILE = BASKETBALL_DIR / "data" / "processed" / "odds.csv"

ELO_VERSION = "elo-v1"
EFFICIENCY_VERSION = "eff-v2"
ODDS_SOURCE = "kaggle"

metadata = MetaData()

teams = Table(
    "teams", metadata,
    Column("id", Integer, primary_key=True),          # NBA.com's team id
    Column("abbreviation", String, nullable=False, unique=True),
    Column("name", String, nullable=False),
)

games = Table(
    "games", metadata,
    Column("id", String, primary_key=True),           # NBA.com's 10-digit game id
    Column("season", String, nullable=False, index=True),
    Column("date", String, nullable=False, index=True),   # YYYY-MM-DD
    Column("game_type", String, nullable=False),      # regular, play_in or playoffs
    Column("home_team_id", Integer, ForeignKey("teams.id"), nullable=False),
    Column("away_team_id", Integer, ForeignKey("teams.id"), nullable=False),
    Column("home_pts", Integer, nullable=False),
    Column("away_pts", Integer, nullable=False),
    Column("is_playoffs", Boolean, nullable=False),
)

odds = Table(
    "odds", metadata,
    Column("game_id", String, ForeignKey("games.id"), primary_key=True),
    Column("source", String, primary_key=True),
    Column("is_closing", Boolean, primary_key=True),
    Column("home_ml", Float),
    Column("away_ml", Float),
    Column("spread", Float),                          # the home team's line: negative = home favored
    Column("total", Float),
)

team_stats = Table(
    "team_stats", metadata,
    Column("game_id", String, ForeignKey("games.id"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.id"), primary_key=True),
    Column("possessions", Float),
    Column("off_rating", Float),                      # points scored per 100 possessions
    Column("def_rating", Float),                      # points allowed per 100 possessions
    Column("rest_days", Integer),                     # days since the team's previous game; empty for its first
)

predictions = Table(
    "predictions", metadata,
    Column("game_id", String, ForeignKey("games.id"), primary_key=True),
    Column("model_version", String, primary_key=True),
    Column("home_win_prob", Float, nullable=False),
    Column("predicted_margin", Float, nullable=False),
    Column("created_at", String, nullable=False),     # UTC time the prediction was stored
)


def connect(db_file: Path = DB_FILE, must_exist: bool = False) -> Engine:
    """Open the database file (created on first use unless must_exist is set)."""
    db_file = Path(db_file)
    if must_exist and not db_file.exists():
        raise FileNotFoundError(f"{db_file} not found. Run basketball/src/db.py first.")
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_file}", poolclass=NullPool)   # close the file after each use


def create_tables(engine: Engine) -> None:
    """Start from empty tables."""
    metadata.drop_all(engine)
    metadata.create_all(engine)


# --- loading -----------------------------------------------------------------

def team_table(games_csv: pd.DataFrame) -> pd.DataFrame:
    """One row per team, using each team's most recent abbreviation."""
    home = games_csv[["date", "home_team_id", "home"]].set_axis(["date", "id", "abbreviation"], axis=1)
    away = games_csv[["date", "away_team_id", "away"]].set_axis(["date", "id", "abbreviation"], axis=1)
    latest = pd.concat([home, away]).sort_values("date").drop_duplicates("id", keep="last")

    def full_name(abbreviation: str) -> str:
        names = market.TEAM_NAMES.get(abbreviation)
        return names[1] if names else abbreviation   # e.g. BOS -> Boston Celtics

    latest = latest.assign(name=latest["abbreviation"].map(full_name))
    return latest[["id", "abbreviation", "name"]].sort_values("abbreviation").reset_index(drop=True)


def team_stats_table(games_csv: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team): possessions, ratings per 100 possessions and rest days."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        frame = pd.DataFrame({
            "game_id": games_csv["game_id"],
            "date": games_csv["date"],
            "team_id": games_csv[f"{side}_team_id"],
            "pts": games_csv[f"{side}_pts"],
            "opp_pts": games_csv[f"{other}_pts"],
        })
        for name, prefix in (("possessions", side), ("opp_possessions", other)):
            needed = [f"{prefix}_{stat}" for stat in ("fga", "fta", "oreb", "tov")]
            if set(needed) <= set(games_csv.columns):
                fga, fta, oreb, tov = (games_csv[column] for column in needed)
                frame[name] = fga + 0.44 * fta - oreb + tov
            else:
                frame[name] = float("nan")
        sides.append(frame)
    stats = pd.concat(sides, ignore_index=True).sort_values(["team_id", "date", "game_id"])

    stats["off_rating"] = 100 * stats["pts"] / stats["possessions"]
    stats["def_rating"] = 100 * stats["opp_pts"] / stats["opp_possessions"]
    played = pd.to_datetime(stats["date"])
    stats["rest_days"] = played.groupby(stats["team_id"]).diff().dt.days.astype("Int64")
    columns = ["game_id", "team_id", "possessions", "off_rating", "def_rating", "rest_days"]
    return stats[columns].sort_values(["game_id", "team_id"]).reset_index(drop=True)


def load_games(engine: Engine, games_csv: pd.DataFrame) -> None:
    """Fill teams, games and team_stats from the table clean.py writes."""
    game_rows = games_csv.rename(columns={"game_id": "id"})[[column.name for column in games.columns]]
    with engine.begin() as connection:
        team_table(games_csv).to_sql("teams", connection, if_exists="append", index=False)
        game_rows.to_sql("games", connection, if_exists="append", index=False)
        team_stats_table(games_csv).to_sql("team_stats", connection, if_exists="append", index=False)


def load_odds(engine: Engine, odds_csv: pd.DataFrame, source: str = ODDS_SOURCE) -> int:
    """Fill odds from the table fetch_odds.py writes. Returns the number of games loaded."""
    with engine.connect() as connection:
        known = set(connection.execute(select(games.c.id)).scalars())
    rows = odds_csv[odds_csv["game_id"].isin(known)].drop_duplicates("game_id")
    table = pd.DataFrame({
        "game_id": rows["game_id"], "source": source, "is_closing": True,
        "home_ml": rows.get("home_ml"), "away_ml": rows.get("away_ml"),
        "spread": rows.get("spread_close"), "total": rows.get("total_close"),
    })
    with engine.begin() as connection:
        table.to_sql("odds", connection, if_exists="append", index=False)
    return len(table)


def save_predictions(engine: Engine, rated: pd.DataFrame, model_version: str) -> int:
    """Store one model's pre-game predictions, replacing its earlier ones."""
    table = rated[["game_id", "home_win_prob", "predicted_margin"]].assign(
        model_version=model_version,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    with engine.begin() as connection:
        connection.execute(delete(predictions).where(predictions.c.model_version == model_version))
        table.to_sql("predictions", connection, if_exists="append", index=False)
    return len(table)


def build(engine: Engine, games_csv: pd.DataFrame, odds_csv: pd.DataFrame | None = None) -> None:
    """Rebuild the whole database: teams, games, team stats, odds and both models' predictions."""
    create_tables(engine)
    load_games(engine, games_csv)
    if odds_csv is not None:
        load_odds(engine, odds_csv)
    played = read_games(engine, possessions=True)
    save_predictions(engine, elo.rate_games(played)[0], ELO_VERSION)
    save_predictions(engine, ratings.rate_games(played)[0], EFFICIENCY_VERSION)


# --- reading -----------------------------------------------------------------

def read_games(engine: Engine, possessions: bool = False, date: str | None = None) -> pd.DataFrame:
    """Every game in date order, in the shape the models expect (the same columns clean.py writes).

    With possessions=True, home_possessions and away_possessions are added from team_stats.
    With date="YYYY-MM-DD", only that day's games are returned.
    """
    home, away = teams.alias("home_team"), teams.alias("away_team")
    columns = [
        games.c.id.label("game_id"), games.c.season, games.c.date, games.c.game_type, games.c.is_playoffs,
        games.c.home_team_id, home.c.abbreviation.label("home"),
        games.c.away_team_id, away.c.abbreviation.label("away"),
        games.c.home_pts, games.c.away_pts,
    ]
    source = (games.join(home, games.c.home_team_id == home.c.id)
                   .join(away, games.c.away_team_id == away.c.id))
    if possessions:
        home_stats, away_stats = team_stats.alias("home_stats"), team_stats.alias("away_stats")
        columns += [home_stats.c.possessions.label("home_possessions"),
                    away_stats.c.possessions.label("away_possessions")]
        source = (source
                  .outerjoin(home_stats, (home_stats.c.game_id == games.c.id)
                             & (home_stats.c.team_id == games.c.home_team_id))
                  .outerjoin(away_stats, (away_stats.c.game_id == games.c.id)
                             & (away_stats.c.team_id == games.c.away_team_id)))
    query = select(*columns).select_from(source).order_by(games.c.date, games.c.id)
    if date is not None:
        query = query.where(games.c.date == date)
    with engine.connect() as connection:
        table = pd.read_sql(query, connection)
    table["is_playoffs"] = table["is_playoffs"].astype(bool)
    for column in ("home_possessions", "away_possessions"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").astype(float)   # empty -> NaN
    return table


def read_odds(engine: Engine, source: str = ODDS_SOURCE) -> pd.DataFrame:
    """Closing odds per game: game_id, spread_close, total_close, home_ml, away_ml."""
    query = (
        select(odds.c.game_id, odds.c.spread.label("spread_close"), odds.c.total.label("total_close"),
               odds.c.home_ml, odds.c.away_ml)
        .where(odds.c.source == source, odds.c.is_closing)
        .order_by(odds.c.game_id)
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def read_predictions(engine: Engine, model_version: str = ELO_VERSION) -> pd.DataFrame:
    query = (
        select(predictions.c.game_id, predictions.c.home_win_prob, predictions.c.predicted_margin,
               predictions.c.created_at)
        .where(predictions.c.model_version == model_version)
        .order_by(predictions.c.game_id)
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def row_counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {table.name: connection.execute(select(func.count()).select_from(table)).scalar_one()
                for table in metadata.sorted_tables}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load games.csv and odds.csv into the SQLite database.")
    parser.add_argument("--games", type=Path, default=GAMES_FILE, help="cleaned games table")
    parser.add_argument("--odds", type=Path, default=ODDS_FILE, help="odds table from fetch_odds.py")
    parser.add_argument("--db", type=Path, default=DB_FILE, help="database file to (re)build")
    args = parser.parse_args(argv)

    if not args.games.exists():
        print(f"Error: {args.games} not found. Run basketball/src/clean.py first.", file=sys.stderr)
        return 1
    games_csv = pd.read_csv(args.games, dtype={"game_id": str})
    odds_csv = pd.read_csv(args.odds, dtype={"game_id": str}) if args.odds.exists() else None
    if odds_csv is None:
        print(f"{args.odds} not found, so the odds table stays empty. "
              "Run basketball/src/ingest/fetch_odds.py to fill it.")

    engine = connect(args.db)
    build(engine, games_csv, odds_csv)

    print(f"Built {args.db}")
    for table, count in row_counts(engine).items():
        print(f"  {table:<12} {count:>6} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
