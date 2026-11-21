"""Turn betting odds into probabilities and attach them to games.

Two jobs:

1. Conversions. A moneyline or a point spread becomes a win probability, with
   the bookmaker's margin removed, so the market can be graded on the same
   scale as the model.
2. Matching. An odds table (one row per game: date, home, away, spread, total)
   is joined to the games table that clean.py writes.

Spreads are always the HOME team's line, the way sportsbooks quote it:
-5.5 means the home team is favored by 5.5 points, +3 means it is a 3-point underdog.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import norm

MARGIN_SD = 12.0   # typical gap, in points, between the closing spread and the real margin


# --- conversions -------------------------------------------------------------

def moneyline_to_prob(moneyline: float) -> float:
    """American odds -> implied probability, bookmaker's margin still included.

    -150 (bet 150 to win 100) -> 0.60;  +130 (bet 100 to win 130) -> 0.435.
    """
    if not math.isfinite(moneyline) or abs(moneyline) < 100:
        raise ValueError(f"American odds must be +100 or more, or -100 or less; got {moneyline}")
    if moneyline < 0:
        return -moneyline / (-moneyline + 100.0)
    return 100.0 / (moneyline + 100.0)


def decimal_to_moneyline(decimal_odds: float) -> float:
    """Decimal odds -> American odds. 2.50 -> +150;  1.40 -> -250."""
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be above 1, got {decimal_odds}")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def prob_to_moneyline(prob: float) -> float:
    """Probability -> the American odds that would pay fairly for it."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def remove_margin(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Scale two implied probabilities so they add up to 1.

    A bookmaker's two prices add up to more than 1 (about 1.045); the extra is its margin.
    """
    total = home_prob + away_prob
    if total <= 0:
        raise ValueError("Probabilities must add up to more than 0")
    return home_prob / total, away_prob / total


def moneylines_to_probs(home_ml: float, away_ml: float) -> tuple[float, float]:
    """Both teams' American odds -> margin-free (home, away) win probabilities."""
    return remove_margin(moneyline_to_prob(home_ml), moneyline_to_prob(away_ml))


def spread_to_prob(home_spread, margin_sd: float = MARGIN_SD):
    """Home spread -> home win probability.

    The spread is the market's predicted margin. Real margins scatter around it
    like a bell curve about `margin_sd` points wide, so the chance the home team
    wins is the share of that curve above zero. Works on one number or a column.
    """
    if margin_sd <= 0:
        raise ValueError("margin_sd must be positive")
    return norm.cdf(-pd.to_numeric(home_spread) / margin_sd)


def spread_to_margin(home_spread):
    """Home spread -> the market's predicted home margin (-5.5 -> home by 5.5)."""
    return -pd.to_numeric(home_spread)


def estimate_margin_sd(games_with_odds: pd.DataFrame, spread_column: str = "spread_close") -> float:
    """How far real margins land from the closing spread, measured on your own data."""
    priced = games_with_odds.dropna(subset=[spread_column])
    if len(priced) < 2:
        raise ValueError("Need at least two games with a spread")
    miss = (priced["home_pts"] - priced["away_pts"]) + priced[spread_column]
    return float((miss ** 2).mean() ** 0.5)


# --- team names --------------------------------------------------------------

# Odds files rarely use NBA.com's abbreviations. Every spelling here maps to the NBA.com one.
TEAM_NAMES = {
    "ATL": ["Atlanta", "Atlanta Hawks", "Hawks"],
    "BOS": ["Boston", "Boston Celtics", "Celtics"],
    "BKN": ["Brooklyn", "Brooklyn Nets", "Nets", "BRK", "BK", "New Jersey", "New Jersey Nets", "NJN", "NJ"],
    "CHA": ["Charlotte", "Charlotte Hornets", "Hornets", "Charlotte Bobcats", "Bobcats", "CHO", "CHH"],
    "CHI": ["Chicago", "Chicago Bulls", "Bulls"],
    "CLE": ["Cleveland", "Cleveland Cavaliers", "Cavaliers", "Cavs"],
    "DAL": ["Dallas", "Dallas Mavericks", "Mavericks", "Mavs"],
    "DEN": ["Denver", "Denver Nuggets", "Nuggets"],
    "DET": ["Detroit", "Detroit Pistons", "Pistons"],
    "GSW": ["Golden State", "Golden State Warriors", "Warriors", "GS"],
    "HOU": ["Houston", "Houston Rockets", "Rockets"],
    "IND": ["Indiana", "Indiana Pacers", "Pacers"],
    "LAC": ["LA Clippers", "Los Angeles Clippers", "Clippers"],
    "LAL": ["LA Lakers", "Los Angeles Lakers", "Lakers"],
    "MEM": ["Memphis", "Memphis Grizzlies", "Grizzlies"],
    "MIA": ["Miami", "Miami Heat", "Heat"],
    "MIL": ["Milwaukee", "Milwaukee Bucks", "Bucks"],
    "MIN": ["Minnesota", "Minnesota Timberwolves", "Timberwolves", "Wolves"],
    "NOP": ["New Orleans", "New Orleans Pelicans", "Pelicans", "New Orleans Hornets", "NO", "NOH", "NOK"],
    "NYK": ["New York", "New York Knicks", "Knicks", "NY"],
    "OKC": ["Oklahoma City", "Oklahoma City Thunder", "Thunder", "Seattle", "Seattle SuperSonics", "SEA"],
    "ORL": ["Orlando", "Orlando Magic", "Magic"],
    "PHI": ["Philadelphia", "Philadelphia 76ers", "76ers", "Sixers"],
    "PHX": ["Phoenix", "Phoenix Suns", "Suns", "PHO"],
    "POR": ["Portland", "Portland Trail Blazers", "Trail Blazers", "Blazers"],
    "SAC": ["Sacramento", "Sacramento Kings", "Kings"],
    "SAS": ["San Antonio", "San Antonio Spurs", "Spurs", "SA"],
    "TOR": ["Toronto", "Toronto Raptors", "Raptors"],
    "UTA": ["Utah", "Utah Jazz", "Jazz"],
    "WAS": ["Washington", "Washington Wizards", "Wizards", "WSH"],
}


def _key(name: str) -> str:
    """'L.A. Clippers' and 'LAClippers' both become 'laclippers'."""
    return "".join(character for character in str(name).lower() if character.isalnum())


_LOOKUP = {_key(spelling): abbreviation
           for abbreviation, spellings in TEAM_NAMES.items()
           for spelling in [abbreviation, *spellings]}


def team_abbreviation(name: str) -> str:
    """Any common spelling of a team -> NBA.com's abbreviation ('Golden State' -> 'GSW')."""
    try:
        return _LOOKUP[_key(name)]
    except KeyError:
        raise ValueError(f"Unknown team name: {name!r}") from None


# --- matching odds to games --------------------------------------------------

ODDS_COLUMNS = ["spread_open", "spread_close", "total_open", "total_close", "home_ml", "away_ml"]


def standardize_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Tidy an odds table: dates as YYYY-MM-DD, team names as NBA.com abbreviations.

    `odds` needs the columns date, home, away and spread_close (the home team's
    line). spread_open, total_open, total_close, home_ml and away_ml are kept if present.
    """
    missing = [column for column in ["date", "home", "away", "spread_close"] if column not in odds.columns]
    if missing:
        raise ValueError(f"Odds table is missing columns: {', '.join(missing)}")
    tidy = pd.DataFrame({
        "date": pd.to_datetime(odds["date"], format="mixed").dt.strftime("%Y-%m-%d"),
        "home": odds["home"].map(team_abbreviation),
        "away": odds["away"].map(team_abbreviation),
    })
    for column in ODDS_COLUMNS:
        if column in odds.columns:
            tidy[column] = pd.to_numeric(odds[column], errors="coerce").to_numpy()
    repeated = tidy.duplicated(subset=["date", "home", "away"], keep=False)
    if repeated.any():
        example = tidy[repeated].iloc[0]
        raise ValueError(f"Odds table lists a game more than once, e.g. {example['away']} at "
                         f"{example['home']} on {example['date']}")
    return tidy


def match_odds(games: pd.DataFrame, odds: pd.DataFrame, margin_sd: float = MARGIN_SD) -> pd.DataFrame:
    """Attach odds to games by date, home team and away team.

    Returns `games` with the odds columns added, plus the market's predictions:
    market_margin (from the closing spread) and market_home_win_prob (from the
    moneylines when both are present, otherwise from the closing spread).
    Games with no odds keep empty values. Some sources date a late game by the
    next or previous day, so a game with no odds on its own date takes odds for
    the same matchup one day either side.
    """
    tidy = standardize_odds(odds)
    value_columns = [column for column in ODDS_COLUMNS if column in tidy.columns]
    keys = ["date", "home", "away"]

    matched = games.merge(tidy, on=keys, how="left", indicator="_found")
    used = set(map(tuple, matched.loc[matched["_found"] == "both", keys].to_numpy()))
    spare = tidy[[tuple(row) not in used for row in tidy[keys].to_numpy()]]

    for shift_days in (1, -1):
        missing = matched["_found"] == "left_only"
        if not missing.any() or spare.empty:
            break
        shifted = spare.assign(
            date=(pd.to_datetime(spare["date"]) - pd.Timedelta(days=shift_days)).dt.strftime("%Y-%m-%d"))
        found = matched.loc[missing, keys].reset_index().merge(shifted, on=keys, how="inner").set_index("index")
        matched.loc[found.index, value_columns] = found[value_columns]
        matched.loc[found.index, "_found"] = "both"
        taken = pd.to_datetime(found["date"]) + pd.Timedelta(days=shift_days)
        taken_keys = set(zip(taken.dt.strftime("%Y-%m-%d"), found["home"], found["away"]))
        spare = spare[[tuple(row) not in taken_keys for row in spare[keys].to_numpy()]]

    return add_market_predictions(matched.drop(columns="_found"), margin_sd)


def add_market_predictions(games_with_odds: pd.DataFrame, margin_sd: float = MARGIN_SD) -> pd.DataFrame:
    """Add market_margin and market_home_win_prob to games that already carry odds columns.

    The margin comes from the closing spread. The win probability comes from the
    two moneylines when a game has both, otherwise from the closing spread.
    """
    table = games_with_odds.copy()
    if "spread_close" not in table.columns:
        table["spread_close"] = float("nan")
    table["market_margin"] = spread_to_margin(table["spread_close"])
    table["market_home_win_prob"] = spread_to_prob(table["spread_close"], margin_sd)
    if {"home_ml", "away_ml"} <= set(table.columns):
        both = table["home_ml"].notna() & table["away_ml"].notna()
        table.loc[both, "market_home_win_prob"] = [
            moneylines_to_probs(home_ml, away_ml)[0]
            for home_ml, away_ml in zip(table.loc[both, "home_ml"], table.loc[both, "away_ml"])
        ]
    return table


def coverage(games_with_odds: pd.DataFrame) -> pd.DataFrame:
    """Per season: how many games have a closing spread, and how many do not.

    When the table has moneylines, a with_moneyline column counts the games that have both.
    """
    has_spread = games_with_odds["spread_close"].notna()
    table = games_with_odds.assign(with_spread=has_spread, without_spread=~has_spread)
    columns = ["with_spread", "without_spread"]
    if {"home_ml", "away_ml"} <= set(table.columns):
        table["with_moneyline"] = table["home_ml"].notna() & table["away_ml"].notna()
        columns.append("with_moneyline")
    table = table.groupby("season")[columns].sum()
    table.insert(0, "games", table["with_spread"] + table["without_spread"])
    return table


def games_without_spread(games_with_odds: pd.DataFrame) -> pd.DataFrame:
    """The games the backtest cannot grade against the market."""
    columns = ["game_id", "season", "date", "home", "away"]
    return games_with_odds.loc[games_with_odds["spread_close"].isna(), columns].reset_index(drop=True)
