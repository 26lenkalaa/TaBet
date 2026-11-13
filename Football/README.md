# TaBet NFL

This module forecasts NFL regular-season winners before kickoff. It uses results-only Elo and compares it with historical moneylines. It needs Python 3.10+ and no third-party packages.

## Run it

From the TaBet repo root, in Bash or Git Bash:

```sh
cd Football
mkdir -p res
curl -fL https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv -o res/games.csv
python cli.py predict KC BAL 2025-09-28 --season 2025
python cli.py backtest 2025
python -m unittest discover -s tests -v
```

Use nflverse team codes and an ISO date (`YYYY-MM-DD`). Home team comes first. Use `--neutral` for a neutral-site matchup. January games belong to the previous calendar year's NFL season. The forecast command accepts hypothetical matchups; it does not check the schedule. Its `history_through` field shows the newest result used. If you have a different local CSV, pass `--data PATH` before the subcommand.

## MVP levels

| Level | Files | What works |
| --- | --- | --- |
| 1. Data | `data.py` | Read and validate completed regular-season games; skip playoffs and unplayed games. |
| 2. Prediction | `model.py`, `cli.py` | Estimate home and away win probabilities from Elo and home advantage. |
| 3. Evaluation | `backtest.py` | Replay a season and compare against moneylines on the same games. |
| 4. UI handoff | CLI JSON and CSV | Use `sport`, `league`, `game_date`, `home_team`, `away_team`, and `outcome_probabilities` for a future shared UI. |

The UI can call `forecast()` or consume the JSON without knowing how Elo works. Keep NFL-specific code here. Add new model inputs in `model.py` and new CSV fields in `data.py` as needed. The soccer deployment still serves the existing soccer site.

## How it works

Teams start at 1500. The home team gets a 55-point adjustment unless the game is neutral. Ratings move after each result; between seasons, 25% of the distance from 1500 is removed. These are fixed starter settings, not tuned values. The backtest predicts a whole date before using any of its results, so neither the target game's result nor earlier games on the same date can leak into a prediction.

Elo updates a tie as half a win. The forecast shows two probabilities **conditional on no tie**. The evaluation excludes ties and games without valid prices from **both** scores and keeps them in `results/backtest.csv` with a reason. The moneylines are converted from American odds and normalized to sum to one. They never enter Elo. Lower binary Brier score is better; 0.25 is the score of a constant 50/50 forecast. The CSV source is [nflverse/nfldata](https://github.com/nflverse/nfldata). Audit odds coverage and timestamps before making claims about a specific bookmaker or forecast cutoff.

With the nflverse snapshot used during development, the 2025 regular-season run scored 271 of 272 games: Elo **0.2295**, moneyline **0.2121**. The remaining game was a tie. This is one retrospective run, not evidence of an edge. The source is updated over time, so keep the exact CSV used when comparing experiments.

Next: evaluate a second untouched season, then test one improvement at a time. Add spreads, exact scores, or player data only after the winner baseline is stable.
