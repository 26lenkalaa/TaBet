# TaBet: ML Model Edition

TaBet is a student project (SWEF Fall 2026, TAMID at Northeastern) that forecasts English Premier
League matches. For each match it gives three probabilities, **home win / draw / away win**, and it
checks how good those probabilities were once the match has been played.

**Live site:** https://ta-bet-ml-model-edition.vercel.app has the model notebook, plus a match
explorer with every forecast next to the betting market and the result.

## The idea in plain words

- **Probabilities, not picks.** "Arsenal 55%, draw 25%, Chelsea 20%" says more than "Arsenal will
  win", and it can be scored fairly even when the favourite loses.
- **Scored with log-loss.** Each forecast is scored on the probability it gave to what actually
  happened: −ln(that probability). Lower is better. Giving every outcome a one-in-three chance scores
  1.10. Football is mostly luck, so real models land around 0.95–1.00, and a gain of 0.003 matters.
- **The betting market is the yardstick.** Bookmakers' prices, with their margin removed, are the
  strongest public forecast. TaBet compares itself with them but **never uses odds as an input**, so
  it can't just copy the market. It doesn't use crowd signals either (FPL ownership, transfers or
  prices).
- **No cheating with hindsight.** A forecast may use only what was known before its cutoff. A result
  counts as known 2.5 hours after kickoff. Models are trained on the past and scored on later matches.
- **Tests are written down before they run** (`docs/PREREGISTRATION.md`), so a lucky result can't be
  dressed up after the fact.

## The models

| Name | What it knows | Log-loss, 2024-26 |
|---|---|---|
| Closing odds (benchmark) | the betting market at kickoff | 0.9893 |
| **V2x**, with lineups | results, chance quality (xG) and the confirmed starting XIs | 0.9973 |
| **V1x**, before team news | results and opponent-adjusted xG; the default early forecast | 1.0011 |
| **V0**, results only | Dixon-Coles team ratings from scores alone; the baseline | 1.0042 |

- **V0** rates every team's attack and defence from past scores, weighting recent matches more.
- **V1x** adds expected goals (xG): how good the chances a team created and allowed were, adjusted for
  the opponents.
- **V2x** adds the starting lineup: how strong tonight's XI is compared with the team's usual one.

None of them beats the market yet. Many other ideas were tested and dropped; the list is in
`docs/IMPROVEMENT_BRIEF.md`.

## What runs automatically

A GitHub Action (`.github/workflows/prospective.yml`) runs about every 30 minutes, triggered by
cron-job.org (setup: `docs/LOGGER_SETUP.md`), with GitHub's own schedule as a backup.

1. **Logs forecasts before kickoff.** About 24–26 hours before each match (or 26–32 hours as a
   fallback), it records TaBet's forecast with and without predicted lineups, from Fantasy Football
   Scout and RotoWire, in `results/prospective/log.jsonl`. The log is append-only, and records are
   never edited. This is the live test of whether predicted lineups help; the rules are in
   `docs/PROSPECTIVE_PROTOCOL.md`.
2. **Keeps the data fresh.** Once a day it refreshes this season's results from football-data.co.uk,
   and it saves the pre-match odds as they're published, for the market comparison only.
3. **Updates the website.** It rebuilds `site/data/matches.json`. Vercel redeploys the site from
   `main` automatically.
4. **Raises alerts.** If a match passes its logging window without a forecast, or the data goes stale,
   it opens a GitHub issue titled "Prospective logger alert".

The bot commits to `main`, so pull before you start work.

## What's in the repo

| Path | What it is |
|---|---|
| `tabet/` | the Python package: data loading, models, forecasting, logging and scoring |
| `res/` | football-data.co.uk season CSVs (results, match stats, odds), the raw input |
| `data/` | downloaded and processed data (`raw/` and `processed/` are not in git and are rebuilt on demand) |
| `results/` | outputs of experiments and the live forecast log (`results/prospective/`) |
| `prototypes/` | one script per experiment; each writes its numbers to `results/` |
| `site/` | the static website (plain HTML/CSS/JS, no build step) |
| `dashboard/` | an older standalone HTML forecast dashboard |
| `tests/` | the test suite (`python -m pytest -q`) |
| `docs/` | design notes, protocols and audits (below) |

**Docs:**
- `docs/SPEC.md`: system design.
- `docs/IMPROVEMENT_BRIEF.md`: the best overview of every experiment and result, written for
  outside reviewers.
- `docs/PREREGISTRATION.md`: tests committed before they were run.
- `docs/PROSPECTIVE_PROTOCOL.md`: the rules for live forecasts, with dated amendments and an
  operations log.
- `docs/LEAKAGE_AUDIT.md`: checks that no future information leaks into forecasts.
- `docs/SOURCE_AUDIT.md`: every data source tried, what it allows, and why some aren't used.
- `docs/LOGGER_SETUP.md`: how to set up the 30-minute trigger.

**Data sources:** football-data.co.uk (results, stats, odds), the Fantasy Premier League API and its
public archive (player minutes, xG, availability), and Fantasy Football Scout and RotoWire (predicted
lineups). Sites behind bot protection aren't used.

## Running it yourself

Needs Python 3.12 or newer.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                      # run the tests

python -m tabet validate                 # load res/*.csv and run data-quality checks
python -m tabet build                    # write data/processed/{matches,odds}.csv + manifest
python -m tabet history Arsenal --cutoff 2026-01-01T12:00Z --last 5   # a team's results before a date
python -m tabet forecast --gameweeks 4   # forecast upcoming fixtures -> results/forecasts/
python -m tabet log-forecasts --dry-run  # which fixtures the logger would record now (writes nothing)
python -m tabet score-forecasts          # score logged forecasts against lineups and results
python -m tabet heartbeat                # the logger's status: next fixture, misses, data freshness
python -m tabet site-data                # rebuild the website's match data
python dashboard/build.py out.html       # build the standalone dashboard from the latest forecast
```

`log-forecasts` without `--dry-run` writes real records. Leave that to the GitHub Action.

## House rules

- **Odds and crowd data are never model inputs.** A test enforces this.
- **Never edit or backfill** `results/prospective/log.jsonl` or `results/prospective/raw/`.
- **Bad source values are blanked, never guessed**, through `data/source_fixes.csv`.
- **New main tests are pre-registered**, by committing them before running them.
- **Work goes on a branch with a pull request to `main`.**
