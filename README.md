# TaBet: ML Model Edition

TaBet predicts the outcome of Premier League matches. For any upcoming fixture, it
estimates the probability of a **home win, draw, or away win**, and shows which factors
pushed the prediction one way or the other.

## Goals

- **Probabilistic predictions:** calibrated home/draw/away probabilities rather than
  a single "pick".
- **Explainable:** a readable breakdown of what drove each prediction, such as team
  strength, recent form, confirmed lineups and absences.
- **Honest evaluation:** predictions use only information available before kickoff,
  and accuracy is tracked against real results over time.
- **Keeps learning:** completed results feed back into updated features, scheduled
  retraining, and error analysis.
- **Handles squad changes:** sensible estimates for newly promoted teams and for
  players arriving from other leagues.

## Status

- **Data:** a validated pipeline over football-data.co.uk results (2021-22 to the current
  season) and FPL player appearances.
- **Model:**
  - Dixon-Coles team strength, plus opponent-adjusted xG form.
  - With confirmed lineups, an XI-strength adjustment, which is under pre-registered
    confirmation.
- **Live forecasts:**
  - Upcoming fixtures are forecast with and without predicted lineups.
  - A scheduled GitHub Action logs each fixture's forecast 24 hours before kickoff, for
    prospective scoring.

Design and evidence:
- `docs/SPEC.md`: system design.
- `docs/SOURCE_AUDIT.md`: data sources.
- `docs/LEAKAGE_AUDIT.md`: evaluation checks and paired results.
- `docs/PREREGISTRATION.md`: the confirmation test.
- `docs/PROSPECTIVE_PROTOCOL.md`: the predicted-lineup protocol.
- `docs/IMPROVEMENT_BRIEF.md`: a self-contained summary for outside review.

## Usage

```
pip install -r requirements.txt
python -m tabet validate                 # load res/*.csv and run data-quality checks
python -m tabet build                    # write data/processed/{matches,odds}.csv + manifest
python -m tabet history Arsenal --cutoff 2026-01-01T12:00Z --last 5
python -m tabet forecast --gameweeks 4   # forecast upcoming fixtures -> results/forecasts/
python -m tabet log-forecasts --dry-run  # which fixtures are due for the 24h prospective log
python -m tabet score-forecasts          # score logged forecasts against confirmed XIs and results
python dashboard/build.py out.html       # build the forecast dashboard from the latest forecast
python -m pytest -q
```

- **No look-ahead:** a result counts as known 2h30m after kickoff, and nothing later than a
  forecast's cutoff is used.
- **No guessed fixes:** bad source values are blanked (never guessed) through
  `data/source_fixes.csv`.
- **Odds are a benchmark only:** bookmaker odds are never a model input.
