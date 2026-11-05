# TaBet

Premier League match-outcome analytics: home-win / draw / away-win probabilities
with a breakdown of the factors behind them. The design is in `docs/SPEC.md`.

Current stage: a clean, validated data pipeline over five seasons (2021-22 to 2025-26)
of football-data.co.uk match data in `res/`.

```
pip install -r requirements.txt
python -m tabet validate                       # load res/*.csv and run data-quality checks
python -m tabet build                          # validate, then write data/processed/{matches,odds}.csv + manifest
python -m tabet history Arsenal --cutoff 2026-01-01T12:00Z --last 5
python -m pytest -q
```

* A result counts as known 2h30m after kickoff, and `history` never returns anything later than the cutoff.
* Bad values in the source files are blanked (never guessed) through `data/source_fixes.csv`, with a reason for each.
* Bookmaker odds go in a separate table and are used only as an accuracy benchmark.
