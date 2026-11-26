# Where TaBet's data lives

A map for reviewing the system design: each dataset, where it is stored, who can read it, and what uses
it. Odds are a benchmark only and never a model input. `tests/test_odds_archive.py` enforces that.

## Supabase (project TaBet)

Two schemas (migration 0007): **`public`** holds only what the site's API reads; **`research`** holds
private research data, reachable only with the secret key. `research.leagues` links the three id
systems in use (football-data codes such as `E0`, API-Football ids such as `39`, the site's
`premier-league`), and API-Football's integer seasons carry a `season_label` such as `2024-25`.
Every table has a description in the dashboard.

| What | Where | Access | Written by | Used by |
|---|---|---|---|---|
| Matches, forecasts, market odds, lineups (read copy of the site data) | tables `sports`, `competitions`, `teams`, `matches`, `forecasts`, `market_odds`, `lineups`, `sync_runs` (migrations 0001, 0002, 0005) | public read, secret-key write | logger workflow, `backend/scripts/sync-soccer.mjs` | the site's API (`backend/`) |
| API-Football raw responses (≈100,000 matches, 2015–26; cups, Europe, sealed holdout) | bucket `api-football` (private) + tables `research.apif_lists`, `apif_fixtures`, `apif_blocked`, `apif_requests` (0003, moved in 0007) | secret key only | `tabet/api_football.py` (Actions) | `prototypes/run_*.py` experiments |
| Wikipedia transfer lists (English clubs, every window from summer 2005; CC BY-SA) | bucket `football-data`, `wikipedia/transfers/<window>.json.gz` (page + revision id) and `transfers.parquet` (parsed); `results/prospective/transfers_state.json` (last build) | secret key only | `tabet/transfers.py`, refreshed by the logger (new FPL players, or daily while a window is open) | research only; no forecast reads it |
| Experiment caches (parsed tables, Dixon-Coles rates, side-rows) | bucket `api-football`, paths `derived/*.parquet` | secret key only | the experiment scripts | the same scripts |
| League lookup (football-data code, API-Football id, site id, country, tier) | table `research.leagues` (0007) | secret key only | the migration | joins across the tables above |
| Raw Odds API responses and per-bookmaker prices | bucket `odds-archive` (private) (0004) | secret key only | the logger (`tabet/odds_archive.py`) | the paper-betting tracker (one Pinnacle price per bet) |
| Injury lists (19 leagues, 2020–26; post-match, so unusable as an input) | bucket `api-football`, paths `injuries/*.json.gz` | secret key only | `tabet/api_football.py injuries` | the 2026-11-19 timing check only |
| **Managerial changes** (10 leagues, 2015–27) | table `research.manager_changes` (0006, moved in 0007) | secret key only | `python -m tabet.managers` | addendum MGR (withdrawn) |

## The repository (public)

| What | Where | Notes |
|---|---|---|
| The forecast log (the live test's record) | `results/prospective/log.jsonl` | Append-only; never edited by hand |
| Raw lineup pages saved with each forecast | `results/prospective/raw/` | Never modified |
| Lineup snapshots | `results/lineup_snapshots/` | Predicted XIs over time |
| Derived odds summaries (margin-free probabilities only) | `results/prospective/odds/` | Raw prices moved to `odds-archive` on 2026-11-16. Earlier raw files (10–16 Nov) remain in git history: accepted by the project owner on 2026-11-22, no history rewrite |
| football-data results and odds | `res/` | Public source |
| Site data | `site/data/` (`index.json`, `matches-*.json`, `lineups-*.json`, `paper-bets*.json`) | Rebuilt by the logger |
| Experiment results | `results/<experiment>/` | Frozen `dev.json` before each test; `results.json` after |
| Data ledger | `docs/data_ledger.csv` | Which league-seasons each experiment may use |

## Credits

- **Managerial changes:** from Wikipedia's season articles (for example "2023–24 Premier League",
  section "Managerial changes"), retrieved through the MediaWiki API. Wikipedia text is licensed CC BY-SA
  4.0; the table is a factual extract with each row naming its source article.
- Other sources and their uses are listed on the site's notebook page ("Where the data comes from").
