# TaBet

TaBet forecasts sports results and compares them with the betting market. The odds are a benchmark only; they are never a model input.

Website: https://ta-bet-ml-model-edition.vercel.app

## The website

Plain HTML, CSS and JavaScript in `site/`, deployed by Vercel from `main`. Each sport writes its own data files to `site/data/`.

| Page | What it shows |
| --- | --- |
| [Pitch](https://ta-bet-ml-model-edition.vercel.app/) (`index.html`) | The home page, with an arcade intro |
| [Matches](https://ta-bet-ml-model-edition.vercel.app/matches.html) | Every Premier League match since 2022-23: TaBet's forecasts next to the market, predicted vs actual lineups, live scores, and a minute-by-minute timeline of goals (with assists) and cards |
| [Notebook](https://ta-bet-ml-model-edition.vercel.app/notebook.html) | How the Premier League model works, in plain words |
| [Research](https://ta-bet-ml-model-edition.vercel.app/research.html) | The gap to the market over time and by season, calibration, promoted teams, the live test, and every pre-registered experiment with its 95% interval |
| [NFL](https://ta-bet-ml-model-edition.vercel.app/football.html) | NFL match explorer |
| [NBA](https://ta-bet-ml-model-edition.vercel.app/basketball.html) | NBA pregame forecasts next to the market |

## Folders

| Folder | What's in it |
| --- | --- |
| [`soccer (premier league)/`](<soccer (premier league)/>) | Premier League home/draw/away forecasts: the model (`tabet/`), data, results, the live forecast log, tests and docs. Start with its [README](<soccer (premier league)/README.md>). |
| [`Football/`](Football/) | NFL pregame Elo forecasts and moneyline backtests. Start with its [README](Football/README.md). |
| [`basketball/`](basketball/) | NBA pregame forecasts. |
| [`backend/`](backend/) | The API (Express on Vercel, `/api/*`) and its Supabase database, shared by every sport. See its [README](backend/README.md). |
| [`site/`](site/) | The shared website (above). |

Also at the top level: `api/index.mjs` (the Vercel entry point for the backend's Express app) and `vercel.json` (deploys `site/` and `/api`).

## What runs automatically

| Workflow (`.github/workflows/`) | When | What it does |
| --- | --- | --- |
| `prospective.yml` | every 30 minutes (cron-job.org; GitHub's schedule every 2 hours as a backup) | Premier League logger: saves forecasts 24–32 h before kickoff and again with confirmed lineups just before kickoff, archives market odds, refreshes the site's data, emails the weekly digest, and opens a GitHub issue if a match is missed. It commits to `main`, so **pull before you start work**. |
| `api-football.yml` | 06:17 and 22:47 UTC | Collects multi-league match data from API-Football into the private database (training only), and the Premier League match events shown on the Matches page. |
| `ci.yml` | every push and pull request | Tests for each project. |
| `delete-my-merged-branches.yml` | when a pull request merges | Deletes the merged branch, only for the repo owner's own branches. |

## Where data lives

- **In git:** the forecast logs (`soccer (premier league)/results/prospective/`, append-only), results, and the site's data files.
- **In Supabase (private):** raw odds, API-Football responses, transfer pages and other bulky or licensed data, in private storage buckets; a read copy of the season data for the API; the newsletter list. Keys live only in GitHub secrets and local `.env` files, never in git.

## The Premier League project in one paragraph

A Dixon-Coles team-strength model plus a second stage that adds opponent-adjusted xG and the starting XI. On 2024-26 the version with confirmed lineups trails closing odds by 0.008 log-loss (the market is hard to beat; the gap is about half that of a results-only model). Since gameweek 10 of 2026-27, forecasts are logged before kickoff and never changed, and every new idea is pre-registered before it is tested: so far 20 tests, one idea kept (score-driven dynamic ratings, which won on two separate test sets) and others still being tested live in a shadow log. Details, results and limitations are in the [soccer README](<soccer (premier league)/README.md>) and its `docs/`.

## Working on it

```sh
git pull                       # the logger commits to main
cd "soccer (premier league)"
python -m pytest -q
python -m tabet --help
```

Each sport's folder has its own README with setup steps. Who built what is in section 13 of the [soccer README](<soccer (premier league)/README.md>).
