# TaBet Rugby

International men's rugby union forecasts: win, draw and loss chances and a predicted score, compared
with the betting market where odds exist. Work in progress.

## Data

[API-Rugby](https://api-sports.io/sports/rugby) from API-Sports, free plan (100 requests a day). The
key is the repo secret `RUGBY_API_KEY`, used only for `v1.rugby.api-sports.io`, so rugby never shares
a quota with the other sports. Odds are a benchmark only; they never go into the model.

## Run it

GitHub: **Actions > Rugby data > Run workflow**. The result appears on the run's summary page.

| Mode | Requests | What it does |
| --- | --- | --- |
| `probe` | about 15 | Plan and usage, the international competitions and their seasons, which seasons the free key can read, and whether odds come back for an upcoming game. |

Locally (needs the key in your shell, never in a file): `cd rugby && python fetch.py probe`.
