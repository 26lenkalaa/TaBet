# Data source audit and first player/style experiments

Audit date: 2026-11-05. Each claim is labelled:
**[retrieved]** a sample was downloaded and validated here; **[documented]** stated by the
provider or library, not verified by us; **[unverified]** a possibility we did not test.
A source counts as *integrated* only if it is [retrieved] and wired into `tabet/`.

## 1. Summary

| Source | Access from here | 2021–26 dev | 2026–27 eval | Status |
|---|---|---|---|---|
| football-data.co.uk CSVs | plain HTTP (terms noted in 2.0) | yes (PL) | yes, 90 matches + team xG | **integrated** |
| FPL archive + live FPL API | plain HTTP / GitHub | yes (player × match) | yes, live | **integrated** (`tabet/fpl.py`) |
| Wyscout public dataset | figshare, CC BY 4.0 | no (2017-18 only) | no | prototype only (`prototypes/wyscout_prototype.py`) |
| StatsBomb Open Data | GitHub | no (full leagues 2015-16 only) | no | sampled, not used |
| WhoScored via soccerdata | blocked (Cloudflare 403) | — | — | not used |
| Sofascore | blocked (API 403) | — | — | not used |
| Understat, FBref (earlier probes) | blocked (maintenance page / Cloudflare challenge) | — | — | not used |

**Recommended minimal combination:** football-data.co.uk (results, team stats, team xG from
2026-27, benchmark odds) plus FPL (player minutes, starts, xG, xA, xGC, ICT, current injury
news). Keep Wyscout 2017-18 for cross-league prototyping only. None of the richer recent sources
(WhoScored, Sofascore, FBref, Understat) is accessible without circumventing bot protection,
which we will not do.

## 2. Per-source findings

### 2.0 football-data.co.uk (results, match stats, odds)
- **Terms [retrieved, data.php, 2026-11-05]:** the data is "intended for private individuals only,
  NOT commerical or data training products using automated bots/scrapers/AI". A student project
  that trains a model on the CSVs, with an AI assistant downloading them, may fall under this.
  The project owner has reviewed this and decided to proceed (2026-11-05).
- **Odds timing [retrieved, notes.txt]:** pre-closing odds are "collected Friday afternoons for
  weekend games, and on Tuesday afternoons for midweek games". The `C` columns are closing odds.
  A confirmed-lineup forecast therefore has *later* information than the pre-closing benchmark,
  and closing odds have later information than our forecast.
- **Benchmark definition:** `p_k = (1/o_k) / sum_j (1/o_j)` applied to the **average odds** columns
  (`AvgH/D/A`, `AvgCH/D/A`). That is not the same as averaging each bookmaker's normalised
  probabilities.
- **Not confirmed:** the claim that "Pinnacle odds have been stale since July 2025 and are
  excluded from market averages". We found no such statement on data.php or in notes.txt. Our
  benchmarks use only the market averages, not the Pinnacle columns.

### 2.1 WhoScored (via `soccerdata` 2026.43.0)
- **[documented]** `soccerdata.WhoScored` exposes `read_leagues`, `read_seasons`,
  `read_season_stages`, `read_schedule`, `read_missing_players` and `read_events`.
- **[retrieved: failure]** A plain GET to `https://www.whoscored.com/` returned **HTTP 403,
  "Attention Required! | Cloudflare"**.
- **Source inspection:** the reader uses `seleniumbase` with `uc=True` (undetected-Chrome
  mode). Its docstring suggests running non-headless "to avoid getting blocked", and it detects
  "Incapsula incident ID" IP blocks. Running it would mean bot-detection evasion, so it was **not
  run**. Leagues, seasons, fields and identifiers are therefore all **[unverified]**.

### 2.2 Sofascore
- **[retrieved: failure]** `api.sofascore.com/api/v1/unique-tournament/17/seasons` and the
  `www.sofascore.com/api/v1/...` equivalent both returned **HTTP 403 `{"error":{"code":403,"reason":"Forbidden"}}`**.
- **[documented]** `soccerdata.Sofascore` offers only `read_leagues`, `read_seasons`,
  `read_league_table` and `read_schedule`, with no player or team statistics. It sends requests
  through `tls_requests` (browser TLS fingerprinting) with proxy rotation, which is circumvention,
  so it was not used.
- **No independent working method found.** Player/team stat endpoints are **[unverified]**.

### 2.3 Wyscout public dataset (Pappalardo et al., 2019)
- **Access [retrieved]:** figshare collection 4415000, CC BY 4.0. We downloaded competitions,
  matches (0.6 MB), events (77 MB zip), players (1.7 MB) and teams.
- **Coverage [retrieved]:** England, Spain, Italy and France with 380 matches each, Germany with
  306, all **2017-18 only** (2017-08-04 → 2018-05-20). Also World Cup 2018 and Euro 2016.
- **Granularity:** event level (England: 643,150 events with x/y positions, tags, sub-events),
  plus match lineups, benches and substitutions.
- **Identifiers:** `wyId` for matches, players and teams, plus `competitionId` and `seasonId`.
  There is no mapping to FPL or football-data.
- **Missingness:** coded as values, not nulls. 7.2–7.5% of events have `playerId=0`, 2.1% of
  players have `height=0` and 1.4% have no foot. `substitutions` is sometimes the string `'null'`.
- **No xG.** Metrics have to be built from events.
- **2021–27 support: none.** Useful only for method prototypes.

### 2.4 StatsBomb Open Data
- **Access [retrieved]:** `github.com/statsbomb/open-data`. Terms: published work must credit
  StatsBomb and use its logo.
- **Full-league coverage [retrieved]:** 2015-16 only, for Premier League (380), La Liga (380),
  Serie A (380) and Ligue 1 (377). Premier League also has 2003-04.
- **Recent seasons are single-club [retrieved]:** Bundesliga 2023-24 (Leverkusen, 34 matches),
  Ligue 1 2021-22 and 2022-23 (PSG, 26 and 32), La Liga 2020-21 (Barcelona, 35).
- **[documented]** event-level data with StatsBomb xG; 360 freeze-frames for some competitions.
- **2021–27 support: no.** Fine for event-processing experiments; not a recent player history.

### 2.5 FPL (archive + live API)
- **Access [retrieved]:** the archive `raw.githubusercontent.com/vaastav/Fantasy-Premier-League`.
  Its code is MIT-licensed, but the LICENSE file adds that "the data provided is property of
  fantasy.premierleague.com and understat.com", so the MIT licence does not cover data-use rights. and the live endpoints
  `fantasy.premierleague.com/api/{bootstrap-static,fixtures,event/<gw>/live,element-summary/<id>}`.
- **Coverage [retrieved]:** Premier League 2021-22 → 2025-26 complete (380/380 fixtures each),
  plus 2026-27 gameweeks 1–9 (90 fixtures; the archive itself only has 60).
- **Granularity:** player × match, with 10.5k–11.6k appearance rows per season (59,019 in total).
- **Identifiers:** `element` is per season; `code` stays stable across seasons (481 players
  appear in 3+ seasons); fixture ids are per season. All 1,990 finished fixtures join to our
  `match_id` with **identical scores**.
- **Fields [retrieved]:**
  - every season: minutes, goals, assists, threat, creativity, influence
  - from 2022-23: starts, xG, xA, xGC
  - from 2025-26: tackles, recoveries, defensive contributions
  - live only: injury `news` and `chance_of_playing`
- **Missingness and fixes:**
  - `starts` and xG/xA/xGC were coded 0 before 12 Nov 2022. We null the xG and infer starters as
    the 11 players with most minutes, flagged `started_inferred`. Checked on 2023-26, the rule
    gets 96.6% of players and 78% of full lineups right.
  - One player is duplicated under two element ids in 2025-26; we deduplicate.
  - The live API's `team` is a player's *current* club, so we take each player's side from the
    fixture's own bonus-point list instead.
  - Two 2021-22 match sides have small source gaps.
- **No historical injury timeline** (news only reflects the present). Only Premier League players.

## 3. Experiments on identical chronological fixtures

The protocol is in `tabet/experiments.py`; run it with `python prototypes/run_fpl_experiments.py OUT_DIR`.
- **Stage 1:** Dixon-Coles refitted every matchday.
- **Stage 2:** a penalised Poisson adjustment fitted only on earlier matchdays. It stays off
  until 300 training matches exist, under the same rule for every variant.
- **Tuning:** each variant's L2 is chosen on 2022-23 and 2023-24. 2024-25 and 2025-26 are
  scored once; 2026-27 (90 matches) is a forward check.
- **Lineups:** features assume **confirmed lineups**.

Δ = log-loss minus V0 (Dixon-Coles), paired bootstrap 95% CI over matches.

| Variant | Dev 22-24 (760) | Test 24-26 (760) | 2026-27 (90) |
|---|---|---|---|
| V0 Dixon-Coles (log-loss) | 0.9624 | 1.0042 | 1.0362 |
| V1 + team xG form | −0.0011 [−0.0032, +0.0008] | −0.0022 [−0.0050, +0.0006] | −0.0002 [−0.0038, +0.0036] |
| V2 + XI strength, position prior | −0.0009 [−0.0031, +0.0013] | **−0.0062 [−0.0107, −0.0020]** | −0.0047 [−0.0164, +0.0078] |
| V3 + XI strength, similar-player prior | −0.0011 [−0.0036, +0.0011] | **−0.0060 [−0.0106, −0.0014]** | −0.0083 [−0.0201, +0.0040] |
| V4 = V2 + style main effects | −0.0018 [−0.0082, +0.0040] | −0.0039 [−0.0102, +0.0022] | +0.0027 [−0.0113, +0.0189] |
| V5 = V4 + style interactions | −0.0021 [−0.0092, +0.0046] | −0.0057 [−0.0137, +0.0022] | +0.0094 [−0.0106, +0.0306] |
| Market, pre-match avg (log-loss, benchmark) | 0.9375 | 0.9929 | 1.0199 |

Key contrasts on the test seasons:
- **XI strength over xG form (V2 − V1):** −0.0040 [−0.0073, −0.0006]
- **Similar-player vs position prior (V3 − V2):** +0.0002 [−0.0011, +0.0016]
- **Style main effects (V4 − V2):** +0.0024 [−0.0011, +0.0059]
- **Style interactions (V5 − V4):** −0.0018 [−0.0057, +0.0024]

Player-level check of the similarity prior: starters with under 600 decayed xG-era minutes,
predicting (xG+xA)/90 in the match, minutes-weighted MSE on identical player-matches.

| Split | n | Position prior | Similar-player prior | Difference (95% CI) |
|---|---|---|---|---|
| Dev | 4,003 | 0.0698 | 0.0692 | [−0.0014, +0.0001] |
| Test | 1,813 | 0.0590 | 0.0569 | [−0.0041, +0.0001] |

Wyscout 2017-18 prototype (5 leagues, gameweek 20 onwards):
- **Style main effects:** −0.0077 [−0.0184, +0.0031].
- **Interactions:** add nothing beyond the main effects.
- **England-only coefficients:** worse than pooled ones.
- **Similar-player priors:** no better than shrinking toward the role mean, except a small gain
  for goals+assists with same-league neighbours.

## 4. Interpretation

- **XI strength (confirmed lineups) is the only feature group with a significant test gain.**
  It improves log-loss by about 0.006 and narrows the gap to the market from 0.011 to 0.005.
  The caveat: it showed **no gain on the dev seasons**, and the test CI is one of several
  contrasts examined. Treat it as promising rather than established, and confirm it on 2026-27
  once more matches are played.
  - Part of the weak dev result may come from the data. Stage 2 only switches on late in 2022-23,
    and early 2022-23 lineups are inferred.
- **The similar-player prior does not beat the position prior at match level** (dev and test
  differences are both about 0). At player level it is slightly better on both splits, but each
  CI just reaches zero. It is not worth the complexity yet.
- **Team-style interactions do not help.** Neither the Premier League data nor the Wyscout
  prototype shows significant gains, and on 2026-27 style variants are slightly worse
  (not significant). The style features available here are coarse (shots, corners, fouls,
  xG/shot). Richer style data, such as possession, passing and pressing for 2021-27, is exactly
  what the blocked sources hold.
- **Sample size still limits every conclusion.** Most CIs are about ±0.004–0.008 wide.

## 5. Next steps

1. Use V2 (Dixon-Coles + xG form + XI strength, position prior) as the working model and keep
   V0 as the fallback. Re-check on 2026-27 as matches accumulate.
2. Evaluate **projected-lineup mode** (XI predicted before team news), which is what a
   pre-lineup prediction would really use.
3. Increase power: add 2016-17 → 2020-21 (FPL ICT and minutes exist; no xG), and score goal
   deviance alongside outcome log-loss.
4. Revisit style and cross-league similarity only if an accessible source with 2021-27
   event/possession data becomes available, for example through a provider API key or a licence.
