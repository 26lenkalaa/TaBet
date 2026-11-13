# TaBet — System Specification (Planning Draft)

Status: planning deliverable, written before implementation. Nothing in here is model output.
The first thing to build is the clean, validated data pipeline (§9). Modelling comes after it.

---

## 1. What the finished system does

You give it one upcoming Premier League fixture. It returns home-win, draw and away-win
probabilities, along with a breakdown of what pushed those numbers up or down. Once the match
has been played, the result goes back into the data. That updates the team and player features,
gets logged for error analysis, and is used at the next scheduled retrain.

```
 fixture request ──► data store (point-in-time) ──► feature builder ──► model ──► probabilities
        ▲                     ▲                                             │        + factor table
        │                     │                                             ▼
        │              ingest completed result ◄──── match played ◄──── prediction log
        │                     │
        └──── scheduled retrain + error report ◄──┘
```

## 2. Inputs supplied for an upcoming fixture

| Field | Required | Example | Notes |
|---|---|---|---|
| `home_team`, `away_team` | yes | `Arsenal`, `Fulham` | Names must match canonical team IDs, via an alias table (e.g. `Nott'm Forest`). |
| `kickoff` (UTC) | yes | `2026-11-22T15:00Z` | Sets the **prediction cutoff**. Data timestamped after this is never used. |
| `cutoff` override | no | `2026-11-21T12:00Z` | Lets you re-run a prediction as it would have looked earlier, which backtesting needs. |
| `lineups` | no | 11 player IDs per side | Only used if confirmed. Otherwise the system falls back to a projected XI (§4). |
| `absences` | no | player ID + reason (injury/suspension) | Manual entry overrides any feed. |

The CLI can stay this simple:

```
$ python -m tabet predict --home "Arsenal" --away "Fulham" --kickoff 2026-11-22T15:00Z
$ python -m tabet predict ... --lineups lineups.json --explain
```

## 3. Historical data retrieved for the fixture

Everything is read "as of" the cutoff.

**Team level** (we have this now: `res/Premier League 21-22.csv` … `25-26.csv`, from football-data.co.uk):
- Results, goals, half-time score, shots, shots on target, corners, cards, referee.
- Pre-match and closing bookmaker odds. These are a **benchmark only**, never a feature (see §8).
- Derived: rolling attack and defence ratings, home/away splits, rest days, and a season weight
  (the README's "weighting the different years" idea becomes a decay parameter we tune, not a fixed guess).
- Promoted teams: a prior built from how earlier promoted sides did. Relegated teams' last top-flight
  seasons calibrate it (README idea).

**Player level** (we **do not have this yet**, see §7):
- Minutes, starts, position, goals, assists, xG, xA, and per-90 rates, by season and by league.
- Transfer history (date, from club/league, to club).
- Availability history (injuries, suspensions).

## 4. How lineups, absences, form and prior-league performance enter

1. **Lineup strength.** Each side's XI is scored by summing player contributions (for example
   minutes-weighted xG+xA per 90 for attackers, and a defensive measure for defenders/GK). The feature
   is *XI strength relative to the team's typical XI*, so a club with a weakened side is marked down
   against its own baseline.
   - Confirmed lineup available: use it.
   - Not available: use a projected XI (most-used available players by position over the last N
     matches, minus listed absences), and mark the prediction `lineup_status = projected`.
2. **Absences.** These are handled through (1). A missing player drops out of the XI and the
   replacement's rating takes their place. Absences are never a separate hand-tuned penalty.
3. **Recent form.** Exponentially weighted results and shot/xG differential over recent matches,
   adjusted for opponent strength. The half-life is tuned on validation data.
4. **Prior-league performance.** Applies to players with little Premier League history. Their
   prior-league stats go in shrunk toward a position-level prior, weighted by minutes played. How
   league strength is used (if at all) is an experiment, not an assumption (§6).

## 5. Output and inspecting the factors

The probabilities always sum to 1. The factor table shows how far each feature group moved the
prediction away from a baseline that knows only the two teams' long-run ratings and home advantage.
The method depends on the model: coefficient × value for a multinomial logistic / Poisson model,
or SHAP values for a tree model.

### Worked example: ALL INPUTS FICTIONAL, PROBABILITIES ILLUSTRATIVE

> **None of the numbers below come from a trained model.** The teams, players and numbers are made
> up to show the layout and logic. The system has not been trained yet.

**Input (fictional):**
```
home_team: Northbridge FC       away_team: Harbour City
kickoff:   2026-11-22T15:00Z    lineups: confirmed (both)
absences:  Northbridge — "D. Okafor" (CB, injury)
           Harbour City — none
new signing: Harbour City "L. Varga" (ST), 1,900 min last season in a fictional "League X"
```

**Output (illustrative):**
```
TaBet prediction — Northbridge FC vs Harbour City — cutoff 2026-11-22 15:00Z
lineup_status: confirmed / confirmed    data freshness: last result ingested 2026-11-16

            Home win   Draw    Away win
Baseline      0.46     0.27     0.27     (team ratings + home advantage only)
Final         0.41     0.27     0.32

Factor contributions (percentage points, home-win / away-win)
  Factor                                   Home    Away   Detail
  Northbridge XI vs usual XI (Okafor out)  -2.5    +2.0   replacement CB rated lower
  Recent form (last 6, opp-adjusted)       -1.5    +1.5   Harbour City xG diff +0.6/match
  Harbour City XI incl. Varga               -0.5    +1.0   prior-league stats, shrunk (1,900 min)
  Rest days (3 vs 6)                        -0.5    +0.5
  Other / interaction                       +0.0    +0.0
                                           -----   -----
  Total shift                              -5.0    +5.0

Uncertainty notes
  • Varga: prior-league rating has wide uncertainty (only one season of data, league-strength input = fallback)
  • Market benchmark (not used as input): H 0.44 / D 0.27 / A 0.29
```

The draw stays around 0.25–0.30. That matches the README note that draws should rarely be the
favourite, and it holds because the model is calibrated, not because of a hard-coded rule.

## 6. League strength and cross-league player histories

### 6.1 Candidate sources

| Source | What it measures | Coverage / history | Free & reproducible? | When available | Limits for judging individual players |
|---|---|---|---|---|---|
| **Opta Power Rankings** ([intro, Jan 2023](https://theanalyst.com/2023/01/power-rankings-your-club-ranked/); [league table](https://theanalyst.com/articles/strongest-football-leagues-in-the-world-opta-power-rankings)) | A hierarchical Elo-style **club** rating from 0 to 100, using results and (when available) xG, weighted 80/20. A league's rating is the **average of its clubs' ratings**. | 13,500+ men's clubs worldwide, so very wide coverage. Publicly launched around Jan 2023. **No public historical archive found** that we could download. | Free to view, but no download or API found, and we haven't checked whether scraping is allowed. **Not reproducible for past dates** unless an article happened to publish that date's table. | Since about early 2023 | Measures team strength, not how hard it is for one player to produce. Averaging hides gaps between the top and bottom of a league. The method has been revised over time. |
| **UEFA association coefficients** ([UEFA explainer](https://www.uefa.com/news-media/news/0252-0cda3876cf40-3b08507f16f2-1000--how-association-club-coefficients-are-calculated/); [Wikipedia](https://en.wikipedia.org/wiki/UEFA_coefficient)) | Sum over 5 seasons of each association's per-season score: points won by its clubs in UEFA competitions (win 2, draw 1, plus bonuses) divided by the number of clubs entered. | All UEFA associations, published season by season for decades (UEFA site; [worldfootball.net](https://worldfootball.net/uefa_country_coefficients)). **No non-UEFA leagues** (MLS, Brazil, Argentina, Saudi, etc.). | Yes. Season-end values are public and stable, so they can be versioned by hand in a CSV. | Updated during the season. Season-end values are final after the season ends. | Reflects only the clubs that play in Europe (the top 4–7 per country), not the league as a whole. The 5-year lag reacts slowly. Bonus rules change between cycles. Countries, not divisions, so second tiers aren't covered. |
| **ClubElo** (clubelo.com; API `api.clubelo.com/<YYYY-MM-DD>`) | Daily Elo rating for each club, based on results. | European clubs (top flights and many second tiers) going back decades. **Date snapshots** make point-in-time lookups straightforward. | Free CSV API. The site returned 403 to our automated fetch, so **terms of use and access need confirming by hand**. | Daily history | Club strength only, Europe only. A league value has to be built from it (e.g. the average Elo of the clubs that season). |
| **FiveThirtyEight SPI** (archived) | Club offensive/defensive ratings used for match forecasts. | Many leagues worldwide, about 2016 to June 2023, then **discontinued**. | Archived CSVs on GitHub (CC BY). | Snapshots in the archive only | Ends in 2023, so it can't cover current seasons. Useful only for backtests. |

**Recommendation.** Use **UEFA association coefficients (season-end, lagged)** as the primary
point-in-time league feature, because they are the only fully reproducible free option. Add a
**ClubElo-derived league average** if its terms allow automated use. Opta is too thin historically
to use as a feature. Treat it as a qualitative cross-check only. For leagues outside UEFA, use
the fallback in §6.4.

### 6.2 Point-in-time rule

For a prediction with cutoff *t*, use only a rating **published before *t***. For UEFA, that means
the coefficient as of the end of the last *completed* season (e.g. a November 2026 prediction uses
end-of-2025/26 values). Each rating row stores `source, league, value, as_of_date, retrieved_date`.
Today's ratings are never applied to past seasons.

### 6.3 Experiment: does league strength help?

**Population:** players arriving in the Premier League from another league, with at least *M*
minutes the season before (*M* to be tuned, starting around 600).
**Target:** later unseen performance. Primary target: PL per-90 xG+xA (or goals+assists if xG is
unavailable) over their first PL season, with minutes as a weight. Secondary target: change in
match-outcome log-loss when these players' ratings feed the match model.

| Variant | Description |
|---|---|
| A — no adjustment | Raw prior-league per-90 stats, shrunk by minutes toward a position prior. |
| B — league strength as a feature | Same raw stats, plus the point-in-time league rating as an extra input. The model learns how much it matters and whether it interacts with each stat. |
| C — learned transfer adjustment | Fit a mapping, separately for each stat and position, from prior-league stats to next-season PL stats on past transfers (e.g. a hierarchical regression with a league effect). Only attempted if enough transfers per league exist (target ≥ ~30 player-seasons per league, otherwise pool leagues into tiers). |

Rules:
- **No assumed ratio multipliers.** Dividing two league ratings and multiplying goals by the result is
  not done unless variant C learns something similar *and* it beats A/B out of sample.
- **Controls:** minutes (also used as a weight), position group, age, the new team's strength
  (moving to a dominant team raises output regardless of league), and the old team's strength.
- **Small samples:** shrink toward the position prior in proportion to minutes. Report uncertainty intervals.
- **Validation:** split by time (train on transfers before season S, test on season S). Never split randomly.
- **Keep the originals:** raw stats are stored unchanged. Adjusted values go in separate,
  clearly labelled columns (`*_raw`, `*_adj_B`, `*_adj_C`).
- **Decision rule:** keep the simplest variant whose out-of-sample error isn't clearly worse.

### 6.4 Fallback when no suitable rating exists

For a league with no point-in-time rating (e.g. outside UEFA, before a source existed, or a
second tier that isn't covered):
- set the league feature to *missing* and add a `league_rating_missing` flag;
- shrink that player's prior-league stats **more** heavily toward the position prior (i.e. treat
  the stats as less informative), and don't make up a league weight;
- log the gap in `data/gaps.csv` so it shows up in reports.

## 7. Unresolved data requirements

| Need | Status | Options to evaluate |
|---|---|---|
| Player match-level stats (minutes, position, xG/xA) | **Missing** | FBref/StatsBomb-derived tables (check licence/scraping terms), Understat (PL + 5 leagues, xG), paid APIs (API-Football, Sportmonks). |
| Confirmed lineups for upcoming fixtures | **Missing** | API-Football/Sportmonks (~1 h pre-kickoff); manual entry as the always-available fallback. |
| Injuries / suspensions | **Missing** | Same APIs; manual entry. Point-in-time history is hard to get. |
| Transfer history | **Missing** | Transfermarkt (check terms), or derive it from changes in a player's club. |
| Team xG | **Missing** | Understat. The current CSVs only have shots and shots on target. |
| League ratings | Partly solved | UEFA coefficients (manual CSV); ClubElo pending terms check. |
| Current season (2026/27) matches | **Missing** from `res/` | football-data.co.uk publishes an in-season file; it needs a scheduled fetch. |
| Player/team ID matching across sources | **Open** | Needs an alias table and a matching step that gets checked by hand. |

## 8. Main assumptions

- Pre-match odds are used only as a **benchmark** for accuracy (the README's "use API to check accuracy").
  We don't use them as an input, so the model has to stand on its own.
- Every feature can be computed using only data timestamped before the cutoff. That is the pipeline's
  main invariant and will be tested.
- Probabilities are judged on calibration and log-loss / Brier score, not on win-pick accuracy.
- A match outcome depends mostly on team strength. Player-level features adjust that baseline at the
  margins rather than replacing it.

## 9. Feedback loop: results, updates, retraining, error analysis

1. **Ingest (after each matchday):** fetch completed results, validate them (schema, team aliases,
   duplicates, score sanity), and append to the store with `ingested_at`.
2. **Feature update:** recompute rolling team and player features. Past predictions are never
   rewritten, because each prediction stores the feature snapshot it used.
3. **Prediction log:** every prediction records inputs, cutoff, feature snapshot, model version,
   probabilities and lineup status. Once the result is in, the outcome is joined on.
4. **Scheduled retrain:** for example monthly and at the end of each season, using a rolling time split.
   A new model is promoted only if it beats the current one on held-out log-loss and isn't worse calibrated.
5. **Error analysis report:** log-loss/Brier against the market benchmark and a naive baseline; a
   calibration table; errors broken down by projected vs confirmed lineup, promoted teams, matches
   involving new signings, and whether a league rating was missing.

## 10. First implementation deliverable: clean, validated pipeline

Scope before any modelling:
- Load all `res/*.csv` (some files start with a UTF-8 BOM, and the bookmaker columns change between
  seasons) into one normalised match table: `match_id, season, date_utc, home_id, away_id, FTHG, FTAG, FTR, HS, AS, HST, AST, …`.
- Canonical team IDs plus an alias table.
- Validation checks: 380 matches per season, 20 teams, each pair meets home and away once, FTR
  consistent with the goals, no future-dated rows, odds kept in a separate benchmark table.
- A point-in-time accessor `get_history(team, cutoff)`, with tests showing it never returns data after the cutoff.
  A result counts as known `RESULT_LAG` (2h30m) after kickoff. If a kickoff time is missing, it's assumed to be 22:00 UK time (the safe direction).
- Known-bad source values are blanked through `data/source_fixes.csv` (with a reason), never replaced with guesses.
  Odds of `0` are read as missing.
- Tooling note: `.github/workflows/main.yml` currently runs Node/npm steps and its branch filter
  (`-master`) doesn't match `main`. It needs replacing with a Python job (lint + pytest).
