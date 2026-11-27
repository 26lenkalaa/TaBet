# Prospective protocol: predicted lineups at a fixed early horizon

Written 2026-11-05, before any 2026-27 gameweek-10 match. It answers one question:

> **Does a learned, conservative use of predicted lineups beat the lineup-free forecast (V1x) at
> 24 hours before kickoff?**

## What is logged

- **Horizon (current rule, amendment A4 from gameweek 11):** each fixture can get two records: a
  **primary** one from the first run that finds its kickoff 24–26 h ahead, and a **fallback** one
  from the first run that finds it 26–32 h ahead. Scoring uses the primary record when it exists and
  the fallback otherwise. An external cron service triggers a run every 30 minutes, with GitHub's
  own schedule as a backup. The hours before kickoff are recorded and results are reported by them.
- **Horizon (gameweek 10 rule, superseded):** each fixture is logged once, by the **first run that finds it 24–26 h before
  kickoff**. Runs are hourly (`python -m tabet log-forecasts`) on a scheduled GitHub Action
  (`.github/workflows/prospective.yml`), so forecasts are normally made 25–26 h before kickoff.
  The actual hours before kickoff are recorded. *(Amended 2026-11-05 before any record was
  logged: the original rule was "the last run at or before kickoff − 24 h" with runs every
  2 hours on a local machine, which is not available. GitHub's scheduler can start runs late, and
  the 2-hour window tolerates one missed run.)*
- **Commits:** the Action commits the log, raw snapshots and the refreshed results file back to
  `main`.
- **Each run:**
  1. refreshes the current-season results file (it is replaced only if the new file validates)
  2. refits every model at that moment
  3. fetches the predicted-lineup pages once
  4. appends one record per fixture to `results/prospective/log.jsonl`
- **The log is append-only.** Raw lineup pages are kept gzipped in `results/prospective/raw/`.

Each record holds:

- **Identity:** fixture id, kickoff, logged time (the cutoff), hours before kickoff.
- **Provenance:** git commit, whether the code had uncommitted changes, results through date,
  training size.
- **Probability vectors** for every candidate (below), plus any candidate that could not be
  produced.
- **Lineups:** each source's predicted XI as FPL player codes, names it could not match, and the
  source's own "last updated" text where shown (Fantasy Football Scout shows one per team;
  RotoWire shows none).
- **Inputs to reproduce each candidate exactly:** Dixon-Coles rates, adjusted-xG features, XI
  features per source, and stage-2 coefficients, means and SDs. Reconstruction was checked
  (difference 0).
- **FPL availability** (status, chance of playing, news) for both squads at the cutoff.

## Candidates

| Name | Forecast |
|---|---|
| `p0` | V1x, lineup-free. **The default early forecast.** |
| `ffs` | V2x with Fantasy Football Scout's predicted XI |
| `rotowire` | V2x with RotoWire's predicted XI |
| `pE` | the plain average of the `ffs` and `rotowire` outcome probabilities (if one is missing, the other alone) |
| `fpl_proj` | V2x with our projection (most-started GK + top-10 outfielders by start probability for the player's *current* FPL club), excluding players FPL flags as unavailable |
| `fpl_proj_hist` | the same projection without the availability filter |
| `pα` | `(1 − α)·p0 + α·pE`, with α ∈ [0, 1] fitted by minimising log-loss on earlier logged matches only |

All V2x candidates apply coefficients trained on *confirmed* XIs. That makes them experimental
plug-in forecasts, which this protocol is designed to test.

**Frozen definitions:**
- **Unavailable** means FPL `status != "a"` **and** chance of playing `< 50`, where the chance is
  read from the field that refers to **the fixture's own gameweek**: `chance_of_playing_next_round`
  before that gameweek's FPL deadline, `chance_of_playing_this_round` after it. A missing chance
  counts as 0 for any status other than `"a"`. Doubtful players (50–75%) are kept, and loans count
  as unavailable. *(Amendment A2.)*
- **Replacement in `fpl_proj`:** a dropped player is replaced by the next player of the same group
  (goalkeeper, or any outfielder) by decayed start probability. Positions within the outfield are
  not matched.
- **An unresolved source name** contributes the position-prior rating.
- **A source without an XI for either team** leaves that candidate missing for the fixture.

## Schedule and decisions

- **Fitting period: gameweeks 10–19.** Everything is logged; nothing is chosen.
- **α is fitted once,** on all gameweek 10–19 records, then frozen.
- **Evaluation period: gameweeks 20–29.** This is a preliminary chronological evaluation, with
  α frozen.

Ten gameweeks each is an engineering choice, not a statistical minimum, and twenty gameweeks may
not be enough to rank close candidates.

**Comparisons on the evaluation period** (log-loss; each fixture counted once; 14-day
calendar-block bootstrap CI):

| | Comparison | Question |
|---|---|---|
| primary | pα − p0 | Does a learned amount of lineup information help? |
| secondary | pE − p0 | Does predicted-lineup information help at all? |
| secondary | pα − pE | Does shrinkage beat trusting predicted XIs fully? |
| secondary | fpl_proj − fpl_proj_hist | Does the availability filter alone help? |
| secondary | each source − p0 | Source-level value |

- **Variance:** each comparison uses its own paired variance. Variance from the confirmed-XI
  backtests is not reused.
- **Rule against shortcuts:** α is never set from lineup accuracy or source agreement.

## Diagnostics after each gameweek (`python -m tabet score-forecasts`)

Per candidate:
- correct starters out of 11
- absolute error of the XI-strength feature against the feature computed from the confirmed XI
- outcome log-loss against `p0`
- coverage and unresolved names

**Controlled comparison:** the same frozen early model and non-lineup inputs, with the confirmed
XI substituted (`controlled_confirmed_xi`). This measures the cost of lineup error and is kept
separate from operational performance.

**Operational comparison (not yet scheduled):** a second horizon at about kickoff − 45 min, using
lineups confirmed by then, would measure the later product's performance. It needs runs every 15
minutes on matchdays and will be reported separately if added.

## What this protocol does not claim

- It does not confirm the pre-registered V2 − V1 effect. That test uses confirmed XIs and is
  defined in `docs/PREREGISTRATION.md`.
- Snapshot ages are recorded but not controlled. A source updated days before the cutoff is still
  that source's latest view at the cutoff.

## Amendment A2 (2026-11-05, before any record was logged)

Made in response to an external review, before the first gameweek-10 record. At the time of the
amendment `results/prospective/log.jsonl` was empty.

1. **Availability field (bug fix).** FPL's "next round" moves to the following gameweek at each
   deadline (gameweek 10's is 2026-11-06 18:30 UTC). A Sunday fixture logged 24 h earlier is logged
   after its own deadline, so the old rule read next week's availability. The rule above now reads
   the fixture's own gameweek. Checked against the live API: both fields exist, and 32 of 670
   players differed between them on 2026-11-05. **Null chances** counted as available under the old
   code; they now count as 0 for any non-"a" status.
2. **Moved kickoffs.** A record is keyed by fixture *and* kickoff. If a kickoff moves after logging,
   the old record is kept, the fixture is logged again by the normal rule against the new kickoff,
   and only the record whose kickoff equals the kickoff actually played is scored. Superseded
   records are counted and reported. Postponed fixtures without a new date are not scored.
3. **Missing sources.** `pE` is the single available source if only one has both XIs, and `p0` if
   neither does (so `pα` = `p0` there). The fallback rate is reported per source. An unmatched
   name keeps the position-prior rating (unchanged).
4. **α.** Fitted once, on all non-superseded gameweek 10–19 records, after gameweek 19 has been
   scored; not refitted walk-forward. Constrained to [0, 1] on a 0.01 grid. **α = 0** if fewer than
   20 fixtures are available or the log-loss curve is flat.
5. **Power and the confirmatory read.** With ~100 evaluation fixtures and a paired SD near 0.047,
   the SE of `pα − p0` is about 0.005, so a gain of 0.001–0.003 cannot be detected. Therefore:
   - the gameweek 20–29 read is **descriptive**;
   - the **confirmatory** read of the same primary contrast is gameweeks 20–38 (end of season,
     ~190 fixtures), with α still frozen at its gameweek-19 value;
   - a **secondary, pre-registered decomposition** is reported at both points (`python -m tabet
     score-forecasts`): per source, feature capture = 1 − mean((predicted − confirmed XI
     feature)²) / var(confirmed XI feature) over team-matches; confirmed gain =
     `controlled_confirmed_xi − p0`; implied gain = capture × confirmed gain, against the observed
     `source − p0`. It is an approximation (it assumes the log-loss gain scales with captured
     feature variance), reported as such.
6. **Code freeze.** Every record carries its git commit. If a bug fix is unavoidable during
   gameweeks 10–38, both versions' predictions are kept where they can be reproduced, and the
   pre-registered (earlier) version is the one reported.
7. **Not comparable to the ledger.** Logged `p0` uses a cutoff 24 h before kickoff, so it can miss
   results from earlier the same weekend. It is never compared with the historical ledger's V1x as
   if they were the same model; all prospective contrasts are within a record.
8. **Live xG snapshots.** Each day the logger re-downloads the four most recent finished
   gameweeks and keeps player xG/xA/xGC in `results/prospective/xg_snapshots/`, so revisions to
   live FPL xG can be measured against the archive values used in the backtests.

## Operations log

- **2026-11-05: Everton v Coventry (fixture 94, kickoff 2026-11-06 20:00 UTC) not logged.** Its window
  was 18:00–20:00 UTC on 2026-11-05. The scheduled workflow had not fired since it was added, and its
  first scheduled run came at 21:06 UTC, after the window. Under the logging rule it is not backfilled.
  A fixture missing entirely does not bias any within-record contrast. Amendment A2 and the
  re-registered schedule (:23 past each hour) reached `main` at 22:10 UTC.
- **2026-11-08: gameweek 10, eight more fixtures not logged.** GitHub started the hourly schedule only
  every 4–8 hours: runs at Nov 5 21:06, Nov 6 02:34 / 11:10 / 18:06 / 23:17, Nov 7 02:55 / 10:27 /
  16:54 / 20:58 and Nov 8 00:46 UTC, all successful in 30–50 s. With the 24–26 h window, only fixture
  97 (Leeds v Tottenham, logged Nov 6 11:10, 25.3 h ahead) fell inside a run. Not backfilled:

  | FPL fixture | Kickoff (UTC) | 24–26 h window | Nearest runs |
  |---|---|---|---|
  | 91, 95 | Nov 7 15:00 | Nov 6 13:00–15:00 | 11:10, 18:06 |
  | 99 | Nov 7 17:30 | Nov 6 15:30–17:30 | 11:10, 18:06 |
  | 92, 93, 96, 100 | Nov 8 14:00 | Nov 7 12:00–14:00 | 10:27, 16:54 |
  | 98 | Nov 8 16:30 | Nov 7 14:30–16:30 | 10:27, 16:54 |

  Gameweek 10 therefore has 1 record out of 10 fixtures. The cause was GitHub's scheduler, not the
  code. Amendment A4 changes the trigger and the window from gameweek 11.
- **2026-11-10: amendment A6 written (odds archive and daily results refresh).** During the break no
  fixture was due, so the results file was last refreshed on 2026-11-02 (90 matches) while
  football-data already had all of gameweek 10 (100). A6 adds a daily refresh and a pre-match odds
  archive for the market benchmark. Both run inside the existing runs. Neither changes any forecast,
  record or rule of the primary contrast. On 2026-11-10 `fixtures.csv` still held gameweek 10's
  fixtures (Last-Modified Fri 6 Nov 11:14 GMT), all already played, so the archive starts empty.
- **2026-11-10: amendment A7 written (The Odds API snapshot at each record's cutoff).** football-data's
  file updates about once a round, so on its own it can't give a market snapshot at our forecast time.
  The key was added as a repository secret the same day. The first A6 run (20:13 UTC) worked: the
  results file reached 100 matches and the first fixtures.csv copy was saved.

## Amendment A3 (2026-11-05, before any record was logged)

From a further external review. `results/prospective/log.jsonl` was still empty.

1. **Valid source XI.** A source's lineup for a team is used only if it lists exactly 11 players and no
   resolved FPL player appears twice. Otherwise that source counts as missing for that team, and the
   A2 fallbacks apply (pE = the other valid source, or p0). Invalid lineups are recorded per record
   (`lineup_invalid`). Every Scout and RotoWire snapshot so far lists 11 per team with no duplicates.
2. **Eligibility details, made explicit.** A chance of exactly 50 is kept (the rule is < 50). A null
   chance counts as 0 only for a non-"a" status, as in A2. FPL leaves the chance empty for many
   injured, suspended and departed players, so "unknown" would keep players FPL has already ruled
   out. Club eligibility is separate: a player counts for a team only if FPL lists him at that club
   at the cutoff (projections), and scraped names are matched only within the team's FPL squad.
3. **What α may see.** α is fitted on gameweek 10–19 records whose results were known (kickoff +
   2 h 30 min) before the first gameweek-20 cutoff. A postponed fitting-period fixture played later
   is excluded from the fit and is not scored in the evaluation period either.
4. **What the cutoff means.** Each record's cutoff is its logged time. Every input (results,
   ratings, FPL availability, both lineup pages) is fetched in that run, so there is no separate
   assembly step and no input is older than the run itself. Each source's own "last updated" text
   is kept where it shows one.
5. **Reporting.** The paired normal SE is the primary uncertainty measure at this sample size; a
   gameweek-clustered SE with a t reference on (gameweeks − 1) degrees of freedom is the
   sensitivity check, and the 14-day block bootstrap is reported as well. If α is fitted at 0,
   pα equals p0 and the result is reported as "the fit chose no lineup weight", with pE − p0
   carrying the information. The primary contrast includes fallback fixtures, so it measures the forecast a
   user would actually get. Alongside it: source coverage, the fallback rate, the complete-case
   contrast (both sources valid), and the contrast with each gameweek left out in turn. With 14-day
   blocks the evaluation window holds only about five blocks, so no source "winner" is declared
   unless the interval clearly separates them.

## Amendment A4 (2026-11-08, written before any gameweek-11 record)

Written during the international break, after the gameweek-10 failure in the operations log. The
first gameweek-11 kickoff is fixture 108 (Man City v Fulham) on 2026-11-21 12:30 UTC. This amendment
must be on `main` before that fixture's first window opens (2026-11-20 04:30 UTC).

1. **Two windows, one scored record.** Each fixture can get two records per kickoff time, from the
   same code at the run's own cutoff:
   - **primary:** the first run that finds the kickoff in [now + 24 h, now + 26 h), stored with
     `horizon_h = 24`;
   - **fallback:** the first run that finds it in [now + 26 h, now + 32 h), stored with
     `horizon_h = 26`.

   Every analysis uses **one record per fixture: the primary if it exists, otherwise the fallback**
   (`select_records`). This rule depends only on which records exist, and both are kept. `hours_before`
   is recorded for every record, and results are also reported by hours before kickoff.
2. **Why.** In gameweek 10 GitHub's schedule ran only every 4–8 hours (one gap was 8.6 h), and only 1
   of 10 fixtures was logged. A single wide window would survive that, but with a run every 30
   minutes it logs every fixture at the *start* of the window, making every forecast about 6 hours
   older than the old rule. The two-window rule keeps forecasts about 25.5 h old when the trigger
   works, and still produces a record when it does not. Simulated on the gameweek-10 run times, the
   fallback would have covered all nine fixtures after the first scheduled run (fixture 94's window
   came before it). Simulated on gameweek 11's fixtures: every 30 minutes gives 10 primary records at
   25.5 h; GitHub's gameweek-10 gaps alone give 1 primary and 9 fallback records at 26–31 h.
3. **Trigger (operations, not protocol).** An external free cron service (cron-job.org) calls the
   workflow's `workflow_dispatch` every 30 minutes; dispatched runs start within about a minute.
   GitHub's own schedule stays as a backup every 2 hours. The interval can change without a new
   amendment as long as it stays well under 2 hours. Setup: `docs/LOGGER_SETUP.md`.
4. **Settings in code.** The windows and the run interval are separate settings: `LOG_HORIZONS`
   (protocol: (24, 2) and (26, 6)) and `SCHEDULE_EVERY_H` (operations). `--horizon`/`--window` on
   `log-forecasts` now log a single test window; `--every` is kept as an alias of `--window`.
5. **Extra runs are harmless.** The workflow's concurrency group runs one job at a time. A queued
   job checks out `main` after the previous job has pushed, `log_due()` re-reads the log just before
   appending, and `python -m tabet check-log` refuses to push if a fixture/horizon/kickoff key ever
   appears twice. Two near-simultaneous runs therefore cannot both append the same record.
6. **Monitoring.** Every run writes a heartbeat to the job summary: the run time, the last logged
   record, the next fixture with no record and when its windows open, and any fixture that only got
   a fallback. If a fixture's primary window closes with no record at all (from gameweek 11 on), or
   a window is open and no run has succeeded for 3 hours, the workflow opens or updates a GitHub
   issue titled "Prospective logger alert".
7. **Actions minutes.** The repo is private. Dispatched runs: 48 a day, about 1,440 minutes a month.
   Backup schedule: at most 12 a day, about 360. Total about 1,800, within GitHub Free's 2,000 (3,000
   with GitHub Pro). Each run bills one minute when nothing is due.
8. **Analysis.** Gameweek 10 (one record, under the old 24–26 h rule) is reported separately. The
   share of fixtures scored from fallback records is reported alongside every result.

## Amendment A5 (2026-11-09, written before any gameweek-11 record)

From the fifth external review. Items 1–3 change only what is reported and how the confirmatory read
is judged; item 4 adds one logged candidate. Nothing about p0, pE, α's fitting rule or the record
windows changes. This must be on `main` before 2026-11-20 04:30 UTC; after the first gameweek-11
record no rule changes.

1. **Results by record type.** A fallback record for a Saturday 15:00 kickoff is made between Friday
   07:00 and 13:00 UK time, which is before most clubs' Friday press conferences. So the lineup
   forecasts can be worth less in fallback records, and the primary/fallback mix depends on the
   scheduler. Every pE − p0 and pα − p0 result is also reported separately for primary and fallback
   records, and, where a source's "last updated" text parses as a date (the sources show a date such
   as "Tue 3rd Nov", not a time), by whether that update was on the record's own UK date or earlier.
   Descriptive.
2. **α's uncertainty.** α is still fitted once on gameweeks 10–19 (A2, A3). Reported with it: a
   likelihood interval (every α on the 0.01 grid whose summed log-loss is within 1.92 of the
   minimum), and the evaluation contrast at α = 0, 0.5 and 1 as sensitivity. If α is fitted at 1,
   pα equals pE and is reported that way; α is never refitted.
3. **Inference for the confirmatory read.** For the gameweek 20–38 read, the **gameweek-clustered SE
   with a t reference on (gameweeks − 1) degrees of freedom is the primary uncertainty measure**; the
   paired normal SE and the 14-day block bootstrap are reported beside it. Reason: fixtures in one
   gameweek share their information timing (same lineup snapshots, same news cycle), and 14-day
   blocks give too few blocks to bootstrap. A sign-flip test on gameweek sums (10,000 random flips,
   seed 0) is reported as a check. The gameweek 20–29 read stays descriptive, with both SEs.
   The measured paired SD (0.047) puts the SE of a 100-fixture read near 0.005, so that read cannot
   detect the gains in question; this is why the confirmatory read runs to gameweek 38.
4. **Secondary prospective contrast pBP − p0.** From gameweek 11, each record also stores `pBP`:
   the same expected goals as p0, turned into a score grid by the bivariate Poisson of addendum M
   instead of the Dixon-Coles grid, with the shared rate λ3 fitted at the record's cutoff on the
   training matches' Dixon-Coles rates (decay 0.004 per day). λ3 is stored in `components.bp_l3`.
   pBP − p0 is read at gameweeks 20–29 (descriptive) and 20–38, with the same inference as item 3.
   It is not blended into pα and does not affect the primary contrast. If pBP fails for a record,
   that record lacks it; pBP − p0 is computed on the records that have it and the missing count is
   reported. Its offline counterpart is addendum B of `docs/PREREGISTRATION.md`.

## Amendment A6 (2026-11-10, written before any gameweek-11 record or odds snapshot)

Adds a market benchmark at matching times and keeps the results file current. **Bookmaker odds stay
a benchmark only.** Nothing here changes p0, pE, pα, α's fitting rule, the record windows, what a
record stores, or the primary contrast. Like A5, it must be on `main` before 2026-11-20 04:30 UTC.

1. **Odds archive (benchmark only).** On every logging run, at most once an hour, the run downloads
   football-data's `fixtures.csv` (`tabet/odds_archive.py`, same client name as the results
   download). Every fetch is listed in `results/prospective/odds/fetch_log.csv`.
   - If the file's SHA-256 differs from the last saved copy, the raw file is kept gzipped as
     `results/prospective/odds/raw/<UTC time>_fixtures.csv.gz`.
   - On a changed file, the Premier League (E0) rows for fixtures that have not kicked off are
     appended to `odds_log.csv`. Each row holds the FPL fixture (matched by home team, away team and
     a kickoff within 36 h), kickoff, retrieval time, hours before kickoff, the raw file's hash, and
     one bookmaker column group's H/D/A, including the average and maximum.
   - An unchanged file adds nothing, so a snapshot stands from its retrieval time until the next
     changed file.
   - Rows that match no FPL fixture are kept with an empty fixture id and listed in the run output.
   - Everything is append-only. No existing record in `log.jsonl` or `raw/` is touched.
2. **Isolation.** Only the archiver, `tabet/market_benchmark.py` and the CLI that runs them may refer
   to the archive. A test (`tests/test_odds_archive.py`) fails if any other module under `tabet/`
   refers to the archive path or modules, if any module outside the existing benchmark code names
   bookmaker odds columns, or if any module names FPL crowd fields (ownership, transfers, price).
3. **Snapshots compared, per scored fixture** (the record `select_records` chooses: primary, else
   fallback):
   - **matched:** the last archived snapshot retrieved at or before that record's cutoff. If there is
     none, the first one after the cutoff and before kickoff, flagged `later`.
   - **last pre-kickoff:** the last archived snapshot retrieved before kickoff.
   - **football-data pre-closing and closing:** the `Avg*` and `AvgC*` columns of the results file,
     published after the match, as in `prototypes/run_market_comparison.py`.

   Archived snapshots use the market average ("Avg") group. Probabilities come from proportional
   margin removal, p_k = (1/o_k) / Σ_j (1/o_j). The hours between each archived snapshot and the
   record's cutoff are reported, along with the number flagged `later`.
4. **Daily results refresh.** On the first run of each UTC day, `refresh_results()` runs whether or
   not a fixture is due. The validate-or-restore rule is unchanged. A failed attempt is retried at
   most hourly until one succeeds that day. Every attempt is listed in
   `results/prospective/results_refresh.csv`, with football-data's and the local file's match counts.
   This changes only how fresh the file is, not what a record may use: each record already refreshes
   the file in its own run and stores `results_refreshed`.
5. **Reporting.** `score-forecasts` prints a separate "market benchmark" section, saved to
   `results/prospective/odds/benchmark.csv`. For p0, pE and pα against each snapshot, on identical
   fixtures, it reports:
   - n and coverage (scored fixtures with that snapshot)
   - each side's log-loss and the paired TaBet − market delta, with paired and gameweek-clustered
     SEs (t on gameweeks − 1)
   - the agreement rate of the top picks, and among disagreements: TaBet only right, market only
     right, both wrong

   pα uses α as fitted at the time and is provisional until gameweek 19 is complete. The benchmark is
   descriptive. It never feeds α, any model choice, or any later amendment's choice of candidate.
6. **Monitoring.** The heartbeat adds:
   - the results file's local and football-data match counts
   - "results file behind" when both today and yesterday (UTC) had refresh attempts and none left the
     local file current (a failed download counts)
   - fixtures kicking off within 24 h with no average-odds snapshot
   - odds coverage for the current gameweek

   The first two conditions open or update the "Prospective logger alert" issue. A comment is added
   only when the alert text changes.

   When `fixtures.csv` first fills before a round is not yet known. Gameweek 10's copy was last
   modified on Friday 6 November at 11:14 GMT, and football-data's notes say pre-closing odds are
   collected on Friday afternoons for weekend games and Tuesday afternoons for midweek games. So a
   Friday-evening or midweek fixture may raise "no odds snapshot" until the file fills. That is a
   coverage note, not a logger failure, and the first gameweek-11 fetches will show the real timing.

## Amendment A7 (2026-11-10, written before any gameweek-11 record)

Adds a time-matched market snapshot from The Odds API. **Odds stay a benchmark only.** Nothing here
changes p0, pE, pα, α's fitting rule, the record windows, what a record stores, or the primary
contrast. It must be on `main` before 2026-11-20 04:30 UTC.

1. **Why.** football-data publishes pre-match odds about once a round (gameweek 10: Friday 11:14
   GMT), often after our 24–26 h record for Friday-evening, midweek and early-Saturday matches. A
   fair "TaBet vs the market at the same moment" comparison needs a snapshot taken at the record's
   cutoff.
2. **What is called.** After a run logs one or more records, it makes one call to The Odds API:
   `GET /v4/sports/soccer_epl/odds`, with 1X2 (`h2h`) odds from UK and EU bookmakers, decimal
   format. That costs 2 credits and returns every upcoming Premier League match. Rules:
   - At most one call per run.
   - If a record logged in the last 6 hours still has no snapshot (because a call failed), a later
     run retries. Its snapshot is then "later", not matched.
   - Calls stop for the rest of the calendar month when fewer than 50 credits remain.
   - The key comes from the `ODDS_API_KEY` repository secret and is never written anywhere.
3. **What is stored.**
   - Every raw response, gzipped: `results/prospective/odds/raw/<UTC time>_oddsapi.json.gz`.
   - In `oddsapi_log.csv`, a separate file from football-data's: one row per match and bookmaker,
     with the bookmaker's own `last_update`, plus an **"Avg" row**. This is the mean decimal odds
     over the bookmakers quoting all three outcomes, as football-data's `Avg` columns are built,
     with `n_books`.
   - Every call, with credits left and the call's cost, in `oddsapi_fetch_log.csv`.
   - Matches that don't match an FPL fixture are kept and listed.
4. **The matched snapshot (replaces A6 item 3's first bullet).** For each scored record, **matched**
   is The Odds API "Avg" snapshot: the last one retrieved at or before the record's cutoff, where a
   snapshot from the same run (up to 15 minutes after the cutoff) also counts. If there is none, the
   first later one before kickoff is used, flagged `later`. A6's rule applied to football-data's
   archive is kept and reported separately as **matched_fd**. The two sources are never pooled, and
   last pre-kickoff and football-data pre-closing and closing are unchanged. Margin removal is
   proportional, as before.
5. **Website.** For a logged forecast, the match explorer shows "Market at our forecast time": the
   matched snapshot, or a football-data snapshot taken by the cutoff if The Odds API has none. It
   includes the source, retrieval time and number of bookmakers, and is the headline market for that
   match. Snapshots flagged `later` are not shown as matched. `site_data.py` reads the archive for
   display only, and the isolation test allows that file alone in addition to A6's.
6. **Monitoring.** The heartbeat shows whether the key is set, the last call and the credits left.
   The alert issue opens when fewer than 50 credits remain this month.
7. **Terms.** The Odds API's terms (31 August 2026) allow storing the data, displaying it on a
   website and using it in analysis. They forbid redistributing it as a standalone data product, so
   the raw archive stays in the private repository and only probabilities are shown on the site
   (`docs/SOURCE_AUDIT.md`).
8. **Daily snapshot for the site (operations).** The first run of each UTC day also makes one call,
   costing 2 credits (about 60 a month), so upcoming matches on the site show current "Latest market
   odds". A daily snapshot is used as **matched** only under item 4's rule. The site labels a snapshot
   "Market at our forecast time" only if it was taken within 1 hour of the record's cutoff.
9. **Site previews (display only).** Before a fixture's record exists, the site's preview also runs
   V2x with the predicted XIs that Fantasy Football Scout and RotoWire have published, usually for
   the coming gameweek. The preview is labelled as such and is not a record. The logging windows
   (24–26 h primary, 26–32 h fallback) and everything scored are unchanged.

## Amendment A8 (2026-11-16, written before any gameweek-11 record)

From the sixth external review. It changes what is reported and how results are judged. Nothing
about p0, pE, pBP, α's fitting rule, the record windows or what is logged changes, and no logger code
changes. This must be on `main` before 2026-11-20 04:30 UTC; after the first gameweek-11 record no
rule changes.

1. **What pBP − p0 actually tests.** pBP keeps p0's expected goals and replaces the Dixon-Coles grid
   with a bivariate Poisson grid, which has no ρ correction. On the 2022-26 replay the fitted λ3 is
   below 0.02 for 86% of matches (median 0 in every season). For those matches, pBP − p0 is mostly
   a test of *removing ρ*, not of adding a shared goal component. pBP stays logged, but it is
   demoted to a descriptive secondary. Reported with it: the distribution of the stored λ3
   (`components.bp_l3`) and pBP − p0 split by λ3 < 0.02 and λ3 ≥ 0.02. A ρ-preserving variant is
   **not** added (decided 2026-11-16, to avoid changing the logger before the freeze).
2. **Pooled decision rule for the bivariate model.** Its replay paired SD is about 0.035, so the SE
   at gameweek 38 (about 280 records) is about 0.002, and no live result alone can be decisive. The
   bivariate model is adopted only if **both** of these hold:
   - (a) the inverse-variance pooled estimate is below zero with a 95% interval excluding zero. It
     pools all pre-registered bivariate estimates:
     - PL 2018-21 (addendum M): −0.0028, 95% [−0.0051, −0.0005];
     - four leagues 2022-26 (addendum B): −0.0006, 95% [−0.0017, +0.0004];
     - the live pBP − p0 at gameweek 38, with its gameweek-clustered SE.

     Each SE is taken as (upper − lower) / 3.92.
   - (b) the live point estimate is no worse than +0.002.

   Otherwise the idea is retired, apart from at most one more offline test (pre-2015 non-PL leagues),
   which would need its own addendum.
3. **Headline market.** The headline benchmark becomes **matched_pinnacle**:
   - **Snapshot:** the same The Odds API snapshot that item 4 of A7 selects for `matched`.
   - **Pricing:** Pinnacle's odds in that snapshot, with the margin removed by **Shin's method**
     (`market_benchmark.shin`).
   - **Fallback:** if Pinnacle is missing from that snapshot, each bookmaker in it is normalised
     proportionally, the per-outcome median is taken, and the result is renormalised to sum to 1.
   - **Records:** which basis was used, and how long after the cutoff the snapshot was taken (in
     hours), are written for every record to `results/prospective/odds/benchmark_records.csv`.

   A7's `matched` mean of decimal odds is kept as a secondary, for continuity with §5.19. The other
   snapshots are unchanged. Pinnacle was quoted for all 20 upcoming fixtures in every snapshot from
   10 to 16 November.
4. **When α's fit counts as degenerate.** This states what `fit_alpha` already does. The fit is
   degenerate, and α = 0, only if:
   - fewer than 20 fixtures (`ALPHA_MIN_FIXTURES`) have both p0 and pE, after A3's results-known
     filter; or
   - the summed log-loss is constant across the 0.01 grid on [0, 1] (range below 1e-12).

   **A minimum on the boundary (α = 0 or α = 1) is not degenerate** and is used as fitted. A flat but
   non-constant likelihood is not degenerate either. Its width shows in A5's likelihood interval,
   and A5's sensitivity contrasts at α = 0, 0.5 and 1 are reported beside it.
5. **Gameweek clusters.** A cluster is the FPL gameweek (`event`) id stored in the scored record. A
   rearranged fixture's scored record is the one matching the kickoff actually played (A4), so it
   carries the gameweek it was played in. In a double gameweek, every fixture with that id is one
   cluster, including both of a team's matches. A postponed fixture has no FPL kickoff time until
   it is rescheduled, and only fixtures with a kickoff time are logged, so it never forms a cluster. α's fitting window
   (gameweeks ≤ 19) and the evaluation windows use the same id.
6. **Where the raw odds are kept (correcting A7 item 7).** A7 assumed a private repository, but the
   repository is public, and The Odds API's terms forbid redistributing its data. So from this
   amendment:
   - **Private:** every raw Odds API response (`oddsapi/raw/<stamp>.json.gz`) and every snapshot's
     per-bookmaker prices (`oddsapi/books/<stamp>.csv.gz`) go to the **private Supabase bucket
     `odds-archive`**. Only the secret key can read it.
   - **Repository:** `oddsapi_log.csv` keeps only three derived, margin-free probability rows per match
     and snapshot, written as fair odds 1/p:
     - `Avg`: A7's mean of the bookmakers' decimal odds, normalised proportionally;
     - `pinnacle_shin`: item 3's headline;
     - `median`: item 3's fallback.

     `raw_file` points at the private copy.
   - **Existing data:** snapshots taken before this amendment are moved the same way, and the move is
     idempotent (`odds_archive.privatise`).
   - **Checks:** on the data to 16 November, every probability the site and the benchmark use is
     unchanged. The site's "latest market" is identical, and the headline is within 1e-7 (rounding).
   - **Failures:** if an upload fails, the derived rows are still written and the failure is recorded
     in `oddsapi_fetch_log.csv`.

   Copies already in the public git history are not rewritten by this amendment. football-data's
   `fixtures.csv` archive is unaffected.

## Amendment A9 (2026-11-16, written before any gameweek-11 record)

From the seventh external review. Like A8, it changes only what is reported and how results are
judged. p0, pE, pBP, α's fitting rule, the record windows and what is logged are unchanged. It must be
on `main` before 2026-11-20 04:30 UTC.

1. **One primary claim.** The protocol makes exactly one confirmatory claim: **pα − p0 on gameweeks
   20–38**, judged with A5's inference.
   - **Descriptive only, never a route to declaring success:**
     - the gameweek 20–29 read;
     - pE − p0, pα − pE, each source − p0, fpl_proj − fpl_proj_hist;
     - the record-type and source-timing splits (A5);
     - α = 0 / 0.5 / 1 (A5);
     - pBP − p0 (A8);
     - every market benchmark.
   - **The confirmed-XI question (V2x − V1x)** was answered offline. Any live confirmed-XI number is
     descriptive.
2. **Who is in the fitting and evaluation sets.**
   - **α's fitting set:** gameweek 10–19 fixtures whose scored record (`select_records`: primary if
     any, else fallback; the one matching the kickoff actually played) has both p0 and pE, and whose
     result was known (kickoff + 2 h 30) before the first gameweek-20 record's cutoff (A3).
     - A fitting fixture whose result is not known by then is left out of α permanently, and
       counted.
     - Every evaluation record is therefore scored with the single frozen α.
   - **Evaluation set:** gameweek 20–38 fixtures with a scored record holding both p0 and pE, and a
     result.
   - **Excluded and counted:** fixtures abandoned or voided without a result, and fixtures whose
     played kickoff has no matching record (A4).
   - **Source failures** follow A2: pE falls back to the one valid source, or to p0. Such fixtures
     stay in the set, and complete cases are reported separately (A5).
3. **α degeneracy, complete.** A8 item 4's rule stands. α is fitted by grid search on [0, 1] in steps
   of 0.01, so there is no optimiser to fail. "Constant" covers the case where pE equals p0 for every
   fitting fixture (summed log-loss range below 1e-12). A minimum at 0 or 1 is a valid fit.
4. **Inference checks beside A5's primary.**
   - **Primary measure (unchanged):** the gameweek-clustered SE with t on (G − 1) df.
   - **Reported beside it:**
     - a **null-imposed wild-cluster bootstrap** p-value: Rademacher weights per gameweek,
       studentised with the clustered SE, 9,999 resamples, seed 0;
     - the 14-day calendar-block bootstrap, as a calendar-dependence sensitivity;
     - the sign-flip test, as a check that assumes symmetry.
   - **α's intervals:** A5's likelihood interval is labelled **model-based**, because it treats
     fixtures as independent. Beside it is a **gameweek-bootstrap interval** (resample the fitting
     gameweeks, refit α, 2,000 resamples, seed 0). Both are descriptive; only the one frozen α is
     evaluated.
5. **Market benchmark wording and fallback.**
   - **"Within 15 minutes":** the headline (A8 item 3) is described as *Pinnacle at the snapshot
     within 15 minutes of the cutoff*, not "at the same information cutoff". Each record's signed
     delay is reported (A8). The site's 1-hour rule is for display only and never used in analysis.
   - **Fallback:** from this amendment, the median fallback excludes betting exchanges (Betfair
     Exchange, Matchbook, Smarkets), because their prices carry commission and depend on liquidity.
     Each bookmaker's margin is removed before the median is taken. Pinnacle coverage and fallback
     coverage are reported separately (`n_pinnacle`).
   - **Exchanges** stay in A7's `Avg` secondary, for continuity.
   - **Order of fetching:** odds are still fetched after the forecast inputs, in the same run. A
     confidence interval that crosses zero does not establish parity with the market.
6. **pBP adoption, one more condition.** Beyond A8 item 2's pooled rule, adoption also requires that
   pBP − p0 is ≤ 0 among records with λ3 ≥ 0.02. That shows any gain is not just from dropping ρ.

## Amendment A10 (2026-11-26, written before any code for it exists): S2 shadow records

Addenda TS and S2PL found that the score-driven model S2 beats the refitted Dixon-Coles D0 as a
results-only model, on 13 new leagues (−0.0024) and on untouched Premier League seasons (−0.0051,
`docs/IMPROVEMENT_BRIEF.md` §5.38–5.39). The live forecasts sit on D0 plus the xG and lineup stage,
and S2 under that stage was worse on 2022-26 (§5.18, development data). A10 asks the remaining
question prospectively:

> **Is the full model built on S2 at least as good as the live model built on D0?**

**Nothing about the live record changes.** `log.jsonl`, its candidates (p0, pE, pα, pBP, the
sources), the windows, α's fitting rule and every A2–A9 decision stay exactly as they are.

### What is logged

- **Where:** a separate append-only file, `results/prospective/shadow.jsonl`. `log.jsonl` is never
  touched by shadow code.
- **When:** in the same run, immediately after a live record is appended, for the same fixture,
  kickoff, cutoff and record role (primary or fallback). There is no shadow record without a live
  record. Shadow records start with the first live record logged after the A10 code is merged; there
  is no backfill.
- **Same inputs:** the shadow uses the live run's in-memory inputs: the same results, FPL
  appearances, availability, and the same parsed lineup snapshots. Nothing is fetched again.
- **Isolation:** shadow code runs after the live record is written, inside its own error handler, with
  a time limit of 120 seconds per run. A shadow failure writes a shadow record that names the error,
  never stops or delays the live record, and never opens a logger alert.
- **Each shadow record holds:** fixture id, kickoff, cutoff, record role, git commit, S2's static
  parameters and pre-match rates, the stage-2 coefficients, and the shadow candidates below.

### Shadow candidates

| Name | Forecast |
|---|---|
| `p0S2` | p0 with S2's rates in place of D0's: the same xG features, stage-2 form and L2 (1.0) |
| `pES2` | pE with S2's rates in place of D0's: the same lineup sources, features, L2 (100) and source averaging |
| `S2` | S2 alone (results only), for a live replication of S2 − D0 |

- **S2 settings (fixed now):** `tabet/dynamic.py` with the summer factor, unchanged. The filter runs
  over every PL match from 2015-16 (`res/history`) to the cutoff. Static parameters are fitted once
  per season, at that season's first kickoff, on all earlier matches; for 2026-27 that is the
  fit at 2026-08-21 on 2015-26.
- **Stage-2 training rows:** the live training rows with log(D0 rate) replaced by each match's
  walk-forward S2 rate (each season's matches use that season's static parameters). The stage-2
  coefficients are refitted at every run, as the live ones are.

### Decision

- **Primary: pES2 − pE**, log-loss on fixtures that have both, one record per fixture (A4's
  primary-over-fallback rule), 14-day calendar-block bootstrap, 2,000 resamples, seed 0.
- **Period:** every shadow record from the start through the end of 2027-28 (about 640 fixtures).
  At the end of 2026-27 the result is reported **descriptively only**; it cannot trigger a decision.
- **Power, stated in advance:** with a per-fixture SD of about 0.115 (addendum S2PL), the SE at 640
  fixtures is about 0.0045. The shadow cannot confirm a gain of S2PL's size (0.005); it is a safety
  check that the full model does not get worse, judged together with the two results-only tests.
- **Rule, at the end of 2027-28:**
  - **adopt** if the mean is below 0 and the upper bound is below +0.005;
  - **reject** if the lower bound is above 0;
  - **otherwise extend once**, through the end of 2028-29 (about 1,020 fixtures), then **adopt** if the
    mean is below 0 and **reject** otherwise. The weaker final rule is deliberate: two results-only
    tests already favour S2, so the shadow only has to show the full model is not worse.
- **Adopt** means S2 replaces D0 as the base of p0 and pE from the next season, through a new
  amendment written before that season. It never changes records already logged.
- **Secondary, never decisive:** p0S2 − p0; S2 − D0 (live replication); pES2 and pE against the
  Pinnacle benchmark (A8); fixtures with a promoted side in its first 10 matches; the first 56 days of
  each season.
- **Ledger:** the 2026-27 shadow records are part of the live test (family `live`); 2027-28 and the
  possible 2028-29 extension are reserved as family S2SH.

**Operations note (2026-11-26):** the Dixon-Coles low-score factors are now floored at 0 when a score
grid is built (`tabet.model.dc_correct`, used by the live forecast, backtests and the shadow log). This
changes no forecast: the smallest factor in every logged record so far is 0.902, and in the 2022-27
backtest rows 0.620. It only keeps a grid valid at rates no forecast has reached.

## Amendment A11 (2026-11-26, written before any code for it exists): kickoff records with confirmed lineups

The protocol's "operational comparison" (a second horizon close to kickoff, with confirmed lineups) was
sketched but never built. A11 builds it. It is **descriptive**: it adds a separate record; the 24-hour
records, their candidates, α, every A2–A10 decision and the shadow log are unchanged.

- **Window:** a fixture gets a kickoff record from the first logger run that finds its kickoff **10 to
  50 minutes ahead** and RotoWire showing **confirmed** XIs for both teams. Runs every 30 minutes put
  at least one run in that window; if none of them sees both XIs confirmed, the fixture has no kickoff
  record (counted and reported).
- **Lineup source:** RotoWire's lineup page, which marks an XI "confirmed" once clubs announce it
  (about an hour before kickoff). Only a confirmed XI for both sides is used; names resolve to FPL
  players as for the 24-hour records. The raw page is kept beside the others.
- **Where:** a separate append-only file, `results/prospective/kickoff.jsonl`, written by its own code
  path inside its own error handler; it never touches `log.jsonl`.
- **Candidates:**
  - **pK:** V2x (the 24-hour pE model) with RotoWire's confirmed XIs, fitted at the kickoff cutoff;
  - **p0K:** the lineup-free p0 at the same cutoff (to separate "later data" from "confirmed XIs").
- **Market:** the run's The Odds API snapshot (A7/A8, Pinnacle with Shin's margin removal) is taken
  for kickoff records as for 24-hour ones; football-data's closing odds are added once published.
- **Reported (descriptive, after gameweeks 20–29 and 20–38, with the protocol's inference methods):**
  pK against Pinnacle at the same snapshot; pK against closing odds; pK − pE (24 h); pK − p0K.
  **No decision rests on it**, and nothing live changes because of it.

**Operations note (2026-11-26): unresolved lineup names (bug fix).** The frozen definitions say an
unresolved source name "contributes the position-prior rating". The code instead dropped unresolved
names, so that player's attacking contribution was missing from the XI. Found by an outside review.
- **Affected:** one fixture, both records: Hull v Brighton (FPL 106, primary and fallback), where one
  Brighton name on RotoWire did not resolve. No other record had an unresolved name.
- **Size:** Brighton's XI feature −0.152 instead of −0.026; RotoWire's candidate 26.7 / 31.2 / 42.1 instead
  of 24.6 / 29.7 / 45.7 (reconstructed from the record's stored inputs, which reproduce the logged
  candidate exactly). pE moved by about half of that. Hull won 2-0, so the logged values scored slightly
  better than the corrected ones would have.
- **Handling:** the log is append-only and these records stay as logged and are scored as logged; this
  note is reported beside the gameweek 20-29 and 20-38 results. From this commit an unresolved name keeps
  its place with the position prior (a placeholder code with no rating), as the protocol specifies.

## Amendment A12 (2026-11-27, written before any code for it exists): squad cost in the shadow records

Addendum TV (`docs/IMPROVEMENT_BRIEF.md` §5.41) was inconclusive on PL 2008-15, with its gain coming from
squad cost (−0.0016 alone, better in 6 of 7 seasons) and none from early-season fees. The PL holdouts are
used up, so the only fair test left is prospective. A12 adds squad cost to the shadow records.

**Nothing about the live record or A10 changes.** The live log, its candidates, α, the A10 candidates
and the A10 decision stay as they are.

- **Where:** the same shadow record as A10 (`results/prospective/shadow.jsonl`), as extra candidates.
  Computed after the A10 candidates inside its own error handler; a failure is stored as `cost_error`
  and leaves the A10 candidates intact.
- **Feature (`dcost`):** for the scoring team T against opponent O, z(log(1 + squad cost of T)) −
  z(log(1 + squad cost of O)), with z taken across that season's clubs at the cutoff;
  squad cost is `tabet.transfers.squad_cost` with the point-in-time fee index (addendum TV's amendment),
  from moves dated before the cutoff's day. Each (season, day) is computed once and cached
  (`results/prospective/shadow_cost_cache.json`), so a day's values are those first computed for it.
- **Candidates** (both on the live D0 base, refitted at every run like the live stage 2):
  - **p0C:** p0's stage 2 with `dcost` added (adjusted xG + dcost, L2 = 1);
  - **pEC:** pE's stage 2 with `dcost` added (adjusted xG + XI + dcost, L2 = 100), same sources and averaging.
- **Primary: pEC − pE**, log-loss, one record per fixture (A4's rule), 14-day blocks, 2,000 resamples,
  seed 0, from the first shadow record with these candidates through the end of 2027-28. The end of
  2026-27 is reported descriptively only.
- **Rule** (FAT's thresholds, stricter than A10's because the prior evidence is only inconclusive):
  - **adopt** if the mean is ≤ −0.0005 and the upper bound is below 0;
  - **reject** if not adopted and the lower bound is above −0.0015;
  - **otherwise extend once** through 2028-29 and apply the same two rules; if still neither, squad
    cost is not adopted.
- **Adopt** means `dcost` joins the live stage 2 from the next season, through an amendment written before
  that season. **Secondary, never decisive:** p0C − p0; matches with a promoted side in its first 10;
  the first 56 days of each season.
- **Ledger:** 2026-27 records are part of the live test (family `live`); 2027-28 and the possible
  extension are reserved as family SC.

## Amendment A13 (2026-11-27, written before any code for it exists): S2 and squad cost together

A10 tests the S2 base and A12 tests squad cost, each against the live model on its own. The two are
different kinds of information (S2 changes how fast ratings follow results; squad cost brings in
spending a results-only rating cannot see), so their gains may add up. Neither has been tested together
with the other, and no historical Premier League data is left to do it (the 2008-15 closed set is used
up, `docs/data_ledger.csv`). A13 adds the combination to the shadow records.

**Nothing about the live record, A10 or A12 changes.** The live log, its candidates, α, the A10 and A12
candidates and their decision rules stay as they are.

- **Where:** the same shadow record (`results/prospective/shadow.jsonl`), as extra candidates. Computed
  after the A10 and A12 candidates inside its own error handler; a failure is stored as `s2c_error` and
  leaves the A10 and A12 candidates intact. Starts with the first shadow record logged after the A13 code
  is merged; there is no backfill.
- **Base:** S2's pre-match rates, exactly as A10 computes them (same static parameters, same filter).
- **Feature:** A12's `dcost`, exactly as A12 computes it (same cache, same point-in-time fee index).
- **Stage-2 training rows:** A10's rows (offset log of the walk-forward S2 rate) with A12's `dcost`
  added. The coefficients are refitted at every run, like the live ones.
- **Candidates:**
  - **p0S2C:** p0's stage 2 on S2's rates with `dcost` (adjusted xG + dcost, L2 = 1);
  - **pES2C:** pE's stage 2 on S2's rates with `dcost` (adjusted xG + XI + dcost, L2 = 100), same
    sources and averaging as pE; the per-source versions are stored too.
- **Primary: pES2C − pE**, log-loss, one record per fixture (A4's rule), 14-day calendar-block
  bootstrap, 2,000 resamples, seed 0, from the first record with these candidates through the end of
  2027-28. The end of 2026-27 is reported descriptively only.
- **Rule** (A12's thresholds):
  - **adopt** if the mean is ≤ −0.0005 and the upper bound is below 0;
  - **reject** if not adopted and the lower bound is above −0.0015;
  - **otherwise extend once** through 2028-29 and apply the same two rules; if still neither, the
    combination is not adopted.
- **How A10, A12 and A13 combine** (fixed now, so the three results cannot be picked from afterwards):
  1. if A13 adopts, S2 replaces D0 as the base **and** `dcost` joins the stage 2;
  2. otherwise A10 and A12 are applied as each is written. If both adopt on their own while A13 does
     not, only A10 (S2 base) is adopted: it has the stronger prior evidence (two results-only tests),
     and the combined model failing its own test is evidence against stacking the two.
  Adoption always takes effect from the next season, through an amendment written before that season,
  and never changes records already logged.
- **Secondary, never decisive:** pES2C − pES2 (does squad cost add on S2); pES2C − pEC (does S2 add with
  squad cost); p0S2C − p0; pES2C against the Pinnacle benchmark (A8); matches with a promoted side in
  its first 10; the first 56 days of each season.
- **Power, stated in advance:** A13 starts on gameweek 13 at the earliest, so it will have about 260
  fixtures by the end of 2026-27 and about 640 by the end of 2027-28. With A10's per-fixture SD for a
  candidate against pE (about 0.115 for a base change), the SE at 640 is about 0.0045, too large to
  confirm a gain of 0.002 (about 0.0036 at 1,020 fixtures after the extension). Unless the combined
  gain is several times the sum of the two historical estimates (S2PL −0.0051, TV squad cost −0.0016),
  A13 will most likely end "not adopted". This is stated so that such an ending is not read as
  evidence against the combination; the full-season descriptive reads are reported either way.
- **Ledger:** 2026-27 records are part of the live test (family `live`); 2027-28 and the possible
  extension are reserved as family S2C.

## Amendment A14 (2026-11-28, written before any code for it exists and before any defensive-contribution data has been analysed): the opponent's defensive XI (F8)

Brief item F8. Addendum DXI (`docs/IMPROVEMENT_BRIEF.md` §5.31) tested a defensive-XI block on four other
leagues with API-Football's tackles, interceptions and duels: −0.0003 [−0.0014, +0.0007], not confirmed,
with what signal there was in ball-winning. F8 asks the Premier League version with a different measure:
FPL's `defensive_contribution`, published per player and match since 2025-26 (clearances, blocks,
interceptions and tackles for defenders; the same plus recoveries for midfielders and forwards). It is
ingested as `def_contrib` (`tabet/fpl.py`) but no TaBet model or analysis has used it so far.

The PL's backtest seasons cannot be used: 2026-27 and 2027-28 are reserved by the live test and the
shadow amendments, and the data only starts in 2025-26. So the test is prospective, on the A11 kickoff
records, which carry confirmed XIs. **Nothing about the live record, A10–A13 or their decisions changes.**

### Feature (fixed now)

- **Player rating:** for player p in FPL position q (DEF or MID only; keepers and forwards are left out),
  from that player's league matches dated before the cutoff, in the current and previous season:
  r_p = (Σ def_contrib + (k/90)·μ_q) / (Σ minutes/90 + k/90), where μ_q is the league's per-90 mean for
  position q over the same earlier matches. A player with no earlier minutes gets μ_q. Positions are
  FPL's for that season.
- **Relative rating:** r_p / μ_q, so defenders and midfielders are on one scale despite FPL's different
  definitions.
- **Block:** DEFXI(T, match) = log(mean relative rating of T's starting DEF and MID ÷ T's mean of the same
  block over its previous 10 league matches with data). Set to 0 (the team's usual XI) when T has fewer
  than 5 such matches. The ratio to the team's own baseline also removes team style and possession,
  which FPL does not report.
- **In the model:** for the scoring team's goals, the opponent's DEFXI enters as `f_def_opp`.

### Development (once, on PL 2025-26, before any 2026-27 value of the feature is computed)

- **Choice (the only one):** k ∈ {300, 900, 2700} minutes, by log-loss of V2x + `f_def_opp` (L2 = 100,
  monthly refits, confirmed XIs) on 2025-26 matches where both teams have a non-zero DEFXI.
- **Withdrawal rule:** if (V2x + `f_def_opp`) − V2x on those matches is not below 0 at the chosen k, F8 is
  withdrawn before any record is logged, and the result is reported.
- **Data check, reported before fitting:** the share of 2025-26 starter rows (DEF and MID) with a
  non-missing `def_contrib`; development runs only if it is at least 95%.

### Logged candidates (in `kickoff.jsonl`, from the first kickoff record after the code is merged; no backfill)

- **pKD:** pK's stage (V2x: adjusted xG + XI features, L2 = 100) plus `f_def_opp`, refitted at each run.
  Training rows before 2025-26, which have no data, enter with `f_def_opp` = 0.
- Computed after pK inside its own error handler; a failure is stored as `def_error` and leaves pK and
  p0K intact. The record also stores both teams' DEFXI and the stage coefficients.
- **Secondary only:** pED in the 24-hour shadow records (pE's stage + `f_def_opp` from the predicted XIs,
  same sources and averaging), to see whether the block survives predicted lineups.

### Decision

- **Primary: pKD − pK**, log-loss, one kickoff record per fixture, 14-day calendar blocks, 2,000
  resamples, seed 0, from the first record with pKD through the end of 2027-28. The end of 2026-27 is
  reported descriptively only.
- **Rule** (A12's thresholds): **adopt** if the mean is ≤ −0.0005 and the upper bound is below 0;
  **reject** if not adopted and the lower bound is above −0.0015; **otherwise extend once** through
  2028-29 and apply the same two rules; if still neither, F8 is not adopted.
- **Coverage, reported:** the share of played fixtures with a kickoff record. If it is below 70% in
  2027-28, that is stated beside the result as a limitation (the kickoff records depend on RotoWire
  marking both XIs confirmed in time).
- **Power, stated in advance:** XI-feature contrasts have a per-match SD of about 0.026 (brief §6.1), so
  about 600 kickoff records give an SE of about 0.0011: enough to see a gain of about 0.002, not 0.0005.
- **Adopt** means `f_def_opp` joins the lineup stage (pE and V2x) from the next season, on whichever base
  A10/A13 have chosen, through an amendment written before that season.
- **Secondary, never decisive:** pED − pE (24-hour shadow); defenders only and midfielders only; the
  first 10 matches of each season; matches where either DEFXI is in its top or bottom 10%.
- **Ledger:** family F8. Development E0 2025-26 (reserved in this commit). Test: the 2026-27 kickoff
  records are part of the live test (family `live`); 2027-28 and the possible 2028-29 extension are
  reserved as family F8.

**Ledger note (same commit), fixing a conflict in A10, A12 and A13:** their 2027-28 and 2028-29 rows each
had a quota of 1, so whichever was scored first would have blocked the others under `tabet.ledger.claim`.
They all score different candidates in the same prospective records, so those seasons are now one closed
test set shared by S2SH, SC, S2C and F8 (quota 4). No 2027-28 data exists, so no decision is affected.
