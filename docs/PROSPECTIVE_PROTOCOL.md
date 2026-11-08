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
