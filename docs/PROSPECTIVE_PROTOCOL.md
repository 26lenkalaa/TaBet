# Prospective protocol: predicted lineups at a fixed early horizon

Written 2026-11-05, before any 2026-27 gameweek-10 match. It answers one question:

> **Does a learned, conservative use of predicted lineups beat the lineup-free forecast (V1x) at
> 24 hours before kickoff?**

## What is logged

- **Horizon:** each fixture is logged once, by the **first run that finds it 24–26 h before
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
- **Unavailable** means FPL `status != "a"` **and** `chance_of_playing_next_round < 50`. Doubtful
  players (50–75%) are kept, and loans count as unavailable.
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
