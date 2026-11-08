# Pre-registration: confirming the XI-strength effect on unseen 2026-27 matches

Registered 2026-11-05, before any of the matches below were played. The 2022-26 seasons have now
been looked at repeatedly, so they count as development evidence. This document fixes one
confirmatory test on fixtures nobody has seen.

- **Fixtures:** every 2026-27 Premier League match kicking off **after 2026-11-05T23:59Z**
  (gameweek 10 onwards), up to the end of the season. Matches that are abandoned or not played are
  excluded.
- **Frozen configuration:** the code at the commit that adds this file. Settings: Dixon-Coles
  `Hyperparams()` defaults; `FeatureParams()` defaults; stage-2 L2 values from
  `results/fpl_v0_v5/manifest.json` (V1: 1.0, V2: 100.0); `MIN_TRAIN_MATCHES = 300`;
  confirmed-lineup mode.
- **Single primary contrast:** V2 (XI strength, position prior) minus V1 (team xG form), on mean
  outcome log-loss.
- **Decision rule:** the XI-strength effect is confirmed if the mean difference is below 0 **and**
  the 95% calendar-block bootstrap interval (14-day blocks, 2,000 resamples, seed 0) excludes 0.
  Otherwise it is reported as not confirmed. Nothing gets re-tuned on these fixtures.
- **Reported alongside, not used for the decision:** V0, V3, V4 and V5; Brier, RPS and goal
  deviance; both market benchmarks (labelled by information timing); draw totals.
- **Expected power (an honest caveat):** about 290 matches remain. With the test-season paired SD
  (0.047), a true −0.004 effect would be detected only about 30% of the time. A null result here
  would therefore be weak evidence against the effect.
- **Prospective logging:** for this to be a genuine prospective test, predictions should be
  generated and stored before each matchday, with cutoff, feature snapshot and model version.
  Until that logger exists, the evaluation will be a replay with the frozen code, and must be
  labelled that way.

## Addendum (2026-11-05, before any gameweek-10 kickoff): secondary contrasts

The primary contrast above (V2 − V1, confirmed XIs) is unchanged. The model has since been
revised (opponent-adjusted xG, §5.6 of the brief), so these **secondary** contrasts are specified
now, on the same fixtures, with the same 14-day block bootstrap. They do not change the primary
decision.

- **S1: V2x − V1x.** XI strength conditional on adjusted xG. Settings: xG half-life 90 days, V1x
  L2 = 1, V2x L2 = 100 (frozen V2 value).
- **S2: V2x − V2.** Adjusted xG conditional on XI features.
- **S3: V1x − V1.** Adjusted xG replacing raw xG form.

Predicted-lineup forecasts are covered by `docs/PROSPECTIVE_PROTOCOL.md`, not by this document.

## Addendum R (2026-11-05, written and committed before any result was computed): pre-xG-era replication

An independent replication of the XI-strength effect on seasons never used for any choice so far.

- **Data.** Results: football-data E0 2015-16 → 2020-21 (`res/history/`, all validated). Players:
  FPL archive 2016-17 → 2020-21 (`tabet/history.py`); all 1,900 fixtures join one-to-one with
  identical scores. Starters are inferred (11 players with most minutes per side; no `starts`
  exists before 2022). Kickoff times exist only from 2019-20; earlier matchdays are grouped by date,
  so same-day results are never used.
- **Player rating mapping (frozen, no tuning).** xG + xA is replaced by FPL **threat + creativity**,
  equal weights, through the unchanged V2 feature code (decayed per 90, shrunk to the position
  prior with 900 minutes, XI ratio = log(XI rating / usual XI rating)). The ratio does not depend
  on the scale of the metric, so the equal weighting is the only mapping choice.
- **Models.** Dixon-Coles at the frozen settings (ξ = 0.004, L2 = 10), walk-forward per matchday,
  with 2015-16 as burn-in. Stage 2 as everywhere else (`MIN_TRAIN_MATCHES = 300`).
  - **B** (base): + opponent-adjusted shots-on-target strength (`f_sot_att`, `f_sot_def_opp`;
    §5.11 model, half-life 90 days), because xG does not exist in these seasons.
  - **X**: B + XI strength (`f_xi_position`, `f_backline_opp`).
- **Development / evaluation split.** 2016-17 is burn-in for player ratings and stage 2 (rows
  built, not scored). Stage-2 L2 for B and for X is chosen separately on **2017-18** from
  {1, 10, 100}. **2018-19, 2019-20 and 2020-21 (1,140 matches) are scored once.**
- **Primary contrast:** X − B mean log-loss on 2018-21. Confirmed if the mean is below 0 **and** the
  14-day calendar-block bootstrap 95% interval (2,000 resamples, seed 0) excludes 0.
- **Reported, not used for the decision:** X − B per season; B − Dixon-Coles; X − B by
  XI-deviation band (cut points from 2017-21); 2020-21 flagged as played without crowds.
- **Expected power.** With a paired SD near 0.047, the SE on 1,140 matches is about 0.0014, so a
  true −0.004 effect would be detected about 80% of the time; a −0.002 effect about 30%.

## Addendum T (2026-11-05, committed before the run): match-level transfer priors

Specified in full in the header of `prototypes/run_transfer_match_test.py`. In short: priors for
every pre-season-eligible arrival (>= 600 previous-league minutes, no condition on PL minutes) from
the frozen P1 mapping trained only on earlier cohorts; the prior replaces the position-prior mean in
the unchanged V2 rating and fades with PL minutes; nothing is tuned (stage-2 L2 = 100).
**Primary:** V2xT − V2x on 2024-25 + 2025-26; **co-primary:** the same on matches where an arrival
starts within his first 10 PL appearances. Confirmed if the mean is below 0 and the 14-day block
bootstrap interval excludes 0. Disclosure: the player-level gate saw the 2023-26 cohorts, and 2024-26
is reused validation.

## Addendum D (2026-11-05, committed before the run): dynamic team strengths

Specified in full in the header of `prototypes/run_dynamic_ratings.py` (model in `tabet/dynamic.py`).
Score-driven attack/defence strengths (Koopman & Lit 2019 style) with weekly persistence and an
offseason factor, same Poisson + Dixon-Coles goal model, static parameters refitted each season
on earlier matches only. **Primary:** S2 − frozen Dixon-Coles on the clean 2018-21 results
(n = 1,140); confirmed if the mean is below 0 and the 14-day block bootstrap interval excludes 0.
Early-season, shorter-half-life, no-summer and recent-season contrasts are secondary; the stage-2
comparison with adjusted xG is reported, not decisive.

## Addendum M (2026-11-08, committed before the run): alternative team-strength models

Specified in full in the header of `prototypes/run_alt_models.py`; one file per model in
`tabet/alt/` (returning-team reset, hierarchical Bayesian Dixon-Coles with a Laplace approximation,
negative binomial and bivariate Poisson goals). Each is compared with the frozen Dixon-Coles model on
the clean 2018-21 results (n = 1,140). With five primary contrasts, a model counts as better only if
its mean is below 0 **and** the Bonferroni 99% 14-day block interval excludes 0. Early-promoted
matches, recent seasons and the effect inside the main V1x/V2x model are secondary. The only tuned
choice is the Bayesian model's first-season prior strength, picked on dev seasons. The Elo blend
was run earlier (addendum E) and is not repeated.
