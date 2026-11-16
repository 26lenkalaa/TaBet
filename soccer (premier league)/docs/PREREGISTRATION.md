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

## Addendum E (2026-11-05, committed before the run): Elo blend (F14)

Specified in full in the header of `prototypes/run_elo_blend.py` (model in `tabet/elo.py`).
Results-only Elo with an ordered-logit outcome mapping, blended linearly with Dixon-Coles; K and
the blend weight chosen on development seasons only. **Primary:** blend(DC, Elo) − DC on the
clean 2018-21 results (n = 1,140); confirmed if the mean is below 0 and the 14-day block
bootstrap interval excludes 0. Secondary: Elo alone − DC, and blends with V0 and V1x on recent
seasons (reused validation).

## Addendum M (2026-11-08, committed before the run): alternative team-strength models

Specified in full in the header of `prototypes/run_alt_models.py`; one file per model in
`tabet/alt/` (returning-team reset, hierarchical Bayesian Dixon-Coles with a Laplace approximation,
negative binomial and bivariate Poisson goals). Each is compared with the frozen Dixon-Coles model on
the clean 2018-21 results (n = 1,140). With five primary contrasts, a model counts as better only if
its mean is below 0 **and** the Bonferroni 99% 14-day block interval excludes 0. Early-promoted
matches, recent seasons and the effect inside the main V1x/V2x model are secondary. The only tuned
choice is the Bayesian model's first-season prior strength, picked on dev seasons. The Elo blend
was run earlier (addendum E) and is not repeated.
*Correction to addendum M before the scored results were written (2026-11-08):* the first run
printed results but crashed before saving them, and showed non-finite Bayesian forecasts on recent
seasons. Cause: the Bayesian model treated every team's first season *in the data* as the first
season of its spell, flagging all teams as promoted in the earliest season (2015-16 / 2021-22),
which makes the promoted-group offset collinear with the scoring level. Fixed to match
`promoted_team_seasons` (no team is "new" in the earliest season). This changes only the Bayesian
model; its pre-fix clean result (+0.0063 vs D0, 95% [+0.0016, +0.0109], lam_new = 100) is reported
alongside the corrected one. The other models are unchanged by the fix.
*Second correction (2026-11-08), before results were saved:* early in the first scored recent
season the training data held no promoted-team matches, so the group offset was unidentified and
its Laplace variance unbounded, giving non-finite forecasts. When no promoted-team matches are in
training, the offset is now held at 0 and not sampled (what the Dixon-Coles promoted model does).
This cannot occur on the clean 2018-21 data, whose results are unchanged.

## Addendum B (2026-11-09, committed before the run): bivariate Poisson on four other leagues

From the fifth external review. The bivariate Poisson passed on clean 2018-21 Premier League data
(addendum M, −0.0028) but did not repeat on 2024-26. A shared goal component is a property of the
goal likelihood, not of one league, so this replicates the mechanism on data no goal-distribution
model has been scored on. It is **not** Premier League evidence (§6.1 of the brief).

- **Data:** La Liga, Serie A, Bundesliga and Ligue 1, 2021-22 → 2026-27 (`res/other_leagues/`).
  These files were used before only to learn shared shots-on-target coefficients and to tune
  Dixon-Coles settings, scored on Premier League matches (`prototypes/run_multileague.py`). No
  forecast *in* these leagues has been scored for a goal-distribution comparison.
- **Models:** `tabet/alt/goal_distributions.py`, unchanged. Per league, at each matchday: the
  frozen Dixon-Coles fit (`Hyperparams()` defaults, ratings never shared across leagues); **DC** is
  its own score grid with ρ; **BP** keeps the same expected goals and replaces the grid with a
  bivariate Poisson whose shared rate λ3 is fitted at the cutoff by decay-weighted maximum
  likelihood on that league's training matches. 2021-22 is burn-in (training only).
- **Single primary:** BP − DC mean log-loss, pooled over the four leagues, on the completed
  seasons 2022-23 → 2025-26 (about 5,500 matches). Confirmed if the mean is below 0 **and** the 95%
  14-day calendar-block bootstrap interval (blocks by date across leagues, 2,000 resamples, seed 0)
  excludes 0.
- **Secondary, not decisive:** each league separately; 2026-27 so far; negative binomial − DC (the
  same run produces it); the fitted λ3 over time.
- **Consequence:** none for p0 or the primary prospective contrast. The result is read next to the
  secondary prospective contrast pBP − p0 (protocol amendment A5).

## Addendum S (2026-11-09, committed before the run): table stakes (F6)

From the fifth external review. Results only, so it can be tested on the clean 2015-21 data.

- **Feature, per side, from standings built only on results before the cutoff:** `settled` = 1 if
  the team has played at least 29 league matches and, for each boundary 1/2, 4/5, 6/7 and 17/18,
  it can no longer cross it on points: a team above a boundary is settled for it if its points
  exceed every team below the boundary's points + 3 × that team's remaining matches; a team below
  is settled for it if its points + 3 × its remaining matches are below the points of the team
  currently on the last place above the boundary. Equal points count as not settled (goal
  difference is ignored). Otherwise `settled` = 0. Two features per side: own `settled` and the
  opponent's `settled`.
- **Model:** V0 Dixon-Coles walk-forward (`Hyperparams()` defaults; 2015-16 burn-in, forecasts
  from 2016-17), then a stage-2 penalised Poisson on goals per side (`tabet.experiments._fit_poisson`)
  with offset log(V0 rate) and the two features, **L2 fixed at 10** (no tuning). Stage 2 is
  refitted at every matchday on all earlier side-rows from 2016-17 on, using their out-of-sample
  V0 rates; probabilities use the Dixon-Coles grid with that matchday's ρ.
- **Single primary:** F6 − V0 mean log-loss on 2018-19 → 2020-21 matches where at least one side
  has played 29 or more matches before the cutoff (about 300 matches). Confirmed if the mean is
  below 0 **and** the 95% 14-day block bootstrap interval excludes 0.
- **Reuse, stated plainly:** clean 2018-21 has now been scored for addenda R, D, E and M (five
  primaries). This is the last new idea sent there; from now on it takes only pre-registered
  primaries that replicate a result found elsewhere, and 2024-26 counts as development data.
- **Secondary, not decisive:** 2017-18; 2018-21 excluding matches played behind closed doors
  (from 2020-06-01); the share of settled sides; the two fitted coefficients over time.

## Addendum P (2026-11-16, committed before the run): promoted-team mapping on 2000-15 (R6)

From the sixth external review (brief §8 item 2, R6). §5.15 learned the Championship → Premier League
mapping inside the decay-weighted Dixon-Coles fit, so in effect it saw only the last few cohorts.
Here the mapping is learned separately, from every earlier promotion, on data never used before.

- **Data (fresh: never scored for any earlier decision).** football-data's E0 (Premier League)
  files for 1999-00 → 2014-15 and E1 (Championship) files for 1999-00 → 2013-14, downloaded
  2026-11-16.
  - **Cleaning:** five files had stray extra fields or a non-UTF-8 byte, so each line is cut to its
    header's width and the file re-encoded. Every season has its full fixture list (380 / 552
    matches).
  - **Shots on target:** 100% from 2000-01.
  - **Kickoff times:** none, so a match's results count only from the next date.
  - **Team ids:** division-prefixed name slugs, linked across divisions by the name slug.
- **Promoted team-season:** a team in E0 season *s* that was in E1 season *s − 1*. There are 3 per
  season, 42 for 2001-02 → 2014-15.
- **Features**, from the team's whole Championship season (known before its PL season):
  - x1, x2: opponent-adjusted shots-on-target attack and defence. Same quasi-Poisson model as §5.15
    (`fit_xg_strength` on `_sot_sides`, half-life 90 days, cutoff one day after the last E1 match).
  - x3: goal difference per match.
- **Mapping (stage 1).**
  - **Targets:** for each earlier promoted team-season, its first-PL-season attack and defence. These
    are the Dixon-Coles ratings (`Hyperparams()` defaults, no promoted prior) fitted one day after the
    last PL match of that season.
  - **Model:** two ridge regressions on standardised features with unpenalised intercepts: attack on
    [x1, x3] and defence on [x2, x3].
  - **Ridge λ:** chosen from {0.1, 1, 10, 100} by leave-one-cohort-out squared error on the
    2001-02 → 2007-08 cohorts only (21 team-seasons).
- **Model D3.**
  - **Mapping refit:** at every match date, refitted with that λ on all cohorts whose PL season had
    ended before the date.
  - **Offsets:** each team promoted this season gets fixed offsets (m_att, m_def) = the mapping's
    prediction. They are added to its attack and defence in its promoted season. The team's own
    ratings are still L2-shrunk to 0, so the mapped value acts as the prior mean.
  - **Implementation:** `DixonColes.fit` with the promoted prior on, `promo_x` = the mapped values,
    slopes fixed at 1 and shared offsets fixed at 0. Otherwise `Hyperparams()` defaults.
- **Comparators** (same walk-forward, same fixtures):
  - **D0:** `Hyperparams()`, no promoted prior.
  - **D1:** the shared promoted offset learned inside the fit (`Hyperparams(promoted_prior=True)`,
    no features), as in §5.15.
- **Walk-forward.**
  - **Fitting:** every model is refitted at each match date, on PL matches completed on earlier
    dates from 1999-00 on.
  - **Scoring:** **2008-09 → 2014-15** (2,660 matches), once.
  - 2000-01 → 2007-08 is development: λ only, and nothing else is looked at.
- **Primaries** (one family; Bonferroni over three, so **98.33%** 14-day calendar-block bootstrap
  intervals, 2,000 resamples, seed 0):
  - **P1:** D3 − D0 mean log-loss on early-promoted matches (a promoted side within its first 10 PL
    matches of the season; about 210).
  - **P2:** D3 − D1 on the same matches.
  - **P3:** D3 − D0 on all matches.
- **Decision.**
  - **Confirmed** if P1 and P2 are both below 0 with intervals excluding 0, **and** P3's mean is
    ≤ 0. If confirmed, D3 becomes a candidate for next season's live model, under its own amendment.
    Nothing changes this season.
  - **Stop rule:** the line is closed until squad-level data exists live if:
    - P1's interval includes 0; and
    - the final mapping's slopes (fitted on all cohorts to 2013-14, 2,000 bootstrap resamples of
      team-seasons) all have 95% intervals including 0.
  - **R3:** run only if the slopes are clearly non-zero.
- **Secondary, not decisive:** each season separately; D1 − D0; the mapping's coefficients over
  time; λ's leave-one-cohort-out curve.
- **Data ledger.**
  - **PL 2000-01 → 2007-08:** development for this line.
  - **PL 2008-09 → 2014-15:** a closed test set with a quota of **three primary families**. This
    addendum uses the first. The other two must be declared in their own addenda before they run.
