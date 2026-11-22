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

## Addendum G (2026-11-16, committed before any code for it exists): boosted stage 2 on the big four

From the seventh external review (its item 7c and experiment 1). Question: does a shallow
gradient-boosted adjustment to Dixon-Coles goal rates beat a pooled linear adjustment on identical
inputs, on league-seasons no TaBet experiment has ever used?

- **Data.**
  - **Source:** football-data's La Liga (SP1), Serie A (I1), Bundesliga (D1) and Ligue 1 (F1) files,
    read through `tabet.fd_cache` (Supabase).
  - **Seasons:** 2005-06 → 2020-21. D1 starts in 2006-07, its first season with shots on target. These
    four leagues' shots on target are ≥ 90% complete in every season used (checked 2026-11-16).
  - **Untouched:** the ledger (`docs/data_ledger.csv`) shows no earlier use of these four leagues'
    2018-19 → 2020-21 seasons. 2017-18 was used once, in a Wyscout prototype, and is used here only as
    training data.
  - **Usable rows:** both teams have at least 5 earlier league matches with shots on target.
- **Base model.** Dixon-Coles per league (`Hyperparams()` defaults), refitted weekly on matches
  completed before that week's first kickoff date, over the previous 3 years. Results count only from
  the next date. Ratings never cross leagues.
- **Inputs per side-row** (identical for both stage-2 models; earlier dates only, same league):
  - f1: own shots-on-target-for form.
  - f2: the opponent's shots-on-target-against form.
  - f3: own goals-for form.
  - f4: the opponent's goals-against form.

    f1–f4 are time-decayed (half-life 120 days), as a log ratio to the league's 365-day average, as
    in §5.29.
  - f5: relative strength, log DC rate (own) − log DC rate (opponent's).
  - f6, f7: own and the opponent's rest days since their previous league match: log(1 + min(days,
    21)).
  - f8, f9: own and the opponent's matches played this season, min(n, 10) / 10.
  - f10: home indicator.

  Inputs are standardised on each fit's training rows only.
- **Models** (stage 2 on Poisson goals, with the log DC rate as offset):
  - **DC:** no adjustment.
  - **LIN:** pooled penalised Poisson regression (`tabet.experiments._fit_poisson`).
    L2 ∈ {0.1, 1, 10, 100}.
  - **GBM:** scikit-learn 1.9.1 `HistGradientBoostingRegressor(loss="poisson")`, fitted to goals ÷
    DC rate with sample weight = DC rate. This has the same likelihood as a Poisson model with offset.
    - **Tuned settings:** `max_depth` ∈ {2, 3}; `min_samples_leaf` ∈ {50, 200}.
    - **Fixed settings:** `learning_rate` 0.05, `l2_regularization` 1.0, `random_state` 0.
    - **Boosting rounds:** chosen at each refit by chronological early stopping. Up to 1,000 rounds;
      stop when the Poisson deviance on the most recent 15% of that fit's training rows has not
      improved for 50 rounds. scikit-learn's own random validation split is not used.
- **Refits and timing.**
  - Both stage-2 models refit at the start of each calendar month, on all four leagues' side-rows
    from dates before the refit date (one global clock). They then forecast that month's matches.
  - **Probabilities:** the Dixon-Coles score grid with that week's ρ.
- **Periods.**
  - **Training:** from 2005-06.
  - **Development:** 2014-15 → 2016-17. These choose LIN's L2 and GBM's settings, by mean log-loss.
    Nothing else is chosen.
  - **2017-18:** training only.
  - **Test:** **2018-19 → 2020-21**, scored once with the frozen settings.
- **Primary** (two co-primaries, both required). For each league, take the mean paired log-loss
  difference on the test seasons, then average the four leagues with equal weight:
  - **C1:** GBM − LIN.
  - **C2:** GBM − DC.
- **Inference.** 14-day calendar-block bootstrap, with blocks spanning all four leagues together,
  2,000 resamples, seed 0, 95% intervals.
- **Decision.** **Confirmed** only if **C1 ≤ −0.001** with its upper bound below 0, **and** C2's upper
  bound is below 0.
  - 0.001, about a fifteenth of our Dixon-Coles-to-market gap, is fixed now as the smallest gain
    worth the added complexity.
  - Anything else is not confirmed.
- **Secondary, never decisive:**
  - LIN − DC;
  - each league; each season;
  - excluding matches played behind closed doors (from 2020-03-01);
  - early-season matches (either side within its first 10 league matches);
  - GBM's choices over time.
- **Not done here:** no Premier League scoring. A Premier League improvement claim would need its own
  frozen prospective comparison in a later season.
- **Dependency:** scikit-learn==1.9.1 is added to `requirements-ml.txt` only. The logger never
  installs it.
- **Ledger:** the rows for family G are added to `docs/data_ledger.csv` in this commit. The test
  seasons are **reserved**, and the script must call `tabet.ledger.claim` before scoring them.

## Addendum DXI (2026-11-16, committed before any player data for it has been looked at): a defensive-XI block

The seventh review's experiment 2. Question: does a fixed block describing the starting XI's
**defensive** output add information beyond an **attacking**-XI rating, on leagues no TaBet
experiment has used? Nothing about the live Premier League model changes.

- **Data.**
  - **Source:** API-Football (`tabet/api_football.py`, stored in Supabase). Per-player match stats:
    `games.minutes`, `games.position` (G/D/M/F), `games.substitute`, `shots.total`, `passes.key`,
    `tackles.total`, `tackles.interceptions`, `duels.total`, `duels.won`. Per-team match stats:
    `Ball Possession`. Plus the starting lineups and final scores.
  - **Development:** La Liga, Serie A, Bundesliga and Ligue 1, 2022-23 → 2024-25. Spent as test
    data by addendum B, and usable for development under the ledger.
  - **Test:** Eredivisie, Primeira Liga, Belgian Pro League and Scottish Premiership, 2022-23 →
    2024-25. **Untouched** per `docs/data_ledger.csv` on 2026-11-16.
  - **Run condition:** the test runs only once at least 95% of each test league-season's finished
    fixtures are stored with player stats. That check, and a check of the provider's field
    definitions (for example whether `duels` mixes aerial and ground contests), happen before any
    model is fitted, and are reported.
- **Base:** Dixon-Coles per league (`Hyperparams()` defaults), refitted weekly on that league's
  API-Football results from earlier dates, over the previous 3 years. A match is used once its
  league has at least 150 earlier matches and both teams have at least 5 earlier matches with
  player stats. Ratings never cross leagues.
- **Player ratings** (from earlier dates only, within the league):
  - **attack:** (shots + key passes) per 90 minutes, shrunk toward that position's league mean with
    a prior weight of `k_att` minutes;
  - **ball-winning:** (tackles + interceptions) per 90 minutes, possession-adjusted (multiplied by
    50 / the opponent's possession % in each match), shrunk toward the position mean with
    `k_def` minutes;
  - **duels:** (won + `a` × position mean rate) / (total + `a`).
- **XI blocks**, each as the log ratio of this match's starting XI to the team's own average for the
  same block over its previous 10 league matches:
  - **ATT** = the mean attack rating of the starting outfielders;
  - **BW** = the mean ball-winning rating of the starting defenders and midfielders;
  - **DUEL** = the mean duel rate of the starting defenders and midfielders.
- **Models** (pooled penalised Poisson stage 2 on goals, offset the log DC rate, refitted monthly on
  all eight leagues' side-rows from earlier dates):
  - **BASE:** the scoring side's ATT;
  - **DEF:** BASE plus the **opponent's** BW and DUEL.
- **Development choices** (only these, by equal-weight mean log-loss over the four development
  leagues, monthly walk-forward on 2022-25):
  - for BASE: L2 ∈ {1, 10, 100} and `k_att` ∈ {300, 900, 2700};
  - then for DEF, with BASE's choices fixed: `k_def` ∈ {300, 900, 2700}, `a` ∈ {20, 60} and its own
    L2 ∈ {1, 10, 100}.
- **Primary.** DEF − BASE: the mean paired log-loss difference per test league, averaged over the
  four test leagues with equal weight, on 2022-23 → 2024-25. Inference: 14-day calendar blocks
  spanning all four leagues, 2,000 resamples, seed 0, 95% interval.
- **Decision.** **Confirmed** only if DEF − BASE ≤ −0.001 **and** its upper bound is below 0.
  - The 0.001 threshold is fixed now, as for addendum G.
  - If confirmed, the block becomes a candidate for the Premier League XI model. That requires its
    own prospective test, with FPL data in a later season.
- **Secondary, never decisive:**
  - each league and each season;
  - BW alone and DUEL alone;
  - early-season matches;
  - BASE − DC, which shows how much the attacking block alone adds.
- **Ledger:** family DXI. Development on SP1/I1/D1/F1 2022-25; test on N1/P1/B1/SC0 2022-25, all
  reserved in this commit. The script must call `tabet.ledger.claim` before scoring.

## Addendum NLX (2026-11-17, committed before any code for it exists and before its test seasons are loaded; revised the same day after the eighth review, still before any code or test data): nonlinear stage 2 with player-level inputs

The retry of the neural network (brief §5.26–§5.29) and of boosting (addendum G, §5.30). Both
reached parity with the linear stage 2 on **team-level** inputs. The new question is whether a
nonlinear model gains once **player-level XI inputs** are available, which give interactions (a
strong attack against a weak ball-winning midfield) for it to find. Nothing about the live Premier
League model changes.

- **Prior expectation (written down so a null is read correctly):** low, especially for the
  network. Development refits train on roughly 6,000-8,000 big-four matches and the test-phase pool
  is about 25,000, below the 30,000-40,000 at which §5.29 found the network only reached parity on
  simpler inputs. DXI found the ball-winning main effect to be about 0.0005, and interactions built
  from weak main effects are usually weaker still.
- **Data:** API-Football only (`tabet/api_football.py`, stored in Supabase): results, team match
  statistics (`Shots on Goal`, `Ball Possession`), starting lineups and per-player match statistics.
  League matches only (status FT).
  - **Training only** (inputs and history, never scored): La Liga, Serie A, Bundesliga, Ligue 1
    2015-16 → 2018-19; Eredivisie, Primeira Liga, Belgian Pro League, Scottish Premiership
    2015-16 → 2019-20 and 2022-23 → 2024-25.
  - **Development:** La Liga, Serie A, Bundesliga, Ligue 1, 2019-20 → 2024-25.
  - **Test:** Eredivisie, Primeira Liga, Belgian Pro League, Scottish Premiership, **2020-21,
    2021-22 and 2025-26**. **Untouched** per `docs/data_ledger.csv` on 2026-11-17. A closed test set
    with a quota of two families: NLX and XIA (below).
  - The development phase loads no data from the four test leagues at all (their history runs
    through the test seasons). The test phase loads all of the above.
  - **Run condition:** at least 95% of each test league-season's finished matches stored with player
    statistics (2020-21 onward; Belgian and Scottish player statistics start in 2020). Checked and
    reported before any model is fitted.
  - **Known handicap:** development is on the big four, the test on four smaller leagues. That is a
    transfer handicap (not leakage), and it probably hurts the flexible models more, since their
    settings are tuned on big-four volumes. It is accepted, not corrected.
- **Base:** Dixon-Coles per league (`Hyperparams()` defaults), refitted weekly on that league's
  earlier matches over the previous 3 years, once the league has at least 150 earlier matches.
- **Inputs, identical for all three models** (per side-row: the scoring side and its opponent; all
  from earlier dates only, within the league):
  - **Team form (addendum G's ten):** shots-on-target and goals for and against, decayed with a
    120-day half-life and scaled to the league's previous-365-day average; DC relative strength;
    rest days for both sides; games played this season for both sides (capped at 10); home.
  - **XI blocks (addendum DXI's definitions and its frozen development settings, `k_att` 300,
    `k_def` 2700, `a` 20):** ATT, BW and DUEL for **both** the scoring side and the opponent: six
    inputs, each the log ratio of this match's starting XI to the team's mean over its previous
    (up to) 10 matches, i.e. deviation from the usual XI, not its level.
  - **League indicators:** one indicator per league (eight), so a pooled fit is not steered by
    big-four levels.
  - **Eligibility, identical for every model:** a DC rate exists; both sides have at least 5 earlier
    matches with shots data and at least 5 earlier matches with player statistics (the minimum XI
    history); all inputs are finite. The number of eligible matches per league-season is reported.
- **Models** (Poisson on goals with offset log DC rate; pooled over all loaded leagues' side-rows
  from earlier dates; refitted monthly; standardised inputs; **no model falls back to Dixon-Coles**):
  - **LIN:** penalised linear, unpenalised intercept, L2 ∈ {0.1, 1, 10, 100}.
  - **GBM:** addendum G's model. scikit-learn `HistGradientBoostingRegressor(loss="poisson")`; the DC
    offset enters exactly through the target y/μ_DC with sample weight μ_DC; learning rate 0.05,
    `l2_regularization` 1, `random_state` 0; built-in early stopping **off**. Depth ∈ {2, 3} and
    minimum leaf ∈ {50, 200}.
  - **NN:** `tabet/neural.py` (`_fit_predict`): two hidden layers, last layer initialised at zero
    (so it starts at Dixon-Coles), AdamW, learning rate 0.01, up to 300 full-batch epochs, mean of
    5 seeds (seeds 0-4); hidden ∈ {16, 64} and weight decay ∈ {0.001, 0.01}. Its Dixon-Coles
    fallback is switched **off**, so no model has that safety valve.
  - **Stopping, identical for GBM and NN:** each refit holds out the most recent 15% of its training
    rows **chronologically**, trains on the rest, and keeps the round (GBM, up to 1,000, patience
    50) or epoch (NN, checked every 10) with the lowest held-out Poisson deviance.
  - A month is scored only if it has at least 500 earlier side-rows to train on (the same months
    for every model).
- **Development choices** (only these), by equal-weight mean log-loss over the four development
  leagues, monthly walk-forward on 2019-20 → 2024-25:
  - each model's own grid;
  - then **which nonlinear model goes to the test**: GBM or NN, whichever has the lower development
    log-loss with its chosen settings.
- **Minimum detectable effect,** computed at the end of development and recorded in `dev.json`
  before the test runs: (1.96 + 0.84) × SE, with SE the 14-day-block bootstrap SE of the chosen
  model − LIN on development, scaled by √(development matches / expected test matches). A null
  smaller than this is read as "not detectable at this size". On addendum G's SE the expectation is
  about 0.004.
- **Primary (one contrast): NL − LIN,** where NL is the model chosen on development. For each test
  league, the mean paired log-loss difference over its test matches; then the four leagues
  averaged with equal weight.
- **Inference.** 14-day calendar blocks spanning all four leagues together, 2,000 resamples,
  seed 0, two-sided 95% percentile interval (so the upper bound is a one-sided 97.5% bound).
- **Decision.**
  - **Confirmed** only if the mean is **≤ −0.001** and the 95% upper bound is below 0. If
    confirmed, NL becomes a candidate for the Premier League stage 2; adopting it needs its own
    frozen prospective comparison in a later season.
  - **Closed** if not confirmed **and** the 95% lower bound is above −0.003 (a gain larger than
    0.003 is ruled out). The nonlinear stage-2 line is then closed until a new kind of input exists
    (for example player xG).
  - **Inconclusive** otherwise: not confirmed, but a practically relevant gain is not ruled out. The
    line is parked, not refuted.
  - No secondary can rescue a failed primary; in particular excluding 2020-21 (behind closed doors)
    is descriptive only.
- **Secondary, never decisive:**
  - the model not chosen on development, minus LIN;
  - LIN − DC, GBM − DC, NN − DC;
  - **LIN-X − LIN:** LIN plus two named products, own ATT × opponent BW and own ATT × opponent DUEL
    (same L2 as LIN), to tell whether a nonlinear gain is just these interactions;
  - each league; each test season; excluding 2020-21;
  - early-season matches (either side within its first 10 league matches);
  - eligible matches per league-season; GBM's rounds; the NN's chosen epochs.
- **Ledger:** family NLX, rows in this commit, test seasons **reserved** (quota 2, shared with XIA).
  The script must call `tabet.ledger.claim` for every role before loading the data.

## Addendum XIA (2026-11-17, committed with the revised NLX, before any code or test data): the attacking-XI block, confirmatory

The eighth review's suggestion. In DXI the attacking-XI block (BASE − DC) gained about 0.001 on both
development and test, as a secondary. This tests it as a primary, on NLX's untouched test cells in
the same run, with its own error control. Nothing in it adapts to NLX's results.

- **Data, periods, base, eligibility, stage-2 fitting and inference:** exactly NLX's.
- **What the block measures:** deviation from the usual XI, not its level (Dixon-Coles already
  absorbs a team's level): the six XI inputs as defined in NLX (log ratio to the team's previous
  up-to-10 matches).
- **Models:** LIN from NLX (G's ten team-form features, league indicators and the six XI inputs)
  versus **LIN−XI**, the same model without the six XI inputs. Each has its own L2 from {0.1, 1, 10,
  100}, chosen on NLX's development seasons by the same rule.
- **Primary:** LIN − LIN−XI, the equal-weight mean over the four test leagues.
- **Decision.** **Confirmed** only if the mean is ≤ −0.001 and the 95% upper bound is below 0. If
  confirmed, the XI block's deviation form is supported across leagues, and its Premier League use
  is judged by the live log (V2 already uses an XI layer).
- **Main secondary: XI-deviation bands.** Each match's deviation is the larger of the two sides'
  |ATT| log ratios. Bands are terciles of that value on the development rows, fixed in `dev.json`
  before the test. The Premier League effect sat in high-deviation matches (§5.16), so the top band
  is the one to watch.
- **Other secondaries:** each league; each test season; early season.
- **Ledger:** family XIA, rows in this commit. Training and development cells as NLX; test cells
  shared with NLX as a closed set with a quota of two. The script calls `tabet.ledger.claim` for
  XIA separately.

## Ledger note: sealed holdout (2026-11-17, before any of it is collected)

The eighth review's advice: keep a fresh cross-league holdout while bulk collection is cheap. Before
API-Football collects them, these cells are registered in `docs/data_ledger.csv` as family
**SEALED**, role test, status reserved, quota 4:

- Austrian Bundesliga (AUT), Swiss Super League (SWZ), Danish Superliga (DNK): 2015-16 → 2025-26
  (never used by any experiment);
- Greek Super League (G1), Süper Lig (T1), Serie B (I2), 2. Bundesliga (D2), Ligue 2 (F2), Segunda
  División (SP2): 2020-21 → 2025-26 (their earlier seasons were used by the learning curve, §5.29).

What this means, enforced by `tabet.ledger.claim`:
- No one may use these cells for training or development, and no one looks at them, even as
  summary statistics, outside a pre-registered test.
- Up to three pre-registered families may score a test on them (SEALED itself takes the fourth
  place in the quota). Each must register its own test rows first.
- They are expected to transfer to the Premier League less well than the leagues already used.

## Amendment to NLX and XIA (2026-11-17, before any model was fitted or any test result seen): Belgian 2020-21 excluded

The test phase's run condition stopped the first test run before any model was fitted: Belgian Pro
League 2020-21 had player statistics for 88 of 332 finished matches (27%; every other test
league-season 97-100%). All 332 matches' details were fetched. API-Football has no player statistics
for that season before February 2021 (0 of 225 matches from August 2020 to January 2021; 88 of 107
from February to May 2021), so the gap is the provider's and re-fetching cannot close it.

- **Change:** B1 2020-21 is removed from both families' test cells and becomes **training** for both
  (history for B1 2021-22's ratings and Dixon-Coles fits, and earlier stage-2 rows); it is never
  scored. Belgium is still one of the four equally weighted test leagues, scored on 2021-22 and
  2025-26.
- **Basis:** only the completeness check, which reads no results. Nothing else changes.

## Addendum FAT (2026-11-19, committed before any code for it exists and before any sealed data is loaded): rest, congestion and XI fatigue (F1, F2)

The eighth review's pick for the new data (brief §8; F1 and F2 in the feature list). Every model so
far has seen only **league** matches, so a team that played a Champions League or cup match three
days earlier looks rested. The question: does knowing each side's full fixture load (all
competitions) improve the forecast beyond its league-only rest? This is the first family to score
on the sealed holdout. Nothing about the live Premier League model changes.

- **Prior expectation (written down so a null is read correctly):** low to moderate, and small if
  present, likely under 0.001 in log-loss. Clubs rotate for congested weeks, and the lineup already
  shows the rotation. The best case is in top flights with European football.
- **The injury lists are not used.** The check of 2026-11-19 (PR #77) found them reconciled after
  the match.
- **Data:** API-Football only (`tabet/api_football.py`, stored in Supabase): fixture dates,
  results, team match statistics (`Shots on Goal`) and starting lineups. **No player statistics are
  needed.** That is deliberate: the sealed leagues' player statistics are patchy (Greece only from
  2022-23, per the coverage flags of 2026-11-19, which are metadata).
  - **Counted competitions:**
    - the team's league;
    - the Champions League (2), Europa League (3) and Conference League (848), qualifying rounds
      included;
    - the domestic cups FA Cup (45), League Cup (48), Copa del Rey (143), Coppa Italia (137),
      DFB Pokal (81), Coupe de France (66), KNVB Beker (90), Taça de Portugal (96), Belgian Cup
      (147), Scottish FA Cup (181), Scottish League Cup (185), Austrian Cup (220), Schweizer Cup
      (209), DBU Pokalen (121), Greek Cup (199) and Türkiye Kupası (206).
    - **Not counted (known gaps):** super cups, the Taça da Liga, the Coupe de la Ligue (ended in
      2020), the Club World Cup, friendlies and national-team matches.
    - The last five cups' ids were checked against the API's own names on 2026-11-19. They are
      collected before the test (they involve sealed teams, so they are treated as sealed: never
      looked at outside the test).
  - **Training only** (inputs and history, never scored): La Liga, Serie A, Bundesliga, Ligue 1,
    Eredivisie, Primeira Liga, Belgian Pro League and Scottish Premiership, 2015-16 → 2020-21; Greek
    Super League, Süper Lig, Serie B, 2. Bundesliga, Ligue 2 and Segunda División, 2015-16 → 2019-20.
  - **Development:** the same eight leagues, 2021-22 → 2025-26. All of these cells are spent test
    or development sets of earlier families, which the ledger allows for development.
  - **Test (the sealed holdout):** Austrian Bundesliga, Swiss Super League, Danish Superliga, Greek
    Super League, Süper Lig, Serie B, 2. Bundesliga, Ligue 2, Segunda División, **2020-21 →
    2025-26 scored**.
    - The Austrian, Swiss and Danish cells for 2015-16 → 2019-20 are loaded as history only and
      never scored. They are part of the sealed set, so they are registered as this family's test
      use.
    - The development phase loads nothing from the nine test leagues or their five new cups.
  - **Run condition,** checked and reported before any model is fitted (as the NLX amendment did):
    - for each scored league-season, at least 95% of finished league matches have lineups and both
      teams' shots on target;
    - at least 90% of counted non-league fixtures involving that league's teams have lineups.
    - A failing league-season stops the run. It is then excluded by a written amendment before any
      fitting.
- **Base:** Dixon-Coles per league (`Hyperparams()` defaults), refitted weekly on that league's
  earlier matches over the previous 3 years, once the league has at least 150 earlier matches.
- **Inputs** (per side-row: the scoring side and its opponent; earlier dates only):
  - **BASE block:** addendum G's ten team-form features (shots on target and goals for and against,
    decayed with a 120-day half-life and scaled to the league's previous-365-day average; DC
    relative strength; **league-only** rest days for both sides; games played this season for both
    sides, capped at 10; home). Plus one indicator per league.
  - **FAT block** (six inputs, three for each side, all from the counted competitions):
    - **rest:** days since the team's previous counted match, capped at 14;
    - **load14:** the team's counted matches in the previous 14 days;
    - **xi7:** for this match's starting XI, the mean number of counted starts each player made in
      the previous 7 days. A player with no recorded start counts 0. It uses this match's lineup, so
      it is a lineup-time input, like V2's XI layer.
  - **Eligibility, identical for both models:**
    - a DC rate exists;
    - both sides have at least 5 earlier league matches with shots data;
    - this match has both lineups;
    - all inputs are finite.
    - The number of eligible matches per league-season is reported.
- **Models** (Poisson on goals with offset log DC rate; pooled over all loaded leagues' side-rows
  from earlier dates; refitted monthly; standardised inputs; unpenalised intercept; no fallback):
  - **LIN:** the BASE block;
  - **LIN+FAT:** the BASE block plus the FAT block.
  - Each model has its own L2 from {0.1, 1, 10, 100}. A month is scored only if it has at least 500
    earlier side-rows, the same months for both models.
  - Only linear models: the nonlinear stage 2 is closed (NLX).
- **Development choices (only these):** each model's L2, by equal-weight mean log-loss over the
  eight development leagues, monthly walk-forward on 2021-22 → 2025-26.
- **Minimum detectable effect,** recorded in `dev.json` before the test, as in NLX: (1.96 + 0.84)
  × SE, with SE the 14-day-block bootstrap SE of LIN+FAT − LIN on development, scaled by
  √(development matches / expected test matches).
- **Primary (one contrast): LIN+FAT − LIN.** For each test league, the mean paired log-loss
  difference over its scored matches; then the nine leagues averaged with equal weight.
- **Inference:** 14-day calendar blocks spanning all nine leagues together, 2,000 resamples, seed
  0, two-sided 95% percentile interval.
- **Decision.** The thresholds are half of NLX's. The expected effect is smaller, and the test has
  roughly 15,000 scored matches, against NLX's 3,109.
  - **Confirmed** only if the mean is **≤ −0.0005** and the 95% upper bound is below 0. Then FAT
    becomes a candidate for the Premier League stage 2. Live use needs a feed of other-competition
    fixtures and lineups. Dates are public, but the free API-Football plan excludes the current
    season. Adoption needs its own frozen prospective comparison.
  - **Closed** if not confirmed **and** the 95% lower bound is above −0.0015. The fatigue line is
    then closed.
  - **Inconclusive** otherwise.
  - No secondary can rescue a failed primary.
- **Secondary, never decisive:**
  - F1 only: LIN plus rest and load14, minus LIN;
  - F2 only: LIN plus xi7, minus LIN (own L2s, chosen on development);
  - top flights (AUT, SWZ, DNK, G1, T1) versus second tiers (I2, D2, F2, SP2);
  - **European hangover:** matches where either side played a European match in the previous 4
    days;
  - each league; each test season; excluding 2020-21 (behind closed doors and unusually congested);
  - counted fixtures per team-season, and the share with lineups.
- **Ledger:** family FAT. Rows are in this commit; the test cells are added to SEALED's closed set
  (quota 4: SEALED itself and FAT, leaving two). The script calls `tabet.ledger.claim` for every role
  before loading data.

## Amendment to FAT (2026-11-22, before any model was fitted or any test result seen): nine sealed league-seasons excluded

The test phase's run condition stopped the first test run before any model was fitted
(`results/fat/test_check.json`). Nine of the 54 scored league-seasons fell short:

| League-season | Shortfall |
|---|---|
| SWZ 2021-22 → 2025-26 | Lineups for 76-85% of counted cup and European fixtures |
| DNK 2020-21 | 66% |
| G1 2024-25 | 88% |
| F2 2025-26 | 41% (cups only) |
| T1 2022-23 | League matches with lineups and shots 91.8%: the earthquake season |

Counting only the league's own team's lineup, rather than both teams', changes none of these
(checked on completeness alone). The gaps are the provider's, so re-fetching cannot close them.

- **Change:** these nine league-seasons are never scored. They stay loaded as history (ratings, rest
  and starts for later matches). All nine leagues remain equally weighted test leagues. Switzerland
  is scored on 2020-21 only, as Belgium was in NLX.
- **Basis:** only the completeness check, which reads no results or forecasts. Nothing else changes:
  the frozen development choices, the primary, the thresholds and the inference.
