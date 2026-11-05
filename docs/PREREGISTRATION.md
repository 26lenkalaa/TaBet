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
