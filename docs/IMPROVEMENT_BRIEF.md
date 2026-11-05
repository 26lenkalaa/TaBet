# TaBet: brief for an outside review of model improvements

> Paste this whole file into a chat with another AI assistant (e.g. ChatGPT) and ask for the review
> described in the last section. It is self-contained: the reviewer does not have the code.
> Last updated 2026-11-05, before gameweek 10. Since the previous review: corrections in §5.4,
> §5.6–5.9, and a new prospective protocol in §5.10.

## 1. What the project is

TaBet predicts **home-win / draw / away-win probabilities** for English Premier League matches and
explains which factors moved them. It is a student software-engineering project (Python, pandas,
scipy). Success is judged on **probability quality**, not on picking winners:

- primary metric: mean **log-loss** (lower is better); also Brier score and ranked probability score (RPS)
- benchmark: bookmaker market probabilities (pre-match and closing averages from football-data.co.uk)
- a naive baseline: historical H/D/A frequencies

## 2. Hard rules (please respect these in suggestions)

1. **No leakage.** Every feature for a match uses only data known before its cutoff. A result is
   treated as known 2h30m after kickoff. Confirmed starting XIs (published ~1h before kickoff) are
   allowed as inputs; post-match information about the predicted match is not.
2. **Bookmaker odds are a benchmark only, never a model input.** The same goes for crowd signals
   such as FPL ownership, transfers and price.
3. **Chronological evaluation only.** Walk-forward refits per matchday; never random splits.
   Hyperparameters are chosen on development seasons and test seasons are scored once.
4. **No bot-protection circumvention.** Sources behind Cloudflare/Incapsula challenges, TLS
   fingerprinting or "undetected" browser automation are out of scope unless the provider grants access.
5. **No hand-set multipliers** (e.g. "La Liga goals × 0.8"). Any adjustment must be learned and must
   beat a simpler alternative out of sample.

## 3. Data we actually have (retrieved and validated)

| Source | Coverage | Granularity | Notes |
|---|---|---|---|
| football-data.co.uk CSVs | PL 2021-22 → 2026-27 (90 matches so far) | match | results, shots, shots on target, corners, fouls, cards, referee, bookmaker odds. 2026-27 file adds team xG (`HxG/AxG`). Other leagues available in the same format (not yet ingested). |
| FPL archive (github.com/vaastav/Fantasy-Premier-League) + live FPL API | PL 2021-22 → 2026-27, ~59k player-match rows | player × match | minutes, starts, goals, assists, xG, xA, xG conceded, FPL "threat/creativity/influence" indices; defensive actions only from 2025-26. All 1,990 fixtures join to our match table with identical scores. |
| Wyscout public dataset (Pappalardo et al. 2019, CC BY 4.0) | top-5 European leagues, **2017-18 only** | event | ~640k events per league with pitch coordinates, lineups, substitutions; no xG. Used only for prototyping. |
| StatsBomb Open Data | full leagues only for 2015-16 (PL, La Liga, Serie A, Ligue 1); recent seasons are single-club (e.g. Leverkusen 2023-24, PSG 2021-23) | event | attribution required. Not usable for 2021-27 player histories. |
| Predicted lineups: Fantasy Football Scout team-news page, RotoWire lineups page | the **next gameweek only**; no history | team × predicted XI | Plain HTTP, and robots.txt allows both pages. Scout's player photos use Premier League ids that equal FPL `code` (100% join). RotoWire names are matched to FPL by name within the team (98.6%). Each fetch is saved with its retrieval time so it can be scored later against confirmed XIs. |
| FPL live availability | current only | player | `status`, `chance_of_playing_next_round`, injury/suspension `news`. No historical archive. |

**FPL data caveats found during validation**

- `starts` and xG/xA/xGC were not published before 12 Nov 2022; the archive codes them as 0. We null
  xG for those matches and infer starters as the 11 players with most minutes (checked against
  seasons with real `starts`: 96.6% of players correct, 78% of team lineups fully correct).
- One player duplicated under two IDs in 2025-26 (deduplicated); two 2021-22 match sides have
  minor source gaps.
- The archive's code is MIT-licensed, but its LICENSE file says the data belongs to FPL and
  Understat, so data-use rights are not covered by that licence.
- football-data.co.uk's terms say its data is for private individuals, "NOT commerical or data
  training products using automated bots/scrapers/AI". The project owner has reviewed this and
  decided to proceed; please don't spend the review on it.

**Sources that blocked plain access (not used)**: Understat (maintenance/block page), FBref
(Cloudflare JS challenge), WhoScored (Cloudflare 403; the `soccerdata` library reaches it via
undetected-Chrome automation), Sofascore API (403; `soccerdata` uses TLS impersonation + proxy
rotation, and its documented Sofascore interface only covers leagues/seasons/schedule/table).
FotMob is not used for lineups because its robots.txt disallows `/api/*` for all agents.

## 4. Current model

**Stage 1, team strength (Dixon-Coles).** Poisson goals with per-team attack/defence ratings,
home advantage, the Dixon-Coles low-score correction, exponential time decay and an L2 penalty.
Tuned on 2022-23 + 2023-24: decay half-life ≈ 173 days, L2 = 10. A promoted-team prior did not
measurably help, so it is off.

**Stage 2, adjustments.** `log(rate) = log(Dixon-Coles rate) + b0 + b·x`, a penalised Poisson
regression fitted walk-forward on earlier matches, using each match's *pre-match* Dixon-Coles
rate as the offset. Kept features:

- **Opponent-adjusted xG strength (V1x):** att_T and −def_O from a decay-weighted quasi-Poisson
  model of team-match xG (§5.6). It replaced raw decayed xG form.
- **XI strength (V2 / V2x):** the starting outfielders' decayed (xG + xA)/90 ratings, shrunk to a
  position prior, divided by the team's usual XI strength, plus the opponent's back-line
  regularity. Needs a lineup.

Tested and not kept (no gain over a simpler alternative on identical fixtures): similar-player
priors, team-style main effects and interactions, lineup departures by role, continuity-only
lineup features, faster-adapting league scoring and home advantage, a multinomial outcome-level
second stage, and temperature/draw calibration.

**Two operating modes:**
1. **Before team news:** V1x, which needs no lineup.
2. **With a lineup:** V2x. With confirmed XIs this is the evaluated model. With *predicted* XIs
   (§5.9), the same stage-2 coefficients are applied. They were trained on confirmed XIs, so this
   assumes the predicted XI is close to the real one.

## 5. Results so far

### 5.1 Team-only baseline on held-out 2024-25 + 2025-26 (760 matches)

| | log-loss | Brier | RPS | accuracy |
|---|---|---|---|---|
| Dixon-Coles | 1.004 | 0.601 | 0.205 | 49.2% |
| Naive frequencies | 1.083 | 0.656 | 0.232 | 41.7% |
| Market, pre-match average | 0.993 | 0.595 | 0.201 | 51.8% |
| Market, closing average | 0.989 | 0.592 | 0.200 | 52.5% |

The gap to the pre-closing market is 0.0113 log-loss, SE 0.0061 (paired, 760 matches), with
95% CI [−0.0007, 0.0233]. That SE is specific to the model-vs-market comparison. Close variants
have much smaller paired variance: for XI strength vs xG form, the paired SD is 0.047 (SE 0.0017),
so precision has to be judged per contrast. A later calibration test (§5.5) found no
miscalibration worth correcting. Draws are never the single most likely outcome (predicted range
7–36%, observed rate 26%), which on its own is not a defect.

### 5.2 Wyscout 2017-18 prototype (5 leagues, gameweek 20 onwards, walk-forward)

Team style (Δ log-loss vs Dixon-Coles, paired bootstrap 95% CI):

| variant | fixtures | Δ log-loss |
|---|---|---|
| + style main effects, pooled over 5 leagues | 895 | −0.0077 [−0.0184, +0.0031] |
| + style × style interactions, pooled | 895 | −0.0066 [−0.0176, +0.0046] |
| + style interactions, pooled, England fixtures only | 190 | +0.0016 [−0.0244, +0.0282] |
| + style interactions, England-only coefficients | 190 | +0.0241 [−0.0154, +0.0637] |

Player similarity (players with 90–600 first-half minutes, predicting second-half per-90 output):
shrinking a player's own rate toward the role mean clearly beats the role mean alone. Shrinking
toward similar players instead did not help for shots + key passes. For goals + assists it gave a
small significant gain with same-league neighbours (wMSE −0.0020 [−0.0038, −0.0003]) but not with
all-league neighbours.

### 5.3 FPL-period experiments (PL 2022-23 → 2026-27)

The full paired results, calendar-block intervals and leakage checklist are in
`docs/LEAKAGE_AUDIT.md`; the frozen per-match predictions are in `results/fpl_v0_v5/`. Every
contrast below is on the same fixtures (dev 760, test 760, 2026-27 90). Δ is log-loss of the
first variant minus the second.

| Contrast (comparator) | Dev | Test | Notes |
|---|---|---|---|
| XI strength: V2 − V1 | +0.0002 (SE 0.0007) | **−0.0040 (SE 0.0017)** | Unadjusted normal 95% CI [−0.0073, −0.0007]; 14-day block CI [−0.0070, −0.0012]; Bonferroni (4 primary) CI ≈ [−0.0082, +0.0002]. Unconfirmed; pre-registered for unseen 2026-27 matches. |
| Similar-player vs position prior: V3 − V2 | −0.0003 (SE 0.0005) | +0.0002 (SE 0.0007) | no demonstrated advantage; keep the position prior |
| Style main effects given xG form: V1s − V1 | −0.0009 (SE 0.0026) | +0.0027 (SE 0.0018) | no benefit |
| Style interactions given main effects: V5 − V4 | −0.0003 (SE 0.0019) | −0.0018 (SE 0.0021) | no benefit |

Draws on test: 197 observed vs 180 predicted by V2, SD ≈ 12, so about 1.4 SD. A diagnostic flag,
not strong evidence of systematic under-prediction. On dev: 169 observed vs 167.5 predicted.

### 5.4 Answers to audit questions from the previous review

- **Stage-1 offsets are pre-match.** Each historical match's Dixon-Coles rate comes from a fit
  at that match's own cutoff (the first kickoff of its local day, using results known 2h30m
  after kickoff). Stage-2 training reuses those stored pre-match rates, so no offset is ever
  fitted retrospectively on its own result. Check: 0 of 3,220 rows have a cutoff after kickoff.
- **Inferred vs genuine starts.** 136 evaluation fixtures (all 2022-23 matches before 12 Nov 2022)
  have starters inferred from that match's minutes. Every one falls before stage 2 switches on
  (first active cutoff 2023-04-16), so all variants give them the identical Dixon-Coles
  prediction, and they add nothing to any contrast. All 760 test fixtures and all 2026-27
  fixtures use FPL's genuine `starts`. Inferred XIs are used only as past training rows.
- **Availability clock.** The 2h30m lag covers results only. FPL xG/xA/ICT and football-data
  match stats are archive values, possibly with later corrections, and their publication times
  are not known. Assumption: they were available by the next matchday's cutoff.
- **Score distribution after stage 2.** The smallest Dixon-Coles correction factor across all
  adjusted predictions is 0.575 (> 0). The score grid is 0–15 goals per side; the largest
  adjusted rate is 5.41, whose tail beyond 15 goals is below 1e-4.
- **Stricter offset audit.** For every one of the 489 Dixon-Coles fits, the latest result in its
  training set was known at least 13.5 h before that fit's cutoff. Stage-2 training rows are
  filtered by the same rule (kickoff + 2h30m ≤ cutoff) and checked.
- **The dev XI contrast restricted to the 457 fixtures where stage 2 is active:** V2 − V1 =
  +0.00038 (SE 0.0011; 14-day block CI [−0.0008, +0.0014]). On the 303 inactive fixtures the
  difference is exactly 0. Inactivity does not explain the dev/test difference.
- **Pre-registration:** the primary contrast stays V2 − V1 (confirmed XIs) on 2026-27 matches
  from gameweek 10 onward. Secondary contrasts V2x − V1x, V2x − V2 and V1x − V1 were added
  before gameweek 10 and are labelled secondary. Predicted-lineup forecasts have their own
  protocol (§5.10).

### 5.5 Restricted calibration layer (first review's #1)

Method: `q_k ∝ exp(log p_k / T + δ·[k = D])`, with penalty λ(log²T + δ²) on summed log-loss.
It is fitted walk-forward on each variant's own earlier out-of-sample ledger predictions (identity
until 200 known results), with λ chosen per variant and mode on dev from {1, 10, 100, 1000}. The
same policy is applied to V0, V1 and V2, on identical fixtures.

| | Dev | Test | 2026-27 |
|---|---|---|---|
| λ chosen | 1000 (largest, i.e. stay at identity) for every variant and mode | | |
| V2: temperature − identity | +0.0000 | +0.0001 | +0.0000 |
| V2: temperature+draw − temperature | +0.0000 | −0.0001 | −0.0000 |
| V2 − V1 under each calibration mode | +0.0002 | −0.0040 | −0.0044 |

Descriptive fits *without* a penalty, with 300-resample bootstrap CIs (not used for selection):

| | Dev | Test |
|---|---|---|
| V2, T | 0.933 [0.769, 1.122] | 0.975 [0.819, 1.197] |
| V2, δ | +0.057 [−0.139, +0.239] | +0.134 [−0.034, +0.310] |

What this shows:
- **No evidence of miscalibration large enough to correct.** T's intervals include 1 everywhere,
  and the point estimates lean slightly *under*-confident.
- **The draw shift points the same way as the draw shortfall,** but its intervals include 0.
- **The XI-strength contrast is unchanged** under every calibration mode.

Decision: keep the identity. The calibration layer stays in the code (`tabet/calibration.py`) to be
re-checked once more data accumulates.

### 5.6 Opponent-adjusted xG strength (second review's #1)

Method: at each cutoff, a regularised log-linear model of team-match xG,
`E[xG_T vs O] = exp(μ + home + att_T − def_O)`. It is fitted by Poisson quasi-likelihood (only the
conditional mean is modelled, so non-integer xG is fine), decay-weighted, with L2 toward 0, on
past matches only. The stage-2 features att_T and −def_O *replace* V1's raw xG form, giving V1x.
The xG half-life {90, 180, 365 days} and stage-2 L2 {1, 10, 100} were chosen for V1x on dev only
(chosen: 90 days, L2 = 1; 90 days is the shortest value tried). V0 and V1 reproduce the frozen
ledger exactly. Raw and adjusted features correlate at 0.96 (attack) and 0.98 (defence).

| Contrast (n) | Dev (760) | Test (760) | 2026-27 (90) |
|---|---|---|---|
| **primary** V1x − V1 | −0.0006 (SE 0.0006), block CI [−0.0017, +0.0005] | **−0.0009 (SE 0.0004)**, block CI [−0.0017, −0.0001] | −0.0002 (SE 0.0007) |
| secondary V1x − V0 | −0.0017 | −0.0031, block CI [−0.0059, −0.0002] | −0.0004 |
| secondary V1+x − V1x (both together) | +0.0005 | +0.0009, block CI [+0.0002, +0.0016] (worse) | +0.0003 |
| descriptive V2x − V2 | −0.0005 | −0.0007, block CI [−0.0013, −0.0001] | −0.0000 |

What this shows:
- **A small gain with the same sign in every aggregate split.** It is only individually
  significant on the (already reused) test seasons, and about a quarter (23%) of the XI-strength
  effect.
- **Using raw and adjusted xG together is worse** than adjusted alone. The two are nearly
  collinear.
- **Decision:** adopt adjusted xG as the xG reference in place of raw. It uses the same number of
  features and was better on every reported aggregate split, which says nothing about every match
  or future period. This is development evidence. The pre-registered V2 is
  unchanged; V2x would be confirmed separately.

### 5.7 Review round 2, items #3–#5 (all on the V1x base, identical fixtures)

Settings were chosen on dev only (`results/review_round2/results.json`). V1x reproduces its frozen
ledger exactly. Δ log-loss with SE; [ ] marks a 14-day calendar-block 95% CI.

| Contrast | Dev (760) | Test (760) | 2026-27 (90) |
|---|---|---|---|
| **#3 primary** departures − ratio (Ld − Lr) | +0.0001 (0.0014) | −0.0003 (0.0013) | +0.0036 (0.0031) |
| #3 departures − continuity-only (Ld − Lc) | −0.0004 | −0.0029 [−0.0072, +0.0009] | −0.0086 [−0.0152, −0.0011] |
| #3 continuity-only − V1x (Lc − L0) | +0.0007 | −0.0015 [−0.0051, +0.0021] | +0.0078 [+0.0021, +0.0151] |
| #3 ratio − V1x (Lr − L0) | +0.0003 | **−0.0041 [−0.0073, −0.0011]** | −0.0044 [−0.0095, −0.0004] |
| **#4 primary** + recent league scoring (A1 − L0), half-life 30 d | −0.0003 (0.0003) | −0.0003 (0.0005) | +0.0004 (0.0006) |
| #4 + recent home advantage (A2 − A1) | +0.0006 [+0.0002, +0.0012] | −0.0002 | +0.0020 |
| **#5 primary** multinomial − Poisson, same features (M − L0) | −0.0005 (0.0012) | +0.0011 (0.0022) | −0.0025 (0.0050) |
| #5 multinomial − V1 (calibrated V1 = identity) | −0.0011 | +0.0002 | −0.0027 |

Departures are defined as the role's (FWD/MID/DEF) starter attack ratings minus
Σ start-probability × rating from preceding selections, plus the chance the opponent's keeper is
not the usual one, plus the opponent's defenders' mean start probability. Continuity-only uses
own outfield continuity plus the two opponent proxies.

What this shows:
- **#3:** departures are no better than V2's ratio. The ratio's gain reappears on the V1x base
  (test −0.0041, 2026-27 −0.0044), but again not on dev. Continuity alone does not help and is
  worse on 2026-27. **Keep the pre-registered ratio.**
- **#4:** no gain from a faster-adapting league scoring level. Adding home-advantage adaptation is
  worse on dev. **Keep a single decay.**
- **#5:** the multinomial outcome model is not better than the Poisson adjustment, so there is no
  evidence that the goals-vs-outcome objective mismatch matters at this sample size. **Keep Poisson.**
- Many contrasts have now been examined. Apart from XI strength (already pre-registered), none
  shows a consistent gain.

**Current working model:** Dixon-Coles + opponent-adjusted xG strength + XI-strength ratio
(position prior), uncalibrated. The pre-registered primary test confirms only the original
V2 − V1 contrast. The current model's XI contrast (V2x − V1x) and xG replacement (V2x − V2,
V1x − V1) are separate secondary contrasts, specified before gameweek 10.

### 5.8 Projected-lineup mode (XI projected from preceding selections only)

For each team and cutoff, the projected XI is the most-started keeper plus the 10 outfielders with
the highest decayed start probability (half-life 120 days) who last played for that team. No
historical injury or suspension news exists, so this projection cannot drop known absentees. It
gets about 7.5 of 11 starters right (2022-23: 7.64, 2023-24: 7.46, 2024-25: 7.83, 2025-26: 7.56,
2026-27: 7.04). V2x settings are frozen from §5.7.

| Contrast | Dev (760) | Test (760) | 2026-27 (90) |
|---|---|---|---|
| **primary** V2x with projected XI − V1x | +0.0004 (SE 0.0015) | +0.0001 (SE 0.0012) | −0.0009 (SE 0.0022) |
| V2x with confirmed XI − V1x (V2x L2 = 100) | +0.0003 (SE 0.0007) | **−0.0038 (SE 0.0017)**, block CI [−0.0069, −0.0010] | −0.0043, block CI [−0.0092, −0.0004] |
| projected − confirmed | +0.0001 | +0.0039, block CI [+0.0003, +0.0079] | +0.0034 |

Why §5.7 shows −0.0041 and this table −0.0038 for the same confirmed-XI contrast: §5.7's "Lr"
re-chose the stage-2 penalty on dev (L2 = 1), while here V2x uses V2's frozen L2 = 100. The
fixtures, features and cutoffs are identical. Both reproduce exactly: −0.0041 with L2 = 1 and
−0.0038 with L2 = 100.

What this shows, narrowly:
- **This particular history-only projection, passed through confirmed-XI coefficients, adds
  nothing over V1x.** It does not show that every useful early lineup signal must come from
  external team news.
- **Two limitations of the projection:**
  - "Last played for that team" can keep departed players and miss new arrivals. The live
    version uses current FPL clubs instead.
  - The top 10 outfielders need not form a plausible formation.

### 5.9 Predicted lineups from public sources, and live forecasts

Forecasts for upcoming fixtures are now produced and saved before kickoff with their cutoff, code
version and lineup snapshot. That is the start of the prospective log the pre-registration needs.
The first snapshot (gameweek 10, retrieved 2026-11-05 06:56 UTC) shows:

- **Scout vs RotoWire agreement:** a mean of 9.9 of 11 starters (minimum 7; 5 of 20 teams agree
  fully).
- **FPL flags** 15 usual starters as unavailable (injury, suspension or loan).
- **Shift from the lineup-free forecast**, as the mean (max) absolute change in any outcome
  probability, by lineup source (the dashboard's XI view now shows the plain Scout + RotoWire
  average, pE):
  - Scout: 1.8 (3.2) points
  - RotoWire: 2.3 (6.8) points
  - our FPL projection that drops flagged absentees: 2.9 (6.7) points

None of these predicted-lineup forecasts has an accuracy record yet. After each gameweek we can
score:
1. each source's lineup accuracy (correct starters out of 11)
2. each source's forecast log-loss vs V1x and vs V2x with the confirmed XI

A dashboard shows V1x forecasts for the next four gameweeks, plus V2x forecasts for the next
gameweek under each lineup source and their average.

### 5.10 Prospective protocol for predicted lineups (from your last review)

Written before gameweek 10 and summarised here.

- **Horizon:** a fixed forecast time of kickoff − 24 h, logged once per fixture by a run every
  2 hours.
- **Each record stores:**
  - every candidate's probability vector: p0 = V1x; V2x with the Scout XI, the RotoWire XI, and
    their probability average pE; V2x with our FPL projection with and without the availability
    filter
  - lineups as player codes, unresolved names, and each source's own "last updated" text
  - FPL availability flags
  - the exact inputs to rebuild every candidate (checked: reconstruction difference 0)
  - git commit and training cutoff
- **Raw pages** are kept gzipped and the log is append-only.
- **Fitting:** gameweeks 10–19 fit a single shrinkage α for pα = (1−α)·p0 + α·pE. α is fitted on
  earlier logged matches only, then frozen.
- **Evaluation:** gameweeks 20–29. Primary: pα − p0. Secondary: pE − p0, pα − pE,
  fpl_proj − fpl_proj_hist, and each source − p0. Each uses its own paired variance, counts each
  fixture once, and reports 14-day block CIs.
- **Diagnostics:** correct starters, XI-feature error against the confirmed XI, and a *controlled*
  comparison (same frozen early model with the confirmed XI substituted), reported separately
  from any operational confirmed-lineup forecast.
- **Frozen "unavailable" rule:** FPL status not "a" and chance of playing < 50.

## 6. Known weaknesses and open questions

1. **Statistical power depends on the contrast.** Use each comparison's own paired variance, not
   the model-vs-market variance. For XI strength vs xG form (paired SD 0.047), 80% power at a
   two-sided 5% level needs roughly 693 / 1,082 / 1,924 matches for true gains of 0.005 / 0.004 /
   0.003 (assuming independence, stable variance and one pre-specified comparison).
   Plans:
   - add 2016-17 → 2020-21 for features that existed then (ICT, minutes, no xG), reported
     separately from xG-era results
   - keep few primary hypotheses
   - accumulate prospective forecasts
   - always report block-bootstrap intervals

   Extra leagues help estimate shared coefficients but are not more Premier League evidence.
   Goal deviance is a diagnostic, not extra independent matches.
2. **Player defence.** We have no good per-player defensive rating; FPL xG-conceded is a team
   quantity while on the pitch. Defensive actions exist only from 2025-26.
3. **Lineups before team news.** Being handled by the protocol in §5.10. It will produce
   evidence from gameweek 20 onward. Until then V1x stays the default early forecast.
4. **XI gain: development and test seasons disagree.** The gain is null on 2022-24 (where stage 2
   is active for only 457 of 760 matches) and about −0.004 on 2024-26 and 2026-27. Is there a
   principled way to tell a real regime difference from noise, beyond the pre-registered test?
5. **Cross-league player priors** (new signings) need non-PL player data for 2021-27, which we do
   not have from an accessible source.
6. **Draw modelling.** Calibration found no reliable draw shift (δ interval includes 0). Is a
   single diagonal-inflation parameter still worth testing, or should draws be left alone?
7. **Model class.** Would a gradient-boosted model on the same point-in-time features, or a
   hierarchical Bayesian version of Dixon-Coles, be a better second stage at this sample size?

## 7. What I want from you (the reviewer)

The predicted-lineup work is now on a fixed prospective schedule (§5.10), so no new results will
exist for about ten gameweeks. Please give me:

1. **A critique of the protocol in §5.10** before data arrives: anything that would bias pα − p0,
   problems with the horizon or logging rule, and whether the frozen "unavailable" rule and
   fallbacks are sensible.
2. **What to do with the waiting time:** up to 3 experiments that use only historical data, don't
   reuse ideas already tested in §5.5–5.8, and don't touch the 2026-27 gameweek 10+ fixtures
   reserved for the pre-registered and prospective tests. For each, give the simpler alternative it
   must beat.
3. **Problems in §5.4–5.8** that remain after the corrections.

Please flag any claim you are unsure about instead of stating it as fact, and do not suggest using
bookmaker odds or crowd data as model inputs.
