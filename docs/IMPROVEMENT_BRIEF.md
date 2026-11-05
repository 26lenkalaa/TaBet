# TaBet: brief for an outside review of model improvements

> Paste this whole file into a chat with another AI assistant (e.g. ChatGPT) and ask for the review
> described in the last section. It is self-contained: the reviewer does not have the code.
> Last updated 2026-11-05 (evening), before gameweek 10. Since the last review: the third review's
> items are done (§5.14–5.18), a market comparison (§5.19) and **an open problem with newly
> promoted teams (§5.20), which is what I want help with this time (§8).**

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
| Hugging Face `eatpizzanot/soccer-dataset` (tagged CC BY 4.0; provenance API-Football, so republication rights are unclear) | top-5 leagues, full 2021-22 → 2024-25, 50–66% of 2025-26, **none for 2026-27** | player × match | lineups, minutes, dribbles, duels, tackles, interceptions, passes, key passes, fouls. Validated against FPL (starters 100%, minutes within 1 min 94.7%) and football-data (team fouls 98.8% and shots on target 99.4% exact). Player shots undercounted; no player xG. **Historical only: cannot feed live forecasts.** |
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

### 5.11 Pooling four more leagues (La Liga, Serie A, Bundesliga, Ligue 1)

Added 7,370 football-data.co.uk matches (2021-22 → 2026-27, all validated as complete double
round-robins). Player data for these leagues was not available from an accessible source when
this was run, so pooling works through a team-level feature every league has:
**opponent-adjusted shots-on-target strength**. It uses the same quasi-Poisson model as the xG
strength, fitted within each league.

Each league has its own walk-forward Dixon-Coles fit; team ratings never cross leagues. Stage 2
is trained on PL rows only (P1), or on all five leagues with a per-league intercept (P2). All
training rows respect the target fixture's cutoff, and **everything is scored on PL fixtures
only**. L2 was chosen on PL dev (P1: 1, P2: 100). P0 reproduces the frozen V0 exactly.

| Contrast | Dev (760) | Test (760) | 2026-27 (90) |
|---|---|---|---|
| **primary** pooled − PL-only coefficients (P2 − P1) | −0.0011 (SE 0.0015) | +0.0001 (SE 0.0004) | −0.0005 (SE 0.0012) |
| SoT strength, PL-only − Dixon-Coles (P1 − P0) | −0.0012 | −0.0014 [−0.0029, +0.0002] | −0.0010 |
| pooled SoT − V1x (FPL xG) | −0.0006 | +0.0018 [−0.0009, +0.0047] | −0.0011 |
| Dixon-Coles settings tuned on all 5 leagues − PL-tuned | +0.0001 | +0.0002 | −0.0046 (SE 0.0051) |

What this shows:
- **Pooling did not improve PL forecasts once enough PL data exists.** On test the pooled and
  PL-only coefficients agree almost exactly.
- **The dev difference is mostly timing.** With 4× the rows, stage 2 switches on much earlier
  (active for 698 vs 457 dev fixtures).
- **Pooled-tuned Dixon-Coles settings** (half-life 231 vs 173 days) make no difference.
- **FPL xG (V1x) stays slightly ahead of shots-on-target strength** on test.
- **Read in context:** with only 2–4 shared coefficients, 3–4 PL seasons are already enough to
  estimate them. Pooling would matter more for richer models, early seasons, or promoted teams,
  none of which are tested here.

### 5.12 Transfer priors for players arriving from other leagues

Question: does a new signing's previous-league output predict their first Premier League season
better than the position prior the XI model uses now? Data: the Hugging Face archive (§3) for
prior-league seasons, joined to FPL by name voting within shared fixtures.

- **Cohort:** 95 players moving to the PL from La Liga, Serie A, Bundesliga or Ligue 1, arriving in
  2023-24 (26), 2024-25 (32) or 2025-26 (37). Each cohort's mapping is trained only on earlier cohorts
  (42 / 68 / 100 training players).
- **Target:** first-PL-season (xG + xA)/90, weighted by minutes.
- **Models:** P0 = position prior (current); P0b = position average of earlier arrival cohorts (no player information);
  P1 = ridge on shrunk prior-league per-90 goals+assists, shots and key passes; P2 = P1 + league dummies.

| Contrast (weighted MSE) | Mean | 95% bootstrap CI |
|---|---|---|
| **primary** P1 − P0 | −0.0068 (0.0170 → 0.0102, −40%) | [−0.0121, −0.0018] |
| P0b − P0 | +0.0036 | [−0.0011, +0.0087] |
| P2 − P1 (league dummies) | −0.0001 | [−0.0009, +0.0008] |

*Corrected 2026-11-05:* the first run used a stale copy of the appearance table in which
pre-November-2022 xG was coded 0 instead of missing (the fix in §3 had not been applied to it).
Rerun on the clean table above; the earlier figures were 0.0193 → 0.0131 (−32%). Same conclusion.

The pre-registered gate passed. Which league a player came from adds nothing once output is
shrunk. **Not yet done:** the match-level test (P1 priors inside the XI ratings, scored by match
log-loss). Live use in 2026-27 would need a different prior-season source (the archive stops in
mid-2025-26).

### 5.13 Match statistics and outcomes (dribbles, duels, passing, defending)

From 3,540 PL team-matches (2021-22 → 2025-26), each stat's home-minus-away difference, against the
result (odds ratio per SD for a win):

| Stat (same match) | Raw | Given pre-match DC strength | Given strength + shots on target |
|---|---|---|---|
| Shots on target | 4.31 | 3.31 | — |
| Key passes | 2.08 | 1.31 | 0.50 |
| Passes | 1.38 | **0.59** | 0.45 |
| Pass completion % | 1.30 | 0.59 | 0.44 |
| Tackles + interceptions | 0.95 | **1.38** | 1.38 |
| Duel win % | 1.25 | 1.21 | 1.10 |
| Dribbles attempted | 1.07 | 0.89 | 0.82 |
| Dribble success % | 1.12 | 1.12 | 1.04 |
| Fouls committed | 0.83 | 0.93 | 0.93 |

These are same-match associations and mostly reflect **game state**: a team that leads stops
passing and starts defending. They are not causal and not usable before kickoff. Distance covered
is not available from any source we can access.

**Predictive test** (stat profile from *earlier* matches, decay-weighted, added to V1x stage 2,
L2 = 100 chosen on dev): dev −0.0006 (SE 0.0021), test 2024-25 −0.0028 (SE 0.0026, 14-day block CI
[−0.0079, +0.0014]). Not kept. Largest coefficients: key passes (+0.06 per SD on log goals) and
dribble success % (+0.045).

### 5.14 Diagnostics from the third review

**Protocol amendment A2** (before any prospective record): availability now reads the FPL field for
the fixture's own gameweek (the "next round" field had moved on for fixtures logged after a
deadline), null chances count as 0 for injured/suspended players, moved kickoffs are re-logged and
only the record matching the kickoff played is scored, `pE` falls back to `p0`, α defaults to 0
when degenerate, the gameweek 20–29 read is descriptive with the confirmatory read at gameweeks
20–38, and an XI-feature decomposition is pre-registered as a secondary analysis. Live FPL xG is now
snapshotted daily to measure revisions.

**XI gain by how far both XIs were from usual** (V2x − V1x log-loss; cut points from all dev + test
matches):

| Band | Dev n (stage 2 active) | Dev | Test n | Test |
|---|---|---|---|---|
| low deviation | 296 (107) | +0.0008 (SE 0.0004) | 211 | −0.0017 (SE 0.0017) |
| middle | 238 (175) | +0.0014 (SE 0.0010) | 268 | +0.0006 (SE 0.0023) |
| high | 226 (175) | −0.0016 (SE 0.0019) | 281 | **−0.0096 (SE 0.0039)** |

Overall dev +0.0003 vs test −0.0038; difference +0.0041, z = 2.23. On test the gain is concentrated
in high-deviation matches. Dev points the same way in that band but is much weaker, and the band
mix is similar in both periods, so "dev simply had fewer unusual XIs" does not explain it. This is
between the two readings you described; the pre-xG-era replication is the intended arbiter.

**Shorter xG half-lives:** dev now picks 60 days over 90 by 0.00015 log-loss, and on test the
difference is −0.0002 [−0.0010, +0.0007]. The edge-of-grid concern is real but immaterial; the
live model keeps 90 days (frozen by the protocol).

**Relabelled splits:** about twenty contrasts have now been read on 2024-26 and on 2026-27
gameweeks 1–9, so those are treated as **reused validation**. The only clean confirmatory data is
2026-27 gameweek 10 onward plus the pre-xG-era replication.

### 5.15 Promoted-team prior from Championship data (F4, your top pick)

All 18 promotions 2021-22 → 2026-27 identified from football-data E1 files (team in the PL season,
in the Championship the season before). Championship feature: opponent-adjusted shots-on-target
attack and defence over the promotion season (§5.11 model, half-life 90 days, not tuned).

Implemented inside the Dixon-Coles fit, not as a separate regression: a promoted side's offset is
`promo_att + b_att·x_att` and `promo_def + b_def·x_def`, learned walk-forward from earlier promoted
sides' matches with the same L2 as the ratings. The team's own ratings still start at 0 and take
over as matches accumulate, so the offset is a prior mean that fades. D0 reproduces frozen V0
exactly.

| Contrast (log-loss) | 2022-26 | 2026-27 GW1–9 |
|---|---|---|
| **primary** D2 − D0, early promoted (a promoted side in its first 10 PL matches; n 111 / 25) | −0.0134 (SE 0.0228) [−0.058, +0.028] | +0.0654 (SE 0.0501) |
| D2 − D1 (Championship slopes over a shared offset), same subset | +0.0004 [−0.0002, +0.0010] | +0.0028 |
| D1 − D0 (shared offset only), same subset | −0.0138 [−0.058, +0.028] | +0.0627 |
| D2 − D0, all matches (n 1,520 / 90) | −0.0008 [−0.0048, +0.0027] | +0.0211 |

**Null.** The learned slopes shrink to about zero (final fit −0.04 and −0.16). Every gain or loss
comes from the shared offset: about −0.17 on promoted defence, which helped in 2022-26 and hurt in
2026-27, when the promoted sides started better than usual. Descriptively, across the 12 promoted
team-seasons of 2022-26, Championship shots-on-target attack correlates 0.10 with PL attack, and
Championship defence correlates 0.26 with PL defence. Promotion-season goal difference does
little better (0.39 / 0.20, not tested). With 3 promotions a season there are too few cohorts to
learn a mapping; pooling other leagues' second tiers would be the only way to get more, which I
have not done.

### 5.16 Pre-xG-era replication of XI strength (your experiment 1; pre-registered, addendum R)

Seasons never used for any earlier choice: results 2015-16 → 2020-21, FPL players 2016-17 →
2020-21. The 2016-18 archive has no fixtures or teams files, so team ids were rebuilt from
players' opponents and home/away flags and checked against football-data. All 1,900 fixtures join
with identical scores. Starters are inferred (11 per side, ~986 minutes). Player rating =
threat + creativity through the unchanged V2 code. Base B = Dixon-Coles + shots-on-target
strength; X = B + XI strength. L2 chosen on 2017-18 (both 100); 2018-21 scored once. The
pre-registration was pushed before the run.

| | Result |
|---|---|
| **primary** X − B, 2018-21 (n 1,140) | **−0.0009 (SE 0.0008), block CI [−0.0022, +0.0004]: not confirmed** |
| 2018-19 / 2019-20 / 2020-21 (no crowds) | +0.0006 / −0.0015 / −0.0017 |
| dev 2017-18 (used only for L2) | X worse than B by +0.0046 |
| B − Dixon-Coles | −0.0016 [−0.0044, +0.0011] |
| X − B by XI-deviation band: low / middle / high | +0.0004 / +0.0011 / **−0.0042 (SE 0.0018)** |

Reading:
- **The effect does not replicate at the size seen on 2024-26.** The interval excludes −0.004.
  The paired SD here (0.026) is lower than in the xG era, so power was better than planned.
- **The pattern replicates.** As on 2024-26 (§5.14), any gain sits in the third of matches where
  the XIs were furthest from usual (−0.0042 here vs −0.0096 there), with nothing elsewhere.
- **Caveats that bias toward zero:** threat + creativity is a noisier player measure than xG + xA,
  and inferred starters are fully right for only about 78% of team lineups.
- **My conclusion:** the average XI effect is probably small (around −0.001 to −0.002), and the
  −0.004 on 2024-26 was at the lucky end. The effect is real mainly when lineups change a lot.

### 5.17 Transfer priors at match level (pre-registered, addendum T)

Priors for every **pre-season-eligible** arrival (≥ 600 previous-league minutes, no condition on PL
minutes, which the player-level gate had used): 32 / 37 / 44 arrivals in 2023-24 / 2024-25 /
2025-26. The P1 mapping is trained on earlier cohorts only. The prior replaces the position-prior
mean in the unchanged V2 rating and fades with PL minutes; debutants are rated at their prior.
Nothing tuned. The recomputed standard XI feature reproduces the frozen rows exactly.

| V2xT − V2x (log-loss) | n | Mean | 14-day block CI |
|---|---|---|---|
| **primary** 2024-26, all matches | 760 | −0.00025 (SE 0.00053) | [−0.0010, +0.0005] |
| **co-primary** 2024-26, an arrival starts within his first 10 apps | 317 | +0.00058 (SE 0.00106) | [−0.0007, +0.0018] |
| 2023-24, all / arrival subset | 380 / 133 | +0.00004 / −0.00037 | — |

**Null.** The prior moves the XI ratio by only 0.014–0.020 on average, even when an arrival starts.
The priors do pick out the obvious names (Wirtz 0.50 vs a 0.30 position prior, Cherki, Ekitiké,
Frimpong, Xavi Simons), but a regular starter is also part of the team's *usual* XI, so raising his
rating raises both sides of the ratio. The XI feature measures deviation from the usual lineup, not
squad quality. A signing's effect on team strength reaches the model through the team's own
results and xG, which Dixon-Coles and V1x pick up within a few matches.

What this suggests instead (not run): a **season-start team adjustment** from squad changes —
the minutes-weighted rating of incoming minus outgoing players, applied to the team's attack prior
at the season boundary — which targets the first weeks, where team ratings still reflect last
season's squad. It could be combined with your season-boundary shrinkage suggestion.

### 5.18 Dynamic (score-driven) team strengths (pre-registered, addendum D)

Your last suggestion, implemented as the score-driven prototype you recommended (`tabet/dynamic.py`):
attack/defence move after each match by the Poisson score of the log-rate (scaled by α_att, α_def),
decay toward 0 with weekly persistence β, and by an extra factor β_summer across a season boundary.
Same goal model (Poisson + Dixon-Coles ρ). Static parameters refitted at each season start by
maximising the filtered one-step log-likelihood on earlier matches. A test checks that no same-day
or later result changes a prediction.

| Contrast (log-loss) | Clean 2018-21 (n 1,140) | Recent |
|---|---|---|
| **primary** S2 − D0 | **+0.0009 (SE 0.0034) [−0.0051, +0.0066]: not confirmed** | 2022-24 +0.0044, 2024-26 +0.0051, 2026-27 −0.0044 |
| S2 − S1 (summer term) | +0.0000 | −0.0008 / +0.0001 / +0.0002 |
| S2 − D0, early season (first 56 days) | −0.0076 [−0.029, +0.011] (n 210) | 2022-26 +0.0071 [−0.006, +0.020] |
| S2 − D0s (half-life 116 d) | −0.0047 [−0.011, +0.001] | — |
| D0s − D0 (shorter half-life) | **+0.0056 [+0.0029, +0.0089]** | — |
| S2 + adjusted xG − V1x | — | 2022-24 +0.0053, 2024-26 +0.0031, 2026-27 −0.0059 |

Reading:
- **No gain over the decayed refit.** On clean data the dynamic model ties Dixon-Coles; on recent
  seasons it is slightly worse, with and without the xG layer.
- **The summer term learns itself away.** β_summer is fitted at about 1.0 (0.98–1.0 on clean data,
  0.90–1.0 recently; the 0.80 for 2022-23 is the starting value, since one season of training has no
  boundary to learn from). The data do not support resetting ratings over the summer.
- **Faster adaptation hurts.** The shorter half-life is clearly worse (+0.0056, interval excludes 0),
  and β ≈ 1 means the filter wants almost no mean reversion. Team strength in the PL is persistent;
  the 173-day refit is already close to the right speed.
- **Early season:** opposite signs on the two data sets, both inside noise. No support for the
  claim that the largest gains come early.

### 5.19 TaBet vs the betting market on the same fixtures (benchmark only)

Frozen out-of-sample V2x against football-data average odds (margin removed proportionally),
2022-26, n = 1,520. Pre-closing odds are collected Friday/Tuesday afternoon (*less* information than
V2x, which sees confirmed XIs); closing odds are at kickoff (*more* information).

| | vs pre-closing | vs closing |
|---|---|---|
| Same top pick | 88.5% | 87.4% |
| Disagreements: TaBet alone right / market alone right / both wrong | 50 / 76 / 48 (n 174) | 52 / 86 / 53 (n 191) |
| Sign test, TaBet-only vs market-only | p = 0.026 | p = 0.005 |
| Log-loss TaBet − market, 14-day block CI | +0.0139 [+0.0056, +0.0230] | +0.0188 [+0.0105, +0.0278] |
| 2024-26 only | +0.0044 [−0.0075, +0.0167] | +0.0080 [−0.0029, +0.0197] |

The six largest disagreements in each direction were all already present in the Dixon-Coles base
(xG and XI layers moved them by ≤ 4 points), and most involve a promoted or returning team early in
its season. Script: `prototypes/run_market_comparison.py`.

### 5.20 Open problem: newly promoted teams

**What we know (documented):**
- **Where we lose to the market.** With a promoted side in its first 10 PL matches, TaBet disagrees
  with the closing market twice as often (23.4% vs 11.9% of matches) and trails it by +0.030
  log-loss [−0.016, +0.075] (n 111), against +0.019 [+0.010, +0.028] without a promoted side.
  The difference is not significant, but the worst individual misses are concentrated here:
  - Burnley v Sunderland, Aug 2025: TaBet gave Sunderland 74%, the market 30%; Burnley won 2-0.
  - Man Utd v Sunderland, Oct 2025: TaBet gave Sunderland 42%, the market 19%; Man Utd won 2-0.
  - Coventry v Hull, Aug 2026: TaBet gave Hull 59%, the market 20%; Hull won 0-1 (the same
    behaviour, right this time).
- **How the model treats them now.**
  - A team never seen before starts at league average (rating 0) and then moves with every result,
    so two or three early results can swing it a long way.
  - A *returning* team keeps its decayed ratings from its last PL season, even if that was two
    seasons and a squad ago. Burnley in 2025-26 was rated partly on its 2023-24 relegation season.
- **What has failed.**
  - A shared promoted-team offset (learned from earlier promoted sides: attack about −0.01, defence
    about −0.17). It helped in 2022-26 and hurt in 2026-27, when promoted sides started well.
  - Team-specific offsets from Championship shots-on-target strength (§5.15). The slopes shrank to
    about 0, because 3 promotions a season are too few to learn from.
  - Faster or dynamic ratings in general (§5.18).

**Candidate approaches (not yet run):**

| # | Idea | Data | What it fixes | Simpler alternative to beat |
|---|---|---|---|---|
| R1 | **Reset returning teams:** if a team missed the previous PL season, drop or heavily shrink its old ratings and treat it as newly promoted | have | Stale ratings (Burnley 2025-26) | current decayed ratings |
| R2 | **Slower early reaction for promoted sides:** a larger L2 penalty on a promoted team's ratings for its first N matches, shrinking toward the promoted prior instead of league average; penalty learned on earlier cohorts | have | Over-reaction to 2–3 results (Sunderland, Hull) | shared offset (D1) |
| R3 | **Pool five second tiers:** Championship, Segunda, Serie B, 2. Bundesliga and Ligue 2 (football-data files confirmed available) give about 15 promotions a season instead of 3, to learn "second-tier strength → first-season top-flight rating", with league intercepts; scored on PL only | free, not yet ingested | Too few cohorts (§5.15) | PL-only mapping |
| R4 | **Squad-based prior:** minutes-weighted player ratings of the promoted squad plus summer signings, using the §5.12 mapping for arrivals. The Hugging Face archive lists Championship player tables (not yet validated); it ends mid-2025-26, so live use needs another source | partly free | The prior knows nothing about who is in the squad | R1 + R2 |
| R5 | **Season-boundary shrinkage for everyone** (your earlier suggestion), with a separate parameter for promoted sides | have | Summer squad change in general | current model |

**Evaluation plan.** Early-promoted matches are about 30 a season, so recent seasons alone cannot
settle this. The 2015-21 results (§5.16) add 15 promotion cohorts never used for team-strength
choices, and E1 files exist for those years. I would pre-register a primary contrast on
early-promoted matches pooled over 2016-26, plus all matches, so that a gain for promoted teams
cannot hide a loss elsewhere.

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
   quantity while on the pitch. FPL defensive actions exist only from 2025-26. The Hugging Face
   archive has per-player tackles, interceptions and duels for 2021-25, but nothing live.
3. **Lineups before team news.** Being handled by the protocol in §5.10. It will produce
   evidence from gameweek 20 onward. Until then V1x stays the default early forecast.
4. **XI gain: development and test seasons disagree.** *(Update: §5.14 and §5.16 suggest a small
   average effect concentrated in high-deviation lineups; the original question follows.)* The gain is null on 2022-24 (where stage 2
   is active for only 457 of 760 matches) and about −0.004 on 2024-26 and 2026-27. Is there a
   principled way to tell a real regime difference from noise, beyond the pre-registered test?
5. **Cross-league player priors.** Solved for history by §5.12; for 2026-27 arrivals we only have
   the season-level Top-5 CSV (no declared licence, merged from sites that block us).
6. **Draw modelling.** Calibration found no reliable draw shift (δ interval includes 0). Is a
   single diagonal-inflation parameter still worth testing, or should draws be left alone?
7. **Model class.** Would a gradient-boosted model on the same point-in-time features, or a
   hierarchical Bayesian version of Dixon-Coles, be a better second stage at this sample size?

## 7. Candidate factors not yet tested

None of these has been run. Data status: **[have]** = already ingested and validated;
**[free]** = a free source exists but has not been retrieved or validated; **[none]** = no
accessible source. Each must beat the listed simpler alternative on identical chronological
fixtures, and every feature must be computable at the match cutoff.

| # | Factor | How it would enter the model | Data | Simpler alternative to beat |
|---|---|---|---|---|
| F1 | **Rest and congestion:** days since each team's last match, matches in the last 14 days, rest difference | stage-2 features | PL dates [have]; European and cup dates [free, not retrieved: the HF fixtures table or openfootball may have them] | V1x |
| F2 | **XI fatigue:** starters' minutes in the previous 7/14 days | XI-level feature in V2x | PL minutes [have]; other competitions [none] | V2x |
| F3 | **[done, §5.17: null]** **Transfer priors in the XI** (finishes §5.12) | replace the position prior for arrivals | [have] for 2022-25 | V2x with position prior |
| F4 | **[done, §5.15: null]** **Promoted-team prior from Championship data:** shots-on-target strength in the Championship mapped to first PL games by ridge, like §5.12 | prior on attack/defence for promoted sides | football-data E1 CSVs [free, same format] | DC with no prior (a fixed promoted prior already failed) |
| F5 | **Manager change:** days since a new permanent manager, decayed | stage-2 feature, or faster DC decay after a change | appointment dates [free: Wikipedia lists, not checked] | DC time decay alone |
| F6 | **Table stakes:** point-in-time gap to relegation, Europe and title places; "nothing to play for" late in the season | stage-2 features, active from about gameweek 30 | [have] (standings from our results) | V1x |
| F7 | **Goalkeeper quality:** saves and goals conceded relative to opponent shots on target, shrunk; flag when the usual keeper is out | XI-level feature | FPL saves and goals conceded [have]; no post-shot xG [none] | V2x (outfielders only) |
| F8 | **Player defensive ratings:** tackles + interceptions, duel win % per 90 for the opponent's back line | XI-level feature | HF 2021-25 [have]; FPL defensive actions 2025-26+ [have] | back-line regularity |
| F9 | **Penalties and finishing:** split xG into non-penalty xG; penalty-taker absence; shrunk goals-minus-xG | XI-level feature | FPL xG and penalties [have, partial] | V2x |
| F10 | **Set pieces:** opponent-adjusted corner strength | stage-2 feature | football-data corners [have] | V1x |
| F11 | **Referee:** shrunk per-referee fouls, cards and penalty rates | stage-2 feature on total goals / draws | football-data `Referee` [have] | none (expected to be null) |
| F12 | **Weather:** rain, wind and temperature at kickoff | stage-2 feature on total goals | Open-Meteo historical [free]; live must use the *forecast*, not the observed value | V1x |
| F13 | **Kickoff slot and travel:** early / evening / midweek, distance between grounds, derbies | stage-2 features | stadium coordinates [free, static] | V1x |
| F14 | **Rating ensemble:** an Elo built only from our results, blended with DC | average of probabilities, weight fitted on dev | [have] | DC alone |
| F15 | **Score distribution:** bivariate Poisson or negative binomial for over-dispersion and draws | stage-1 likelihood | [have] | DC with ρ |

Out of scope by the rules in §2: bookmaker odds, FPL ownership/transfers/prices, and anything
behind bot protection. Distance covered and other tracking data: [none].

My concern is power (§6.1): most of these probably move log-loss by under 0.002, which needs
well over 1,000 matches to detect. I would rather pre-register two or three than test fifteen.

## 8. What I want from you (the reviewer)

Your previous requests are done (§5.14–5.18) and the prospective log starts at gameweek 10. This
time I want help with one problem: **forecasting newly promoted and returning teams** (§5.20).
Please give me:

1. **A critique of the diagnosis in §5.20.** Is the problem really early-season over-reaction and
   stale ratings, or does the evidence point elsewhere (for example promoted sides' home advantage,
   or ordinary noise in a small subgroup)? What check would distinguish these?
2. **A ranking of R1–R5** by expected gain per unit of effort, which to combine, and which are likely
   null. If you would build something not on the list, describe it and the simpler alternative it
   must beat.
3. **How to learn the promoted prior with so few cohorts:** whether pooling second tiers (R3) is
   sound given that leagues differ in strength, and how to set the shrinkage for promoted teams'
   early matches (R2) without fitting it to the same handful of seasons I evaluate on.
4. **An evaluation design:** a primary contrast and subgroup definition (first N matches? first
   season?), how to use 2016-21 as clean data, and the power I can expect.

Please flag any claim you are unsure about instead of stating it as fact, and do not suggest using
bookmaker odds or crowd data as model inputs (pre-season odds and bookmaker relegation prices
included).
