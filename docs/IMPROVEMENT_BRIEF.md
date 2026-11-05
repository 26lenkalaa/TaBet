# TaBet: brief for an outside review of model improvements

> Paste this whole file into a chat with another AI assistant (e.g. ChatGPT) and ask for the review
> described in the last section. It is self-contained: the reviewer does not have the code.
> Numbers marked **pending** were not yet available when this was written.

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

## 4. Current model

**Stage 1, team strength (Dixon-Coles).** Poisson goals with per-team attack/defence ratings,
home advantage, the Dixon-Coles low-score correction, exponential time decay and an L2 penalty.
Tuned on 2022-23 + 2023-24: decay half-life ≈ 173 days, L2 = 10. A "promoted team" prior offset was
tried and did not measurably help, so it is off.

**Stage 2, adjustments.** `log(rate) = log(Dixon-Coles rate) + b0 + b·x`: a penalised Poisson
regression fitted walk-forward on earlier matches only. Feature groups:

- team xG form: decayed xG for/against (from summed FPL player xG)
- XI strength: starting outfield players' decayed (xG + xA)/90 ratings, shrunk toward a prior,
  divided by the team's usual XI strength; plus the opponent's back-line regularity
- player prior: either the **position mean** or the mean of the **10 most similar established
  players** (by per-90 FPL threat/creativity/influence within position)
- team style: decayed, z-scored shots for/against, corners for/against, fouls and xG per shot, as
  main effects and as attack-style × opponent-defence-style interactions

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
so precision has to be judged per contrast. Calibration is good overall, slightly overconfident on strong home
favourites. Draws are never predicted as the favourite (predicted range 7–36%, observed rate 26%).

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
- **Pre-registration** stays as written: V2 − V1 on 2026-27 matches from gameweek 10 onward,
  14-day block CI excluding 0. Any revised lineup feature will be reported separately.

### 5.5 Restricted calibration layer (your previous #1)

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

### 5.6 Opponent-adjusted xG strength (your previous #1)

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
- **A small, consistently signed gain** in every split. It is only individually significant on
  the (already reused) test seasons, and about one tenth the size of the XI-strength effect.
- **Using raw and adjusted xG together is worse** than adjusted alone. The two are nearly
  collinear.
- **Proposed decision:** adopt adjusted xG as the xG reference in place of raw (same number of
  features, never worse), and report this as development evidence. The pre-registered V2 is
  unchanged; V2x would be confirmed separately.

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
3. **Lineups.** Results assume confirmed XIs. We have not evaluated projected lineups, and we have
   no historical injury/suspension timeline (FPL only exposes *current* injury news).
4. **Cross-league player priors** (new signings) need non-PL player data for 2021-27, which we do
   not have from an accessible source.
5. **Draw modelling.** Dixon-Coles under-represents some low-scoring draws. Is a bivariate Poisson,
   a draw-inflation term, or an ordered-probit layer worth trying?
6. **Model class.** Would a gradient-boosted model on the same point-in-time features, or a
   hierarchical Bayesian version of Dixon-Coles, be a better second stage at this sample size?

## 7. What I want from you (the reviewer)

Please give me:

1. **The 5 highest-value improvements** to try next, ranked by expected log-loss gain relative to
   effort, each with *why* it should help, *what data* it needs (only from the sources in §3, or a
   named source that allows plain programmatic access), and *how to test it* under the rules in §2.
2. **Problems with the evaluation design** in §4–5: leakage risks, multiple-comparison issues,
   anything that would make the reported gains unreliable.
3. **Ideas to increase statistical power** (question 6.1) that do not break the rules.
4. For each suggestion, the **simpler alternative it must beat** on identical chronological fixtures.

Please flag any claim you are unsure about instead of stating it as fact, and do not suggest using
bookmaker odds or crowd data as model inputs.
