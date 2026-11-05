# Leakage and evaluation audit: V0–V5 pipeline

Audit date: 2026-11-05. It responds to the external review checklist. Each item's status is one of:
**verified (test)**, an automated test in `tests/`; **verified (check)**, a one-off check on the
real data whose result is recorded here; **by construction**, enforced by the code path; or
**limitation**, which cannot be verified and is disclosed.

| # | Risk | Status | Evidence / handling |
|---|---|---|---|
| 1 | Stage-2 offsets fitted in-sample | **by construction, verified (check)** | Each match's Dixon-Coles offset comes from a fit at *that match's* cutoff: the first kickoff of its local day, using results known 2h30m after kickoff (`experiments.build_dataset`). Stage-2 training rows reuse those stored offsets, so no offset is ever fitted on the match itself. Checks: 0 of 3,220 rows have a cutoff after kickoff, and in all 489 fits the latest training result was known ≥ 13.5 h before the fit's cutoff. |
| 2 | Reconstructed XIs use the predicted match's minutes | **verified (check)** | 136 matches (all before 12 Nov 2022) have starters inferred as the 11 players with most minutes. Every one falls before stage 2 switches on (first active cutoff 2023-04-16), so all variants score them as identical Dixon-Coles predictions. After that they serve only as *past* training rows. They play no part in any contrast. |
| 3 | Features not published by kickoff + 2h30m | **limitation** | The 2h30m lag is our convention. FPL xG/xA/ICT and football-data match stats are end-of-season archive values, which may include later corrections. Their publication times cannot be confirmed without historical snapshots. The live 2026-27 data is fetched after the fact, too. |
| 4 | Matchday processing | **by construction** | Cutoffs come from actual kickoff timestamps; rescheduled games carry their played date. All matches on one local day share the day's earliest kickoff as cutoff, which is conservative for later games. Archive double gameweeks are keyed by fixture id. Live-API double gameweeks would be dropped and counted (0 so far). |
| 5 | Preprocessing fitted on future data | **by construction, verified (test)** | Stage-2 scalers and imputers are fitted on training rows only (`run_variant`). Position priors, the similar-player pool and team usage are built from appearances known at the cutoff. `tests/test_features.py` shows that adding extreme future appearances leaves ratings and match features unchanged. |
| 6 | Club and position from current or end-of-season records | **by construction** | Club comes from each historical appearance row. Position is the player's last FPL position before the cutoff. In the live season, a player's side comes from the fixture's own bonus-point list, not the current-club field. |
| 7 | Invalid score distribution after stage 2 | **verified (check)** | The smallest Dixon-Coles correction factor across all adjusted predictions is 0.575 (> 0). The score grid now runs 0–15 goals per side. The largest adjusted rate is 5.41, whose one-side tail beyond 15 goals is 1.7e-4 (corrected from "below 1e-4"; it was 2.3% beyond 10). Grids are renormalised; the largest omitted joint mass across the ledger forecasts is 1.2e-5. Results changed by under 0.0001. |
| 8 | Penalty meaning drifts with sample size | **limitation (defined)** | Both stages minimise a *summed* loss plus a fixed penalty (Dixon-Coles weights normalised to mean 1). As the training window grows, the penalty therefore weakens relative to the data, as a fixed prior would. This is recorded in `results/fpl_v0_v5/manifest.json`. |
| 9 | Market benchmark mislabelled | **fixed** | Pre-closing averages (collected Friday/Tuesday afternoon) have *earlier* information than a confirmed-XI forecast; closing averages have *later* information. Both are labelled. Probabilities are `(1/o_k)/Σ(1/o_j)` on the average-odds columns. |
| 10 | Promoted-team flags | **by construction** | Built from each season's team list, which is known before the season starts. The flag is unused by default anyway. |
| 11 | Multiple comparisons / forking paths | **limitation** | Four primary contrasts were fixed only *after* the first test-season run, so 2024-26 now counts as development evidence for anything chosen next. Feature definitions, decay settings, shrinkage and neighbour counts were set by hand, not tuned, but they were not pre-registered. Confirmation moves to a pre-registered prospective set (`docs/PREREGISTRATION.md`). |
| 12 | Bootstrap ignores dependence | **addressed** | Each contrast reports fixture-level intervals and calendar-block intervals (7/14/28-day blocks). The 2026-27 block intervals rest on only 3–12 blocks and are unreliable. |

## Paired incremental results (log-loss; lower is better)

These come from the frozen ledger `results/fpl_v0_v5/ledger.csv`. "SE" is the paired SD / √n.

| Contrast | Dev 2022-24 (760; stage 2 on for 457) | Test 2024-26 (760) | 2026-27 (90) |
|---|---|---|---|
| **P1** XI strength given xG form (V2−V1) | +0.0002, SE 0.0007 | **−0.0040, SE 0.0017**; 14-day block CI [−0.0070, −0.0012] | −0.0044 [−0.0160, +0.0077] |
| **P2** similar-player vs position prior (V3−V2) | −0.0003, SE 0.0005 | +0.0002, SE 0.0007 | −0.0036 [−0.0089, +0.0016] |
| **P3** style main effects given xG form (V1s−V1) | −0.0009, SE 0.0026 | +0.0027, SE 0.0018 | +0.0083 [−0.0014, +0.0189] |
| **P4** style interactions given main effects (V5−V4) | −0.0003, SE 0.0019 | −0.0018, SE 0.0021 | +0.0066 [−0.0055, +0.0186] |
| xG form (V1−V0) | −0.0011 | −0.0022 [−0.0050, +0.0006] | −0.0002 |
| continuity only (V1c−V1) | +0.0004 | −0.0009 [−0.0023, +0.0006] | +0.0019 |

How to read these:
- **P1 is the only contrast with a non-zero test effect.** Every fixture and block interval
  excludes zero (z ≈ 2.3, p ≈ 0.02). It does **not** survive a Bonferroni correction across the
  four primary contrasts (threshold 0.0125), and it was null on the dev seasons. Status:
  *candidate effect, unconfirmed.*
- **P2–P4 show no detectable benefit.** The similarity prior's player-level surrogate gain (MSE)
  does not carry through to match log-loss.
- **Power depends on the contrast.** Paired SDs run from 0.013 (P2) to 0.070 (P3), against
  roughly 0.17 for model-vs-market, so close variants can be compared much more precisely than
  the brief first claimed.
- **Market:**
  - Test: V2 − pre-closing = +0.0051 [−0.0072, +0.0176]; V2 − closing = +0.0087 [−0.0041, +0.0216].
  - Dev: V2 trails the pre-closing market by 0.024 [0.011, 0.037]. That gap is unexplained.
- **Draws:** test seasons had 197 observed vs 180.2 predicted by V2 (SD ≈ 11.8); dev had 169
  observed vs 167.5. There is mild evidence of draw under-prediction on test, which supports the
  calibration experiment.
