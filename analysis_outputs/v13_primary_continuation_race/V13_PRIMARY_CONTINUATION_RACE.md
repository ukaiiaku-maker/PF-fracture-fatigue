# V13 inherited companion versus renewed primary

**BRANCHING_KINETICS_MODEL_UNCALIBRATED**

The inherited-clock/primary-continuation frozen pattern passes. The older d325b95 audit and its trajectories are immutable. No pair barrier, commitment rate, arrest rate, material change, or branch RNG is introduced. The option is default-off (`--v13-inherited-primary-race`).

## Clock semantics

Both clocks are read at the canonical accepted endpoint. Thresholds are cumulative: remaining primary action is the next cumulative threshold minus current action, not the absolute threshold. Pending completed events have zero remaining time without consuming or renewing them. A companion must strictly beat both the primary and tau_c; exact pair admissibility remains required. With multiple candidates, the earliest admissible completion wins; exact admissible ties fall back to the canonical single arm.

The raw first-passage reconstruction is reported separately, using both clocks at that common time origin. It is not mixed with endpoint action. The raw reconstruction has the archived absolute-time roundoff bound. Both origins give the same eight outcomes. No subgrid time or extra emission update is applied.

## Frozen physical states

| Case | Reach (µm) | Primary next (µs), endpoint | Companion (µs), endpoint | Primary next (µs), raw origin | Companion (µs), raw origin | Pair margin (J/m) | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Peak_1000K | 7.04416 | 0.46357175 | 5.122944e+27 | 0.51449763 | 5.1253438e+27 | 0.013205688 | single |
| Peak_1000K | 26.94407 | 0.72624634 | 0.11814375 | 0.73118041 | 0.12307796 | 0.12024524 | branch opportunity |
| Peak_1000K | 52.52306 | 1.7837801 | 2.6965687 | 1.8152888 | 2.7280774 | 0.15196801 | single |
| Peak_1000K | 101.83218 | 0.44312941 | 0.2103242 | 0.46698258 | 0.23417726 | 0.26732023 | branch opportunity |
| weakT_1000K | 7.04416 | 0.46344202 | 1.0181289e+23 | 0.51449763 | 1.01813e+23 | 0.0066977367 | single |
| weakT_1000K | 26.94407 | 0.72611662 | 0.12165527 | 0.73118041 | 0.12671908 | 0.061474338 | branch opportunity |
| weakT_1000K | 52.52306 | 1.7836504 | 2.7000802 | 1.8152888 | 2.7317186 | 0.077690819 | single |
| weakT_1000K | 101.83218 | 0.44664093 | 0.21019447 | 0.46698258 | 0.23053597 | 0.13667487 | branch opportunity |

The correlation window is 1 µs. All eight original first-event controls also lose to expiry independently of the primary rate. Every candidate clock, action, threshold, ordinal, seed identity, full process/RNG fingerprint, raw/effective balance and source hash is in the per-event JSON.

Primary kinetics follow the actual production selection: valid local J when admissible, otherwise the independently evaluated same-plane marginal energy. The companion uses the previously qualified single-to-pair marginal energy. These are model-native/discrete mechanics, not applied remote K or continuum-qualified G. Fixed-state primary-increment solves were performed; no parent trajectory was regenerated. Existing pair mechanics were reused.

The source callback was exercised on all eight exact archived states with hash-verified cached FEM results and exact request-topology checks. Accepted marks preserve baseline clock/RNG/process/event-counter objects; losing states retain the exact canonical single-state object. An AST regression verifies that the default-off canonical parent body is unchanged apart from explicit observer/overlay hooks and the already-qualified diagnostic aliases.

## Ideal exponential ensemble identity

For independent exponential residual clocks at constant rates, P_branch = lambda_j/(lambda_i+lambda_j) × [1−exp(−(lambda_i+lambda_j)tau_c)] × A_pair. Deterministic quadrature verifies the identity. Equal saturated rates give 0.43233235838169365; primary-first has the same probability, and neither completes with probability exp(−2). This ideal identity is not an estimated probability for conditioned archived histories or a calibrated material probability.

## Short physical gate

Preregistered: Peak/weak-T at 300/1000 K; common seeds 3621–3624; at most two workers. Each case stops at first branch plus 20 µm additional growth of a daughter (25 µm total daughter length), 125 µm maximum forward reach, or an existing legitimate gate. No automatic retry, seed screen, parameter tuning, or long atlas. The V13 mark ledger is distinct from the unchanged baseline event counters.

Terminal records available: 0/16. The short ensemble is not complete; no final incidence conclusion is claimed.

The frozen evidence supports history-dependent, nonmonotonic branch opportunities, not calibrated recursive branching physics. Earlier attribution found constant 1 µm radius and at most 2.12 ppm direct shielding/blunting correction: the later rate increase tracks geometry-dependent native mechanics, not demonstrated strong direct dislocation-state control. Four seeds per material/temperature support descriptive incidence and onset spacing only.
