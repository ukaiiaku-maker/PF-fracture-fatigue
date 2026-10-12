# Supporting Information

## Temperature- and rate-dependent fracture from competing crack-tip kinetics

## S1. Experimental-image scope

The two experimental composites contain selected transmission-electron-microscopy frames of crack opening in nominally 500 nm, 99.9% Ag leaf. One sequence shows extensive evolving line and band contrast described as dislocation motion and twinning; the second shows less visible plastic contrast. Both carry 200 nm scale bars. Frame selection is nonuniform, and synchronized force and raw timebase data are unavailable. The images therefore motivate the coexistence of opening and accommodation pathways but do not calibrate an activation landscape or validate a quantitative load–temperature curve.

![Companion crack-opening sequence with limited visible plastic contrast.](figures/Figure_S1_experimental_companion.png)

**Figure S1.** Selected frames from the companion Ag sequence with limited visible plastic contrast. Visibility depends on thickness, diffraction condition, orientation, and temporal resolution; limited contrast is not proof of absent plasticity.

## S2. Mechanical, stochastic, and state definitions

The common specimen, loading convention, and \(K_J\) sign convention are defined by Eqs. (1)--(5) of the main text. Force is reported per unit out-of-plane thickness. \(J_{\rm signed}\) is the native signed domain integral and \(\varsigma\) selects positive forward mode-I release. The common-domain observer used for the elastic cross-code control is evaluated on identical nodes, elements, damage, elastic tensor, and supports.

The opening mark is sampled once per renewal as \(\Xi=-\ln R\). The survival sum contains the represented discrete cleavage candidates. Persistent source population evolution is deterministic conditional on the current state and does not add another sampled point process. Event length is threshold-correlated by the native geometry law and is committed only after its opening clock crosses.

## S3. Executed activation surfaces and bounds

The EXP-floor surface is Eq. (4). The implementation uses \(T_{\rm ref}=481.33\) K, \(G_0\ge10^{-9}\) eV, \(\sigma_c\ge1\) Pa, and

\[
G_f=\min[0.95G_0,\max(10^{-4}\ {\rm eV},fG_0)] . \qquad (S1)
\]

Only \(\max(\sigma,0)\) enters this surface, and the exponent is clipped to \([-700,0]\). The DBTT emission stress scale reaches its 1 Pa lower bound near 1001 K. Consequently, the high-temperature branch samples a strongly softened, bounded emission surface; it is not a named-material DBTT prediction.

## S4. Persistent source and signed process-zone operator

The front state has two reduced slip systems and 80 bins over 50 µm. The executed source model is persistent_areal_sites_backstress_limited_no_inventory. Its multiplicity, implicit mean-population update, and complementarity gate are Eqs. (6) and (7). The source/channel value 141.0591 in Table S4 is a legacy-compatible parameter-field value and is not a finite consumable inventory in these runs. Runtime diagnostics require persistent_source_inventory_active=false and persistent_source_refresh_active=false.

Activation-to-line-content factors come from the selected two-dimensional signed unit-perturbation kernel. Positive and negative mobile, retained, accumulated-slip, and wake arrays remain separate. The exact exchange in Eq. (8) is followed by physical mobile advection; escape, wake transfer, and discarded wake content have separate ledgers. The selected monotonic rows set explicit temporal mobile and retained recovery to zero. The vacancy-transport positivity operator used by the cavity calculation is a different conservative field operator and is not this dislocation exchange step.

The selected Peierls and Taylor surfaces use the direct row parameters in Tables S3--S5, including their temperature/entropy terms and runtime bounds. Taylor completion supplies the retained-to-mobile release rate used in Eq. (8). Local adaptive integration accepts a full interval only when the full-step versus two-half-step state error is within the declared tolerance; the outer mechanics/event transaction then accepts or retries without partially committing geometry.

## S5. Moving coordinate, geometry commitment, and observers

The hazard-linear translation \(d\xi=L_c dB\) is an executed coarse-graining assumption. It translates signed state and accumulated slip into the wake but does not renew persistent sites. The sharp-wake geometry changes only at clock completion. \(J\), \(K_J\), reaction, and stress belong to their equilibrated or exact pre-event state; length belongs to the paired committed geometry. No post-commit \(K_J\) is assigned unless a new equilibrium was actually observed.

The cavity family uses current-front ownership, positive conservative vacancy transport, and topology-specific candidates. A local nested \(J\) and a finite complete-ligament operation are different observables. Their ratio is not reported as a shielding factor.


## S6. Material parameters

Energies are in eV, energy slopes in eV K\(^{-1}\), stresses in GPa, stress slopes in GPa K\(^{-1}\), and densities in m\(^{-2}\). A machine-readable table supplied with the article preserves all digits.

### Table S1. Opening parameters

| Family | \(G_{00}\) | \(g_T\) | \(\sigma_{c0}\) | \(s_T\) | \(\alpha\) | \(n\) | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Peak | 4.0118 | 0.0081468 | 7.3496 | -0.0012100 | 1.3228 | 1.2558 | 0.018451 |
| DBTT | 2.3577 | 0.0056085 | 6.7501 | -0.0012041 | 0.49684 | 2.6188 | 0.026239 |
| Weak-T | 3.0409 | 0.0064702 | 5.0254 | -0.0014868 | 0.97559 | 1.0790 | 0.025475 |
| Ceramic-like | 3.2733 | 0.0085807 | 4.6647 | -0.0020771 | 1.0759 | 1.0818 | 0.022991 |

### Table S2. Emission parameters

| Family | \(G_{00}\) | \(g_T\) | \(\sigma_{c0}\) | \(s_T\) | \(\alpha\) | \(n\) | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Peak | 2.0446 | 0.0009420 | 7.2052 | 0.0013326 | 0.085872 | 1.1423 | 0.0058420 |
| DBTT | 3.3449 | 0.0061668 | 2.7143 | -0.0052233 | 0.15725 | 0.51562 | 0.010368 |
| Weak-T | 2.9331 | -0.00042943 | 1.4962 | -0.000042793 | 0.26802 | 1.5518 | 0.064524 |
| Ceramic-like | 3.1909 | -0.0024372 | 8.7375 | 0.0040232 | 0.43370 | 0.74084 | 0.0083828 |

### Table S3. Peierls and Taylor parameters

| Family | \(H_{0,p}\) | \(\Delta S_p/k_B\) | \(\alpha_p\) | \(n_p\) | \(H_{0,t}\) | \(\Delta S_t/k_B\) | \(\alpha_t\) | \(n_t\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Peak | 6.3684 | 12.2266 | 1.9098 | 1.1373 | 1.6987 | 38.4538 | 0.45612 | 1.2507 |
| DBTT | 3.5257 | -36.3393 | 1.4484 | 1.8713 | 1.4998 | 31.3659 | 0.70489 | 0.53896 |
| Weak-T | 4.8567 | -30.9531 | 1.0104 | 2.3834 | 1.8921 | 39.5363 | 0.52685 | 1.4233 |
| Ceramic-like | 6.5849 | -5.1265 | 1.4449 | 0.79892 | 1.1080 | -9.2766 | 0.74570 | 2.2320 |

### Table S4. Correlation and feedback parameters

| Family | \(\rho_c\) | correlation scale | source/channel | encounter efficiency | \(c_{\rm blunt}\) |
|---|---:|---:|---:|---:|---:|
| Peak | \(5.8780\times10^{13}\) | 0.86390 | 141.0591 | 9.16025 | 2.6873 |
| DBTT | \(2.2042\times10^{13}\) | 1.2690 | 141.0591 | 9.16025 | 2.9292 |
| Weak-T | \(2.0520\times10^{16}\) | 0.10446 | 141.0591 | 9.16025 | 1.5308 |
| Ceramic-like | \(1.2482\times10^{17}\) | 1.8464 | 141.0591 | 9.16025 | 0.46276 |

### Table S5. Common state parameters

| Quantity | Value |
|---|---:|
| Reference temperature | 481.33 K |
| Slip channels | 2 |
| Process-zone bins / length | 80 / 50 μm |
| Forest-density floor | \(5.0\times10^{12}\ {\rm m}^{-2}\) |
| Peierls and Taylor stress fractions | \(1/\sqrt{3}\) |
| Opening / emission attempt frequency | \(10^{12}\) / \(10^{11}\ {\rm s}^{-1}\) |
| Peierls / Taylor attempt frequency | \(10^{12}\) / \(10^{11}\ {\rm s}^{-1}\) |
| Mobile shielding fraction | 0 |
| Temporal source, mobile, and retained recovery | 0 |
| Source reference area / front width / zone length | 25 μm² / 10 μm / 2 μm |

## S7. Current 96-case baseline and seed uncertainty

The current baseline contains four fixed rows, twelve temperatures (300, 600, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, and 1300 K), and two arms. Seed 1733, 0° orientation, unit loading-rate factor, and tip-surface source geography are held fixed. All 96 cases have verified runtime attestations and physical first events.

| Row | Enabled range (MPa√m) | Opening-only range (MPa√m) | Largest paired change |
|---|---:|---:|---:|
| DBTT | 22.08–55.11 | 20.54–23.23 | 156.1% at 1050 K |
| Peak | 22.17–54.90 | 19.35–23.33 | 163.7% at 900 K |
| Weak-T | 18.52–20.14 | 12.71–13.15 | 53.4% at 600 K |
| Ceramic-like | 9.08–13.38 | 8.45–13.38 | 11.7% at 950 K |

Eight paired seeds at DBTT/1200 K and Peak/900 K were chosen before inspecting outcomes. The two-sided Student-t intervals use 7 degrees of freedom and assume independent seeds and approximate normality of paired gaps. They estimate the mean gap, not population tails.

| Anchor | mean gap | sample SD | empirical range | 95% t interval |
|---|---:|---:|---:|---:|
| DBTT, 1200 K | 32.45 | 0.88 | 31.24–33.50 | 31.71–33.19 |
| Peak, 900 K | 34.74 | 0.59 | 33.63–35.22 | 34.25–35.23 |

All numerical entries are in MPa√m. Full precision and individual seeds are in `figure_data/PF_FOCUSED_SEEDS_PAIRED.csv`.

## S8. Current rate, orientation, and source matrix

Forty predeclared rows test rate factors 0.01, 1, and 100 at five anchors, 0°/45° orientation at four selected conditions, and three source geographies at DBTT/1200 K. Every row reaches a physical first event. The largest absolute rate change relative to 1× is 52.12%; the 45° minus 0° changes range from −61.91% to 0.00%. At DBTT/1200 K the tip-surface, bulk-uniform, and tip-plus-bulk results are 52.376, 20.934, and 52.414 MPa√m. The 45° operator is the original two-pass self-consistent signed-kernel family; the target-family kernel arrays are unchanged between capture/build passes.

## S9. Developed-growth observer contract

The 300 and 1200 K paths each contain 203 geometry events and 3695 chronological observation rows across the paired dataset. A resistance row uses the native J-derived \(K_J\) measured at the exact pre-event state and associates it with only the subsequent realized committed jump. The committed coordinate has no invented post-event J or K. Queries at 100, 500, and 1000 µm preserve the measured lower pre-event and upper committed coordinates; interpolation is false for every query.

| Temperature | Initial event coordinate | observed range | terminal event coordinate | projected / arclength |
|---:|---:|---:|---:|---:|
| 300 K | 23.234 | 23.234–67.807 | 49.039 | 1004.926 / 1008.560 µm |
| 1200 K | 52.376 | 52.376–153.307 | 110.873 | 1004.926 / 1008.560 µm |

The paths have one monotonic loading segment and zero independent reload-resistance points. Active mobile-plus-retained population is a current state and is never relabeled cumulative emission.

## S10. Common-specimen PF–FEM/CZM contract

The shared specimen is plane strain, 2 mm × 4 mm, with a 0.5 mm initial notch and total opening imposed as +U/2 and −U/2. Force is per unit out-of-plane thickness. Identical-mesh elastic calculations match reaction, stored energy, ahead-tip stress, common-domain J, and K at total openings 0.2, 10, and 30 µm with maximum relative difference zero at stored precision.

The production histories were additionally compared at common pre-event openings of 0.2, 2, 5, 10, and 12 µm. Across these points, the PF/FEM reaction ratio is 1.09227. The native PF/FEM \(K_J\) ratio is 0.58646 for tip-only and 0.44091 for full-field states, reflecting different production observer supports. The differences exist before fracture and cannot be inferred from event endpoints at different openings.

| Scope | PF first-event K | FEM/CZM first-event K | PF / FEM event opening |
|---|---:|---:|---:|
| Tip-only | 53.878 | 54.212 | 31.723 / 18.720 µm |
| Full-field | 51.110 | 54.407 | 26.610 / 12.492 µm |

K values are MPa√m. The event values are reported as native first-passage observations, not parity metrics. The retained production data do not contain a two-level mesh sequence for each backend, so no production convergence rate is claimed. The FEM/CZM accepted state contains four physical events, 26 fully separated numerical cohesive subsegments, and 20.828 µm total cohesive length.

## S11. Recovered physical cavity continuation

The DBTT/1200 K/seed-3621 pair uses the qualified full-field plasticity and positive conservative vacancy transport. After the unchanged migration dwell and accepted ramp suffixes, the final matched continuation runs from physical time 6.0010201 to 6.2010201 s and from total opening 6.4010101 to 6.6010101 µm.

| Quantity | Cavity enabled | Cavity suppressed |
|---|---:|---:|
| Graph-length change | 0 µm | 80 µm |
| Terminal reaction | 575712.14 N m−1 | 501039.17 N m−1 |
| Terminal admissible energy drive | 43.6472 J m−2 | 4155.24 J m−2 |
| Terminal effective K | 3.17371 MPa√m | 30.9661 MPa√m |
| Terminal effective rate | 5.39093×10−58 s−1 | 3.99577 s−1 |

The enabled cavity grows from 42.4796461 to 42.4878668 nm. No connection forms, and all fractured/free-span ledgers remain zero. The suppressed arm accepts four 20 µm ordinary increments. The exact native pair checkpoint survives a round trip. Because a generic JSON normalizer failed after execution, exact event times within this 0.2 s suffix are unavailable and are not reconstructed.

## S12. Preserved active restart and combined topology

At 15.462475 MPa√m and R=0.1, interrupted and uninterrupted V3.6 terminal payloads are exactly equal. The terminal state contains maximum mobile density 5.348994×10^15 m−2, cumulative source activation 1.097165×10^4, and nonzero material-frame plastic shear/cumulative swept ledgers (1.882753×10−13). The positive-R waveform does not reverse slip sign.

The public combined branch–cavity route preserves root branch `b00000000`, active child `void-front-1`, and cavity `void:site-1`. Its terminal graph length is 795 µm and cavity radius 55 µm. Restart reproduces the uninterrupted terminal fingerprint exactly while conserving the field-transfer ledgers to their declared tolerances. This is a compatibility qualification of existing laws, not a recursive-branch calibration or production fracture result.

## S13. Cohesive and energetic comparison

For cohesive energy \(\phi(\delta)\), traction \(t_n\), and activation area \(A_*\),

\[
\mathcal F(\delta)=A_*[\phi(\delta)-t_n\delta],
\]

\[
\Delta G_c^\ddagger=A_*\int_{\delta_m}^{\delta_s}
[t_{\rm coh}(\delta)-t_n]\,d\delta,\qquad
V_c^\ddagger=A_*(\delta_s-\delta_m).
\]

The derivative is taken at fixed cohesive landscape and traction-controlled loading ensemble. Different cohesive shapes with equal peak strength and work can still produce different barrier trajectories and activation volumes.

Baseline AT2 regularization uses \(w(d)=d^2\), whereas AT1 uses \(w(d)=d\). Their different behavior near \(d=0\) gives AT1 a finite elastic onset and AT2 no finite onset in the baseline formulation. Both can be extended with viscosity, thermal fields, plasticity, or cohesive degradation. The present model differs in assigning conditional physical-time rates to opening and accommodation and in retaining a front-local population history.

![Separate branching morphology example.](figures/Figure_10_branching.png)

**Figure S4.** Accepted branching stress fields and crack graphs from the separate morphology calculation. The plate demonstrates a broader spatial crack capability but is not part of the single-front temperature and rate dataset.

## S14. Figure and data provenance

Every current quantitative plot is copied from, or regenerated solely from, the accepted CSV/JSON records in `review_packet/final_fracture_campaign_a5a5883`. Historical C19 field plates retained in Figures 9 and 10 are actual original full-order fields and are labeled for their limited geometric/provenance role. `reproducibility/CLAIM_TO_EVIDENCE_INDEX.json` maps major claims to exact files and `SOURCE_AND_INPUT_MANIFEST.json` records SHA-256 hashes and producer roots. No reduced/hybrid result is included in a numerical table or figure.
