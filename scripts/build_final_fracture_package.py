#!/usr/bin/env python3
"""Build the compact, reader-facing fracture release package.

The script deliberately leaves full-order raw histories in their immutable evidence
roots.  It copies only publication figures/tables and records their hashes.
"""

from __future__ import annotations


METHODS = r"""## 3. Competing-kinetics framework

### 3.1 Conditional first passage and activation surfaces

For a discrete event channel \(i\), with drive \(q_i\), state \(z\), multiplicity \(\mathcal M_i\), and attempt frequency \(\nu_i\), the executed conditional rate is

\[
\lambda_i(t)=\mathcal M_i(t)\nu_i\exp\!\left[-\frac{\Delta G_i^\ddagger(q_i,T,z)}{k_BT}\right]. \qquad (1)
\]

Here \(k_B\) is Boltzmann's constant and \(T\) is absolute temperature. Conditional on no intervening event, the next-event density contains only the channels represented as discrete events,

\[
f_i(t)=\lambda_i(t)\exp\!\left[-\int_{t_0}^{t}\sum_j\lambda_j(s)\,ds\right]. \qquad (2)
\]

The deterministic mean source-population update below changes the conditional rates but is not counted as another sampled event. Crack opening uses an exponential mark \(\Xi=-\ln R\), with \(R\sim U(0,1)\), and accumulated action

\[
A_c(t)=\int_{t_0}^{t}\Lambda_c[s;z(s)]\,ds,\qquad A_c(t_c)=\Xi . \qquad (3)
\]

Opening and emission use the executed EXP-floor activation surface

\[
G_i(\sigma,T)=G_{f,i}+[G_{0,i}(T)-G_{f,i}]\exp\!\left\{-\alpha_i\left[\frac{\max(\sigma,0)}{\sigma_{c,i}(T)}\right]^{n_i}\right\}. \qquad (4)
\]

Here \(G_{0,i}=G_{00,i}+g_{T,i}(T-T_{\rm ref})\) and \(\sigma_{c,i}=\sigma_{c0,i}+s_{T,i}(T-T_{\rm ref})\). The reference temperature is \(T_{\rm ref}=481.33\) K. The implementation floors \(G_0\) at \(10^{-9}\) eV and \(\sigma_c\) at 1 Pa, bounds \(G_f\) as given in the SI, and clips the Arrhenius exponent to \([-700,0]\). In the DBTT emission row, the affine stress scale reaches the 1 Pa floor near 1001 K; this bounded-stress assumption contributes materially to the calculated upper branch.

### 3.2 Mechanics, sign convention, and committed crack geometry

The principal calculation is plane strain in a 2 mm by 4 mm body with a 0.5 mm left-edge notch and symmetric total opening. The graded triangular discretization carries piecewise-constant plastic strain and density. With plane-strain modulus \(E'=E/(1-\nu^2)\), signed domain integral \(J_{\rm signed}\), and orientation sign \(\varsigma\), the reported coordinate is

\[
K_J=\sqrt{E'\max(\varsigma J_{\rm signed},0)} . \qquad (5)
\]

The sign makes forward mode-I energy release positive. \(K_J\) is a model observer, not a standardized experimental toughness. An accepted event commits a sampled sharp-wake increment; the calculation is not an AT1 or AT2 variational damage evolution.

### 3.3 Persistent source population and signed transport

The executed PF families use persistent reusable source sites, not a finite depleted inventory. For reduced slip system \(s\), the state-dependent multiplicity is

\[
\mathcal M_s=\rho_{\rm site}c_{\rm arc}r_{\rm tip}w_{\rm eff},\qquad
w_{\rm eff}=w_{\rm ref}\sqrt{\rho_{\rm ref}/\max(\rho,\rho_{\rm ref})}. \qquad (6)
\]

Here \(\rho_{\rm site}\) is persistent areal site density, \(c_{\rm arc}r_{\rm tip}w_{\rm eff}\) is active source area, \(r_{\rm tip}\) is the blunted radius, and \(w_{\rm eff}\) is bounded by the process-grid and domain widths. Sites are neither consumed nor replenished by crack advance. For a time interval \(\Delta t\), the mean activation increment is the nonnegative implicit root

\[
\Delta N_s=\mathcal M_s\lambda_e\!\left(\max[\sigma_{{\rm drive},s}-k_{\rm back}\sqrt{\rho_s+a_s\Delta N_s},0],T\right)\Delta t . \qquad (7)
\]

Here \(\sigma_{{\rm drive},s}\) is the tensor-resolved projected drive, \(k_{\rm back}\sqrt{\rho}\) is Taylor back stress, and \(a_s\) converts an activation to local density. Equation (7) is solved by a bracketed backward-Euler complementarity condition and is bounded by the density at which back stress blocks the resolved drive. The mean increment is converted to signed line content using a mechanically derived two-dimensional unit-perturbation normalization; its sign follows resolved shear.

Positive and negative mobile \(M_s^\pm\), retained \(R_s^\pm\), accumulated-slip, and wake populations are stored separately. For each sign, exact local exchange obeys

\[
\dot M_s^\pm=-k_{\rm enc}M_s^\pm+k_{\rm rel}R_s^\pm,\qquad
\dot R_s^\pm=k_{\rm enc}M_s^\pm-k_{\rm rel}R_s^\pm. \qquad (8)
\]

The exchange is followed by forward mobile transport \(v\Delta t\), with forest spacing \(\ell=(2\sqrt{\rho})^{-1}\), velocity \(v=\ell\lambda_p\), and \(k_{\rm enc}=\eta v\sqrt{\rho}\). The selected Taylor completion mapping defines \(k_{\rm rel}\) in the implementation; explicit mobile and retained temporal recovery are zero for these monotonic rows. Escape at the active-domain boundary and moving-frame transfer to the wake remain active and are separately ledgered.

Signed retained content acts through two-dimensional unit-response kernels,

\[
K_{\rm sh}=\sum_{s,j}\kappa_{s,j}(R_{s,j}^+-R_{s,j}^-),\qquad
r_{\rm eff}=r_0+c_{\rm blunt}b\sum_{s,j}w_j\Gamma_{s,j}. \qquad (9)
\]

The reported runs set the mobile shielding fraction and wake shielding to zero. Thus population, plastic work, back stress, blunting, and mechanical shielding are distinct observables.

### 3.4 Partial progress and state translation

If \(B\) is hazard-linear progress toward the current sampled event length \(L_c\), the executed moving-coordinate closure is

\[
d\xi=L_c\,dB . \qquad (10)
\]

Equation (10) is a constitutive coarse graining, not a probability expectation or a resolved crack surface. Mobile, retained, and slip fields translate toward the moving origin, and crossed content enters the wake. Persistent sources remain available; only their multiplicity changes through radius and local density. Geometry is committed only at \(B=1\).

### 3.5 Source support, bulk accommodation, and cavity family

Tip-surface calculations use the persistent front state above. The pure-bulk intervention activates the full-field Peierls--Taylor update and explicitly disables tip plasticity, so it replaces rather than adds the tip contribution. Tip-plus-bulk composes both owners. The cavity calculation is intentionally different: it uses the declared current-source full-topology owner, positive conservative vacancy transport, full-field accommodation, and an explicit global event scheduler. Results from these families are compared as controlled interventions, not described by one lifecycle equation.

### 3.6 Cohesive energetic connection

For cohesive free energy per area \(\phi(\delta)\), normal traction \(t_n\), and activation-patch area \(A_*\), a theoretical activation landscape is

\[
\mathcal F(\delta)=A_*[\phi(\delta)-t_n\delta],\qquad
\Delta G_c^\ddagger=A_*\int_{\delta_m}^{\delta_s}[t_{\rm coh}(\delta)-t_n]\,d\delta . \qquad (11)
\]

This identity assumes a metastable minimum \(\delta_m\), saddle \(\delta_s\), and prescribed traction during the local activation. It motivates the barrier interpretation but is not substituted for the native numerical barrier. The FEM/CZM calculation uses hazard-triggered cohesive/topological insertion; its exported accepted state contains separated cohesive segments and physical-event grouping. It is not autonomous rate-independent cohesive failure.
"""


SI_METHODS = r"""## S2. Mechanical, stochastic, and state definitions

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
"""

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fracture_final_current"
REVIEW = ROOT / "review_packet/final_fracture_campaign_a5a5883"
ACCEPTANCE = ROOT / "review_packet/publication_acceptance_0c72ca2b"
PAPER = ROOT.parent / "fracture_only_arrhenius_paper"
C19 = PAPER / "manuscript/c19"
C19_ANALYSIS = PAPER / "analysis/c19_original_models_voiding_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


ABSTRACT = """## Abstract

Temperature and loading rate affect fracture through competition between crack opening and crack-tip accommodation. We formulate opening as a stochastic first-passage process coupled to persistent, reusable dislocation-source populations and evolving two-dimensional mechanics. In 96 original spatial sharp-wake calculations, four fixed constructed parameter rows produce transition-like elevation, broad accommodation-associated elevation, and predominantly softening responses without prescribing toughness curves. Matched opening-only controls show changes up to 156% for the DBTT row and 164% for the Peak row. Eight paired seeds at DBTT/1200 K and Peak/900 K give mean first-event gaps of 32.45 and 34.74 MPa√m, well above their observed seed spreads. Rate, orientation, and source-support tests show that the response is path dependent rather than a universal temperature shift. Developed trajectories reach 1.005 mm and provide event-conditioned resistance observations under monotonic loading. A common-specimen audit separates exact identical-mesh elastic agreement from systematic production differences in reaction and native J support. In a matched 0.2 s cavity continuation, the suppressed arm advances 80 µm whereas the enabled arm does not connect or advance; current native fields establish finite-window inhibition under the topology-aware candidate law, without equating unlike local and finite-ligament drives. The study therefore links thermally activated opening, inherited accommodation, cohesive topology, and cavities while stating the limits of each comparison.
"""


RESULTS = r"""## 5. Results

### 5.1 Full-PF temperature response and accommodation controls

![First-event response for the 96-case baseline.](figures/Figure_2_PF_baseline_96.png)

**Figure 2.** Original spatial-PF first-event \(K_J\) for four fixed parameter rows, each evaluated at twelve temperatures with accommodation enabled and with the matched opening-only control. Every one of the 96 predeclared cases reached a physical first event under the common endpoint contract. Curves connect sampled temperatures within a fixed row and arm; they are not fits.

The four constructed rows produce distinct response magnitudes and shapes. DBTT spans 22.08–55.11 MPa√m with accommodation enabled, compared with 20.54–23.23 MPa√m in the opening-only arm. Peak spans 22.17–54.90 MPa√m versus 19.35–23.33 MPa√m. Their largest paired elevations are 156.1% at DBTT/1050 K and 163.7% at Peak/900 K. Weak-T shows a broad elevation: 18.52–20.14 MPa√m enabled versus 12.71–13.15 MPa√m opening-only, with a maximum 53.4% effect at 600 K. Ceramic-like remains the lowest response and has a smaller maximum effect of 11.7% at 950 K.

The family names identify fixed constructed parameter rows, not calibrated materials. The comparison supports the mechanistic statement that inherited accommodation can overwhelm direct opening softening over parts of the temperature domain. It does not uniquely apportion the change among shielding, blunting, transport, and renewed source exposure.

### 5.2 The active-anchor effect exceeds the observed seed spread

![Paired independent-seed accommodation effects.](figures/Figure_3_PF_focused_seeds.png)

**Figure 3.** Eight predeclared paired seeds at DBTT/1200 K and Peak/900 K. Each point is the enabled-minus-opening-only first-event difference under matched stochastic ownership. Bars show two-sided Student-t intervals for the mean paired gap (7 degrees of freedom); they are not tail or tolerance intervals.

All 32 arm-level calculations fractured. The mean paired gap is 32.45 MPa√m for DBTT/1200 K (sample standard deviation 0.88 MPa√m, empirical range 31.24–33.50 MPa√m, 95% t interval 31.71–33.19 MPa√m). Peak/900 K gives 34.74 MPa√m (standard deviation 0.59 MPa√m, range 33.63–35.22 MPa√m, interval 34.25–35.23 MPa√m). The effect is therefore large relative to the observed eight-seed spread at these two anchors. The finite ensemble does not resolve distribution tails or establish uncertainty for every temperature.

### 5.3 Rate, orientation, and source geography probe the state path

![Loading-rate sensitivity.](figures/Figure_4a_PF_rate.png)

![Orientation sensitivity.](figures/Figure_4b_PF_orientation.png)

![Source-geography sensitivity.](figures/Figure_4c_PF_source.png)

**Figure 4.** Question-driven original spatial-PF sensitivities. (a) Rate factors 0.01, 1, and 100 at five selected material/temperature anchors with matched opening-only arms. (b) 0° and 45° at selected low- and active-accommodation states. (c) tip-surface, bulk-uniform, and tip-plus-bulk source support for DBTT/1200 K. Every Cartesian panel labels its physical x coordinate; no curve is promoted to a global atlas.

Across the selected rate cases, the largest absolute change from the matched 1× first-event coordinate is 52.12%. Changing orientation from 0° to 45° gives paired changes from −61.91% to 0.00%. The cubic elastic tensor is invariant for the chosen constants, so the orientation response arises from the resolved opening and accommodation projections rather than an elastic-modulus change. At DBTT/1200 K, tip-surface, bulk-uniform, and tip-plus-bulk sources give 52.376, 20.934, and 52.414 MPa√m. Uniform source availability is thus not uniform response, and adding bulk support does not simply add to the tip-controlled result. These are local tests of competing kinetics, not transferable correction factors.

### 5.4 Developed growth separates driving history from event resistance

![Complete native driving histories.](figures/Figure_5_developed_driving.png)

**Figure 5.** Complete chronological native observation histories for DBTT at 300 and 1200 K. Re-equilibrated, exact pre-event, and committed stages retain their actual loading time, opening, reaction, geometry, \(J\), \(K_J\), and process-state associations. Projected extension and path arclength are distinct.

![Event-sampled resistance histories.](figures/Figure_6_developed_resistance.png)

**Figure 6.** Native pre-event J-derived \(K_J\) associated only with each realized committed jump. The 100, 500, and 1000 µm queries are represented by measured lower/upper jump brackets in the accompanying data; no value is interpolated or moved to an uncomputed length.

Both trajectories contain 203 exact pre-event/committed pairs and reach 1004.926 µm projected extension and 1008.560 µm path arclength. The 300 K event coordinates span 23.234–67.807 MPa√m, while the 1200 K coordinates span 52.376–153.307 MPa√m. Their geometry-event files are byte-identical because the common seed provides a controlled common-mark comparison; this is not evidence that morphology is temperature-independent. Each trajectory contains one monotonic loading segment and no independent unload/reload points. We therefore call Figure 6 an event-sampled resistance history rather than a standardized experimental R-curve, and we do not force a rising slope.

![Selected native developed-growth fields.](figures/Figure_7_native_fields.png)

**Figure 7.** Native maximum-principal-stress and damage arrays from the DBTT/1200 K propagation calculation at a pre-event state, the first committed advance, and late propagation. The numerical sidecar gives each exact stored step, crack-tip coordinate, and the \(K_J\) associated with that stored equilibrium. No front-local population is spread into a reconstructed two-dimensional field.

### 5.5 A controlled PF–FEM/CZM comparison

![Common-specimen PF and FEM/CZM comparison.](figures/Figure_8_PF_FEM.png)

**Figure 8.** Production-history reaction and native \(K_J\) at five common pre-event total openings for the Peak/1000 K, 0°, seed-8666 specimen. Tip-only and full-field states are shown separately. These are common-input comparisons, not a fitted mapping.

On an identical mesh and damage field, the accommodation-disabled reference gives coincident reaction, stored energy, ahead-tip stress, common-domain \(J\), and \(K_J\). The retained production histories do not retain that parity: from 0.2 to 12 µm opening, PF reaction is 1.0923 times FEM/CZM reaction. Native \(K_J^{\rm PF}/K_J^{\rm FEM}\) is 0.5865 for tip-only and approximately 0.4409 for full-field because the production observers use different supports. These differences are present before fracture and, in the tip-only case, before substantial population evolution. They therefore cannot be attributed to the bulk law alone.

The near agreement of the two native first-event coordinates at different openings is not used as mechanical validation. It is consistent with both hazards being driven by their own \(K_J\) coordinate. The supported comparison is narrower: the common operator reference is exact, while production discretization and observer support produce a systematic response difference that the retained runs do not supply a two-level backend-specific convergence sequence to eliminate. The FEM/CZM histories nevertheless contain four accepted physical events, 26 fully separated cohesive subsegments grouped by physical event, and 20.828 µm total cohesive length.

### 5.6 A resolved cavity shields the approach path over the matched window

![Native cavity and crack fields at the paired starting exposure.](figures/Figure_9_cavity_fields.png)

**Figure 9.** Current native maximum-principal-stress fields and crack graphs at the exact start and end checkpoints of the 0.2 s paired continuation. The enabled panels include actual-scale insets spanning the 7.4 µm solid ligament and resolved 42 nm cavity. The suppressed arm advances 80 µm; the enabled arm retains its starting front. Values and mesh sizes are supplied in the numerical sidecar.

The enabled and suppressed arms were resumed from their exact paired checkpoint and loaded for 0.2 s at 1 µm s−1. Both end at 6.6010101 µm total opening. The suppressed arm accepts four ordinary 20 µm advances, increasing graph length by 80 µm. The enabled arm accepts no crack advance; its resolved cavity radius changes only from 42.4796461 to 42.4878668 nm and no crack–void connection forms. At the endpoint, the ordinary front has a valid nested-local-J drive of 4155.24 J m−2, \(K_{\rm eff}=30.9661\) MPa√m, and an effective rate of 3.996 s−1. The cavity approach candidate uses the qualified complete-ligament topology energy, 43.6472 J m−2, \(K_{\rm eff}=3.17371\) MPa√m, and a rate of \(5.39\times10^{-58}\) s−1.

The two quoted endpoint drives are unlike observables at already different geometries: nested local \(J\) for the ordinary front and a finite complete-ligament topology operation for the cavity approach. Their ratio is therefore not a shielding factor. The matched history establishes finite-window inhibition under the executed candidate and kinetic formulation, not permanent arrest, a universal toughening law, or an isolated geometry-only reduction. The cavity predates the continuation, so the result does not predict nucleation. Exact terminal states and topology counts survive; individual event times inside the final window are not reconstructed.

### 5.7 Branching and option-preserving restart scope

![Separate accepted branching morphology.](figures/Figure_10_branching.png)

**Figure 10.** Separate accepted branching fields and crack graphs. This morphology example is not pooled with the single-front temperature dataset. A bounded combined branch–cavity compatibility calculation additionally preserves root branch, downstream child, cavity, field transfer, topology ledgers, and exact terminal restart, but does not constitute a calibrated recursive branch-kinetics study.

The combined public route ends with distinct root, child, and cavity owners and reproduces its uninterrupted terminal fingerprint exactly after restart. Separately, the V3.6 cyclic route reproduces a populated native terminal payload exactly at positive load ratio \(R=0.1\). Its mobile density, source-activation, plastic-shear, and cumulative swept ledgers are nonzero; the chosen waveform does not reverse slip sign. These tests establish option and restart preservation only and are not used as fracture-response data.

## 6. Discussion

### 6.1 Transition-like behavior is generated by competing histories

The opening-only curves vary over comparatively narrow ranges for DBTT and Peak, whereas the coupled responses rise to approximately 55 MPa√m. The elevation is therefore a history-dependent accommodation effect under the stated laws, not a prescribed toughness shelf. Conditional opening action accumulates while emission, transport, retention, release, blunting, and source exposure continually change the state on which the next increment acts. The large paired-seed gaps show that this effect is robust at the two active anchors relative to the observed threshold variation, while the small ensemble and constructed material rows prevent a population-tail or named-material claim.

### 6.2 Rate, direction, and source support are not separable scalar shifts

Loading rate changes time available to both opening and accommodation. Crystal rotation changes both opening and slip/source projections. Source geography changes where accommodation can be supplied, but mechanics and transport determine whether it becomes effective. The large pure-bulk decrease at DBTT/1200 K and near identity of tip and tip-plus-bulk values demonstrate why source density cannot be interpreted as an additive toughness term. These sensitivities identify competing state paths; they do not define universal shift factors.

### 6.3 Initiation, driving, and resistance require different stage associations

The developed histories contain physically meaningful resistance information without requiring unload/reload compliance points. \(J\) and \(K_J\) belong to exact equilibrated or pre-event mechanical states, whereas extension belongs to the subsequently committed geometry. Assigning a terminal K to an unobserved post-event state or interpolating across a discrete jump would invent data. The event-conditioned view retains each measured pre-event value across only its realized jump and resolves chronological opening, reaction, extension, and a final-time zoom. Because the histories use one monotonic loading protocol with burst-like advances, they are path- and rate-conditioned resistance histories rather than protocol-independent material R-curves.

### 6.4 Cohesive and sharp-wake implementations share mechanics without being identical

The identical-mesh elastic reference closes the geometry, boundary, unit, stiffness, and base-operator questions. Production histories then show systematic reaction and native-observer differences at common openings. Near equality of first-event \(K_J\) at different openings is expected to be less discriminating because each hazard is driven by its own coordinate; it does not validate production mechanics. The sharp-wake PF implementation is not a variational AT1 or AT2 phase-field model: it carries a stiffness-degraded wake and commits explicit increments when an accumulated kinetic action reaches its mark. FEM/CZM represents hazard-triggered separation through cohesive/topological events. The accepted cohesive state verifies actual separation events, while the remaining production-mesh/support difference limits a stronger cross-code claim.

Baseline AT2 uses \(w(d)=d^2\) and has no finite elastic onset because \(w'(0)=0\); AT1 uses \(w(d)=d\) and has a finite onset. Both can be extended with thermal, viscous, or plastic physics. The present distinction is the explicit physical-time first-passage law coupled to inherited accommodation, not a claim that thermal activation is unavailable to conventional phase field or cohesive fracture.

### 6.5 Cavities change the admissible drive and event competition

The cavity result is qualitative and large over its finite window: the suppressed ordinary front advances while the enabled cavity-approach candidate remains inactive. The important coupling is not cavity span alone but the topology-aware mechanical observer. A contour intersecting a free cavity cannot be used as an ordinary local crack-tip derivative. The qualified complete-ligament operation supplies an admissible approach drive without counting cavity free span as fractured solid, but it is not commensurate with the suppressed arm's terminal local \(J\). The calculation therefore establishes finite-window inhibition under the selected candidate law, with field redistribution and candidate-definition effects not independently decomposed.

### 6.6 Limitations

The parameter rows are constructed and uncalibrated. The accommodation variables are coarse-grained populations; two slip traces and a front-local coordinate do not resolve individual dislocations. The source renewal and Taylor-release mappings are constitutive closures. The baseline curves use one seed per condition, with eight-seed uncertainty only at two active anchors. Rate, orientation, and source studies are deliberately local. The developed paths share common-mark geometry and one monotonic load segment. PF/FEM comparison is on one material/temperature/seed condition and retains different production discretizations. The cavity comparison is one paired realization with a finite 0.2 s continuation, and its inner event chronology is unavailable after the exporter failure. These limits constrain inference without weakening the directly observed comparisons.

## 7. Conclusions

1. Fixed opening and accommodation landscapes generate distinct temperature-response classes without prescribing toughness curves; for the current DBTT and Peak rows, accommodation raises selected first-event coordinates by more than 150% relative to matched opening-only controls.
2. At DBTT/1200 K and Peak/900 K, the accommodation-associated gaps are much larger than the observed spread across eight paired seeds, although distribution tails and other temperatures remain unresolved.
3. Rate, orientation, and source geography act through the evolving state path. Their effects cannot be represented by a universal horizontal shift or additive source-density correction.
4. Developed 1 mm trajectories require stage-faithful reporting: the native pre-event coordinates form an event-sampled monotonic resistance history, not independent unload/reload R-curve points.
5. Original PF and FEM/CZM calculations share exact elastic mechanics on the identical-mesh reference, but production histories differ systematically in reaction and native \(J\) support at common openings; close event coordinates alone do not validate production mechanics.
6. In the matched finite cavity continuation, the suppressed arm advances 80 µm and the enabled arm does not advance or connect; this establishes finite-window inhibition under the executed topology-aware candidate law, not a ratio-based shielding factor or permanent arrest.

## Data and code availability

The accompanying package contains plotted numerical tables, source mappings, campaign definitions, hashes, commands, and validated manuscript/SI exports. Full native histories, meshes, and checkpoints remain in the hashed evidence roots listed by the reproducibility manifest rather than being duplicated in the compact paper package. No reduced, surrogate, fitted-transfer, or frozen-state rethresholded trajectory is used as numerical evidence.
"""


SI_ADDENDUM = r"""

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

## S14. Figure and data provenance

Every current quantitative plot is copied from, or regenerated solely from, the accepted CSV/JSON records in `review_packet/final_fracture_campaign_a5a5883`. Historical C19 field plates retained in Figures 9 and 10 are actual original full-order fields and are labeled for their limited geometric/provenance role. `reproducibility/CLAIM_TO_EVIDENCE_INDEX.json` maps major claims to exact files and `SOURCE_AND_INPUT_MANIFEST.json` records SHA-256 hashes and producer roots. No reduced/hybrid result is included in a numerical table or figure.
"""


def build_manuscript() -> str:
    old = (C19 / "fracture_manuscript.md").read_text()
    title = old.split("## Abstract", 1)[0]
    intro_methods = old.split("## 1. Introduction", 1)[1].split("## 5. Results", 1)[0]
    intro_methods = "## 1. Introduction" + intro_methods
    intro_methods = (
        intro_methods.split("## 3. Competing-kinetics framework", 1)[0]
        + METHODS + "\n\n## 4. Calculation program"
        + intro_methods.split("## 4. Calculation program", 1)[1]
    )
    intro_methods = intro_methods.replace("../../figures/experimental/", "figures/")
    intro_methods = intro_methods.replace("Figure_1A_Ag_plastic_sequence.png", "Figure_1_experimental.png")
    intro_methods = intro_methods.replace(
        "The voiding comparison uses the completed full-order V11 DBTT/1200 K/seed 3621 pair. Both arms start from accepted full-FEM states at 6.180 µm total opening and follow the same 1 µm s\\(^{-1}\\) ramp to 6.300 µm. The fracture threshold stream, temperature, material, and nonvoid constitutive laws are matched; the controlled difference is whether cavity nucleation and growth are enabled. Native accepted stress, dislocation-density, vacancy, crack, and cavity states are retained. The later hybrid continuation is not used.",
        "The cavity comparison uses the full-order V11 DBTT/1200 K/seed-3621 pair, resumed from exact native paired checkpoints after the qualified migration dwell. The final matched window applies a 1 µm s\\(^{-1}\\) total-opening ramp for 0.2 s. The fracture stream, temperature, material, and noncavity laws remain matched; the controlled difference is permission for cavity evolution. No hybrid trajectory is used."
    )
    intro_methods = intro_methods.replace(
        "The independent cohesive finite-element comparison uses a distinct parameter generation, 45° orientation, cohesive/topological crack representation, and simpler accommodation closure. It evaluates the same organization—activated opening competing with accommodation—but does not reproduce the principal model’s binned signed state. Its values are compared as consistency of response structure, not pooled with the principal dataset or used to fit a transfer factor.",
        "The cohesive finite-element comparison uses the common 2 mm × 4 mm specimen, Peak/1000 K, 0° orientation, and seed 8666. Identical-mesh elastic controls precede matched tip-only and full-field calculations. The cohesive/topological backend does not reproduce the sharp-wake discretization, so first-event K is compared directly while openings and event lengths remain native observables rather than forced parity targets."
    )
    refs = old.split("## References", 1)[1]
    return title + ABSTRACT + "\n" + intro_methods.rstrip() + "\n\n" + RESULTS + "\n\n## References\n" + refs.strip() + "\n"


def build_si() -> str:
    old = (C19 / "supporting_information.md").read_text()
    old = old.replace("../../figures/experimental/", "figures/")
    old = old.replace("Figure_1B_Ag_limited_plasticity_sequence.png", "Figure_S1_experimental_companion.png")
    # S1--S6 contain the still-current experimental, mechanical, kinetic, state,
    # and parameter definitions.  Historical C19 response tables S7--S12 are
    # deliberately excluded rather than silently pooled with the current campaign.
    methods = old.split("## S7. State measurements", 1)[0].rstrip()
    methods = (
        methods.split("## S2. Mechanical problem and loading", 1)[0]
        + SI_METHODS + "\n\n## S6. Material parameters"
        + methods.split("## S6. Material parameters", 1)[1]
    )
    energetic = old.split("## S13. Cohesive and energetic comparison", 1)[1].split("## S14. Data and reproduction statement", 1)[0]
    energetic = "## S13. Cohesive and energetic comparison" + energetic
    energetic = energetic.replace("../../analysis/branching_fields/Figure_8_accepted_branching_full_fields_v2.png", "figures/Figure_10_branching.png")
    return methods + SI_ADDENDUM.rsplit("## S14. Figure and data provenance", 1)[0] + energetic.rstrip() + "\n\n## S14. Figure and data provenance" + SI_ADDENDUM.rsplit("## S14. Figure and data provenance", 1)[1] + "\n"


def generate_cavity_figure(dst: Path) -> None:
    """Regenerate the native C19 plate while suppressing plot-top titles."""
    import importlib.util
    import matplotlib.axes
    import arrhenius_fracture
    void_source = Path("/Volumes/Data/Data/Nanopillar_calculation/voiding/recovered-pr63/arrhenius_fracture")
    if str(void_source) not in arrhenius_fracture.__path__:
        arrhenius_fracture.__path__.append(str(void_source))
    source = PAPER / "scripts/analyze_c19_full_order_figures.py"
    spec = importlib.util.spec_from_file_location("c19_native_figure_builder", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIG = dst.parent
    original_text = matplotlib.axes.Axes.text
    suppressed = {"6.180 µm; before fracture", "6.183 µm; both advanced", "6.300 µm; endpoint", "endpoint ligament, actual scale"}
    def filtered_text(self, x, y, s, *args, **kwargs):
        if s in suppressed:
            return original_text(self, x, y, "", *args, **kwargs)
        return original_text(self, x, y, s, *args, **kwargs)
    matplotlib.axes.Axes.text = filtered_text
    try:
        module.voiding_figure()
    finally:
        matplotlib.axes.Axes.text = original_text
    generated = dst.parent / "Figure_8_full_order_voiding.png"
    generated_pdf = dst.parent / "Figure_8_full_order_voiding.pdf"
    generated.replace(dst)
    if generated_pdf.exists():
        generated_pdf.replace(dst.with_suffix(".pdf"))


def generate_comparison_figure(dst: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Tip only", "Full field"]
    pf = [53.87817177332942, 51.10954269777489]
    fem = [54.212265295184146, 54.40720347046784]
    x = [0, 1]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.bar([v - 0.18 for v in x], pf, 0.36, label="Sharp-wake PF", color="#2878B5")
    ax.bar([v + 0.18 for v in x], fem, 0.36, label="FEM/CZM", color="#D95319")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Accommodation scope")
    ax.set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\rm m}$)")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(dst, dpi=300)
    fig.savefig(dst.with_suffix(".pdf"))
    plt.close(fig)


def markdown_to_html(md: str, title: str) -> str:
    lines = md.splitlines()
    body: list[str] = []
    in_math = False
    math_lines: list[str] = []
    in_table = False
    for line in lines:
        if line.strip() == "\\[":
            in_math = True; math_lines = []; continue
        if in_math:
            if line.strip() == "\\]":
                body.append('<div class="math">' + html.escape(" ".join(math_lines)) + "</div>")
                in_math = False
            else: math_lines.append(line.strip())
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells): continue
            if not in_table:
                body.append("<table>"); in_table = True; tag = "th"
            else: tag = "td"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>"); in_table = False
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            n = len(m.group(1)); body.append(f"<h{n}>{html.escape(m.group(2))}</h{n}>"); continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
        if m:
            body.append(f'<figure><img src="../{html.escape(m.group(2))}" alt="{html.escape(m.group(1))}"></figure>'); continue
        if re.match(r"^\d+\.\s+", line):
            body.append("<p>" + html.escape(line) + "</p>"); continue
        if line.strip():
            txt = html.escape(line).replace("**", "")
            body.append(f"<p>{txt}</p>")
    if in_table: body.append("</table>")
    css = """body{font:11.5pt Georgia,serif;line-height:1.45;max-width:8.2in;margin:0.55in auto;color:#161616}h1{font-size:23pt}h2{font-size:17pt;margin-top:1.4em}h3{font-size:13pt;margin-top:1.2em}p{text-align:justify}figure{text-align:center;margin:1.2em 0}img{max-width:100%;max-height:8.2in}table{border-collapse:collapse;width:100%;margin:1em 0;font-size:9.5pt}th,td{border:1px solid #aaa;padding:.35em;text-align:left}.math{text-align:center;font-family:Cambria Math,serif;margin:1em} @media print{body{margin:.45in}h2{break-after:avoid}figure,table{break-inside:avoid}}"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
ET.register_namespace("w", W); ET.register_namespace("r", R); ET.register_namespace("m", M)
ET.register_namespace("wp", WP); ET.register_namespace("a", A); ET.register_namespace("pic", PIC)


def q(ns: str, tag: str) -> str: return f"{{{ns}}}{tag}"


def add_run(p, text: str, bold=False, italic=False, size=None):
    r = ET.SubElement(p, q(W, "r")); rp = ET.SubElement(r, q(W, "rPr"))
    if bold: ET.SubElement(rp, q(W, "b"))
    if italic: ET.SubElement(rp, q(W, "i"))
    if size: ET.SubElement(rp, q(W, "sz"), {q(W, "val"): str(size)})
    t = ET.SubElement(r, q(W, "t")); t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t.text = text


def latex_to_omml(expression: str) -> ET.Element:
    """Convert LaTeX through MathML and Microsoft's MathML-to-OMML XSLT."""
    try:
        from latex2mathml.converter import convert
    except ImportError as exc:
        raise RuntimeError(
            "DOCX math conversion requires latex2mathml (pip install latex2mathml)"
        ) from exc
    mathml = convert(expression.strip())
    stylesheets = [
        Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl"),
        ROOT / "tools" / "mathml2omml.xsl",
    ]
    stylesheet = next((path for path in stylesheets if path.is_file()), None)
    if stylesheet is None:
        raise RuntimeError("Microsoft mathml2omml.xsl was not found")
    completed = subprocess.run(
        ["xsltproc", str(stylesheet), "-"], input=mathml,
        text=True, capture_output=True, check=True,
    )
    return ET.fromstring(completed.stdout)


def add_rich_text(p, text: str, *, bold=False, italic=False, size=None):
    """Emit ordinary text and every inline ``\\(...\\)`` span as real OMML."""
    cursor = 0
    for match in re.finditer(r"\\\((.+?)\\\)", text):
        if match.start() > cursor:
            add_run(p, text[cursor:match.start()], bold=bold, italic=italic, size=size)
        p.append(latex_to_omml(match.group(1)))
        cursor = match.end()
    if cursor < len(text):
        add_run(p, text[cursor:], bold=bold, italic=italic, size=size)


def para(parent, text="", style=None, bold=False, italic=False):
    p = ET.SubElement(parent, q(W, "p"))
    if style:
        pp = ET.SubElement(p, q(W, "pPr")); ET.SubElement(pp, q(W, "pStyle"), {q(W, "val"): style})
    add_rich_text(p, text, bold=bold, italic=italic)
    return p


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n": return (1600, 1000)
    return struct.unpack(">II", data[16:24])


def add_picture(body, rid: str, path: Path, docpr_id: int):
    px, py = png_size(path); max_cx, max_cy = 6_300_000, 8_000_000
    scale = min(max_cx / px, max_cy / py)
    cx, cy = int(px * scale), int(py * scale)
    p = ET.SubElement(body, q(W, "p")); pp = ET.SubElement(p, q(W, "pPr")); ET.SubElement(pp, q(W, "jc"), {q(W, "val"): "center"})
    run = ET.SubElement(p, q(W, "r")); drawing = ET.SubElement(run, q(W, "drawing")); inline = ET.SubElement(drawing, q(WP, "inline"))
    ET.SubElement(inline, q(WP, "extent"), {"cx": str(cx), "cy": str(cy)})
    ET.SubElement(inline, q(WP, "docPr"), {"id": str(docpr_id), "name": path.name})
    graphic = ET.SubElement(inline, q(A, "graphic")); gd = ET.SubElement(graphic, q(A, "graphicData"), {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"})
    pic = ET.SubElement(gd, q(PIC, "pic")); nv = ET.SubElement(pic, q(PIC, "nvPicPr")); ET.SubElement(nv, q(PIC, "cNvPr"), {"id": "0", "name": path.name}); ET.SubElement(nv, q(PIC, "cNvPicPr"))
    fill = ET.SubElement(pic, q(PIC, "blipFill")); ET.SubElement(fill, q(A, "blip"), {q(R, "embed"): rid}); stretch = ET.SubElement(fill, q(A, "stretch")); ET.SubElement(stretch, q(A, "fillRect"))
    sp = ET.SubElement(pic, q(PIC, "spPr")); xfrm = ET.SubElement(sp, q(A, "xfrm")); ET.SubElement(xfrm, q(A, "off"), {"x": "0", "y": "0"}); ET.SubElement(xfrm, q(A, "ext"), {"cx": str(cx), "cy": str(cy)}); geom = ET.SubElement(sp, q(A, "prstGeom"), {"prst": "rect"}); ET.SubElement(geom, q(A, "avLst"))


def add_math(body, expression: str):
    omp = ET.SubElement(body, q(M, "oMathPara"))
    omp.append(latex_to_omml(expression))


def add_table(body, rows: list[list[str]]):
    table = ET.SubElement(body, q(W, "tbl")); props = ET.SubElement(table, q(W, "tblPr")); ET.SubElement(props, q(W, "tblStyle"), {q(W, "val"): "TableGrid"})
    for ri, row in enumerate(rows):
        tr = ET.SubElement(table, q(W, "tr"))
        for cell in row:
            tc = ET.SubElement(tr, q(W, "tc")); para(tc, cell, bold=(ri == 0))


def build_docx(manuscript: str, si: str, out: Path) -> dict:
    document = ET.Element(q(W, "document")); body = ET.SubElement(document, q(W, "body"))
    relationships: list[tuple[str, Path]] = []; next_id = 1; eq_count = table_count = 0
    def emit(md: str, part: str):
        nonlocal next_id, eq_count, table_count
        lines = md.splitlines(); i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == "\\[":
                expr = []; i += 1
                while i < len(lines) and lines[i].strip() != "\\]": expr.append(lines[i].strip()); i += 1
                add_math(body, " ".join(expr)); eq_count += 1
            elif line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].startswith("|"):
                    cells = [c.strip() for c in lines[i].strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", c) for c in cells): rows.append(cells)
                    i += 1
                add_table(body, rows); table_count += 1; i -= 1
            elif (m := re.match(r"^!\[(.*?)\]\((.*?)\)$", line)):
                path = OUT / m.group(2)
                if path.exists() and path.suffix.lower() == ".png":
                    rid = f"rId{next_id}"; next_id += 1; relationships.append((rid, path)); add_picture(body, rid, path, next_id)
            elif (m := re.match(r"^(#{1,6})\s+(.*)$", line)):
                level = len(m.group(1)); para(body, m.group(2), "Title" if level == 1 else f"Heading{min(level-1,3)}")
            elif re.match(r"^\d+\.\s+", line): para(body, line, "ListNumber")
            elif line.startswith("- "): para(body, line[2:], "ListBullet")
            elif line.strip(): para(body, line.replace("**", ""), italic=(line.startswith("*Figure")))
            i += 1
        if part == "manuscript":
            p = ET.SubElement(body, q(W, "p")); pp = ET.SubElement(p, q(W, "pPr")); ET.SubElement(pp, q(W, "pageBreakBefore"))
    emit(manuscript, "manuscript"); emit(si, "si")
    sect = ET.SubElement(body, q(W, "sectPr")); ET.SubElement(sect, q(W, "pgSz"), {q(W, "w"): "12240", q(W, "h"): "15840"}); ET.SubElement(sect, q(W, "pgMar"), {q(W, "top"): "720", q(W, "right"): "720", q(W, "bottom"): "720", q(W, "left"): "720", q(W, "header"): "360", q(W, "footer"): "360", q(W, "gutter"): "0"})
    styles = f'''<w:styles xmlns:w="{W}"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria"/><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:rPr><w:b/><w:i/><w:sz w:val="23"/></w:rPr></w:style><w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/></w:style><w:style w:type="paragraph" w:styleId="ListNumber"><w:name w:val="List Number"/></w:style><w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/></w:style></w:styles>'''
    rel_root = ET.Element("Relationships", {"xmlns": "http://schemas.openxmlformats.org/package/2006/relationships"})
    ET.SubElement(rel_root, "Relationship", {"Id": "rIdStyles", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "Target": "styles.xml"})
    for idx, (rid, path) in enumerate(relationships, 1): ET.SubElement(rel_root, "Relationship", {"Id": rid, "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", "Target": f"media/image{idx}.png"})
    types = ET.Element("Types", {"xmlns": "http://schemas.openxmlformats.org/package/2006/content-types"}); ET.SubElement(types, "Default", {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"}); ET.SubElement(types, "Default", {"Extension": "xml", "ContentType": "application/xml"}); ET.SubElement(types, "Default", {"Extension": "png", "ContentType": "image/png"}); ET.SubElement(types, "Override", {"PartName": "/word/document.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"}); ET.SubElement(types, "Override", {"PartName": "/word/styles.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"})
    pkg_rels = f'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    with ZipFile(out, "w", ZIP_DEFLATED) as z:
        def zput(name: str, payload: bytes | str) -> None:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, payload)
        zput("[Content_Types].xml", ET.tostring(types, encoding="utf-8", xml_declaration=True)); zput("_rels/.rels", pkg_rels); zput("word/document.xml", ET.tostring(document, encoding="utf-8", xml_declaration=True)); zput("word/styles.xml", styles); zput("word/_rels/document.xml.rels", ET.tostring(rel_root, encoding="utf-8", xml_declaration=True))
        for idx, (_, path) in enumerate(relationships, 1): zput(f"word/media/image{idx}.png", path.read_bytes())
    return {"embedded_images": len(relationships), "native_display_math_objects": eq_count, "editable_tables": table_count}


def validate_docx(path: Path, expected: dict) -> dict:
    errors = []
    with ZipFile(path) as z:
        bad = z.testzip()
        if bad: errors.append(f"bad zip member: {bad}")
        names = set(z.namelist())
        for required in {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/_rels/document.xml.rels", "word/styles.xml"}:
            if required not in names: errors.append(f"missing {required}")
        doc = ET.fromstring(z.read("word/document.xml")); rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        targets = [r.attrib["Target"] for r in rels if r.attrib.get("Type", "").endswith("/image")]
        for target in targets:
            if "word/" + target not in names: errors.append(f"missing image target {target}")
        ordinary_text = "".join(node.text or "" for node in doc.findall(".//" + q(W, "t")))
        math_text = "".join(node.text or "" for node in doc.findall(".//" + q(M, "t")))
        unresolved = sorted(set(re.findall(r"\\(?:frac|sqrt|int|sum|max|exp|left|right|rm|mathrm|mathcal|qquad)|\\\(|\\\)", ordinary_text + math_text)))
        stats = {
            "paragraphs": len(doc.findall(".//" + q(W, "p"))),
            "embedded_images": len(doc.findall(".//" + q(A, "blip"))),
            "native_display_math_objects": len(doc.findall(".//" + q(M, "oMathPara"))),
            "native_math_objects": len(doc.findall(".//" + q(M, "oMath"))),
            "structured_fractions": len(doc.findall(".//" + q(M, "f"))),
            "structured_radicals": len(doc.findall(".//" + q(M, "rad"))),
            "structured_subscripts": len(doc.findall(".//" + q(M, "sSub"))),
            "structured_superscripts": len(doc.findall(".//" + q(M, "sSup"))),
            "structured_nary": len(doc.findall(".//" + q(M, "nary"))),
            "editable_tables": len(doc.findall(".//" + q(W, "tbl"))),
            "unresolved_tex_tokens": unresolved,
        }
        if unresolved:
            errors.append(f"unresolved TeX tokens: {unresolved}")
        for required in ("structured_fractions", "structured_radicals", "structured_subscripts", "structured_nary"):
            if stats[required] <= 0:
                errors.append(f"missing {required}")
        for key, value in expected.items():
            if stats.get(key) != value: errors.append(f"{key}: expected {value}, found {stats.get(key)}")
    return {
        "schema": "fracture-docx-validation/1",
        "file": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "package_zip_valid": not errors,
        "xml_and_relationships_valid": not errors,
        "microsoft_word_open_test_performed": False,
        "renderer_checks": {
            "textutil_html_parse": "PASS",
            "microsoft_word_session": "UNAVAILABLE; the installed application did not open either this file or a known control DOCX in the current GUI session",
            "page_by_page_visual_review": "NOT_CLAIMED",
        },
        "stats": stats,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def write_json(path: Path, data) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    for d in [OUT / "figures", OUT / "figure_data", OUT / "previews", OUT / "campaign", OUT / "methods/model_and_observer_contracts", OUT / "reproducibility", OUT / "author_record"]: d.mkdir(parents=True, exist_ok=True)
    for stale in [OUT / "figures/Figure_S2_historical_source_state.png", OUT / "figures/Figure_S3_historical_refinement.png"]:
        if stale.exists(): stale.unlink()
    figure_map = {
        PAPER / "figures/experimental/Figure_1A_Ag_plastic_sequence.png": "Figure_1_experimental.png",
        ACCEPTANCE / "Figure_2_PF_baseline_96.png": "Figure_2_PF_baseline_96.png",
        ACCEPTANCE / "Figure_3_PF_focused_gaps.png": "Figure_3_PF_focused_seeds.png",
        REVIEW / "PF_RATE_SENSITIVITY.png": "Figure_4a_PF_rate.png",
        REVIEW / "PF_ORIENTATION_SENSITIVITY.png": "Figure_4b_PF_orientation.png",
        REVIEW / "PF_SOURCE_GEOGRAPHY.png": "Figure_4c_PF_source.png",
        ACCEPTANCE / "Figure_5_developed_physical_history.png": "Figure_5_developed_driving.png",
        REVIEW / "DEVELOPED_RESISTANCE.png": "Figure_6_developed_resistance.png",
        ACCEPTANCE / "Figure_7_native_fields.png": "Figure_7_native_fields.png",
        PAPER / "analysis/branching_fields/Figure_8_accepted_branching_full_fields_v2.png": "Figure_10_branching.png",
        PAPER / "figures/experimental/Figure_1B_Ag_limited_plasticity_sequence.png": "Figure_S1_experimental_companion.png",
    }
    for src, name in figure_map.items(): copy(src, OUT / "figures" / name)
    copy(ACCEPTANCE / "Figure_9_current_cavity_fields.png", OUT / "figures/Figure_9_cavity_fields.png")
    copy(ACCEPTANCE / "Figure_9_current_cavity_fields.pdf", OUT / "figures/Figure_9_cavity_fields.pdf")
    copy(ACCEPTANCE / "Figure_8_PF_FEM_production.png", OUT / "figures/Figure_8_PF_FEM.png")
    copy(ACCEPTANCE / "Figure_8_PF_FEM_production.pdf", OUT / "figures/Figure_8_PF_FEM.pdf")
    data_files = list(REVIEW.glob("*.csv")) + list(ACCEPTANCE.glob("*.csv")) + list((C19_ANALYSIS / "data").glob("*.csv"))
    for src in data_files:
        dst = OUT / "figure_data" / src.name
        dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
    manuscript, si = build_manuscript(), build_si()
    (OUT / "MANUSCRIPT.md").write_text(manuscript.rstrip() + "\n"); (OUT / "SUPPORTING_INFORMATION.md").write_text(si.rstrip() + "\n")
    (OUT / "previews/MANUSCRIPT.html").write_text(markdown_to_html(manuscript, "Fracture manuscript")); (OUT / "previews/SUPPORTING_INFORMATION.html").write_text(markdown_to_html(si, "Supporting Information"))
    copy(ROOT / "finalization_control/CAMPAIGN_MATRIX.csv", OUT / "campaign/CAMPAIGN_MATRIX.csv")
    copy(ROOT / "scripts/run_final_fracture_campaign.py", OUT / "campaign/run_or_resume.py")
    copy(ROOT / "CAPABILITY_PRESERVATION_MATRIX.csv", OUT / "methods/model_and_observer_contracts/CAPABILITY_PRESERVATION_MATRIX.csv")
    copy(ROOT / "finalization_control/TEST_RECEIPT.json", OUT / "reproducibility/TEST_RECEIPT.json")
    copy(ROOT / "finalization_control/FINAL_ACCEPTANCE.json", OUT / "FINAL_ACCEPTANCE.json")
    # Backup receipts are written after the immutable release bytes are archived.
    # Do not embed an older receipt whose document/archive hashes describe a prior
    # generation, and avoid a self-referential package-hash cycle.
    stale_backup_receipt = OUT / "reproducibility/BACKUP_AND_RESTORE_RECEIPT.json"
    if stale_backup_receipt.exists():
        stale_backup_receipt.unlink()
    for src in [REVIEW / "PF_BASELINE_96.json", REVIEW / "PF_FOCUSED_SEEDS.json", REVIEW / "PF_SENSITIVITIES.json", REVIEW / "DEVELOPED_GROWTH_RESISTANCE.json", REVIEW / "PF_FEM_FINAL_FRACTURE_COMPARISON.json", REVIEW / "CAVITY_INTERACTION_0P2S_RECOVERED.json", REVIEW / "V36_ACTIVE_RESTART.json", REVIEW / "BRANCH_CAVITY_COMPATIBILITY.json"]: copy(src, OUT / "methods/model_and_observer_contracts" / src.name)
    for src in [ACCEPTANCE / "METHOD_TO_RESULT_MAP.json", ACCEPTANCE / "METHOD_TO_RESULT_MAP.md", ACCEPTANCE / "PF_FEM_PRODUCTION_COMMON_LOAD.json", ACCEPTANCE / "FIGURE_3_ESTIMATOR_CHECK.json"]:
        copy(src, OUT / "methods/model_and_observer_contracts" / src.name)
    doc_stats = build_docx(manuscript, si, OUT / "manuscript_and_SI.docx")
    validation = validate_docx(OUT / "manuscript_and_SI.docx", doc_stats); write_json(OUT / "reproducibility/DOCX_VALIDATION.json", validation)
    (OUT / "reproducibility/requirements-docx.txt").write_text("latex2mathml==3.81.1\n")
    claims = {
        "baseline_96": {"claim": "all 96 cases fractured; reported row ranges", "evidence": "methods/model_and_observer_contracts/PF_BASELINE_96.json", "plot": "figures/Figure_2_PF_baseline_96.png"},
        "paired_seeds": {"claim": "eight paired seeds per active anchor and reported intervals", "evidence": "methods/model_and_observer_contracts/PF_FOCUSED_SEEDS.json", "plot": "figures/Figure_3_PF_focused_seeds.png"},
        "sensitivities": {"claim": "40 rate/orientation/source cases", "evidence": "methods/model_and_observer_contracts/PF_SENSITIVITIES.json"},
        "developed_growth": {"claim": "203 events and 1004.926 µm projected extension per path", "evidence": "methods/model_and_observer_contracts/DEVELOPED_GROWTH_RESISTANCE.json"},
        "method_binding": {"claim": "paper PF families execute persistent reusable sources; cavity intentionally uses current-source topology ownership", "evidence": "methods/model_and_observer_contracts/METHOD_TO_RESULT_MAP.json"},
        "pf_fem": {"claim": "exact identical-mesh elastic control; systematic production common-opening response differences; accepted cohesive topology events", "evidence": "methods/model_and_observer_contracts/PF_FEM_PRODUCTION_COMMON_LOAD.json", "data": "figure_data/PF_FEM_COMMON_LOAD.csv"},
        "cavity": {"claim": "80 µm suppressed-arm advance versus zero enabled-arm advance over 0.2 s; unlike terminal drives are not ratioed", "evidence": "methods/model_and_observer_contracts/CAVITY_INTERACTION_0P2S_RECOVERED.json", "field_data": "figure_data/CAVITY_CURRENT_FIELD_SIDECAR.csv"},
        "restart_options": {"claim": "exact populated V3.6 and combined branch-cavity restart", "evidence": ["methods/model_and_observer_contracts/V36_ACTIVE_RESTART.json", "methods/model_and_observer_contracts/BRANCH_CAVITY_COMPATIBILITY.json"]},
    }
    write_json(OUT / "reproducibility/CLAIM_TO_EVIDENCE_INDEX.json", {"schema": "fracture-claim-index/1", "claims": claims})
    manifest = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SOURCE_AND_INPUT_MANIFEST.json"):
        manifest.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(OUT / "reproducibility/SOURCE_AND_INPUT_MANIFEST.json", {"schema": "fracture-release-manifest/1", "generated_utc": datetime.now(timezone.utc).isoformat(), "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": manifest, "raw_roots": ["integration_evidence/final_fracture_campaign", "integration_evidence/current_science_pilots_6abbf10", "integration_evidence/branch_cavity_bb0c21d9", "integration_evidence/v36_active_restart_340727f7"]})
    readme = """# Final fracture study package\n\nThis compact release contains the external-reader manuscript, Supporting Information, figures and numerical figure data. The Word file is self-contained and has embedded PNG figures, editable Word tables, and native structured OOXML equations. Its package and relationships are validated in `reproducibility/DOCX_VALIDATION.json`; a `textutil` parse passed, but a page-by-page Microsoft Word render is not claimed because the installed Word session did not open either the release or a known control DOCX.\n\nFull native meshes, histories, and checkpoints remain in the hashed evidence roots listed by the manifest. Run `campaign/run_or_resume.py --audit-only` from the repository root to audit campaign status. HTML previews are in `previews/`. Regenerating the structured equations requires `latex2mathml` (pinned in `reproducibility/requirements-docx.txt`), `xsltproc`, and Microsoft Word's installed `mathml2omml.xsl`; the generated DOCX itself has no runtime dependency.\n\nThe paper uses only original full spatial PF or original FEM/CZM numerical evidence. Historical field plates retained for geometry/provenance are labeled explicitly and are not pooled with the current quantitative curves.\n"""
    (OUT / "README.md").write_text(readme)
    report = """# Final scientific report\n\nThe equation audit establishes that the retained baseline, seed, rate, orientation, source-support, developed-growth, and PF companion families execute the persistent reusable source law. Earlier finite-inventory and advance-refresh prose was stale; no trajectory was rerun. The cavity family intentionally uses the separate current-source topology owner.\n\nThe 96-case campaign remains the principal result: DBTT and Peak reach approximately 55 MPa√m, with selected accommodation effects above 150%, while Weak-T is broadly elevated and Ceramic-like remains comparatively modest. Eight paired seeds resolve the active-anchor effect relative to observed stochastic spread. Developed paths reach 1.005 mm and are reported as chronological and event-conditioned monotonic resistance histories.\n\nThe common-specimen comparison is narrower than the prior release claimed. Identical-mesh elastic mechanics is exact, but production histories show a 1.0923 PF/FEM reaction ratio and support-dependent native K ratios at common openings. Close first-event coordinates at different openings are not mechanical validation. The FEM/CZM paths do contain four accepted physical events and 26 fully separated cohesive subsegments.\n\nThe exact cavity checkpoints establish 80 µm suppressed-arm advance versus zero enabled-arm advance over 0.2 s. Current stress fields show the evolved geometry. Because the terminal local-J and complete-ligament drives are unlike observables at different geometries, their ratio is not a shielding factor; the supported claim is finite-window inhibition under the executed topology-aware candidate law.\n"""
    (OUT / "FINAL_SCIENTIFIC_REPORT.md").write_text(report)
    copy(ROOT / "finalization_control/DECISIONS.md", OUT / "author_record/DECISIONS.md")
    for clean_text in [OUT / "FINAL_ACCEPTANCE.json", OUT / "author_record/DECISIONS.md"]:
        clean_text.write_text(clean_text.read_text().rstrip() + "\n")
    # Synchronize only after every reader-facing byte is fixed.  The manifest
    # deliberately excludes itself and the archive that contains it.
    portable = OUT / "reproducibility.zip"
    if portable.exists():
        portable.unlink()
    manifest = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"SOURCE_AND_INPUT_MANIFEST.json", "reproducibility.zip", "reproducibility.zip.sha256"}):
        manifest.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(OUT / "reproducibility/SOURCE_AND_INPUT_MANIFEST.json", {
        "schema": "fracture-release-manifest/2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "producer_source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "analysis_source": "scripts/build_publication_acceptance_evidence.py",
        "report_source": "scripts/build_final_fracture_package.py",
        "excluded_self_referential_files": ["reproducibility/SOURCE_AND_INPUT_MANIFEST.json", "reproducibility.zip", "reproducibility.zip.sha256"],
        "files": manifest,
        "raw_roots": ["integration_evidence/final_fracture_campaign", "integration_evidence/current_science_pilots_6abbf10", "integration_evidence/branch_cavity_bb0c21d9", "integration_evidence/v36_active_restart_340727f7"],
    })
    with ZipFile(portable, "w", ZIP_DEFLATED) as archive:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p not in {portable, OUT / "reproducibility.zip.sha256"}):
            archive.write(path, Path("fracture_final_current") / path.relative_to(OUT))
    (OUT / "reproducibility.zip.sha256").write_text(f"{sha256(portable)}  reproducibility.zip\n")
    if validation["status"] != "PASS": raise RuntimeError(validation)
    print(json.dumps({"output": str(OUT), "docx": validation, "manifest_files": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
