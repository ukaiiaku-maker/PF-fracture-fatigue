# Voiding V2 static FEM report

`EXPLICIT_VOID_MECHANICS_QUALIFIED = OPEN` unless every row in the generated acceptance table passes. Missing calculations remain `OPEN`, `NOT_RUN`, or `NOT_CONVERGED`.

## Solver and geometry

The runner uses the unmodified production plane-strain CST assembler and SciPy `spsolve`/SuperLU. A source-native benchmark adapter replaces the production bottom-corner pins with one mid-plane horizontal rigid-body pin; the corner pins break top/bottom field symmetry. The body-fitted polar-to-rectangle mesh is a static benchmark mesh, not a production local-remesh implementation. Its actual multiplicity-one internal boundary comes from retained connectivity. Boundary matching uses exact node-set equality plus a bidirectional Hausdorff distance, never independent coordinate sorting. Validation covers topology, triangle/open-disk intersection, orphan DOFs, area, perimeter, mesh quality, reactions, local mirrored stress fields, and `2U = P Delta`.

The cavity traction diagnostic is

`sqrt(integral_boundary |sigma_CST n|^2 ds) / (measured_remote_stress sqrt(perimeter))`.

It uses the raw stress of the unique CST adjacent to each cavity edge. The separate weak residual is `||(Ku-f)_Gamma_void||/||reaction||`. Four refinement values are published. Qualification uses both finest meshes, not only the best value. The worst mesh elements occur in the polar-to-rectangle transition. Plate-size results distinguish refinement behavior from the residual finite-width displacement-controlled Kirsch effect.

## Far-void diagnosis

Each void case has its own no-void control made by filling only the cavity cycle with a center fan. Every pre-existing node, element, crack/wake definition, probe location, material property, external boundary, and imposed opening is identical. This isolates the cavity patch and removes the V1/V2 comparison's crack-tip discretization mismatch.

The matrix separates fixed-opening specimen response from reaction-normalized local response. It publishes reactions, compliance, energy, three finite-radius tensors, probe-derived `K_I` and `K_II`, and a common-element annular L2 stress norm for `d/R = 8, 16, 32, 64` at two `R/W` values and four refinements. Finite-radius probes remain diagnostics. The production intrinsic-isotropy signed interaction integral is executed on three bounded contours that do not enclose the void. Its contour plateau, centered `K_II/K_I`, matched-control reproduction, and reaction-normalized far response are separate gates. Exact point monotonicity is replaced by a declared fine/finer uncertainty envelope.

The old 81.7% point-probe failure is preserved in the preceding evidence commit. The matched-mesh calculation diagnoses it primarily as an inconsistent-discretization comparison. The new farthest reaction-normalized K mismatch is small, but strict monotonic convergence is evaluated independently and is not assumed.

## Virtual derivatives

At fixed imposed opening, centered differences calculate the elastic mechanical contributions

`G_crack_el = -(1/B) dU/da` at fixed cavity radius, and

`f_R = -d(U/B)/dR` at fixed crack/wake geometry.

The crack result has units `J/m^2`. The radial force density `f_R` has units `J/m^2`; total `F_R = B f_R` has units `J/m = N`, and `G_void_surface = f_R/(2 pi)` has units `J/m^2`. The evidence uses `B=1 m` and reports all three roles explicitly. It excludes capillarity and vacancy chemical-potential terms; in particular, no `2 pi B gamma` derivative is included.

Both derivatives use four mesh levels and common dimensionless perturbation sets. Radius perturbations retain identical connectivity, radial-layer count, and P0 wake element identities while smoothly moving paired nodes. Crack perturbations retain identical nodes/elements and divide by the measured change in the P0 element front, not the nominal cutoff change. A fixed physical wake-width convention is used at every refinement. Energy and compliance derivatives are cross-checked, crack `G` is compared with `(K_I^2+K_II^2)/E'`, and the SVG plots centered derivatives against `delta^2`. Mesh and perturbation convergence remain independent fail-closed checks.

## Environment and reproducibility

CI is pinned to Python 3.12 and exact NumPy, SciPy, Matplotlib, and pytest versions. `environment.json` records actual runtime versions and sparse solver identity. `qualification_test_inventory.txt` lists every selected test node ID. The local default Python 3.13.2 signal-139 failure is recorded as an environment observation, not a physics failure or a compatibility claim.

The exact V11 base `2b5e5351add0bf0db67f2cda35a1480c3e7efc91` reproduces the six GitHub legacy failures in v10 schema/model-ID, zero-event summary, and fatigue delegate expectations. A seventh local-only status-script failure is caused by sandbox denial of `ps`. None of those paths is modified by V2. They are inherited CI debt, not evidence for or against the static-cavity mechanics gate.

The authoritative outputs are generated under `analysis_outputs/voiding_v2_static_fem/`. Evidence is generated twice and compared byte-for-byte before commit. This scope includes no production crack-void topology, ligament rupture, nucleation coupling, promotion, resolved growth, fatigue, or material calibration.
