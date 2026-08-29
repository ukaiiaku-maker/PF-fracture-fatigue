# Voiding V2 static FEM report

`EXPLICIT_VOID_MECHANICS_QUALIFIED = OPEN` unless every row in the generated acceptance table passes. Missing calculations remain `OPEN`, `NOT_RUN`, or `NOT_CONVERGED`.

## Solver and geometry

The runner uses the unmodified production plane-strain CST assembler and SciPy `spsolve`/SuperLU. The body-fitted polar-to-rectangle mesh is a static benchmark mesh, not a production local-remesh implementation. Its actual multiplicity-one internal boundary comes from retained connectivity. Validation covers boundary components, triangle/open-disk intersection, orphan DOFs, cycle integrity, area, perimeter, angle, quality, aspect ratio, reactions, free residual, and the identity `2U = P Delta`.

The cavity traction diagnostic is

`sqrt(integral_boundary |sigma_CST n|^2 ds) / (measured_remote_stress sqrt(perimeter))`.

It uses the raw stress of the unique CST adjacent to each cavity edge. The natural traction-free condition is imposed weakly, so this is a recovered strong-form diagnostic rather than the weak boundary residual. All three refinement values are published. Qualification uses both finest meshes, not only the best value. The worst mesh elements occur in the polar-to-rectangle transition.

## Far-void diagnosis

Each void case has its own no-void control made by filling only the cavity cycle with a center fan. Every pre-existing node, element, crack/wake definition, probe location, material property, external boundary, and imposed opening is identical. This isolates the cavity patch and removes the V1/V2 comparison's crack-tip discretization mismatch.

The matrix separates fixed-opening specimen response from reaction-normalized local response. It publishes reactions, compliance, energy, three finite-radius tensors, probe-derived `K_I` and `K_II`, and a common-element annular L2 stress norm for `d/R = 8, 16, 32, 64` at two `R/W` values and three refinements. The probe-derived K values are finite-radius diagnostics, not interaction-integral results. A contour interaction integral remains `NOT_RUN` and therefore open.

The old 81.7% point-probe failure is preserved in the preceding evidence commit. The matched-mesh calculation diagnoses it primarily as an inconsistent-discretization comparison. The new farthest reaction-normalized K mismatch is small, but strict monotonic convergence is evaluated independently and is not assumed.

## Virtual derivatives

At fixed imposed opening, centered differences calculate the elastic mechanical contributions

`G_crack_el = -(1/B) dU/da` at fixed cavity radius, and

`F_void_el = -dU/dR` at fixed crack/wake geometry.

The crack result has units `J/m^2` for the unit-thickness model. The radial cavity force is reported in `J/m`; `F_void_el/(2 pi B)` is also reported as energy per newly created cylindrical cavity surface (`J/m^2`). It excludes capillarity and vacancy chemical-potential terms; in particular, no `2 pi B gamma` surface-energy derivative is included.

Both derivatives use three mesh levels and three perturbations per level. Radius perturbations retain identical connectivity and radial-layer count while smoothly moving the paired nodes. Crack perturbations retain identical nodes and elements while changing the production P0 wake extent. Energy differences are cross-checked against compliance differences, and the generated SVG plots each centered derivative against `delta^2`. Mesh and perturbation convergence receive separate numerical spreads; absence of convergence cannot emit PASS.

## Environment and reproducibility

CI is pinned to Python 3.12 and exact NumPy, SciPy, Matplotlib, and pytest versions. `environment.json` records actual runtime versions and sparse solver identity. `qualification_test_inventory.txt` lists every selected test node ID. The local default Python 3.13.2 signal-139 failure is recorded as an environment observation, not a physics failure or a compatibility claim.

The authoritative outputs are generated under `analysis_outputs/voiding_v2_static_fem/`. Evidence is generated twice and compared byte-for-byte before commit. This scope includes no production crack-void topology, ligament rupture, nucleation coupling, promotion, resolved growth, fatigue, or material calibration.
