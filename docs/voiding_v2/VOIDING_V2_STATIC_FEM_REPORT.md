# Voiding V2 static FEM report

The calculation runner uses the unmodified production plane-strain CST assembly and solver. The cavity is a body-fitted, circumscribed polygon: its actual multiplicity-one internal boundary is extracted from retained element connectivity. Validation checks actual boundary components, triangle/disk intersection, orphan DOFs, polygon matching, angle, quality, aspect ratio, and local edge size.

All numerical statuses are computed by `scripts/qualify_voiding_v2_static_fem.py`. Missing calculations are `OPEN` or `NOT_RUN`; no renderer contains prescribed response arrays.

The hole-only matrix records reactions, reaction balance, stored energy, compliance, free residual, solution-derived boundary traction norm, hoop-stress concentration, area, perimeter, symmetry, and mesh quality at three refinements. The prescribed crack–void matrix uses the V11 production assembler's P0 sharp-wake channel and records crack-only controls, centered/far cavities, mirror offsets, and tensor probes. Virtual crack and void energy derivatives are executed separately, although their mesh and perturbation convergence remains an open check.

The authoritative gate is `analysis_outputs/voiding_v2_static_fem/decision.json`. A failed or open criterion leaves `EXPLICIT_VOID_MECHANICS_QUALIFIED = OPEN`. This is numerical software evidence only—not material calibration or validation.
