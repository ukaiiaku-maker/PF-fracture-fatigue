# Cavity boundary patch recovery V1: prospective protocol

This specification is frozen before sentinel evaluation. It introduces a new
postprocessing representation, never replacement or reinterpretation of retained
adjacent-CST evidence. It does not modify displacement, stiffness, stress storage,
geometry, clocks, thresholds or RNG and does not authorize an event.

At the exact polygonal boundary site, select its unique solid owner (or all
incident owners at a boundary vertex), then add two shared-edge adjacency rings.
Exclude every explicitly non-intact element and never traverse excluded elements.
Sort element IDs. Use centroid samples and fit an affine function of the source
normal/tangent coordinates independently for each symmetric tensor component.
Coordinates are divided by the maximum stencil-centroid distance. Positive
geometric weights are element area divided by squared distance plus squared RMS
centroid distance, normalized to sum to one. No traction-free constraint or
stress-dependent sample selection is imposed. Retain sample IDs, raw tensors,
coordinates, regression weights, evaluation weights, singular values and residuals.

Reject nonfinite inputs, nonunit normal, non-symmetric stresses, invalid elements,
non-boundary source sites, invalid ownership, fewer than three samples, rank < 3,
or weighted design condition number > 1e6. Rank uses the standard SVD numerical
criterion max(matrix.shape)*machine_epsilon*largest_singular_value. The condition
ceiling prevents amplification above roughly 1e-10 of unit-scaled floating error;
it is a numerical admissibility screen, not an accuracy certificate.

Manufactured constant/affine reproduction and rigid rotation must agree within
1e-11 relative to the maximum field magnitude, with 1e-12 absolute only for
unit-scale manufactured data. Repeated A/B application and reversed input boundary
edge order must produce exact identical records. Analytic Kirsch samples require
final-level boundary tensor relative error <= the existing 0.03 Kirsch limit and
strictly decreasing error over 32/12, 64/24, 128/48, 512/192 levels. This tests the
operator on an analytic field, not FEM correctness or finite-specimen equivalence
to an infinite Kirsch plate.

Retained actual FEM captures provide cavity-only and crack-plus-cavity comparisons
at five fixed arc fractions (0, 1/8, 1/4, 3/8, 1/2), with 512/192 as the reference.
Report full tensors, tangential/normal/shear components, principal stresses and
four candidate-plane opening/shear pairs. Practical 32/12, 64/24, 128/48 production
checkpoints are compared to the same owned site's 512/192 checkpoint. Frozen tensor
limit is 0.05; traction 0.05 and global mesh quality 0.05 remain separate mandatory
gates. Positive/negative fixed-crack offset source captures are measured when
available and cannot be inferred from cavity-only symmetry.

A successful fit is only NUMERICALLY_ADMISSIBLE. Production source qualification
also requires all external geometry, mechanics, topology, refinement and timing
gates. Root protocol owns the prospective 5% maximum first-passage uncertainty,
log-rate/barrier error budgets, exact candidate/threshold/RNG/provenance checks,
and event integration. No new operator is used for event selection by this module.

## Independent annular Kirsch FEM sentinel (frozen before its first solve)

In addition to analytic-sample recovery, solve a genuine traction-free unit
cavity in an annulus of outer radius 5, with exact plane-strain Kirsch displacement
prescribed on the outer boundary. Use E=210 GPa, nu=0.3 and remote stress 1 MPa;
levels 32/12, 64/24 and 128/48, geometric radial spacing from radius 1 to 5.
Require monotonically decreasing maximum fixed-arc tensor error and final error
<=0.03, normalized by max(exact tensor norm, remote stress). No traction condition
is imposed on recovered tensors. Retain nodes, elements, stiffness, displacement,
CST stress and residual for independent reconstruction. The outer-circle
Dirichlet problem is not the finite rectangular production specimen.
