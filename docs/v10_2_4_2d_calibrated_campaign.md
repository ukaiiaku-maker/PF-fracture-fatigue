# v10.2.4 2-D-calibrated reduced campaign

This stage replaces the temporary constant-factor monotonic closure by a mechanical closure calibrated from cap-free 2-D tensor-probe histories. The shared spatial MPZ constitutive state introduced in v10.2.3 remains unchanged.

The campaign sequence is:

1. Build a candidate-independent mechanical atlas from cap-free 2-D histories using the preserved A0002277, A0002333, and A0003837 anchors at their low/high endpoint temperatures.
2. Interpolate the two tensor-derived channel factors as functions of applied K and continuous crack-progress state. The interpolation is constrained to the convex support of the atlas; extrapolation is explicitly diagnosed.
3. Run a local Sobol campaign around the three preserved parameterizations using the shared production state and the atlas closure.
4. Score the full response at 300, 700, 900, and 1200 K and endpoint ablations for plasticity, shielding, and back stress.
5. Promote only candidates whose reduced response has the desired shielding-history mechanism.
6. Run cap-free 2-D endpoint validation on the promoted candidates. Reduced-model acceptance alone is never sufficient.

The replay gate is an implementation audit only and is not part of the parameter search objective.