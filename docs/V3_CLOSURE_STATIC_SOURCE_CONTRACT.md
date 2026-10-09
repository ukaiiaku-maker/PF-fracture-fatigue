# V3 closure static source contract, revision 2

This is a prospective replacement execution under the closure directory. It
does not overwrite traction_a/b or accepted blocked V3 evidence. Eleven unique
physical solves feed shared derived rows; family membership is not a solve.
Every configuration key is passed explicitly to the solver. Static provenance
is input -> realized mesh/support -> assembled system -> solved u/stress -> raw
measurements -> predicates; no lifecycle states are invented.

The static branch of `validate_closure_evidence` loads captured arrays, verifies
their fingerprints, reassembles the production CST system at the captured u,
checks stress/residual/system equality and prescribed opening, recomputes raw
measurements, and recomputes the complete registered derived-row inventory.
This source validation is distinct from scientific acceptance of any predicate.

Fixed physical arc fractions are 0, 1/8, 1/4, 3/8, 1/2. Recovery is periodic
linear interpolation of adjacent-CST tensors using cavity edge midpoint
angles, with physical radial normal and counterclockwise tangent. No component
is projected to zero. Supporting edge arrays and interpolation stencils are
retained. The full crack-tip probe is the area-weighted containing live CST
tensor at 25 micrometers ahead along the final crack segment, not a singular
tip sample or changing nearest-element centroid.

The nonzero tangential field, complete tensor, and normal/shear components
are compared to the finest reference using its tensor norm; the frozen 0.05
tensor tolerance is unchanged. Candidate-plane diagnostic normals are fixed
laboratory axes and +/-45 degrees; they are not a substitute for the actual
source-native production cleavage candidate registry or direction eligibility.

The numerical resolution SCREEN is a prospective hypothesis based explicitly
on the already published crossed matrix, not an independent blind prediction:
eta_n_max <= 0.03, eta_t_max <= 0.025, local edge-owner aspect ratio <= 7,
minimum quality >= 0.05. Its agreement with observed crossed traction outcomes
must itself be checked. These bounds do not change the scientific traction
limit 0.05 and do not qualify production tensors, barriers, rates, or events.
No fitting of material parameters is performed.

Production transfer remains NOT_RUN until actual production candidate evidence
exists. A fine traction endpoint or this mesh screen alone cannot authorize a
downstream first-passage event. Complete static qualification and the overall
closure decision remain OPEN.
