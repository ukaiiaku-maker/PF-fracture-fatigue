# Prospective unique static mechanics registry

The source-bound registry in closure_mechanics_evidence.py is frozen before
execution. Exact physical configurations, not labels, identify base solves.
Overlapping centered/3R cases share a base; near-equal floating representations
of the same prescribed center are canonicalized before registration.

The fixed laboratory crack is (0,0) -> (0.5 mm,0). Mesh pairs are 128/48 and
256/96. Radii 40/50/60 micrometers crossed with ligament/R=1/2/3 give eighteen
unique crack/cavity configurations. Each has a filled-patch, otherwise matched
crack-only control. Offsets +/-40 micrometers retain the identical crack.
Far-void centers 1.2 and 1.5 mm use a fixed 3 mm by 2 mm domain and matched
crack-only peers; changing cavity distance does not move an outer boundary.

Crack extension and cavity radius are separately perturbed by +/-2.5 and
1.25 micrometers on each mesh. Central energy and compliance derivatives and
perturbation convergence are evaluated directly with frozen 0.1 limits.
Derivatives are explicitly with respect to crack length or cavity radius;
radius derivatives are not mislabeled as standard absolute K or crack G.

The validator reassembles the captured sparse system and recomputes reactions,
energy, compliance, cavity traction/area/perimeter, fixed full tensors, vertex
geometry, independent intact-path certificates, support/polygon overlap, and
registered predicates. Scientific failures remain false. The fine traction
endpoint does not substitute for these gates, and none of this enables an
unqualified production first passage.
