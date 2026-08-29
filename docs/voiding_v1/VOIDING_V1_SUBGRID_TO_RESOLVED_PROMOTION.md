# Subgrid-to-resolved promotion

Gate: **VOID_PROMOTION_AND_GROWTH_QUALIFIED** for deterministic representation transitions.

Promotion requires a stable cavity, a minimum boundary-segment count, bounded `h/R`, and enough element layers across the ligament. It preserves void/site IDs, center, equivalent radius, area target, inventory, and lineage. It does not create a site, birth, stabilization, fracture event, or population increment. Polygon chordal area error is checked before commit; the expected local mechanics change is accepted because explicit geometry resolves a stress concentration absent from the subgrid representation.

Resolved growth uses a newly qualified polygon/remesh at the next radius and transfers the same cavity record only after geometry conservation and quality gates pass.
