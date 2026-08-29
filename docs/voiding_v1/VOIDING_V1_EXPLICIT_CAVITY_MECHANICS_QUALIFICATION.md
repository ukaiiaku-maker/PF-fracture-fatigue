# Explicit cavity mechanics qualification

The V1 cavity is a finite-area plane-strain hole per unit thickness. Its polygon is a closed internal boundary, triangles with a vertex or centroid inside it are rejected, and no stiffness, damage, porous-plasticity, area-loss, or stress-amplification surrogate is used. Natural FEM boundary conditions make it traction free.

The deterministic qualification covers 24, 48, and 96 boundary segments. Polygon area and perimeter converge to `pi R^2` and `2 pi R`; topology validation requires one closed cycle, no interior triangle, and no wake overlap. The present compact runner emits analytical Kirsch reference trends and independent virtual-derivative bookkeeping. It does not yet execute the requested finite-plate FEM reaction/compliance/residual convergence matrix.

Gate: **EXPLICIT_VOID_GEOMETRY_QUALIFIED**. The stronger requested **EXPLICIT_VOID_MECHANICS_QUALIFIED** gate remains open until the FEM matrix closes. This is software/numerical qualification, not material validation.
