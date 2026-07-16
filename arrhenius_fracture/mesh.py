"""
Mesh generation and boundary conditions for triangular FE mesh.

Produces an unstructured Delaunay triangulation with controlled randomness
for a notched rectangular plate.
"""

import numpy as np
from scipy.spatial import Delaunay
from dataclasses import dataclass
from typing import Tuple

from .config import GeometryConfig, MeshConfig


@dataclass
class TriMesh:
    """Triangular finite element mesh with precomputed quantities."""
    nodes: np.ndarray          # (nn, 2) node coordinates
    elems: np.ndarray          # (ne, 3) element connectivity (0-based)
    nn: int                    # number of nodes
    ne: int                    # number of elements
    ndof: int                  # number of displacement DOFs (2*nn)
    hbar: float                # characteristic element size (global mean)
    area_e: np.ndarray         # (ne,) element areas
    dNdx_e: np.ndarray         # (ne, 2, 3) shape function gradients per element
    B_e: np.ndarray            # (ne, 3, 6) strain-displacement matrices
    hbar_tip: float = 0.0      # tip-local mean edge length (process-zone resolution)


@dataclass
class BoundaryData:
    """Boundary node sets and notch damage."""
    top_nodes: np.ndarray      # node indices on top edge
    bot_nodes: np.ndarray      # node indices on bottom edge
    left_bot: int              # bottom-left corner node
    right_bot: int             # bottom-right corner node
    notch_nodes: np.ndarray    # nodes in initial notch band (d=1)


def _radial_ring_nodes(Lx, Ly, xt, yt, h_fine, slope, h_far, R_fine):
    """Generate well-shaped graded node placement around the tip (xt, yt):
    concentric rings whose radial AND angular spacing both follow the local
    size h(r) = min(h_fine + slope*r, h_far), unioned with a coarse background
    grid outside R_fine.  Delaunay of this point set gives ISOTROPIC triangles
    (min angle ~25-30 deg) -- a tensor-product graded grid cannot, because a
    cell's dx and dy are set by different radii, producing 100:1 slivers.
    """
    def size(r):
        return min(h_fine + slope * r, h_far)
    pts = [(xt, yt)]
    r = h_fine
    while r < R_fine:
        h = size(r)
        ntheta = max(8, int(2 * np.pi * r / h))
        for k in range(ntheta):
            th = 2 * np.pi * k / ntheta
            x, y = xt + r * np.cos(th), yt + r * np.sin(th)
            if 0 <= x <= Lx and -Ly / 2 <= y <= Ly / 2:
                pts.append((x, y))
        r += h
    # coarse background grid outside the fine disk
    nbx = max(2, int(Lx / h_far))
    nby = max(2, int(Ly / h_far))
    gx, gy = np.meshgrid(np.linspace(0, Lx, nbx + 1),
                         np.linspace(-Ly / 2, Ly / 2, nby + 1))
    bg = np.c_[gx.ravel(), gy.ravel()]
    keep = np.hypot(bg[:, 0] - xt, bg[:, 1] - yt) > 1.05 * R_fine
    pts = np.vstack([np.array(pts), bg[keep]])
    # ensure the four domain corners and edges are represented for clean BCs
    return pts


def make_tri_mesh(geom: GeometryConfig, mesh_cfg: MeshConfig,
                  seed: int = None, tip_center=None) -> TriMesh:
    """Generate a triangular mesh of the plate.

    If mesh_cfg.tip_h_fine > 0, an ADAPTIVE radial mesh is built: concentric
    rings around the refinement center (default the notch tip (a0, 0), or an
    arbitrary `tip_center=(xc,yc)` for tip-following remeshing) spaced
    ~tip_h_fine at the center and coarsening linearly with radius (slope
    tip_ratio-1) up to a far-field cap, unioned with a coarse background grid and
    Delaunay-triangulated.  This gives isotropic, well-shaped triangles down to
    sub-L_pz resolution at ~log node cost.  Otherwise a uniform jittered grid.
    """
    if seed is not None:
        np.random.seed(seed)

    Lx, Ly = geom.Lx, geom.Ly
    nx, ny = mesh_cfg.nx, mesh_cfg.ny
    h_fine = getattr(mesh_cfg, 'tip_h_fine', 0.0)
    ratio = getattr(mesh_cfg, 'tip_ratio', 1.15)
    graded = bool(h_fine and h_fine > 0)
    if tip_center is None:
        centers = np.array([[geom.a0, 0.0]], float)
    else:
        tc = np.asarray(tip_center, dtype=float)
        centers = tc.reshape(1, 2) if tc.ndim == 1 else tc[:, :2]
    xc, yc = float(centers[0, 0]), float(centers[0, 1])

    if graded:
        slope = max(ratio - 1.0, 0.02)        # linear growth rate of h with r
        h_far = max(Lx, Ly) / 40.0            # far-field background size
        R_fine = min(0.15 * max(Lx, Ly),      # extent of the refined disk
                     max(40 * h_fine, 0.05e-3))
        # Multi-tip refinement: union radial fine patches around all active
        # crack tips.  This is essential once branches separate; otherwise only
        # the leading x-tip is process-zone resolved and daughter J-integrals are
        # dominated by coarse background elements.
        clouds = [_radial_ring_nodes(Lx, Ly, float(c[0]), float(c[1]), h_fine,
                                     slope, h_far, R_fine) for c in centers]
        nodes = np.vstack(clouds)
        key = np.round(nodes / max(1e-12, 0.05 * h_fine)).astype(np.int64)
        _, keep_idx = np.unique(key, axis=0, return_index=True)
        nodes = nodes[np.sort(keep_idx)]
        tri = Delaunay(nodes)
        elems = tri.simplices
        centroids = nodes[elems].mean(axis=1)
        inside = ((centroids[:, 0] >= 0) & (centroids[:, 0] <= Lx) &
                  (centroids[:, 1] >= -Ly / 2) & (centroids[:, 1] <= Ly / 2))
        elems = elems[inside]
    else:
        xv = np.linspace(0, Lx, nx + 1)
        yv = np.linspace(-Ly / 2, Ly / 2, ny + 1)
        hx = xv[1] - xv[0]
        hy = yv[1] - yv[0]
        nodes_list = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = xv[i], yv[j]
                on_boundary = (i == 0 or i == nx or j == 0 or j == ny)
                if not on_boundary:
                    x += mesh_cfg.jitter * hx * (2 * np.random.rand() - 1)
                    y += mesh_cfg.jitter * hy * (2 * np.random.rand() - 1)
                nodes_list.append([x, y])
        nodes = np.array(nodes_list)
        tri = Delaunay(nodes)
        elems = tri.simplices
        centroids = nodes[elems].mean(axis=1)
        inside = ((centroids[:, 0] >= 0) & (centroids[:, 0] <= Lx) &
                  (centroids[:, 1] >= -Ly / 2) & (centroids[:, 1] <= Ly / 2))
        elems = elems[inside]

    nn = nodes.shape[0]
    ne = elems.shape[0]
    ndof = 2 * nn

    # Precompute element quantities
    hbar = _estimate_hbar(nodes, elems)
    # tip-local resolution: mean edge length among elements within a small patch
    # around the crack tip (a0, 0).  For a graded mesh this is the number that
    # matters for the process zone, not the global-mean hbar.
    hbar_tip = float(min(_estimate_hbar_tip(nodes, elems, float(c[0]), float(c[1]))
                          for c in centers))
    area_e, dNdx_e, B_e = _precompute_element_data(nodes, elems)

    return TriMesh(
        nodes=nodes, elems=elems, nn=nn, ne=ne, ndof=ndof,
        hbar=hbar, hbar_tip=hbar_tip, area_e=area_e, dNdx_e=dNdx_e, B_e=B_e
    )



def rebuild_tri_mesh(nodes: np.ndarray, elems: np.ndarray, tip_centers=None, validate: bool = True) -> TriMesh:
    """Rebuild a :class:`TriMesh` after topology-only node splitting.

    Element ordering is preserved exactly so Gauss-point material histories remain
    aligned with their original element indices.  This is the key operation used
    by the edge-split CZM migration backend.
    """
    nodes = np.ascontiguousarray(np.asarray(nodes, dtype=float))
    elems = np.ascontiguousarray(np.asarray(elems, dtype=int))
    if validate:
        if elems.ndim != 2 or elems.shape[1] != 3:
            raise ValueError("triangular connectivity must have shape (ne,3)")
        if np.min(elems) < 0 or np.max(elems) >= len(nodes):
            raise ValueError("element connectivity references invalid node index")
    area_e, dNdx_e, B_e = _precompute_element_data(nodes, elems)
    if validate and np.any(~np.isfinite(area_e)):
        raise ValueError("non-finite element area after topology update")
    if validate and np.any(area_e <= 1e-24):
        bad = np.where(area_e <= 1e-24)[0][:10]
        raise ValueError(f"degenerate elements after topology update: {bad.tolist()}")
    hbar = _estimate_hbar(nodes, elems)
    if tip_centers is None:
        hbar_tip = hbar
    else:
        tc = np.asarray(tip_centers, dtype=float)
        tc = tc.reshape(1,2) if tc.ndim == 1 else tc[:, :2]
        hbar_tip = float(min(_estimate_hbar_tip(nodes, elems, float(c[0]), float(c[1])) for c in tc))
    return TriMesh(
        nodes=nodes, elems=elems, nn=len(nodes), ne=len(elems), ndof=2*len(nodes),
        hbar=float(hbar), area_e=area_e, dNdx_e=dNdx_e, B_e=B_e,
        hbar_tip=float(hbar_tip),
    )

def make_boundary_data(mesh: TriMesh, geom: GeometryConfig) -> BoundaryData:
    """Identify boundary node sets and notch region."""
    tol = 0.3 * mesh.hbar
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    top_nodes = np.where(np.abs(y - geom.Ly / 2) < tol)[0]
    bot_nodes = np.where(np.abs(y + geom.Ly / 2) < tol)[0]

    # Corner nodes
    left_bot = np.argmin((x - 0)**2 + (y + geom.Ly / 2)**2)
    right_bot = np.argmin((x - geom.Lx)**2 + (y + geom.Ly / 2)**2)

    # Initial notch band
    notch = (x <= geom.a0) & (np.abs(y) <= geom.notch_half_thickness)
    notch_nodes = np.where(notch)[0]

    return BoundaryData(
        top_nodes=top_nodes, bot_nodes=bot_nodes,
        left_bot=left_bot, right_bot=right_bot,
        notch_nodes=notch_nodes
    )


def _estimate_hbar_tip(nodes, elems, xt, yt):
    """Mean edge length among elements whose centroid lies within a small patch
    around the crack tip (xt, yt).  Patch radius adapts to the local mesh so it
    always captures a handful of tip elements.
    """
    cent = nodes[elems].mean(axis=1)
    d = np.hypot(cent[:, 0] - xt, cent[:, 1] - yt)
    # take the nearest ~2% of elements (at least 4) to the tip
    k = max(4, int(0.02 * len(elems)))
    idx = np.argsort(d)[:k]
    sub = elems[idx]
    edges = np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
    L = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    return float(np.mean(L))


def _estimate_hbar(nodes: np.ndarray, elems: np.ndarray) -> float:
    """Average edge length."""
    edges = np.vstack([
        elems[:, [0, 1]],
        elems[:, [1, 2]],
        elems[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    return float(np.mean(lengths))


def tri_shape_grad(Xe: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute area and shape function gradients for a single triangle.

    Parameters
    ----------
    Xe : (3, 2) array of nodal coordinates

    Returns
    -------
    area : element area
    dNdx : (2, 3) shape function gradients [dN/dx; dN/dy]
    """
    x1, y1 = Xe[0]
    x2, y2 = Xe[1]
    x3, y3 = Xe[2]

    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    b1 = y2 - y3;  c1 = x3 - x2
    b2 = y3 - y1;  c2 = x1 - x3
    b3 = y1 - y2;  c3 = x2 - x1

    dNdx = np.array([[b1, b2, b3],
                      [c1, c2, c3]]) / (2 * area)

    return area, dNdx


def _precompute_element_data(nodes: np.ndarray, elems: np.ndarray):
    """Precompute areas, shape function gradients, and B matrices."""
    ne = elems.shape[0]
    area_e = np.zeros(ne)
    dNdx_e = np.zeros((ne, 2, 3))
    B_e = np.zeros((ne, 3, 6))

    for e in range(ne):
        Xe = nodes[elems[e]]
        area, dNdx = tri_shape_grad(Xe)
        area_e[e] = area
        dNdx_e[e] = dNdx

        # Strain-displacement matrix B (3x6) for plane strain
        # eps = [eps_xx, eps_yy, gamma_xy]^T = B * u_e
        B = np.zeros((3, 6))
        for a in range(3):
            B[0, 2*a]     = dNdx[0, a]   # dN/dx -> eps_xx
            B[1, 2*a + 1] = dNdx[1, a]   # dN/dy -> eps_yy
            B[2, 2*a]     = dNdx[1, a]   # dN/dy -> gamma_xy
            B[2, 2*a + 1] = dNdx[0, a]   # dN/dx -> gamma_xy
        B_e[e] = B

    return area_e, dNdx_e, B_e
