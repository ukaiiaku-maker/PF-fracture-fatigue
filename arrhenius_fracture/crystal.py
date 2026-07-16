"""
crystal.py -- BCC crystal anisotropy for the sharp-front fracture engine
========================================================================

Foundation for ANISOTROPIC crack advance (deflection + branching) built on the
hazard / activation-work formulation of the group's yield-surface paper
(W = sigma:A, A = phi V*, with Schmid + non-Schmid coupling; hazards add over
systems, Lambda = sum_s Lambda_s).  The crack tip will run a first-passage
cleavage clock PER candidate crystallographic plane; the advance direction(s)
follow from which plane hazards fire (one -> deflection, two co-critical ->
branch, sharing the single-front energy budget).

This module provides the two anisotropy sources that make the per-plane hazards
DIFFER (without that, the near-tip field is a single smooth lobe and only
deflection -- never branching -- is possible):

  (1) ANISOTROPIC ELASTICITY  -- cubic C_ijkl(orientation) so the near-tip stress
      carries the crystal orientation.
  (2) DISCRETE CLEAVAGE / SLIP PLANES -- the candidate advance directions and
      their Schmid / non-Schmid activation tensors.

Note on tungsten: real W is elastically NEAR-ISOTROPIC
(Zener A = 2 C44/(C11-C12) = 2*160/(523-203) ~ 1.0), so for W the directional
anisotropy comes mainly from the discrete {100} cleavage planes, not elasticity.
The elastic anisotropy is left tunable (set C44 to change the Zener ratio) so a
model BCC can also branch elastically and so the machinery is exercised.

Plane-strain, in-plane (2D) crystallography is used: a crystal in-plane
orientation angle ``theta_deg`` rotates the cubic axes about the out-of-plane z.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  Cubic elastic constants (GPa).  W is the near-isotropic baseline.
# --------------------------------------------------------------------------- #
# Tungsten (single crystal, ~300 K):  Zener A ~ 1.0  (elastically isotropic)
W_C11, W_C12, W_C44 = 523.0e9, 203.0e9, 160.0e9


def zener_ratio(C11, C12, C44):
    """Zener anisotropy A = 2 C44 / (C11 - C12).  A = 1 is elastically isotropic."""
    return 2.0 * C44 / (C11 - C12)


def cubic_stiffness_6x6(C11, C12, C44):
    """Cubic stiffness in Voigt 6x6 (crystal frame), order [xx,yy,zz,yz,xz,xy]."""
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = C12
        C[i, i] = C11
    for k in (3, 4, 5):
        C[k, k] = C44
    return C


def _cubic_tensor(C11, C12, C44):
    """Full 4th-order cubic stiffness C_ijkl (3x3x3x3) in the crystal frame."""
    C = np.zeros((3, 3, 3, 3))
    d = np.eye(3)
    # cubic: C_ijkl = C12 d_ij d_kl + C44 (d_ik d_jl + d_il d_jk)
    #                 + (C11 - C12 - 2 C44) sum_m d_im d_jm d_km d_lm
    H = C11 - C12 - 2.0 * C44
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (C12 * d[i, j] * d[k, l]
                                     + C44 * (d[i, k] * d[j, l] + d[i, l] * d[j, k])
                                     + H * sum(d[i, m] * d[j, m] * d[k, m] * d[l, m]
                                               for m in range(3)))
    return C


def cubic_plane_strain_D(C11=W_C11, C12=W_C12, C44=W_C44, theta_deg=0.0):
    """Plane-strain 3x3 elasticity D for a cubic crystal rotated by theta_deg
    (in-plane, about z).  Maps [eps_xx, eps_yy, gamma_xy] -> [sig_xx, sig_yy, sig_xy].

    Built by rotating the full 4th-order tensor (no Voigt-convention ambiguity):
    C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs, then the plane-strain (eps_zz=0) D uses
    the in-plane components; the gamma_xy column equals C'_..xy (engineering shear).
    """
    C = _cubic_tensor(C11, C12, C44)
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # specimen <- crystal
    Cr = np.einsum('ip,jq,kr,ls,pqrs->ijkl', R, R, R, R, C)
    x, y = 0, 1
    D = np.array([
        [Cr[x, x, x, x], Cr[x, x, y, y], Cr[x, x, x, y]],
        [Cr[y, y, x, x], Cr[y, y, y, y], Cr[y, y, x, y]],
        [Cr[x, y, x, x], Cr[x, y, y, y], Cr[x, y, x, y]],
    ])
    return D


# --------------------------------------------------------------------------- #
#  BCC cleavage / slip planes -- the candidate advance directions (2D traces)
# --------------------------------------------------------------------------- #
# In 2D plane strain we use the in-plane TRACES of the 3D planes: each candidate
# is a line in the x-y plane along which the crack can advance, defined by its
# in-plane normal n_hat (the plane normal projected in-plane) and the advance
# direction t_hat = perpendicular to n_hat in-plane.  Crack opens against n_hat;
# it advances along t_hat.  Angles are measured from the specimen +x axis and
# include the crystal orientation theta_deg.
#
# BCC fracture: {100} are the cleavage planes (brittle cleavage).  {110} and
# {112} are the slip planes (plasticity).  In a 2D section normal to a <001>
# zone axis, the {100} cleavage traces lie at 0 and 90 deg (crystal frame), and
# {110} traces at +/-45 deg -- a natural set that can produce 90-deg cleavage
# steps and 45-deg shear deflection / branching.

def bcc_cleavage_traces(theta_deg=0.0, include_110=False, gamma_110_rel=1.3):
    """In-plane advance directions (unit t_hat) for {100} cleavage, plus the
    crack-normal n_hat.  Returns list of dicts with 'name','t','n','angle_deg',
    'gamma_rel'.  For a <001> section the {100} traces are at 0 and 90 deg
    (crystal frame); the {110} traces (secondary cleavage in many BCC metals)
    are at 45 and 135 deg.  gamma_rel = gamma_plane / gamma_{100} sets the
    RELATIVE cleavage resistance (>1 for the less-favored {110}); the per-plane
    overdrive is sigma_nn / sqrt(gamma_rel), so a FINITE gamma anisotropy makes
    off-{100} planes admissible under the right mechanical drive instead of the
    argmax treating {100} as infinitely preferred."""
    base = [('(100)', 0.0, 1.0), ('(010)', 90.0, 1.0)]
    if include_110:
        base += [('(110)', 45.0, gamma_110_rel), ('(1-10)', 135.0, gamma_110_rel)]
    return _make_traces(base, theta_deg, family='cleavage')


def bcc_slip_traces(theta_deg=0.0):
    """In-plane traces for {110}/{112} slip systems (45 / ~35-deg families) used
    for the anisotropic plasticity and the shear-deflection channel."""
    base = [('(110)', 45.0), ('(1-10)', -45.0)]
    return _make_traces(base, theta_deg, family='slip')


def _make_traces(base, theta_deg, family):
    out = []
    for entry in base:
        # entry is (name, angle) or (name, angle, gamma_rel)
        name, ang = entry[0], entry[1]
        gamma_rel = entry[2] if len(entry) > 2 else 1.0
        a = np.deg2rad(ang + theta_deg)
        t = np.array([np.cos(a), np.sin(a)])         # advance direction (in plane)
        n = np.array([-np.sin(a), np.cos(a)])        # crack-normal (opening) direction
        out.append({'name': name, 'family': family, 'gamma_rel': float(gamma_rel),
                    'angle_deg': ang + theta_deg, 't': t, 'n': n})
    return out


# --------------------------------------------------------------------------- #
#  Directional cleavage selection (max resolved opening stress = W=sigma:A with
#  the cleavage activation tensor A_s ~ phi v* (n_s (x) n_s), the normal projector)
# --------------------------------------------------------------------------- #
def near_tip_stress_tensor(sigma_gp, mesh, tip_xy, radius):
    """Average 2x2 stress tensor over GPs within `radius` of the tip.
    sigma_gp is (3, ne) = [sxx, syy, sxy]."""
    cx = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
    cy = mesh.nodes[mesh.elems].mean(axis=1)[:, 1]
    sel = (cx - tip_xy[0]) ** 2 + (cy - tip_xy[1]) ** 2 <= radius ** 2
    if not sel.any():
        # fall back to the single nearest element
        sel = np.zeros(mesh.ne, bool)
        sel[np.argmin((cx - tip_xy[0]) ** 2 + (cy - tip_xy[1]) ** 2)] = True
    sxx = float(sigma_gp[0, sel].mean())
    syy = float(sigma_gp[1, sel].mean())
    sxy = float(sigma_gp[2, sel].mean())
    return np.array([[sxx, sxy], [sxy, syy]])


def pick_cleavage_plane(sigma2x2, planes, forward=None, min_forward=0.0):
    """Choose the cleavage plane with the largest resolved OPENING (normal) stress
    sigma_nn,s = n_s . sigma . n_s  -- the crystallographic max-normal-stress
    criterion (== W=sigma:A_s for the cleavage normal projector).  Optionally
    require the advance direction t_s to have forward component >= min_forward
    along `forward` (prevents the crack from reversing into its own wake).

    Returns (winner_dict, sigma_nn_list).  winner carries 't' (advance dir),
    'n' (opening normal), 'angle_deg', 'name'.
    """
    snn = []
    cand = []
    for p in planes:
        n = p['n']
        s_nn = float(n @ sigma2x2 @ n)        # resolved opening stress on plane s
        t = p['t']
        # allow advance along +t or -t (a plane trace has two directions); pick the
        # one most forward, and apply the no-reversal gate
        if forward is not None:
            if np.dot(t, forward) < np.dot(-t, forward):
                t = -t
            if np.dot(t, forward) < min_forward:
                snn.append(s_nn)
                continue                       # this plane would reverse -> skip as a winner
        snn.append(s_nn)
        cand.append((s_nn, dict(p, t=t)))
    if not cand:                               # all reversed -> keep straightest forward plane
        # fall back: the plane whose t is most aligned with `forward`
        best = max(planes, key=lambda p: abs(np.dot(p['t'], forward)) if forward is not None else 0.0)
        tt = best['t'] if (forward is None or np.dot(best['t'], forward) >= 0) else -best['t']
        return dict(best, t=tt), snn
    winner = max(cand, key=lambda c: c[0])[1]
    return winner, snn


def cleavage_branch_candidates(sigma2x2, planes, forward=None, min_forward=0.2,
                               branch_ratio=0.92):
    """Return the cleavage plane(s) eligible to advance THIS step.

    - Compute resolved opening stress sigma_nn,s on each forward-admissible plane.
    - The winner is max sigma_nn.  A SECOND plane is co-critical (a branch) if its
      sigma_nn >= branch_ratio * winner AND it is well-separated in direction
      (not the same plane / not nearly collinear).
    Returns a list of 1 (deflection) or 2 (branch) plane dicts, each with 't'
    oriented forward.  Branching as a competing-hazard outcome (paper Eq 30:
    Lambda = sum_s Lambda_s) -- two systems near-equally driven.
    """
    scored = []
    for p in planes:
        n = p['n']
        s_nn = float(n @ sigma2x2 @ n)
        t = p['t']
        if forward is not None and np.dot(t, forward) < np.dot(-t, forward):
            t = -t
        if forward is not None and np.dot(t, forward) < min_forward:
            continue
        if s_nn <= 0:
            continue
        g = float(p.get('gamma_rel', 1.0))
        overdrive = max(s_nn, 0.0) / np.sqrt(max(g, 1e-12))
        scored.append((overdrive, dict(p, t=t, sigma_nn=s_nn,
                                       overdrive=overdrive, gamma_rel=g)))
    if not scored:
        return []
    scored.sort(key=lambda c: -c[0])
    winners = [scored[0][1]]
    s_top = scored[0][0]
    for s_nn, p in scored[1:]:
        if s_nn >= branch_ratio * s_top:
            # require directional separation from already-selected winners
            if all(abs(float(p['t'] @ w['t'])) < 0.97 for w in winners):
                winners.append(p)
                if len(winners) >= 2:    # cap at a binary branch for now
                    break
    return winners


def admissible_openings(sigma2x2, planes, forward=None, min_forward=0.2):
    """Forward-admissible cleavage planes with their resolved opening and OVERDRIVE.

    For each plane: sigma_nn,s = n_s . sigma . n_s  (resolved opening stress) and
    the cleavage OVERDRIVE  O_s = max(sigma_nn,s, 0) / sqrt(gamma_rel,s).  The
    sqrt(gamma_rel) is the Griffith resistance scaling (sigma_threshold ~
    sqrt(2 E gamma)), so a plane with higher surface energy needs proportionally
    more opening stress to be equally driven.  Returns a list (sorted by O,
    descending) of dicts carrying 't' (forward-oriented), 'n', 'name',
    'sigma_nn', 'overdrive', 'gamma_rel'.  The ENGINE turns each overdrive into a
    per-plane cleavage hazard lambda_s and competes them (Lambda = sum_s lambda_s)
    -- finite, T-dependent selection instead of an infinitely sharp argmax.
    """
    out = []
    for p in planes:
        n = p['n']; t = p['t']
        s_nn = float(n @ sigma2x2 @ n)
        if forward is not None:
            if np.dot(t, forward) < np.dot(-t, forward):
                t = -t
            if np.dot(t, forward) < min_forward:
                continue
        g = float(p.get('gamma_rel', 1.0))
        O = max(s_nn, 0.0) / np.sqrt(max(g, 1e-12))
        out.append(dict(p, t=t, sigma_nn=s_nn, overdrive=O, gamma_rel=g))
    out.sort(key=lambda c: -c['overdrive'])
    return out


def cubic_cleavage_gamma(psi_rad, theta_deg, gamma_aniso):
    """Relative cleavage surface energy gamma(psi)/gamma_{100} for a crack whose
    OPENING NORMAL points at lab angle psi.  Cubic 4-fold anisotropy with minima
    on the <100> cube axes (the {100} cleavage planes) and maxima at 45 deg
    ({110}).  gamma_aniso = gamma_{110}/gamma_{100} - 1 is the anisotropy
    amplitude: 0 = isotropic (crack follows the mechanical/mode-I direction),
    large = deep {100} wells (crack locks to {100}).  Finite = {100} preferred
    but off-axis directions admissible under sufficient mechanical drive."""
    chi = psi_rad - np.deg2rad(theta_deg)            # normal angle in crystal frame
    return 1.0 + gamma_aniso * np.sin(2.0 * chi) ** 2


def cleave_direction_competition(sigma2x2, theta_deg, forward, min_forward=0.2,
                                 gamma_aniso=0.3, branch_ratio=0.9, n_phi=181,
                                 sep_deg=15.0):
    """Continuous competition for the cleavage ADVANCE direction.

    Scans advance directions t(alpha) over the forward half-plane (|alpha| up to
    the no-reversal limit set by min_forward) and ranks them by the cleavage
    OVERDRIVE  O(t) = max(sigma_nn, 0) / sqrt(gamma(psi)),  where sigma_nn =
    n.sigma.n is the resolved opening on the plane with normal n perp t, and
    gamma(psi) is the cubic cleavage-energy anisotropy (minima on <100>).  This
    is the driving/resistance ratio: the crack runs where opening stress most
    exceeds cleavage resistance.

    Returns a list (sorted by overdrive, descending) of local-maximum directions,
    each a dict with 't','n','angle_deg','sigma_nn','overdrive','gamma'.  The
    first is the deflection winner; any later entry with overdrive >=
    branch_ratio*max (and >= sep_deg from the winner) is a co-critical BRANCH.
    Branching is thus emergent wherever O(t) is multi-modal -- no knife-edge.
    """
    f = np.asarray(forward, float); f = f / (np.linalg.norm(f) + 1e-30)
    a_max = np.arccos(np.clip(min_forward, -1.0, 1.0))     # no-reversal cone
    alphas = np.linspace(-a_max, a_max, n_phi)
    ca, sa = np.cos(alphas), np.sin(alphas)
    # advance dirs t = R(alpha) f  (rotate heading); normal n = perp(t)
    tx = ca * f[0] - sa * f[1]
    ty = sa * f[0] + ca * f[1]
    nx, ny = -ty, tx
    s = sigma2x2
    s_nn = nx * (s[0, 0] * nx + s[0, 1] * ny) + ny * (s[1, 0] * nx + s[1, 1] * ny)
    psi = np.arctan2(ny, nx)
    gam = cubic_cleavage_gamma(psi, theta_deg, gamma_aniso)
    O = np.maximum(s_nn, 0.0) / np.sqrt(np.maximum(gam, 1e-12))
    # interior local maxima (plus rising endpoints)
    loc = []
    for i in range(len(alphas)):
        left = O[i - 1] if i > 0 else -np.inf
        right = O[i + 1] if i < len(alphas) - 1 else -np.inf
        if O[i] >= left and O[i] >= right and O[i] > 0.0:
            loc.append(i)
    if not loc:
        i = int(np.argmax(O)); loc = [i]
    loc.sort(key=lambda i: -O[i])
    out = []
    for i in loc:
        ang = np.rad2deg(np.arctan2(ty[i], tx[i]))
        out.append({'name': 'cleave', 'family': 'cleavage',
                    't': np.array([tx[i], ty[i]]), 'n': np.array([nx[i], ny[i]]),
                    'angle_deg': float(ang), 'sigma_nn': float(s_nn[i]),
                    'overdrive': float(O[i]), 'gamma': float(gam[i])})
    # keep winner + well-separated secondaries above the branch ratio
    sel = [out[0]]
    Otop = out[0]['overdrive']
    for c in out[1:]:
        if c['overdrive'] >= branch_ratio * Otop:
            if all(abs(float(c['t'] @ w['t'])) < np.cos(np.deg2rad(sep_deg)) for w in sel):
                sel.append(c)
    return sel, out
