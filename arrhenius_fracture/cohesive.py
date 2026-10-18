"""Irreversible exponential cohesive interfaces for the R10 kinetic-CZM arm.

The Arrhenius event owner calls :meth:`CohesiveElement.accept_rupture`; the
mechanical law never introduces a second traction threshold.  Before that
accepted event the physical exponential envelope is reversible.  Afterwards
tension and shear remain released, while a distinct numerical contact penalty
acts only in compression.

The node-pair topology follows the audited v10.0.5 donor at commit
``6cf5627f288ae90bb6c78e08d61b1aae545f435c``.  Its penalty/degraded-damage
constitutive law is not copied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import sparse

from .cohesive_arrhenius import (
    BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL,
    COHESIVE_ATHERMAL_MECHANICAL, COHESIVE_MIXED_MECHANICAL_THERMAL,
    COHESIVE_KINETIC_CZM, COHESIVE_RATE_INDEPENDENT_CONTROL,
    CohesiveThermalLaw, cohesive_shape_response,
)


@dataclass
class CohesiveElement:
    plus_nodes: tuple[int, int]
    minus_nodes: tuple[int, int]
    normal: np.ndarray
    tangent: np.ndarray
    length: float
    thermal: CohesiveThermalLaw
    temperature_K: float = 300.0
    rupture_state: str = "INTACT"
    accepted_event_index: int | None = None
    front_id: int = -1
    metadata: dict = field(default_factory=dict)
    cohesive_shape: str = SMOOTH_EXPONENTIAL

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=float)
        self.tangent = np.asarray(self.tangent, dtype=float)
        for name in ("normal", "tangent"):
            value = getattr(self, name)
            norm = float(np.linalg.norm(value))
            if norm <= 0.0:
                raise ValueError(f"cohesive {name} must be non-zero")
            setattr(self, name, value / norm)
        if self.length <= 0.0:
            raise ValueError("cohesive element length must be positive")
        if self.rupture_state not in {"INTACT", "BROKEN"}:
            raise ValueError("rupture_state must be INTACT or BROKEN")
        if self.cohesive_shape not in {SMOOTH_EXPONENTIAL, BILINEAR,
                                       COMPACT_INDEPENDENT}:
            raise ValueError("unsupported cohesive_shape")

    @property
    def nodes4(self) -> tuple[int, int, int, int]:
        return (*self.plus_nodes, *self.minus_nodes)

    @property
    def broken(self) -> bool:
        return self.rupture_state == "BROKEN"

    def accept_rupture(self, event_index: int,
                       separation_m: float | None = None) -> None:
        if int(event_index) < 0:
            raise ValueError("accepted event index must be nonnegative")
        if not self.broken:
            if separation_m is not None:
                intact=float(cohesive_shape_response(
                    self.cohesive_shape,max(float(separation_m),0.0),self.thermal,
                    self.temperature_K)[2])*self.length
                broken=self.thermal.gamma(self.temperature_K)*self.length
                self.metadata.update({
                    "rupture_surface_energy_owner":"COHESIVE_ELEMENT_BROKEN_STATE",
                    "surface_energy_before_J_per_m":intact,
                    "surface_energy_after_J_per_m":broken,
                    "surface_transition_J_per_m":broken-intact,
                    "activation_barrier_charged_as_surface_energy":False,
                })
            self.rupture_state = "BROKEN"
            self.accepted_event_index = int(event_index)

    def accept_rate_independent_spinodal(self, separation_m: float) -> bool:
        if float(separation_m) < self.thermal.delta0_m:
            return False
        if not self.broken:
            self.rupture_state = "BROKEN"
            self.accepted_event_index = -1
            self.metadata["rupture_owner"] = "PEAK_CUTOFF_DIAGNOSTIC"
        return True

    def accept_mechanical_separation(self, separation_m: float,
                                     event_index: int) -> None:
        """Commit geometry-owned separation without invoking a kinetic clock."""
        self.accept_rupture(event_index,separation_m)
        self.metadata.update({"rupture_owner":"MECHANICAL_COHESIVE_COMPLETION",
                              "legacy_cooperative_clock_used":False})


@dataclass
class CohesiveNetwork:
    elements: List[CohesiveElement] = field(default_factory=list)
    contact_penalty_Pa_per_m: float = 1.0e22
    intact_shear_penalty_Pa_per_m: float = 0.0
    execution_identity: str = COHESIVE_KINETIC_CZM
    plastic_closure_Pa: float = 0.0

    def __post_init__(self) -> None:
        if self.execution_identity not in {
            COHESIVE_KINETIC_CZM, COHESIVE_RATE_INDEPENDENT_CONTROL,
            COHESIVE_ATHERMAL_MECHANICAL, COHESIVE_MIXED_MECHANICAL_THERMAL,
        }:
            raise ValueError("unsupported cohesive network identity")
        if self.contact_penalty_Pa_per_m < 0.0 or self.intact_shear_penalty_Pa_per_m < 0.0:
            raise ValueError("numerical penalties must be nonnegative")
        if not np.isfinite(self.plastic_closure_Pa):
            raise ValueError("plastic closure traction must be finite")

    def add(self, elem: CohesiveElement) -> None:
        self.elements.append(elem)

    def active_count(self) -> int:
        return sum(not e.broken for e in self.elements)

    def failed_count(self) -> int:
        return sum(e.broken for e in self.elements)

    def state_rows(self) -> list[dict]:
        return [{"element_index": i, "rupture_state": e.rupture_state,
                 "accepted_event_index": e.accepted_event_index,
                 "front_id": e.front_id, "length_m": e.length,
                 "Gamma_J_m2": e.thermal.gamma(e.temperature_K),
                 "delta0_m": e.thermal.delta0_m,
                 "cohesive_shape": e.cohesive_shape,
                 "compact_variant": (e.thermal.compact_variant
                    if e.cohesive_shape == COMPACT_INDEPENDENT else None),
                 **e.metadata} for i, e in enumerate(self.elements)]

    def energy_ledger(self,u:np.ndarray) -> dict:
        cohesive=0.0;contact=0.0;shear=0.0;plastic_closure_work=0.0
        A=_jump_operator()
        for elem in self.elements:
            nodes=np.asarray(elem.nodes4,dtype=int);edofs=np.empty(8,dtype=int)
            edofs[0::2]=2*nodes;edofs[1::2]=2*nodes+1
            jump=A@np.asarray(u,dtype=float)[edofs]
            dn=float(elem.normal@jump);dt=float(elem.tangent@jump)
            if elem.broken:
                cohesive+=elem.thermal.gamma(elem.temperature_K)*elem.length
            else:
                cohesive+=float(cohesive_shape_response(
                    elem.cohesive_shape,max(dn,0.0),elem.thermal,
                    elem.temperature_K)[2])*elem.length
                shear+=0.5*self.intact_shear_penalty_Pa_per_m*dt*dt*elem.length
                plastic_closure_work+=self.plastic_closure_Pa*dn*elem.length
            if dn<0.0:
                contact+=0.5*self.contact_penalty_Pa_per_m*dn*dn*elem.length
        return {"cohesive_surface_and_storage_J_per_m":cohesive,
                "contact_numerical_energy_J_per_m":contact,
                "intact_shear_numerical_energy_J_per_m":shear,
                "frozen_plastic_closure_work_J_per_m":plastic_closure_work,
                "activation_barrier_surface_charge_J_per_m":0.0,
                "surface_energy_owner":"COHESIVE_NETWORK_STATE",
                "contact_energy_owner":"NUMERICAL_CONTACT_ONLY"}


def _jump_operator() -> np.ndarray:
    A = np.zeros((2, 8), dtype=float)
    A[0, (0, 2)] = 0.5; A[1, (1, 3)] = 0.5
    A[0, (4, 6)] = -0.5; A[1, (5, 7)] = -0.5
    return A


def _response(elem: CohesiveElement, network: CohesiveNetwork,
              jump: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    dn = float(elem.normal @ jump)
    dt = float(elem.tangent @ jump)
    if dn < 0.0:
        # Retain the frozen plastic closing traction on the intact closed
        # branch.  This makes the law continuous at zero opening and lets the
        # contact reaction balance the pre-stress instead of creating a false
        # zero-displacement branch-loss censor.
        closure = 0.0 if elem.broken else float(network.plastic_closure_Pa)
        tn = network.contact_penalty_Pa_per_m * dn + closure
        kn = network.contact_penalty_Pa_per_m
        contact = True
    elif elem.broken:
        tn = 0.0; kn = 0.0; contact = False
    else:
        tn, kn, _ = cohesive_shape_response(
            elem.cohesive_shape,dn,elem.thermal,elem.temperature_K)
        tn=float(tn)+float(network.plastic_closure_Pa);kn=float(kn)
        contact = False
    kt = 0.0 if elem.broken else network.intact_shear_penalty_Pa_per_m
    tt = kt * dt
    traction = tn * elem.normal + tt * elem.tangent
    tangent = kn * np.outer(elem.normal, elem.normal) + kt * np.outer(
        elem.tangent, elem.tangent,
    )
    diag = {"normal_jump_m": dn, "tangential_jump_m": dt,
            "normal_traction_Pa": tn, "tangential_traction_Pa": tt,
            "normal_tangent_Pa_per_m": kn, "contact_active": contact,
            "rupture_state": elem.rupture_state,
            "execution_identity": network.execution_identity,
            "cohesive_shape":elem.cohesive_shape,
            "compact_variant":(elem.thermal.compact_variant
                if elem.cohesive_shape == COMPACT_INDEPENDENT else None),
            "plastic_closure_traction_Pa":(0.0 if elem.broken else
                                             float(network.plastic_closure_Pa)),
            "surface_free_energy_J_m2": (elem.thermal.gamma(elem.temperature_K)
                if elem.broken else float(cohesive_shape_response(
                    elem.cohesive_shape,max(dn,0.0),elem.thermal,
                    elem.temperature_K)[2])),
            "surface_energy_owner":"COHESIVE_ELEMENT_STATE",
            "activation_barrier_charged_as_surface_energy":False}
    return traction, tangent, diag


def cohesive_element_diagnostics(elem: CohesiveElement, u: np.ndarray,
                                 network: CohesiveNetwork) -> dict:
    nodes = np.asarray(elem.nodes4, dtype=int)
    edofs = np.empty(8, dtype=int)
    edofs[0::2] = 2 * nodes; edofs[1::2] = 2 * nodes + 1
    jump = _jump_operator() @ np.asarray(u, dtype=float)[edofs]
    return _response(elem, network, jump)[2]


def cohesive_contribution(network: Optional[CohesiveNetwork], u: np.ndarray,
                          ndof: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    if network is None or not network.elements:
        return sparse.csr_matrix((ndof, ndof)), np.zeros(ndof)
    A = _jump_operator(); rows = []; cols = []; vals = []
    residual = np.zeros(ndof, dtype=float)
    for elem in network.elements:
        nodes = np.asarray(elem.nodes4, dtype=int)
        edofs = np.empty(8, dtype=int)
        edofs[0::2] = 2 * nodes; edofs[1::2] = 2 * nodes + 1
        jump = A @ np.asarray(u, dtype=float)[edofs]
        traction, tangent, diagnostics = _response(elem, network, jump)
        length = float(elem.length)
        local_R = A.T @ traction * length
        local_K = A.T @ tangent @ A * length
        np.add.at(residual, edofs, local_R)
        elem.metadata.update(diagnostics)
        ii = np.repeat(edofs[:, None], 8, axis=1)
        jj = np.repeat(edofs[None, :], 8, axis=0)
        rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(local_K.ravel())
    matrix = sparse.csr_matrix((np.concatenate(vals), (
        np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof))
    return matrix, residual


__all__ = ["CohesiveElement", "CohesiveNetwork", "cohesive_contribution",
           "cohesive_element_diagnostics"]
