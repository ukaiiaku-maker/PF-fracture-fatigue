"""Fail-closed local qualification manifest and capacity admission check."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any


SCHEMA = "common-fracture-qualification-only/1"


@dataclass(frozen=True)
class QualificationJob:
    job_id: str
    goal_ids: tuple[str, ...]
    engineering_question: str
    native_entrypoint: str
    native_args: tuple[str, ...]
    physical_endpoint: str
    numerical_guards: tuple[str, ...]
    required_observables: tuple[str, ...]
    expected_peak_GiB: float
    retained_output_policy: str
    evidence_status: str = "CONFIGURED_NOT_EXECUTED"
    substantial_worker_slots: int = 1
    production_job: bool = False

    def __post_init__(self) -> None:
        if self.production_job:
            raise ValueError("production jobs are forbidden by the qualification-only schema")
        if self.substantial_worker_slots != 1:
            raise ValueError("local readiness permits exactly one substantial worker")
        if not self.native_entrypoint or self.expected_peak_GiB <= 0.0:
            raise ValueError("qualification jobs require an entry point and positive peak estimate")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("goal_ids", "native_args", "numerical_guards", "required_observables"):
            d[key] = list(d[key])
        return d


@dataclass(frozen=True)
class CapacityReceipt:
    path: str
    free_GiB: float
    reserve_GiB: float
    peak_GiB: float
    admitted: bool
    reason: str


def capacity_preflight(path: str | Path, peak_GiB: float, *, reserve_GiB: float = 12.0) -> CapacityReceipt:
    root = Path(path).resolve()
    usage = shutil.disk_usage(root)
    free = usage.free / 2**30
    admitted = free >= float(peak_GiB) + float(reserve_GiB)
    return CapacityReceipt(str(root), free, reserve_GiB, peak_GiB, admitted,
                           "capacity admitted" if admitted else "DEFERRED_RESOURCE: peak plus reserve exceeds free space")


def qualification_jobs() -> tuple[QualificationJob, ...]:
    compact = "retain config/source identity, scalar history, residual table, accepted state and restart input; no movie/dense sequence"
    return (
        QualificationJob("Q-MECH-PATCH", ("FG-08", "FG-09", "FG-10"),
                         "Do plane stress, plane strain and specified transverse constraint deliver the declared full tensor?",
                         "pytest:tests/test_readiness_mechanics_state.py", (), "material-point assertions pass; spatial patch remains open",
                         ("300 s wall", "2 GiB memory"), ("sigma_zz", "tensor rotation", "work", "derivative"), 0.05, compact,
                         "PARTIAL_CONTRACT_QUALIFIED_SPATIAL_PENDING"),
        QualificationJob("Q-PARAM-DOSSIER", ("FG-02", "FG-03", "FG-05", "FG-15"),
                         "Are all four parent rows valid for every active mechanism over their requested domain?",
                         "scripts/build_common_readiness_handoff.py", ("--parameter-dossier",), "complete declared temperature grid",
                         ("no solver",), ("barriers", "slopes", "curvatures", "rates", "domains"), 0.05, compact,
                         "QUALIFIED_COMPACT"),
        QualificationJob("Q-PF-FEM-3LEVEL", ("FG-12",),
                         "Do original PF and FEM backends trend to the same pre-event observables on independent meshes?",
                         "scripts/run_pf_fem_readiness_qualification.py", ("--levels", "3", "--opening-only"),
                         "three fixed pre-event common loads and one short accepted/restart event",
                         ("one worker", "capacity preflight", "wall cap 6 h"),
                         ("reaction", "stored energy", "stress", "displacement", "J/support", "restart"), 18.0, compact,
                         "UNRESOLVED_GENERIC_FEM_ADAPTER"),
        QualificationJob("Q-CAVITY-LOCAL", ("FG-13", "FG-16"),
                         "Does opt-in local_incremental_front preserve solid-area accounting, marks, clocks and rollback?",
                         "pytest:tests/test_candidate_drive_contract.py", (), "solid-area and transaction/refinement contract checks; spatial geometry cases remain open",
                         ("no production exposure",), ("released work", "new solid area", "clock", "ownership", "restart"), 0.1, compact,
                         "PARTIAL_CONTRACT_QUALIFIED_SPATIAL_PENDING"),
        QualificationJob("Q-BRANCH-RECURSIVE", ("FG-13", "FG-14", "FG-16"),
                         "Can native birth, daughter continuation and a second-generation birth survive restart?",
                         "scripts/run_recursive_branch_readiness.py", ("--second-generation",),
                         "second-generation native birth followed by one accepted continuation",
                         ("one worker", "capacity preflight", "wall cap 8 h"),
                         ("birth", "owner", "cluster/local", "inherited clock", "restart"), 24.0, compact,
                         "UNRESOLVED_SECOND_GENERATION_NATIVE_ADAPTER"),
    )


def manifest_payload(path: str | Path) -> dict[str, Any]:
    jobs = qualification_jobs()
    return {"schema": SCHEMA, "production_execution_authorized": False,
            "maximum_substantial_workers": 1,
            "capacity": asdict(capacity_preflight(path, max(j.expected_peak_GiB for j in jobs))),
            "jobs": [j.to_dict() for j in jobs]}


def assert_qualification_launch(job_id: str, *, production_execution_authorized: bool = False) -> QualificationJob:
    if production_execution_authorized:
        raise PermissionError("production execution is not authorized in readiness stage")
    matches = [j for j in qualification_jobs() if j.job_id == job_id]
    if not matches:
        raise KeyError(f"job is not in qualification-only manifest: {job_id}")
    job = matches[0]
    if not job.evidence_status.startswith("QUALIFIED_COMPACT"):
        raise RuntimeError(f"{job_id} is {job.evidence_status}; configuration is portable but local launch is disabled")
    return job


__all__ = ["CapacityReceipt", "QualificationJob", "assert_qualification_launch",
           "capacity_preflight", "manifest_payload", "qualification_jobs"]
