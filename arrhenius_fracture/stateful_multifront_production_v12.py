"""Persistent, fail-closed current-source V12 production composition.

The context in this module is the only mutable lifecycle owner.  It does not
start a worker and it never substitutes an uncached mechanics calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .general_multifront_v12 import (
    FrontCandidateObservation, FrontRuntimeState, MultiFrontRuntimeState,
    ProcessEngineState, ProcessRegionReservoir, TopologyProposal, canonical_hash,
)
from .directional_competition_v11 import competition_state_from_dict
from .production_multifront_v12 import (
    AdaptedAcceptedState, DirectionalObservationBatch, ExactTrialDeltaV12,
    ProductionHooks, ProposalTrialOutcome, SolvedAcceptedState,
    accepted_fem_state_fingerprint, apply_exact_trial_delta,
)


SCHEMA = "v12.current-source-stateful-production-context/2"
BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


class StatefulProductionInterlock(RuntimeError):
    """Raised before an unqualified solve, stale trial, or mixed transaction."""


def event_endpoint_iteration_limit(tolerance_fraction: float) -> int:
    """Return a convergence-derived search budget without relaxing tolerance.

    The endpoint search contains safeguarded local expansion steps as well as
    bisection.  Forty iterations are insufficient for the production absolute
    time tolerance when those safeguards are exercised.  The extra 32 steps
    cover that bounded startup overhead; termination still requires the exact
    configured fraction and absolute-time tolerances below.
    """
    tolerance = float(tolerance_fraction)
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0:
        raise StatefulProductionInterlock(
            "event endpoint tolerance fraction must be finite and in (0, 1]"
        )
    return max(40, int(math.ceil(math.log2(1.0 / tolerance))) + 32)


def load_cached_provider_result_for_topology(
    cache_root: str | Path, topology_fingerprint: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Read one exact accepted cache entry; never fall through to evaluation."""
    root = Path(cache_root)
    matches = []
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("topology_fingerprint") == topology_fingerprint
            and manifest.get("accepted") is True
            and manifest.get("interpolation_permitted") is False
        ):
            matches.append((manifest_path, manifest))
    if len(matches) != 1:
        raise StatefulProductionInterlock(
            f"exact cache lookup requires one hit, found {len(matches)}"
        )
    manifest_path, manifest = matches[0]
    state_path = manifest_path.with_name("provider_state.pkl")
    data = state_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest.get("state_sha256"):
        raise StatefulProductionInterlock("cached provider state hash mismatch")
    result = pickle.loads(data)
    if result.get("topology_fingerprint") != topology_fingerprint:
        raise StatefulProductionInterlock("cached result topology mismatch")
    return result, {
        **manifest, "manifest_path": str(manifest_path.resolve()),
        "provider_state_path": str(state_path.resolve()),
    }


def _pickle_hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _topology_hash(state: Any) -> str:
    return hashlib.sha256(state.crack_network.to_json().encode()).hexdigest()


def _mesh_identity(state: Any) -> str:
    mesh = getattr(state, "mesh", None)
    if mesh is None:
        return canonical_hash({
            "mesh_discretization_status": "absent_in_source_level_fixture",
            "state_type": type(state).__name__,
        })
    digest = hashlib.sha256()
    for name in ("nodes", "elems", "area_e", "dNdx_e", "B_e"):
        value = getattr(mesh, name, None)
        if value is not None:
            array = np.ascontiguousarray(value)
            digest.update(name.encode()); digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def accepted_fem_state_identity(state: Any) -> str:
    return "accepted:" + accepted_fem_state_fingerprint(state)


def accepted_context_state_identity(
    state: Any, runtime: MultiFrontRuntimeState, *, physical_time_s: float,
    accepted_opening_m: float, step_count: int,
) -> str:
    return "accepted:" + canonical_hash({
        "fem_state_sha256": accepted_fem_state_fingerprint(state),
        "registry_fingerprint": runtime.registry_fingerprint,
        "physical_time_s": float(physical_time_s),
        "accepted_opening_m": float(accepted_opening_m),
        "step_count": int(step_count),
    })


def stress_field_identity(
    fem_state: Any, sigma_gp: Any, *, mechanics_source_identity: str,
    accepted_state_id: str | None = None,
) -> str:
    sigma = np.asarray(sigma_gp, dtype=float)
    if sigma.ndim != 2 or not sigma.size or not np.all(np.isfinite(sigma)):
        raise StatefulProductionInterlock("accepted sigma_gp is absent or invalid")
    return "stress:" + canonical_hash({
        "accepted_fem_state_id": (
            accepted_fem_state_identity(fem_state)
            if accepted_state_id is None else str(accepted_state_id)
        ),
        "sigma_gp_sha256": hashlib.sha256(np.ascontiguousarray(sigma).tobytes()).hexdigest(),
        "sigma_shape": list(sigma.shape), "sigma_dtype": str(sigma.dtype),
        "topology_fingerprint": _topology_hash(fem_state),
        "mesh_discretization_identity": _mesh_identity(fem_state),
        "mechanics_source_identity": str(mechanics_source_identity),
    })


def unavailable_stress_identity(
    accepted_state_id: str, mechanics_source_identity: str,
) -> str:
    return "stress:UNAVAILABLE:" + canonical_hash({
        "accepted_state_id": accepted_state_id,
        "mechanics_source_identity": mechanics_source_identity,
    })


@dataclass(frozen=True, order=True)
class AcceptedTrialKey:
    accepted_state_id: str
    stress_field_state_id: str
    topology_fingerprint: str
    proposal_id: str
    selected_front_id: str
    process_owner_id: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


class ExactAcceptedTrialCacheV12:
    """Own isolated exact trials for exactly one accepted-state generation."""

    def __init__(self) -> None:
        self._entries: dict[AcceptedTrialKey, ProposalTrialOutcome] = {}
        self.creation_count = 0
        self.selection_count = 0
        self.invalidation_count = 0
        self.destruction_count = 0
        self._selected_key: AcceptedTrialKey | None = None

    @staticmethod
    def key(runtime: MultiFrontRuntimeState, proposal: TopologyProposal) -> AcceptedTrialKey:
        return AcceptedTrialKey(
            runtime.accepted_state_id, runtime.stress_field_state_id,
            runtime.topology_fingerprint, proposal.proposal_id,
            proposal.front_id, proposal.owner_id,
        )

    def create(
        self, runtime: MultiFrontRuntimeState, proposal: TopologyProposal,
        outcome: ProposalTrialOutcome,
    ) -> AcceptedTrialKey:
        key = self.key(runtime, proposal)
        if outcome.proposal != proposal:
            raise StatefulProductionInterlock("trial outcome changed proposal identity")
        if key in self._entries:
            raise StatefulProductionInterlock("duplicate exact accepted-trial cache key")
        self._entries[key] = outcome
        self.creation_count += 1
        return key

    def select(
        self, runtime: MultiFrontRuntimeState, proposal: TopologyProposal,
    ) -> ProposalTrialOutcome:
        key = self.key(runtime, proposal)
        try:
            outcome = self._entries[key]
        except KeyError as exc:
            raise StatefulProductionInterlock(
                "selected proposal has no exact trial for the accepted state"
            ) from exc
        if not outcome.accepted:
            raise StatefulProductionInterlock("a rejected topology trial cannot be selected")
        self._selected_key = key
        self.selection_count += 1
        return outcome

    def invalidate_after_interval(self, accepted_state_id_after: str) -> None:
        stale = tuple(self._entries)
        self.invalidation_count += len(stale)
        self._entries.clear()
        self.destruction_count += len(stale)
        self._selected_key = None
        if any(key.accepted_state_id == accepted_state_id_after for key in stale):
            raise StatefulProductionInterlock("accepted-state rollover reused a stale identity")

    def require_empty(self) -> None:
        if self._entries:
            raise StatefulProductionInterlock("stale trial objects remain reachable")

    def audit(self) -> dict[str, Any]:
        return {
            "schema": "v12.exact-accepted-trial-cache/1",
            "creation_count": self.creation_count,
            "selection_count": self.selection_count,
            "invalidation_count": self.invalidation_count,
            "destruction_count": self.destruction_count,
            "live_entry_count": len(self._entries),
            "selected_key": None if self._selected_key is None else self._selected_key.to_dict(),
            "live_keys": [key.to_dict() for key in sorted(self._entries)],
            "live_content_sha256": {
                canonical_hash(key.to_dict()): _pickle_hash(value)
                for key, value in sorted(self._entries.items())
            },
        }


class TransactionalMultiFrontWriterV12:
    """Buffer accepted rows and publish an interval as one durable JSONL unit."""

    CATEGORIES = ("interval", "observation", "trial", "event", "owner", "ledger")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._accepted_buffers = {name: [] for name in self.CATEGORIES}
        self._readonly_trials: list[dict[str, Any]] = []
        self.transaction_id: str | None = None
        self.accepted_state_id: str | None = None
        self.topology_fingerprint: str | None = None
        self.flush_count = 0
        self.row_counts = {name: 0 for name in self.CATEGORIES}
        self.readonly_trial_count = 0

    def begin(self, transaction_id: str, *, accepted_state_id: str, topology_fingerprint: str) -> None:
        if self.transaction_id is not None:
            raise StatefulProductionInterlock("an output transaction is already open")
        self.transaction_id = str(transaction_id)
        self.accepted_state_id = str(accepted_state_id)
        self.topology_fingerprint = str(topology_fingerprint)

    def buffer(self, category: str, row: Mapping[str, Any], *, readonly_rejected_trial: bool = False) -> None:
        if self.transaction_id is None:
            raise StatefulProductionInterlock("output rows require an open transaction")
        payload = {
            **dict(row), "transaction_id": self.transaction_id,
            "accepted_state_id": self.accepted_state_id,
            "topology_fingerprint": self.topology_fingerprint,
        }
        if readonly_rejected_trial:
            if category != "trial" or bool(payload.get("accepted", False)):
                raise ValueError("only rejected trials may enter the read-only trial buffer")
            self._readonly_trials.append(payload)
            return
        if category not in self._accepted_buffers:
            raise ValueError(f"unsupported output row category: {category}")
        self._accepted_buffers[category].append(payload)

    @staticmethod
    def _atomic_append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        old = path.read_bytes() if path.exists() else b""
        addition = b"".join(
            (json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n").encode()
            for row in rows
        )
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(old + addition)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)

    def flush_accepted(self) -> dict[str, Any]:
        if self.transaction_id is None:
            raise StatefulProductionInterlock("no output transaction is open")
        self.root.mkdir(parents=True, exist_ok=True)
        published = {}
        for category in self.CATEGORIES:
            rows = self._accepted_buffers[category]
            if rows:
                path = self.root / f"{category}.jsonl"
                self._atomic_append(path, rows)
                published[category] = len(rows)
                self.row_counts[category] += len(rows)
        if self._readonly_trials:
            self._atomic_append(self.root / "rejected_trials_readonly.jsonl", self._readonly_trials)
            self.readonly_trial_count += len(self._readonly_trials)
            published["rejected_trials_readonly"] = len(self._readonly_trials)
        self.flush_count += 1
        self._clear()
        return {"published": published, "flush_count": self.flush_count}

    def rollback(self) -> None:
        # Rejected-trial diagnostics are explicitly separate; accepted rows are
        # discarded.  They can be flushed only by an explicit read-only publish.
        self._accepted_buffers = {name: [] for name in self.CATEGORIES}
        self.transaction_id = None
        self.accepted_state_id = None
        self.topology_fingerprint = None

    def _clear(self) -> None:
        self._accepted_buffers = {name: [] for name in self.CATEGORIES}
        self._readonly_trials = []
        self.transaction_id = None
        self.accepted_state_id = None
        self.topology_fingerprint = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "root": str(self.root.resolve()), "transaction_id": self.transaction_id,
            "accepted_state_id": self.accepted_state_id,
            "topology_fingerprint": self.topology_fingerprint,
            "accepted_buffers": copy.deepcopy(self._accepted_buffers),
            "readonly_trials": copy.deepcopy(self._readonly_trials),
            "flush_count": self.flush_count, "row_counts": dict(self.row_counts),
            "readonly_trial_count": self.readonly_trial_count,
        }

    @classmethod
    def restore(cls, value: Mapping[str, Any]) -> "TransactionalMultiFrontWriterV12":
        result = cls(value["root"])
        result.transaction_id = value.get("transaction_id")
        result.accepted_state_id = value.get("accepted_state_id")
        result.topology_fingerprint = value.get("topology_fingerprint")
        result._accepted_buffers = copy.deepcopy(value["accepted_buffers"])
        result._readonly_trials = copy.deepcopy(value["readonly_trials"])
        result.flush_count = int(value["flush_count"])
        result.row_counts = {str(k): int(v) for k, v in value["row_counts"].items()}
        result.readonly_trial_count = int(value["readonly_trial_count"])
        return result


class AtomicIntervalTransactionWriterV12:
    """Publish one complete interval directory behind one atomic pointer."""

    def __init__(self, root: str | Path, *, staging_parent: str | Path | None = None):
        self.root = Path(root)
        self.staging_parent = Path(staging_parent) if staging_parent is not None else self.root / ".staging"
        self.staging_path: Path | None = None
        self.transaction_id: str | None = None
        self.write_count = 0
        self.publication_count = 0

    def begin(self, transaction_id: str) -> None:
        if self.staging_path is not None:
            raise StatefulProductionInterlock("atomic interval transaction already open")
        self.transaction_id = str(transaction_id)
        self.staging_path = self.staging_parent / self.transaction_id
        if self.staging_path.exists():
            raise StatefulProductionInterlock("atomic staging transaction already exists")
        self.staging_path.mkdir(parents=True)

    def write_member(self, name: str, data: bytes, *, fail_after_write: int | None = None) -> None:
        if self.staging_path is None or "/" in name or name.startswith("."):
            raise StatefulProductionInterlock("invalid atomic transaction member")
        path = self.staging_path / name
        with path.open("wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        self.write_count += 1
        if fail_after_write is not None and self.write_count == int(fail_after_write):
            raise StatefulProductionInterlock("injected atomic staging failure")

    def stage_complete(
        self, *, rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
        runtime: MultiFrontRuntimeState, accepted_fem_state: Any,
        context_payload: Mapping[str, Any], fail_after_write: int | None = None,
        inject_after_outputs: bool = False,
    ) -> dict[str, Any]:
        if self.staging_path is None:
            raise StatefulProductionInterlock("atomic transaction has not begun")
        members: dict[str, bytes] = {}
        for category in TransactionalMultiFrontWriterV12.CATEGORIES:
            rows = tuple(rows_by_category.get(category, ()))
            members[f"{category}.jsonl"] = b"".join(
                (json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n").encode()
                for row in rows
            )
        fem = pickle.dumps(accepted_fem_state, protocol=5)
        runtime_bytes = (json.dumps(
            runtime.to_dict(), sort_keys=True, allow_nan=False
        ) + "\n").encode()
        context_bytes = (json.dumps(
            dict(context_payload), sort_keys=True, allow_nan=False
        ) + "\n").encode()
        members["accepted_fem.pkl"] = fem
        members["runtime.json"] = runtime_bytes
        members["context.json"] = context_bytes
        manifest = {
            "schema": "v12.atomic-accepted-interval-transaction/1",
            "transaction_id": self.transaction_id,
            "accepted_state_id": runtime.accepted_state_id,
            "stress_field_state_id": runtime.stress_field_state_id,
            "topology_fingerprint": runtime.topology_fingerprint,
            "members": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                for name, data in sorted(members.items())
            },
        }
        members["transaction_manifest.json"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        ordered_names = [
            f"{category}.jsonl"
            for category in TransactionalMultiFrontWriterV12.CATEGORIES
        ]
        for name in ordered_names:
            self.write_member(name, members[name], fail_after_write=fail_after_write)
        if inject_after_outputs:
            raise StatefulProductionInterlock(
                "injected output/checkpoint staging interruption"
            )
        for name in ("accepted_fem.pkl", "runtime.json", "context.json"):
            self.write_member(name, members[name], fail_after_write=fail_after_write)
        for name in ("transaction_manifest.json",):
            self.write_member(name, members[name], fail_after_write=fail_after_write)
        return manifest

    def publish(self, *, inject_before_marker: bool = False) -> Path:
        if self.staging_path is None or self.transaction_id is None:
            raise StatefulProductionInterlock("no staged interval to publish")
        if inject_before_marker:
            raise StatefulProductionInterlock("injected pre-publication failure")
        transactions = self.root / "transactions"
        transactions.mkdir(parents=True, exist_ok=True)
        destination = transactions / self.transaction_id
        os.replace(self.staging_path, destination)
        marker = destination / "COMMITTED"
        marker.write_text(self.transaction_id + "\n")
        with marker.open("rb") as stream:
            os.fsync(stream.fileno())
        pointer_tmp = self.root / "LATEST.tmp"
        pointer_tmp.write_text(self.transaction_id + "\n")
        with pointer_tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pointer_tmp, self.root / "LATEST")
        self.staging_path = None; self.transaction_id = None
        self.publication_count += 1
        return destination

    def discard(self) -> None:
        if self.staging_path is not None and self.staging_path.exists():
            import shutil
            shutil.rmtree(self.staging_path)
        self.staging_path = None; self.transaction_id = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "root": str(self.root.resolve()),
            "staging_parent": str(self.staging_parent.resolve()),
            "staging_path": None if self.staging_path is None else str(self.staging_path.resolve()),
            "transaction_id": self.transaction_id, "write_count": self.write_count,
            "publication_count": self.publication_count,
        }

    def read_latest_committed(self) -> tuple[Path, Mapping[str, Any]] | None:
        """Return only a fully committed transaction; ignore private staging."""
        pointer = self.root / "LATEST"
        if not pointer.is_file():
            return None
        transaction_id = pointer.read_text().strip()
        directory = self.root / "transactions" / transaction_id
        marker = directory / "COMMITTED"
        manifest_path = directory / "transaction_manifest.json"
        if not marker.is_file() or marker.read_text().strip() != transaction_id:
            raise StatefulProductionInterlock("latest transaction lacks its commit marker")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("transaction_id") != transaction_id:
            raise StatefulProductionInterlock("latest transaction manifest identity mismatch")
        for name, evidence in manifest.get("members", {}).items():
            data = (directory / name).read_bytes()
            if (
                len(data) != evidence["size_bytes"]
                or hashlib.sha256(data).hexdigest() != evidence["sha256"]
            ):
                raise StatefulProductionInterlock(
                    f"committed transaction member mismatch: {name}"
                )
        return directory, manifest


class CurrentSourceMultiFrontProductionContextV12:
    """Single explicit owner for accepted mechanics, stochastic state and I/O."""

    def __init__(
        self, *, args: Any, mechanical_configuration: Any,
        accepted_fem_state: Any, accepted_stress_field: Any,
        runtime: MultiFrontRuntimeState, provider_runtime: Any,
        destination_cache_root: str | Path, output_root: str | Path,
        checkpoint_path: str | Path, candidates: Sequence[Any],
        clusters: Mapping[str, Any] | None = None,
        physical_time_s: float = 0.0, accepted_opening_m: float = 0.0,
        step_count: int = 0, adapter_configuration: Mapping[str, Any] | None = None,
        stress_available: bool | None = None,
        mechanics_source_identity: str = "legacy_context_unspecified",
    ):
        if _topology_hash(accepted_fem_state) != runtime.topology_fingerprint:
            raise StatefulProductionInterlock("accepted FEM and runtime topology differ")
        self.args = copy.deepcopy(args)
        self.mechanical_configuration = copy.deepcopy(mechanical_configuration)
        self.accepted_fem_state = accepted_fem_state
        if accepted_stress_field is None:
            if stress_available is True:
                raise StatefulProductionInterlock(
                    "available stress requires an explicit sigma_gp array"
                )
            sigma = None
            self.stress_available = False
        else:
            sigma = np.asarray(accepted_stress_field, dtype=float).copy()
            if sigma.ndim != 2 or not sigma.size or not np.all(np.isfinite(sigma)):
                raise StatefulProductionInterlock("accepted sigma_gp is absent or invalid")
            if stress_available is False:
                raise StatefulProductionInterlock(
                    "unavailable stress must not carry a numerical sentinel array"
                )
            sigma.setflags(write=False)
            self.stress_available = True
        self.accepted_stress_field = sigma
        self.mechanics_source_identity = str(mechanics_source_identity)
        accepted_id = accepted_context_state_identity(
            accepted_fem_state, runtime, physical_time_s=physical_time_s,
            accepted_opening_m=accepted_opening_m, step_count=step_count,
        )
        stress_id = (
            stress_field_identity(
                accepted_fem_state, sigma,
                mechanics_source_identity=self.mechanics_source_identity,
                accepted_state_id=accepted_id,
            ) if self.stress_available else
            unavailable_stress_identity(accepted_id, self.mechanics_source_identity)
        )
        self.runtime = replace(
            runtime, accepted_state_id=accepted_id, stress_field_state_id=stress_id,
        )
        self.provider_runtime = provider_runtime
        self.destination_cache_root = Path(destination_cache_root)
        self.candidates = tuple(candidates)
        self.clusters = dict(clusters or {})
        self.physical_time_s = float(physical_time_s)
        self.accepted_opening_m = float(accepted_opening_m)
        self.step_count = int(step_count)
        self.adapter_configuration = dict(adapter_configuration or {})
        self.actual_engines = {}
        self.previewed_clock_states: dict[str, Any] = {}
        self.trial_cache = ExactAcceptedTrialCacheV12()
        self.output_writer = TransactionalMultiFrontWriterV12(output_root)
        self.atomic_writer = AtomicIntervalTransactionWriterV12(output_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.latest_durable_checkpoint: Mapping[str, Any] | None = None
        self.lifecycle_records: list[dict[str, Any]] = []
        self.interval_info_by_owner: dict[str, dict[str, Any]] = {}
        self.provider_lookup_count = 0
        self.provider_solve_count = 0
        self.mechanics_solve_count = 0
        self.pf_workers_started = 0
        self.transaction_phase = "accepted"
        self.provisional_solved_state: SolvedAcceptedState | None = None
        self.provisional_runtime: MultiFrontRuntimeState | None = None
        self.provisional_observation_batch: DirectionalObservationBatch | None = None
        self._assert_identity()

    @property
    def accepted_state_id(self) -> str:
        return self.runtime.accepted_state_id

    @property
    def stress_field_state_id(self) -> str:
        return self.runtime.stress_field_state_id

    def _assert_identity(self) -> None:
        if _topology_hash(self.accepted_fem_state) != self.runtime.topology_fingerprint:
            raise StatefulProductionInterlock("context accepted topology identity mismatch")
        if not self.accepted_state_id or not self.stress_field_state_id:
            raise StatefulProductionInterlock("context state identities must be explicit")
        expected_accepted = accepted_context_state_identity(
            self.accepted_fem_state, self.runtime,
            physical_time_s=self.physical_time_s,
            accepted_opening_m=self.accepted_opening_m,
            step_count=self.step_count,
        )
        if self.accepted_state_id != expected_accepted:
            raise StatefulProductionInterlock("accepted FEM state identity is stale")
        if self.stress_available:
            if self.accepted_stress_field is None:
                raise StatefulProductionInterlock("available stress has no sigma_gp array")
            expected_stress = stress_field_identity(
                self.accepted_fem_state, self.accepted_stress_field,
                mechanics_source_identity=self.mechanics_source_identity,
                accepted_state_id=self.accepted_state_id,
            )
            if self.stress_field_state_id != expected_stress:
                raise StatefulProductionInterlock("accepted stress identity is stale")
        elif self.accepted_stress_field is not None:
            raise StatefulProductionInterlock(
                "unavailable stress carries a forbidden numerical sentinel"
            )

    def require_usable_stress(self) -> None:
        self._assert_identity()
        if not self.stress_available:
            raise StatefulProductionInterlock("accepted stress is explicitly unavailable")

    @property
    def fingerprint(self) -> str:
        self._assert_identity()
        return canonical_hash({
            "schema": SCHEMA,
            "immutable_args_sha256": _pickle_hash(self.args),
            "mechanical_configuration_sha256": _pickle_hash(self.mechanical_configuration),
            "accepted_fem_state_sha256": accepted_fem_state_fingerprint(self.accepted_fem_state),
            "accepted_stress_sha256": _pickle_hash(self.accepted_stress_field),
            "stress_available": self.stress_available,
            "mechanics_source_identity": self.mechanics_source_identity,
            "runtime_sha256": canonical_hash(self.runtime.to_dict()),
            "provider_runtime_sha256": _pickle_hash(self.provider_runtime),
            "destination_cache_root": str(self.destination_cache_root.resolve()),
            "clusters_sha256": _pickle_hash(self.clusters),
            "preview_sha256": _pickle_hash(self.previewed_clock_states),
            "actual_physical_engines_sha256": {
                key: _pickle_hash(value) for key, value in sorted(self.actual_engines.items())
            },
            "interval_info_sha256": _pickle_hash(self.interval_info_by_owner),
            "trial_cache": self.trial_cache.audit(),
            "writer": self.output_writer.snapshot(),
            "atomic_writer": self.atomic_writer.snapshot(),
            "physical_time_s": self.physical_time_s,
            "accepted_opening_m": self.accepted_opening_m,
            "step_count": self.step_count,
            "transaction_phase": self.transaction_phase,
            "provider_lookup_count": self.provider_lookup_count,
            "provider_solve_count": self.provider_solve_count,
            "mechanics_solve_count": self.mechanics_solve_count,
            "pf_workers_started": self.pf_workers_started,
            "checkpoint_publication_sha256": _pickle_hash(self.latest_durable_checkpoint),
            "provisional_solved_state_sha256": _pickle_hash(self.provisional_solved_state),
            "provisional_runtime_sha256": (
                None if self.provisional_runtime is None
                else canonical_hash(self.provisional_runtime.to_dict())
            ),
            "provisional_observation_batch_sha256": _pickle_hash(
                self.provisional_observation_batch
            ),
        })

    def invoke(self, hook_name: str, operation: Callable[[], Any], *, pure: bool = False) -> Any:
        before = self.fingerprint
        result = operation()
        after = self.fingerprint
        if pure and after != before:
            raise StatefulProductionInterlock(f"pure hook mutated context: {hook_name}")
        self.lifecycle_records.append({
            "sequence": len(self.lifecycle_records) + 1, "hook": hook_name,
            "context_sha256_before": before, "context_sha256_after": after,
            "pure": pure,
        })
        return result

    def begin_interval(self, interval_id: str) -> None:
        self.output_writer.begin(
            interval_id, accepted_state_id=self.accepted_state_id,
            topology_fingerprint=self.runtime.topology_fingerprint,
        )
        self.transaction_phase = "trial"

    def commit_rollover(
        self, runtime: MultiFrontRuntimeState, accepted_fem_state: Any,
        accepted_stress_field: Any, *, mechanics_source_identity: str | None = None,
    ) -> None:
        prior = self.accepted_state_id
        self.accepted_fem_state = accepted_fem_state
        if accepted_stress_field is None:
            sigma = None
            self.stress_available = False
        else:
            sigma = np.asarray(accepted_stress_field, dtype=float).copy()
            if sigma.ndim != 2 or not sigma.size or not np.all(np.isfinite(sigma)):
                raise StatefulProductionInterlock("accepted sigma_gp is absent or invalid")
            sigma.setflags(write=False)
            self.stress_available = True
        self.accepted_stress_field = sigma
        if mechanics_source_identity is not None:
            self.mechanics_source_identity = str(mechanics_source_identity)
        next_step = self.step_count + 1
        accepted_id = accepted_context_state_identity(
            accepted_fem_state, runtime,
            physical_time_s=self.physical_time_s,
            accepted_opening_m=self.accepted_opening_m, step_count=next_step,
        )
        stress_id = (
            stress_field_identity(
                accepted_fem_state, sigma,
                mechanics_source_identity=self.mechanics_source_identity,
                accepted_state_id=accepted_id,
            ) if self.stress_available else
            unavailable_stress_identity(accepted_id, self.mechanics_source_identity)
        )
        self.runtime = replace(
            runtime, accepted_state_id=accepted_id, stress_field_state_id=stress_id,
        )
        if self.runtime.accepted_state_id == prior:
            raise StatefulProductionInterlock("accepted interval did not roll state identity")
        self.trial_cache.invalidate_after_interval(self.runtime.accepted_state_id)
        self.trial_cache.require_empty()
        self.step_count = next_step
        self.transaction_phase = "renewed"
        self._assert_identity()

    def flush_before_checkpoint(self) -> dict[str, Any]:
        if self.transaction_phase != "renewed":
            raise StatefulProductionInterlock("output flush requires a renewed accepted state")
        record = self.output_writer.flush_accepted()
        self.transaction_phase = "output_flushed"
        return record

    def publish_checkpoint(self) -> dict[str, Any]:
        if self.transaction_phase != "output_flushed":
            raise StatefulProductionInterlock("checkpoint pointer requires flushed outputs")
        from .multifront_checkpoint_v12 import write_accepted_boundary_checkpoint_v12
        restart = {
            "args": self.args,
            "mechanical_configuration": self.mechanical_configuration,
            "provider_runtime": self.provider_runtime,
            "candidates": self.candidates,
            "destination_cache_root": str(self.destination_cache_root.resolve()),
            "clusters": self.clusters,
            "writer": self.output_writer.snapshot(),
            "adapter_configuration": self.adapter_configuration,
            "solve_counters": {
                "provider_lookup_count": self.provider_lookup_count,
                "provider_solve_count": self.provider_solve_count,
                "mechanics_solve_count": self.mechanics_solve_count,
                "pf_workers_started": self.pf_workers_started,
            },
        }
        record = write_accepted_boundary_checkpoint_v12(
            self.runtime, self.checkpoint_path,
            accepted_fem_state=self.accepted_fem_state,
            accepted_stress_field=self.accepted_stress_field,
            mechanics_source_identity=self.mechanics_source_identity,
            physical_time_s=self.physical_time_s,
            accepted_opening_m=self.accepted_opening_m,
            step_count=self.step_count, context_restart=restart,
        )
        self.latest_durable_checkpoint = record
        self.transaction_phase = "accepted"
        return record

    def checkpoint_payload(self, production_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "boundary": BOUNDARY,
            "runtime_sha256": canonical_hash(self.runtime.to_dict()),
            "accepted_state_id": self.accepted_state_id,
            "stress_field_state_id": self.stress_field_state_id,
            "stress_available": self.stress_available,
            "mechanics_source_identity": self.mechanics_source_identity,
            "accepted_stress_field_pickle_b64": __import__("base64").b64encode(
                pickle.dumps(self.accepted_stress_field, protocol=5)
            ).decode("ascii"),
            "accepted_stress_field_sha256": _pickle_hash(self.accepted_stress_field),
            "immutable_args_pickle_b64": __import__("base64").b64encode(
                pickle.dumps(self.args, protocol=5)
            ).decode("ascii"),
            "mechanical_configuration_pickle_b64": __import__("base64").b64encode(
                pickle.dumps(self.mechanical_configuration, protocol=5)
            ).decode("ascii"),
            "provider_runtime_pickle_b64": __import__("base64").b64encode(
                pickle.dumps(self.provider_runtime, protocol=5)
            ).decode("ascii"),
            "candidates_pickle_b64": __import__("base64").b64encode(
                pickle.dumps(self.candidates, protocol=5)
            ).decode("ascii"),
            "destination_cache_root": str(self.destination_cache_root.resolve()),
            "clusters_pickle_b64": __import__("base64").b64encode(
                pickle.dumps(self.clusters, protocol=5)
            ).decode("ascii"),
            "writer": self.output_writer.snapshot(),
            "physical_time_s": self.physical_time_s,
            "accepted_opening_m": self.accepted_opening_m,
            "step_count": self.step_count,
            "transaction_phase": (
                "accepted" if production_record is not None else self.transaction_phase
            ),
            "adapter_configuration": self.adapter_configuration,
            "production_checkpoint": dict(production_record or {}),
            "solve_counters": {
                "provider_lookup_count": self.provider_lookup_count,
                "provider_solve_count": self.provider_solve_count,
                "mechanics_solve_count": self.mechanics_solve_count,
                "pf_workers_started": self.pf_workers_started,
            },
        }


def build_arbitrary_region_request_v12(
    solved: SolvedAcceptedState, runtime: MultiFrontRuntimeState,
    context: CurrentSourceMultiFrontProductionContextV12,
    *, active_front_ids: Sequence[str] | None = None,
):
    """Construct the exact V11 request with a front-count-independent frame."""
    from .sharp_front_v11_branching import _request
    cfg = context.adapter_configuration
    if "cfg" not in cfg:
        raise StatefulProductionInterlock("exact request requires immutable mechanical cfg")
    request = _request(
        solved.fem_state, context.candidates, args=context.args, cfg=cfg["cfg"],
        runtime_step=context.step_count, cluster=None,
    )
    requested_fronts = tuple(sorted(
        runtime.active_front_ids if active_front_ids is None else active_front_ids
    ))
    if not set(requested_fronts).issubset(runtime.active_front_ids):
        raise StatefulProductionInterlock(
            "arbitrary-region request contains a front outside the accepted runtime"
        )
    candidate_by_id = {item.candidate_id: item for item in context.candidates}
    active_candidates_by_tip = None
    if candidate_by_id:
        active_candidates_by_tip = {
            front_id: tuple(
                candidate_by_id[candidate_id]
                for candidate_id in runtime.front_runtimes[
                    front_id
                ].mechanically_active_candidate_ids
            )
            for front_id in requested_fronts
        }
        if any(not values for values in active_candidates_by_tip.values()):
            raise StatefulProductionInterlock(
                "every active front requires at least one mechanically active candidate"
            )
    frame_by_tip = {}
    process_region_frame_by_owner = {}
    for owner_id, region in runtime.process_regions.items():
        requested_members = sorted(
            set(region.member_front_ids).intersection(requested_fronts)
        )
        if not requested_members:
            continue
        owner_frame = {
            "owner_id": owner_id,
            "member_front_ids": requested_members,
            "unresolved_junction_ids": sorted(region.unresolved_junction_ids),
            "local_process_coordinate_m": region.cumulative_process_advance_m,
            "frame_kind": (
                "unresolved_shared_region" if region.unresolved_junction_ids
                else "independent_region"
            ),
        }
        process_region_frame_by_owner[owner_id] = owner_frame
        for front_id in requested_members:
            frame_by_tip[front_id] = owner_frame
    if set(frame_by_tip) != set(requested_fronts):
        raise StatefulProductionInterlock("arbitrary-region request omitted an active front")
    return replace(
        request,
        **(
            {} if active_candidates_by_tip is None
            else {"candidates_by_tip": active_candidates_by_tip}
        ),
        cluster_frame={
            "mode": "arbitrary_process_regions",
            "process_region_frame_by_owner": process_region_frame_by_owner,
            "frame_by_tip": frame_by_tip,
        },
    )


def restore_stateful_production_context(
    checkpoint_path: str | Path,
) -> CurrentSourceMultiFrontProductionContextV12:
    """Destroy/recreate-safe restore with no implicit closure reconstruction."""
    from .multifront_checkpoint_v12 import load_accepted_boundary_checkpoint_v12

    target = Path(checkpoint_path)
    restored = load_accepted_boundary_checkpoint_v12(target)
    payload = restored.context_restart
    result = CurrentSourceMultiFrontProductionContextV12(
        args=payload["args"],
        mechanical_configuration=payload["mechanical_configuration"],
        accepted_fem_state=restored.accepted_fem_state,
        accepted_stress_field=restored.accepted_stress_field,
        runtime=restored.runtime,
        provider_runtime=payload["provider_runtime"],
        destination_cache_root=payload["destination_cache_root"],
        output_root=payload["writer"]["root"], checkpoint_path=target,
        candidates=payload["candidates"], clusters=payload["clusters"],
        physical_time_s=restored.physical_time_s,
        accepted_opening_m=restored.accepted_opening_m,
        step_count=restored.step_count,
        adapter_configuration=payload.get("adapter_configuration", {}),
        stress_available=True,
        mechanics_source_identity=restored.mechanics_source_identity,
    )
    result.output_writer = TransactionalMultiFrontWriterV12.restore(payload["writer"])
    counters = payload["solve_counters"]
    result.provider_lookup_count = int(counters["provider_lookup_count"])
    result.provider_solve_count = int(counters["provider_solve_count"])
    result.mechanics_solve_count = int(counters["mechanics_solve_count"])
    result.pf_workers_started = int(counters["pf_workers_started"])
    result.latest_durable_checkpoint = {
        "schema": "v12.accepted-boundary-checkpoint/1",
        "path": str(target.resolve()),
        "payload_sha256": restored.payload_sha256,
    }
    from .current_source_multifront_hooks_v12 import (
        CurrentSourceHookBindingError, restore_complete_current_source_engine,
    )
    for engine_id, engine in result.runtime.process_engines.items():
        if engine.checkpoint_payload_b64 is None:
            continue
        try:
            result.actual_engines[engine_id] = restore_complete_current_source_engine(engine)
        except CurrentSourceHookBindingError:
            # Source-only test doubles stay represented by their complete
            # immutable payload.  The production class allow-list remains
            # fail-closed when an evolution hook requests restoration.
            continue
    result.transaction_phase = "accepted"
    return result


def _front_observations(
    context: CurrentSourceMultiFrontProductionContextV12,
    provider_result: Mapping[str, Any], runtime: MultiFrontRuntimeState,
):
    from .tip_directional_observation_v11 import (
        observations_from_provider_by_front_candidate,
    )
    tips = provider_result.get("tips", provider_result.get("directional_tips", ()))
    source = observations_from_provider_by_front_candidate(
        runtime.crack_network, tips, {
            front_id: front.mechanically_active_candidate_ids
            for front_id, front in runtime.front_runtimes.items()
        },
        accepted_state_id=runtime.accepted_state_id,
        stress_field_state_id=runtime.stress_field_state_id,
        topology_fingerprint=runtime.topology_fingerprint,
    )
    candidate_objects = {item.candidate_id: item for item in context.candidates}
    rows = []
    rates: dict[str, dict[str, float]] = {
        front_id: {} for front_id in runtime.active_front_ids
    }
    endpoints: dict[str, dict[str, tuple[float, float]]] = {
        front_id: {} for front_id in runtime.active_front_ids
    }
    mechanics = []
    da = float(context.adapter_configuration.get("da_phys_m", 0.0))
    if da <= 0.0:
        raise StatefulProductionInterlock("directional endpoint construction requires da_phys_m")
    for item in source:
        owner = runtime.owner_by_front[item.tip_id]
        competition = competition_state_from_dict(
            runtime.front_runtimes[item.tip_id].competition_state
        )
        hazard = {row.candidate_id: row for row in competition.hazard_states}[item.candidate_id]
        if not item.local_contour_valid:
            marginal_evaluator = context.adapter_configuration.get(
                "marginal_kinetic_evaluator"
            )
            if not callable(marginal_evaluator):
                raise StatefulProductionInterlock(
                    "invalid local contour requires a candidate-marginal kinetic evaluator"
                )
            marginal = float(marginal_evaluator(
                item.tip_id, candidate_objects[item.candidate_id], runtime, context,
            ))
            kinetic = max(marginal, 0.0)
            Eprime = float(context.provisional_solved_state.fem_state.material.Eprime)
            item = item.with_kinetics(
                kinetic_J_J_per_m2=kinetic,
                marginal_J_J_per_m2=marginal,
                directional_K_Pa_sqrt_m=(kinetic * Eprime) ** 0.5,
                rate_per_s=0.0,
            )
        rate_adapter = context.adapter_configuration.get("directional_rate_adapter")
        rate = (
            float(rate_adapter(item, candidate_objects[item.candidate_id], context))
            if callable(rate_adapter) else float(hazard.previous_rate_per_s or 0.0)
        )
        candidate = candidate_objects[item.candidate_id]
        endpoints[item.tip_id][item.candidate_id] = (
            item.tip_xy_m[0] + da * float(candidate.direction_xy[0]),
            item.tip_xy_m[1] + da * float(candidate.direction_xy[1]),
        )
        rates[item.tip_id][item.candidate_id] = rate
        rows.append(FrontCandidateObservation(
            accepted_state_id=item.accepted_state_id,
            stress_field_state_id=item.stress_field_state_id,
            front_id=item.tip_id, owner_id=owner, candidate_id=item.candidate_id,
            tip_coordinates_m=item.tip_xy_m,
            signed_local_J_J_per_m2=item.signed_local_J_J_per_m2,
            marginal_G_J_per_m2=float(item.marginal_J_J_per_m2 or 0.0),
            kinetic_J_used_J_per_m2=item.kinetic_J_J_per_m2,
            directional_K_MPa_sqrt_m=item.directional_K_Pa_sqrt_m / 1.0e6,
            directional_rate_per_s=rate,
            tensor=(0.0,), tensor_reliability="accepted_tensor_field_explicit",
            controlling_scalar_K_tip_id=item.tip_id, tensor_probe_tip_id=item.tip_id,
        ))
        mechanics.append({
            "front_id": item.tip_id, "owner_id": owner,
            "candidate_id": item.candidate_id,
            "candidate": candidate.__dict__,
            "signed_local_J_J_per_m2": item.signed_local_J_J_per_m2,
            "local_J_valid": item.local_contour_valid,
            "local_J_invalid_reason": item.local_contour_invalid_reason,
            "marginal_G_J_per_m2": item.marginal_J_J_per_m2,
            "kinetic_J_J_per_m2": item.kinetic_J_J_per_m2,
            "kinetic_K_Pa_sqrt_m": item.directional_K_Pa_sqrt_m,
            "directional_rate_per_s": rate,
            "candidate_endpoint_m": list(endpoints[item.tip_id][item.candidate_id]),
            "tensor_reliability": "accepted_tensor_field_explicit",
            "accepted_state_id": item.accepted_state_id,
            "stress_field_state_id": item.stress_field_state_id,
        })
    batch = DirectionalObservationBatch(
        tuple(rows), rates, endpoints, tuple(mechanics),
    )
    batch.validate(runtime)
    return batch


def build_stateful_production_hooks(
    context: CurrentSourceMultiFrontProductionContextV12,
) -> ProductionHooks:
    """Bind every reviewed leaf to closures over exactly one context."""
    cfg = context.adapter_configuration

    def adapt(state, candidates_by_front):
        from .adaptive_multitip_mesh_v11 import adapt_accepted_state_for_trials
        def operation():
            pre_adapted_sha256 = cfg.get("accepted_pre_adapted_source_sha256")
            if (
                pre_adapted_sha256 is not None
                and accepted_fem_state_fingerprint(state) == pre_adapted_sha256
            ):
                return AdaptedAcceptedState(state, {
                    "schema": "v12.pre_adapted-source-state/1",
                    "accepted_fem_state_sha256": pre_adapted_sha256,
                    "additional_refinement_performed": False,
                })
            cached_adapter = cfg.get("accepted_mesh_adapter")
            if callable(cached_adapter):
                raw = cached_adapter(state, candidates_by_front, context)
                if isinstance(raw, AdaptedAcceptedState):
                    return raw
                if not isinstance(raw, tuple) or len(raw) != 2:
                    raise StatefulProductionInterlock("accepted mesh adapter returned an invalid result")
                return AdaptedAcceptedState(raw[0], raw[1])
            required = (
                "da_phys_m", "tip_h_fine_m", "contour_radius_m",
                "crack_band_radius_m",
            )
            missing = [name for name in required if name not in cfg]
            if missing:
                raise StatefulProductionInterlock(f"mesh adapter configuration missing {missing}")
            candidate_objects = {
                item.candidate_id: item for item in context.candidates
            }
            physical_inventory = {
                front_id: tuple(candidate_objects[candidate_id] for candidate_id in ids)
                for front_id, ids in candidates_by_front.items()
            }
            raw = adapt_accepted_state_for_trials(
                state, physical_inventory,
                da_phys_m=cfg["da_phys_m"], tip_h_fine_m=cfg["tip_h_fine_m"],
                contour_radius_m=cfg["contour_radius_m"],
                crack_band_radius_m=cfg["crack_band_radius_m"],
                accepted_load_m=context.accepted_opening_m,
                starting_generation=int(cfg.get("starting_generation", 0)),
                starting_operation_index=int(cfg.get("starting_operation_index", 0)),
            )
            return AdaptedAcceptedState(raw[0], raw[1])
        return context.invoke("accepted_mesh_adaptation", operation, pure=True)

    def solve(adapted, trial_fraction: float | None = None):
        from .sharp_front_v11_branching import solve_accepted_state_v12_hook
        if not isinstance(adapted, AdaptedAcceptedState):
            raise StatefulProductionInterlock("solve hook requires AdaptedAcceptedState")
        fraction = float(
            cfg.get("trial_fraction", 1.0)
            if trial_fraction is None else trial_fraction
        )
        if not 0.0 < fraction <= 1.0:
            raise StatefulProductionInterlock("accepted mechanics fraction must lie in (0, 1]")
        def operation():
            fractional_loader = cfg.get("accepted_mechanics_loader_fractional")
            if callable(fractional_loader):
                value = fractional_loader(adapted, fraction, context)
                if not isinstance(value, SolvedAcceptedState):
                    raise StatefulProductionInterlock(
                        "fractional mechanics loader must return SolvedAcceptedState"
                    )
                return value
            cached_solver = cfg.get("accepted_mechanics_loader")
            if callable(cached_solver):
                if fraction != 1.0 and not cfg.get(
                    "fractional_mechanics_invariant_fixture", False
                ):
                    raise StatefulProductionInterlock(
                        "event-end alignment requires a fractional accepted mechanics loader"
                    )
                value = cached_solver(adapted, context)
                if not isinstance(value, SolvedAcceptedState):
                    raise StatefulProductionInterlock("mechanics loader must return SolvedAcceptedState")
                if fraction != 1.0:
                    value = replace(
                        value,
                        accepted_load_m=context.accepted_opening_m + (
                            value.accepted_load_m - context.accepted_opening_m
                        ) * fraction,
                        physical_time_s=context.physical_time_s + (
                            value.physical_time_s - context.physical_time_s
                        ) * fraction,
                    )
                return value
            needed = ("cfg", "base", "material", "elasticity_D")
            missing = [name for name in needed if name not in cfg]
            if missing:
                raise StatefulProductionInterlock(f"accepted solve configuration missing {missing}")
            context.mechanics_solve_count += 1
            from .production_step_loop_v11 import AcceptedStepContext
            step_context = cfg.get("step_context")
            requested_duration = float(
                cfg.get("requested_interval_duration_s", getattr(context.args, "dt", 0.0))
            )
            if requested_duration <= 0.0:
                requested_duration = float(
                    context.args.get("duration_s", 0.0)
                    if isinstance(context.args, Mapping) else 0.0
                )
            if requested_duration <= 0.0:
                raise StatefulProductionInterlock(
                    "production fractional solve requires requested interval duration"
                )
            step_context = AcceptedStepContext(
                context.step_count + 1, context.physical_time_s,
                requested_duration * fraction,
                canonical_hash({
                    "accepted_state_id": context.accepted_state_id,
                    "step": context.step_count + 1, "fraction": fraction,
                }),
            )
            raw = solve_accepted_state_v12_hook(
                adapted.fem_state, step_context, accepted_load=context.accepted_opening_m,
                trial_fraction=fraction,
                args=context.args, cfg=cfg["cfg"], base=cfg["base"],
                material=cfg["material"], elasticity_D=cfg["elasticity_D"],
                candidates=context.candidates,
            )
            measurement = dict(raw[2])
            measurement["accepted_fem_state_sha256"] = (
                accepted_fem_state_fingerprint(raw[0])
            )
            return SolvedAcceptedState(
                raw[0], raw[1], measurement,
                context.accepted_opening_m + float(context.args.dU) * fraction,
                context.physical_time_s + requested_duration * fraction,
                str(cfg.get("mechanics_source_identity", "production_solve")),
            )
        return context.invoke("accepted_fem_solve", operation)

    def request(solved, runtime):
        if not isinstance(solved, SolvedAcceptedState):
            raise StatefulProductionInterlock("request hook requires SolvedAcceptedState")
        request_builder = cfg.get(
            "exact_request_builder", build_arbitrary_region_request_v12
        )
        return context.invoke(
            "exact_request", lambda: request_builder(solved, runtime, context), pure=True,
        )

    def observations(provider_result, runtime):
        return context.invoke(
            "directional_observations",
            lambda: _front_observations(context, provider_result, runtime),
        )

    def evolve(state, observation, duration):
        from .sharp_front_v11_branching import evolve_process_engine_v12_hook
        configured = cfg.get("process_engine_evolver")
        if callable(configured):
            return context.invoke(
                "process_interval_evolution",
                lambda: configured(state, observation, duration, context),
            )
        from .tip_directional_observation_v11 import TipDirectionalObservation
        engine = context.actual_engines.get(state.engine_id)
        if engine is None:
            from .current_source_multifront_hooks_v12 import restore_complete_current_source_engine
            engine = restore_complete_current_source_engine(state)
        tip_observation = TipDirectionalObservation(
            tip_id=observation.front_id, parent_branch_id=None,
            candidate_id=observation.candidate_id,
            tip_xy_m=observation.tip_coordinates_m, branch_arclength_m=0.0,
            projected_reach_m=0.0,
            signed_local_J_J_per_m2=observation.signed_local_J_J_per_m2,
            kinetic_J_J_per_m2=observation.kinetic_J_used_J_per_m2,
            marginal_J_J_per_m2=observation.marginal_G_J_per_m2,
            directional_K_Pa_sqrt_m=observation.directional_K_MPa_sqrt_m * 1.0e6,
            directional_rate_per_s=observation.directional_rate_per_s,
            local_contour_valid=True, local_contour_invalid_reason=None,
            accepted_state_id=observation.accepted_state_id,
            stress_field_state_id=observation.stress_field_state_id,
            pre_event_topology_fingerprint=_topology_hash(
                context.provisional_solved_state.fem_state
                if context.provisional_solved_state is not None
                else context.accepted_fem_state
            ),
        )
        physical_coordinate = float(getattr(engine.mpz, "advance_total_m", 0.0))
        if not np.isclose(
            physical_coordinate, state.local_process_coordinate_m,
            rtol=0.0, atol=1.0e-15,
        ):
            raise StatefulProductionInterlock(
                "owner-local V12 coordinate differs from restored physical engine"
            )
        solved = context.provisional_solved_state
        if solved is None:
            solved = SolvedAcceptedState(
                context.accepted_fem_state, context.accepted_stress_field,
                {"accepted_fem_state_sha256": accepted_fem_state_fingerprint(
                    context.accepted_fem_state
                )}, context.accepted_opening_m, context.physical_time_s,
                context.mechanics_source_identity,
            )
        if solved.sigma_gp is None:
            raise StatefulProductionInterlock("provisional accepted stress is unavailable")
        provisional_sigma = np.asarray(solved.sigma_gp, dtype=float)
        if (
            provisional_sigma.ndim != 2 or not provisional_sigma.size
            or not np.all(np.isfinite(provisional_sigma))
        ):
            raise StatefulProductionInterlock("provisional accepted stress is unavailable")
        def operation():
            updated, info = evolve_process_engine_v12_hook(
                engine, tip_observation, solved.fem_state,
                solved.sigma_gp,
                accepted_state_id=observation.accepted_state_id,
                stress_field_state_id=observation.stress_field_state_id,
                pre_event_topology_fingerprint=_topology_hash(solved.fem_state),
                duration_s=duration, temperature_K=float(cfg["temperature_K"]),
                pre_progress=state.local_process_coordinate_m,
                permitted_physical_hazard_action=float(
                    cfg.get("permitted_physical_hazard_action", 0.075)
                ),
            )
            context.actual_engines[state.engine_id] = updated
            context.interval_info_by_owner[state.engine_id] = info
            from .sharp_front_v11_branching import _capture_shared_engine
            return ProcessEngineState.from_v11_payload(
                engine_id=state.engine_id, source_state_id=state.source_state_id,
                payload=_capture_shared_engine(updated),
                active_ledgers=state.active_ledgers, wake_ledgers=state.wake_ledgers,
                signed_system_ledgers=state.signed_system_ledgers,
                update_count=state.update_count + 1,
                event_renewal_count=state.event_renewal_count,
                local_process_coordinate_m=state.local_process_coordinate_m,
                family_identity=state.family_identity,
            )
        return context.invoke("process_interval_evolution", operation)

    def advance_clock(front: FrontRuntimeState, _observations, _duration):
        return replace(front, interval_count=front.interval_count + 1)

    def trial(state, proposal):
        executor = cfg.get("topology_trial_executor")
        def operation():
            outcome = (
                executor(state, proposal)
                if callable(executor) else
                execute_selected_topology_trial_v12(state, proposal, context)
            )
            trial_runtime = context.provisional_runtime
            if trial_runtime is None:
                raise StatefulProductionInterlock(
                    "topology trial lacks provisional runtime ownership"
                )
            context.trial_cache.create(trial_runtime, proposal, outcome)
            return outcome
        return context.invoke("topology_trial", operation)

    def preview(*args, **kwargs):
        from .multifront_competition_v12 import preview_competitions
        def operation():
            value = preview_competitions(*args, **kwargs)
            context.previewed_clock_states[value.accepted_registry_sha256] = value
            return value
        return context.invoke("competition_preview", operation)

    def finalize(*args, **kwargs):
        from .multifront_competition_v12 import finalize_competitions
        return context.invoke(
            "competition_finalization", lambda: finalize_competitions(*args, **kwargs)
        )

    def renew(engine_id: str, *, selected: bool, distance_m: float):
        from .tip_directional_observation_v11 import apply_post_interval_event_renewal
        configured = cfg.get("event_renewer")
        if callable(configured):
            return context.invoke(
                "event_renewal",
                lambda: configured(engine_id, selected, distance_m, context),
            )
        def operation():
            engine = context.actual_engines[engine_id]
            info = context.interval_info_by_owner[engine_id]
            return apply_post_interval_event_renewal(
                engine.mpz, info, event_selected=selected,
                event_distance_m=float(distance_m),
            )
        return context.invoke("event_renewal", operation)

    def validate(info, duration):
        from .process_update_semantics_v11 import require_full_accepted_interval_consumption
        return context.invoke(
            "process_interval_validation",
            lambda: require_full_accepted_interval_consumption(info, duration), pure=True,
        )

    def write_output(runtime, interval):
        def operation():
            context.output_writer.buffer("interval", {
                "selected_proposal_id": (
                    None if interval.selected_proposal is None
                    else interval.selected_proposal.proposal_id
                ),
                "active_front_ids": list(runtime.active_front_ids),
            })
        return context.invoke("output_buffer", operation)

    def write_checkpoint(runtime, accepted_state):
        if runtime is not context.runtime or accepted_state is not context.accepted_fem_state:
            raise StatefulProductionInterlock("checkpoint must publish the context accepted objects")
        return context.invoke("checkpoint", context.publish_checkpoint)

    return ProductionHooks(
        adapt_accepted_mesh=adapt, solve_accepted_fem=solve,
        build_exact_topology_request=request,
        extract_directional_observations=observations,
        evolve_process_engine=evolve, advance_front_clock=advance_clock,
        trial_topology_proposal=trial, write_output=write_output,
        write_checkpoint=write_checkpoint, context=context,
        preview_competition=preview, finalize_competition=finalize,
        renew_selected_event=renew, validate_process_interval=validate,
        destroy_trial_cache=context.trial_cache.invalidate_after_interval,
    )


@dataclass(frozen=True)
class IntegratedAcceptedIntervalV12:
    transaction_id: str
    disposition: str
    runtime_before_sha256: str
    runtime_after_sha256: str
    selected_proposal_id: str | None
    selected_event_ids: tuple[str, ...]
    accepted_duration_s: float
    selected_completion_time_s: float | None
    event_endpoint_alignment_iterations: int
    output_published: bool
    lifecycle: tuple[Mapping[str, Any], ...]


def _engine_ledgers(engine: Any) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    mpz = engine.mpz
    def total(name: str) -> float:
        value = getattr(mpz, name, None)
        return 0.0 if value is None else float(np.asarray(value, dtype=float).sum())
    active = {"mobile": total("mobile"), "retained": total("retained")}
    wake = {"mobile": total("wake_mobile"), "retained": total("wake_retained")}
    signed = {}
    for region, stem in (("active", ""), ("wake", "wake_")):
        for kind in ("mobile", "retained"):
            positive = getattr(mpz, f"{stem}{kind}_positive", None)
            negative = getattr(mpz, f"{stem}{kind}_negative", None)
            if positive is not None and negative is not None:
                signed[f"{region}_{kind}"] = float(
                    np.asarray(positive, dtype=float).sum()
                    - np.asarray(negative, dtype=float).sum()
                )
    return active, wake, signed


def create_fresh_independent_process_engine_v12(
    context: CurrentSourceMultiFrontProductionContextV12,
    new_owner_id: str,
    engine_id: str,
    prior_region: Any,
    component: frozenset[str],
) -> ProcessEngineState:
    """Create and capture a real fresh V11 engine for an independent arm.

    No historical process state is divided or copied.  The old physical engine
    remains owned by the residual junction-attached component; this factory
    initializes a new production engine and immediately converts its complete
    engine+MPZ payload into the V12 owner record.
    """
    configured = context.adapter_configuration.get("fresh_process_engine_factory")
    if callable(configured):
        actual = configured(new_owner_id, component, context)
    else:
        cfg = context.adapter_configuration
        base = cfg.get("base")
        material = cfg.get("material")
        if base is None or material is None or not callable(
            getattr(base, "build_engine", None)
        ):
            raise StatefulProductionInterlock(
                "independent-tip handoff requires a production fresh-engine factory"
            )
        actual = base.build_engine(context.args, material)
    coordinate = float(getattr(actual.mpz, "advance_total_m", float("nan")))
    if not np.isfinite(coordinate) or abs(coordinate) > 1.0e-18:
        raise StatefulProductionInterlock(
            "fresh independent physical engine must begin at local coordinate zero"
        )
    from .sharp_front_v11_branching import _capture_shared_engine
    payload = _capture_shared_engine(actual)
    active, wake, signed = _engine_ledgers(actual)
    source_id = "fresh:" + canonical_hash({
        "new_owner_id": new_owner_id,
        "member_front_ids": sorted(component),
        "detached_from_owner_id": prior_region.owner_id,
        "engine_class": type(actual).__name__,
        "payload_sha256": _pickle_hash(payload),
    })
    captured = ProcessEngineState.from_v11_payload(
        engine_id=engine_id, source_state_id=source_id, payload=payload,
        active_ledgers=active, wake_ledgers=wake,
        signed_system_ledgers=signed, update_count=0,
        event_renewal_count=0, local_process_coordinate_m=0.0,
        family_identity=context.runtime.process_engines[
            prior_region.process_engine_id
        ].family_identity,
    )
    context.actual_engines[engine_id] = actual
    return captured


def evaluate_candidate_marginal_kinetic_v12(
    front_id: str, candidate: Any, runtime: MultiFrontRuntimeState,
    context: CurrentSourceMultiFrontProductionContextV12,
) -> float:
    """Evaluate the reviewed V11 fixed-load candidate-marginal energy drive."""
    from .live_topology_kernel_v12 import DynamicExactTopologyProviderV12
    from .topology_transaction_v11 import (
        TopologyArm, apply_causal_sharp_wake_trial_geometry,
        clip_arm_at_first_intersection, extend_network_arm, mark_coalesced,
    )
    solved = context.provisional_solved_state
    if solved is None:
        raise StatefulProductionInterlock("candidate-marginal trial lacks accepted mechanics")
    start = runtime.crack_network.branch(front_id).tip
    distance = float(context.adapter_configuration["da_phys_m"])
    raw = TopologyArm(
        candidate.candidate_id, front_id, start,
        (
            start[0] + distance * float(candidate.direction_xy[0]),
            start[1] + distance * float(candidate.direction_xy[1]),
        ),
        distance, 0.0,
    )
    arm, target = clip_arm_at_first_intersection(runtime.crack_network, raw)
    if arm.event_reward_m <= 0.0:
        return 0.0
    network = extend_network_arm(runtime.crack_network, arm)
    if target is not None:
        network = mark_coalesced(network, arm.branch_id, target)
    trial_fem = replace(solved.fem_state, crack_network=network)
    trial_fem = apply_causal_sharp_wake_trial_geometry(trial_fem, (arm,))
    trial_solved = replace(
        solved, fem_state=trial_fem,
        measurement={
            "accepted_fem_state_sha256": accepted_fem_state_fingerprint(trial_fem)
        },
    )
    request = build_arbitrary_region_request_v12(
        trial_solved, runtime, context,
        active_front_ids=network.active_tip_ids,
    )
    result = DynamicExactTopologyProviderV12(
        runtime.resource_policy
    ).evaluate(request)
    context.provider_solve_count += 1
    trial_energy = float(
        result["base_equilibrium"]["recoverable_potential_energy_J_per_m"]
    )
    return (
        float(solved.fem_state.stored_energy_J_per_m) - trial_energy
    ) / float(arm.event_reward_m)


def _apply_realized_cleavage_intersections_v12(
    before: MultiFrontRuntimeState,
    nominal_post: MultiFrontRuntimeState,
    realized_network: Any,
    arms: Sequence[Any],
    targets_by_front: Mapping[str, str],
) -> MultiFrontRuntimeState:
    """Atomically project V11 arm clipping/coalescence into V12 registries.

    A cleavage transaction remains one transaction even when one of its
    realized arms terminates at an existing crack.  The incoming tip is
    removed from the owner/runtime registries and its process engine is
    archived only when that removal empties the owner region.
    """
    if not targets_by_front:
        realized_fingerprint = hashlib.sha256(
            realized_network.to_json().encode()
        ).hexdigest()
        if realized_fingerprint != nominal_post.topology_fingerprint:
            raise StatefulProductionInterlock(
                "unclipped realized cleavage geometry differs from its atomic commit"
            )
        return nominal_post
    if not nominal_post.transaction_records:
        raise StatefulProductionInterlock(
            "realized cleavage intersection lacks its atomic transaction record"
        )
    if (
        nominal_post.transaction_records[-1].pre_topology_fingerprint
        != before.topology_fingerprint
    ):
        raise StatefulProductionInterlock(
            "realized cleavage intersection does not belong to the accepted precursor"
        )
    front_runtimes = dict(nominal_post.front_runtimes)
    owner_by_front = dict(nominal_post.owner_by_front)
    regions = dict(nominal_post.process_regions)
    engines = dict(nominal_post.process_engines)
    reservoirs = dict(nominal_post.reservoirs)
    transition = dict(nominal_post.transaction_records[-1].owner_region_transition)
    archived = {}
    transaction_id = nominal_post.transaction_records[-1].transaction_id
    for front_id, target_id in sorted(targets_by_front.items()):
        if front_id not in front_runtimes or front_id not in owner_by_front:
            raise StatefulProductionInterlock(
                "realized coalescence refers to a non-active incoming front"
            )
        front_runtimes.pop(front_id)
        owner_id = owner_by_front.pop(front_id)
        region = regions[owner_id]
        remaining = region.member_front_ids - {front_id}
        if remaining:
            regions[owner_id] = replace(region, member_front_ids=remaining)
        else:
            engine = engines.pop(region.process_engine_id)
            regions.pop(owner_id)
            reservoir_id = "reservoir:" + canonical_hash({
                "owner": owner_id,
                "tx": transaction_id,
                "coalesced_front": front_id,
            })[:20]
            reservoirs[reservoir_id] = ProcessRegionReservoir(
                reservoir_id=reservoir_id,
                archived_owner_id=owner_id,
                archived_engine=engine,
                member_front_ids_at_archive=(front_id,),
                junction_ids=tuple(region.unresolved_junction_ids),
                archive_transaction_id=transaction_id,
            )
            archived[front_id] = reservoir_id
    transition.update({
        "coalescence_targets_by_front": dict(sorted(targets_by_front.items())),
        "coalescence_archived_reservoir_by_front": dict(sorted(archived.items())),
    })
    provisional = replace(
        nominal_post,
        crack_network=realized_network,
        front_runtimes=front_runtimes,
        owner_by_front=owner_by_front,
        process_regions=regions,
        process_engines=engines,
        reservoirs=reservoirs,
        cumulative_coalescences=(
            nominal_post.cumulative_coalescences + len(targets_by_front)
        ),
    )
    arm_by_front = {arm.branch_id: arm for arm in arms}
    if len(arm_by_front) != len(arms):
        raise StatefulProductionInterlock(
            "realized cleavage arm identities are not unique"
        )
    record = nominal_post.transaction_records[-1]
    target_values = tuple(sorted(set(targets_by_front.values())))
    record = replace(
        record,
        retired_front_ids=tuple(sorted(
            set(record.retired_front_ids).union(targets_by_front)
        )),
        post_active_front_count=len(provisional.active_front_ids),
        realized_lengths_m=tuple(float(arm.event_reward_m) for arm in arms),
        realized_endpoints_m=tuple(tuple(arm.end_xy_m) for arm in arms),
        renewal_distance_m=max(float(arm.event_reward_m) for arm in arms),
        owner_region_transition=transition,
        post_topology_fingerprint=provisional.topology_fingerprint,
        post_registry_fingerprint=provisional.registry_fingerprint,
        coalescence_target_front_id=(
            target_values[0] if len(target_values) == 1 else None
        ),
        clipped_or_coalesced_disposition=(
            "clipped_at_first_intersection_and_coalesced:"
            + ";".join(
                f"{front}->{target}"
                for front, target in sorted(targets_by_front.items())
            )
        ),
    )
    return replace(
        provisional,
        transaction_records=nominal_post.transaction_records[:-1] + (record,),
    )


def execute_selected_topology_trial_v12(
    state: Any, proposal: TopologyProposal,
    context: CurrentSourceMultiFrontProductionContextV12,
) -> ProposalTrialOutcome:
    """Run one isolated current-source sharp-wake trial for the selected action."""
    import math
    from .fem import assemble_mechanics
    from .general_multifront_v12 import commit_selected_proposal
    from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
    from .multifront_competition_v12 import finalize_competitions
    from .production_multifront_v12 import ExactTrialDeltaV12
    from .topology_transaction_v11 import (
        TopologyArm, apply_causal_sharp_wake_trial_geometry,
        clip_arm_at_first_intersection, extend_network_arm, mark_coalesced,
        equilibrate_fixed_load_with_production_fem,
    )
    runtime = context.provisional_runtime
    batch = context.provisional_observation_batch
    if runtime is None or batch is None or not context.previewed_clock_states:
        raise StatefulProductionInterlock("exact trial lacks its accepted preview state")
    preview = next(iter(context.previewed_clock_states.values()))
    finalized = finalize_competitions(
        runtime, preview, selected_proposal=proposal, accepted=True,
    ).runtime
    post = commit_selected_proposal(finalized, proposal)
    regions = dict(post.process_regions)
    engines = dict(post.process_engines)
    for owner_id, prior_region in runtime.process_regions.items():
        if owner_id in regions and prior_region.process_engine_id in engines:
            engines[prior_region.process_engine_id] = runtime.process_engines[
                prior_region.process_engine_id
            ]
            regions[owner_id] = replace(
                regions[owner_id],
                cumulative_process_advance_m=runtime.process_engines[
                    prior_region.process_engine_id
                ].local_process_coordinate_m,
            )
    post = replace(post, process_regions=regions, process_engines=engines)
    if proposal.action_type not in ("one_arm", "two_arm"):
        raise StatefulProductionInterlock(
            "physical V12 trial currently accepts only one- or two-arm cleavage"
        )
    if proposal.action_type == "one_arm":
        branch_ids = (proposal.front_id,)
        realized_network = runtime.crack_network
    else:
        branch_ids = tuple(
            next(
                front_id for front_id in post.active_front_ids
                if post.crack_network.branch(front_id).local_state.get(
                    "candidate_id"
                ) == candidate_id
                and post.crack_network.branch(front_id).parent_branch_id
                == proposal.front_id
            )
            for candidate_id in proposal.candidate_ids
        )
        parent = runtime.crack_network.branch(proposal.front_id)
        child_ids = set(branch_ids)
        stripped = []
        for branch in post.crack_network.branches:
            if branch.branch_id not in child_ids:
                stripped.append(branch)
                continue
            local_state = dict(branch.local_state)
            local_state.pop("committed_edges", None)
            stripped.append(replace(
                branch,
                path=(parent.tip,),
                orientation_history_rad=(parent.current_orientation_rad,),
                local_state=local_state,
            ))
        realized_network = replace(
            post.crack_network,
            branches=tuple(stripped),
            geometry_generation=runtime.crack_network.geometry_generation,
        )
    observations = {
        (item.front_id, item.candidate_id): item for item in batch.observations
    }
    owner_engine_id = runtime.process_regions[
        proposal.owner_id
    ].process_engine_id
    engine = context.actual_engines[owner_engine_id]
    candidate_objects = {item.candidate_id: item for item in context.candidates}
    arms = []
    targets_by_front = {}
    for branch_id, candidate_id, endpoint in zip(
        branch_ids, proposal.candidate_ids, proposal.end_points_m
    ):
        candidate = candidate_objects[candidate_id]
        start = runtime.crack_network.branch(proposal.front_id).tip
        raw = TopologyArm(
            candidate_id, branch_id, start, endpoint,
            math.dist(start, endpoint), 0.0,
        )
        arm, target = clip_arm_at_first_intersection(realized_network, raw)
        observation = observations[(proposal.front_id, candidate_id)]
        sigma_tip = engine.sigma_tip(
            observation.directional_K_MPa_sqrt_m * 1.0e6
            / math.sqrt(float(candidate.gamma_rel))
        )
        _, _, barrier = engine.lambda_cleave(
            sigma_tip, float(context.adapter_configuration["temperature_K"])
        )
        resistance = hazard_resistance_J_per_m2(
            barrier_J=barrier, cooperative_hits=float(engine.f.m_hits),
            burgers_vector_m=float(engine.b),
            gamma_relative=float(candidate.gamma_rel),
        )
        arm = replace(
            arm,
            hazard_dissipation_J_per_m=resistance * arm.event_reward_m,
        )
        realized_network = extend_network_arm(realized_network, arm)
        if target is not None:
            realized_network = mark_coalesced(
                realized_network, arm.branch_id, target,
            )
            targets_by_front[arm.branch_id] = target
        arms.append(arm)
    post = _apply_realized_cleavage_intersections_v12(
        runtime, post, realized_network, tuple(arms), targets_by_front,
    )
    trial = replace(state.isolated_copy(), crack_network=post.crack_network)
    trial = apply_causal_sharp_wake_trial_geometry(trial, tuple(arms))
    trial = equilibrate_fixed_load_with_production_fem(trial)
    released = float(state.stored_energy_J_per_m - trial.stored_energy_J_per_m)
    cost = math.fsum(item.hazard_dissipation_J_per_m for item in arms)
    tolerance = max(
        1.0e-12,
        1.0e-8 * max(
            abs(float(state.stored_energy_J_per_m)),
            abs(float(trial.stored_energy_J_per_m)), cost,
        ),
    )
    if released + tolerance < cost:
        return ProposalTrialOutcome(
            proposal, False, state,
            "insufficient_whole_topology_energy_release",
            stored_energy_release_J_per_m=released,
            stored_energy_cost_J_per_m=cost,
        )
    counters = dict(trial.event_counters)
    counters["topology_actions"] = int(counters.get("topology_actions", 0)) + 1
    trial = replace(trial, event_counters=counters)
    sigma = assemble_mechanics(
        trial.mesh, trial.displacement, trial.ep_gp, trial.rho_gp,
        trial.damage, trial.elasticity_D, trial.material,
        cohesive_network=trial.cohesive_network,
    )[2]
    exact_topology = hashlib.sha256(
        post.crack_network.to_json().encode()
    ).hexdigest()
    exact_state = accepted_fem_state_fingerprint(trial)
    record = replace(
        post.transaction_records[-1],
        realized_endpoints_m=tuple(item.end_xy_m for item in arms),
        realized_lengths_m=tuple(item.event_reward_m for item in arms),
        wake_mutation=dict(trial.junction_process_state),
        stored_energy_release_J_per_m=released,
        stored_energy_cost_J_per_m=cost,
        post_topology_fingerprint=exact_topology,
        accepted_fem_topology_fingerprint=exact_topology,
        geometry_fingerprint=exact_topology,
        exact_accepted_trial_fingerprint=exact_state,
    )
    post = replace(
        post,
        transaction_records=post.transaction_records[:-1] + (record,),
    )
    before_active = set(runtime.active_front_ids)
    after_active = set(post.active_front_ids)
    lengths = tuple(item.event_reward_m for item in arms)
    delta = ExactTrialDeltaV12(
        pre_crack_network=runtime.crack_network,
        post_crack_network=post.crack_network,
        continued_front_ids=tuple(sorted(before_active & after_active)),
        created_front_ids=tuple(sorted(after_active - before_active)),
        retired_front_ids=tuple(sorted(before_active - after_active)),
        coalescence_target_front_id=(
            post.transaction_records[-1].coalescence_target_front_id
        ),
        junctions_after=post.junctions,
        owner_by_front_after=post.owner_by_front,
        process_regions_after=post.process_regions,
        process_engines_after=post.process_engines,
        front_runtimes_after=post.front_runtimes,
        reservoirs_after=post.reservoirs,
        scheduler_after=post.scheduler,
        cumulative_branch_births_after=post.cumulative_branch_births,
        cumulative_coalescences_after=post.cumulative_coalescences,
        cumulative_retirements_after=post.cumulative_retirements,
        transaction_records_after=post.transaction_records,
        output_counters_after=post.output_counters,
        termination_reason_after=post.termination_reason,
        policy_bound_after=post.policy_bound,
        realized_endpoints_m=tuple(item.end_xy_m for item in arms),
        realized_lengths_m=lengths,
        wake_mutation=dict(trial.junction_process_state),
        released_energy_J_per_m=released,
        dissipative_cost_J_per_m=cost,
    )
    return ProposalTrialOutcome(
        proposal, True, trial, "accepted_exact_current_source_trial",
        exact_realized_crack_network=post.crack_network,
        realized_endpoints_m=tuple(item.end_xy_m for item in arms),
        realized_arm_lengths_m=lengths,
        stored_energy_release_J_per_m=released,
        stored_energy_cost_J_per_m=cost,
        topology_fingerprint=post.topology_fingerprint,
        exact_delta=delta, exact_post_sigma_gp=np.asarray(sigma).copy(),
    )


def production_directional_rate_adapter_v12(
    observation: Any, candidate: Any,
    context: CurrentSourceMultiFrontProductionContextV12,
) -> float:
    """Delegate a front-owned rate to the unchanged V11 production equation."""
    from .directional_competition_v11 import preview_production_cleavage_rate
    front_id = getattr(observation, "front_id", None)
    if front_id is None:
        front_id = getattr(observation, "tip_id")
    front_id = str(front_id)
    owner = context.provisional_runtime.owner_by_front[front_id]
    engine_id = context.provisional_runtime.process_regions[
        owner
    ].process_engine_id
    engine = context.actual_engines.get(engine_id)
    if engine is None:
        from .current_source_multifront_hooks_v12 import (
            restore_complete_current_source_engine,
        )
        engine = restore_complete_current_source_engine(
            context.provisional_runtime.process_engines[engine_id]
        )
        context.actual_engines[engine_id] = engine
    return float(preview_production_cleavage_rate(
        engine, candidate,
        signed_J_J_per_m2=float(observation.kinetic_J_J_per_m2),
        Eprime_Pa=float(context.provisional_solved_state.fem_state.material.Eprime),
        temperature_K=float(context.adapter_configuration["temperature_K"]),
    ).lambda_per_s)


def recompute_physical_process_connectivity_v12(
    staged: MultiFrontRuntimeState,
    selected_outcome: ProposalTrialOutcome | None,
    context: CurrentSourceMultiFrontProductionContextV12,
) -> MultiFrontRuntimeState:
    """Apply the V11 handoff scales to each unresolved V12 junction."""
    import math
    from .general_multifront_v12 import (
        CouplingEvidence, recompute_process_region_connectivity,
    )
    if not any(
        region.unresolved_junction_ids
        for region in staged.process_regions.values()
    ):
        return staged
    handoff = float(context.adapter_configuration["branch_handoff_length_m"])
    process_zone = float(context.adapter_configuration["process_zone_length_m"])
    contour = float(context.adapter_configuration["contour_radius_m"])
    pre_runtime = context.provisional_runtime
    batch = context.provisional_observation_batch
    valid_by_front = {}
    if pre_runtime is not None and batch is not None:
        for front_id in pre_runtime.active_front_ids:
            rows = [
                row for row in batch.local_mechanics_records
                if row["front_id"] == front_id
            ]
            valid_by_front[front_id] = bool(
                rows and all(row["local_J_valid"] for row in rows)
            )

    by_id = {item.branch_id: item for item in staged.crack_network.branches}

    def descendants(branch_id):
        result = []
        for front_id in staged.active_front_ids:
            cursor = by_id[front_id]
            while True:
                if cursor.branch_id == branch_id:
                    result.append(front_id)
                    break
                if cursor.parent_branch_id is None:
                    break
                cursor = by_id[cursor.parent_branch_id]
        return tuple(sorted(result))

    def distance_from_junction(front_id, junction_xy):
        chain = []
        cursor = by_id[front_id]
        while cursor.parent_branch_id is not None:
            chain.append(cursor)
            if cursor.root == junction_xy:
                break
            cursor = by_id[cursor.parent_branch_id]
        return math.fsum(
            math.dist(a, b)
            for branch in reversed(chain)
            for a, b in zip(branch.path, branch.path[1:])
        )

    evidence = {}
    for junction_id, junction in staged.junctions.items():
        if junction.status != "unresolved":
            continue
        owning_regions = [
            region for region in staged.process_regions.values()
            if junction_id in region.unresolved_junction_ids
        ]
        if len(owning_regions) != 1:
            raise StatefulProductionInterlock(
                "an unresolved junction must belong to exactly one process region"
            )
        owned_fronts = owning_regions[0].member_front_ids
        sides = tuple(descendants(child) for child in junction.child_branch_ids)
        fronts = tuple(front for side in sides for front in side)
        if not fronts:
            continue
        lengths = {
            front: distance_from_junction(front, junction.junction_xy_m)
            for front in fronts
        }
        separations = [
            math.dist(
                staged.crack_network.branch(left).tip,
                staged.crack_network.branch(right).tip,
            )
            for left in sides[0] for right in sides[1]
        ]
        separation = min(separations) if separations else 0.0
        detached = tuple(sorted(
            front for front in fronts
            if front in owned_fronts
            if lengths[front] >= max(handoff, contour)
            and separation >= process_zone
            and valid_by_front.get(front, False)
        ))
        evidence[junction_id] = CouplingEvidence(
            junction_id, handoff, process_zone, contour, separation,
            min(lengths.values()), separation < 2.0 * contour, detached,
        )
    if not evidence:
        return staged
    return recompute_process_region_connectivity(
        staged, evidence,
        fresh_engine_factory=lambda owner, engine, region, members:
            create_fresh_independent_process_engine_v12(
                context, owner, engine, region, members
            ),
    )


def _recapture_process_engine(
    prior: ProcessEngineState, engine: Any, *, renewed: bool,
) -> ProcessEngineState:
    from .sharp_front_v11_branching import _capture_shared_engine
    active, wake, signed = _engine_ledgers(engine)
    coordinate = float(getattr(engine.mpz, "advance_total_m", prior.local_process_coordinate_m))
    return ProcessEngineState.from_v11_payload(
        engine_id=prior.engine_id, source_state_id=prior.source_state_id,
        payload=_capture_shared_engine(engine), active_ledgers=active,
        wake_ledgers=wake, signed_system_ledgers=signed,
        update_count=prior.update_count,
        event_renewal_count=prior.event_renewal_count + int(renewed),
        local_process_coordinate_m=coordinate,
        family_identity=prior.family_identity,
    )


def run_stateful_accepted_interval_v12(
    context: CurrentSourceMultiFrontProductionContextV12, duration_s: float, *,
    dry_run_discard: bool = False, failure_stage: str | None = None,
) -> IntegratedAcceptedIntervalV12:
    """Execute one authoritative accepted interval or restore the prior context.

    All physical leaf operations remain supplied by the reviewed hook factory.
    A cache-only mechanics loader/provider lookup can therefore execute this
    function without authorizing a solve.
    """
    from .general_multifront_v12 import select_global_topology_proposal
    if float(duration_s) <= 0.0:
        raise ValueError("accepted interval duration must be positive")
    hooks = build_stateful_production_hooks(context)
    before_runtime = context.runtime
    before_fem = context.accepted_fem_state
    before_sigma = (
        None if context.accepted_stress_field is None
        else context.accepted_stress_field.copy()
    )
    before_stress_available = context.stress_available
    before_mechanics_source = context.mechanics_source_identity
    before_physical_time = context.physical_time_s
    before_opening = context.accepted_opening_m
    before_step = context.step_count
    before_engines = copy.deepcopy(context.actual_engines)
    before_info = copy.deepcopy(context.interval_info_by_owner)
    before_previews = copy.deepcopy(context.previewed_clock_states)
    before_trial_cache = copy.deepcopy(context.trial_cache)
    before_output_writer = context.output_writer.snapshot()
    before_observation_batch = context.provisional_observation_batch
    before_counts = (
        context.provider_lookup_count, context.provider_solve_count,
        context.mechanics_solve_count, context.pf_workers_started,
    )
    before_fingerprint = context.fingerprint
    transaction_id = f"interval-{before_step + 1:08d}"
    atomic = AtomicIntervalTransactionWriterV12(
        context.output_writer.root,
        staging_parent=(
            Path("/private/tmp/v6_4_discarded_atomic_staging")
            if dry_run_discard else None
        ),
    )
    try:
        context.begin_interval(transaction_id)
        candidate_inventory = {
            key: value.mechanically_active_candidate_ids
            for key, value in before_runtime.front_runtimes.items()
        }
        adapted = hooks.adapt_accepted_mesh(before_fem, candidate_inventory)
        if not isinstance(adapted, AdaptedAcceptedState):
            raise StatefulProductionInterlock("adapt hook returned a raw tuple")
        def evaluate_fraction(fraction: float):
            if float(fraction) == 0.0:
                if before_sigma is None:
                    raise StatefulProductionInterlock(
                        "accepted-boundary pending event requires accepted stress"
                    )
                boundary_state = adapted.fem_state
                boundary_hash = accepted_fem_state_fingerprint(boundary_state)
                before_hash = accepted_fem_state_fingerprint(before_fem)
                if boundary_hash == before_hash:
                    boundary_sigma = before_sigma
                else:
                    # A pending clock freezes time/opening, not discretization.
                    # Use the graph-authoritative adapted state at that same
                    # boundary; reverting to before_fem here would silently
                    # discard a required topology-damage remap.
                    stress_rebuilder = context.adapter_configuration.get(
                        "accepted_boundary_stress_rebuilder"
                    )
                    if callable(stress_rebuilder):
                        boundary_sigma = stress_rebuilder(boundary_state, context)
                    else:
                        from .fem import assemble_mechanics
                        boundary_sigma = assemble_mechanics(
                            boundary_state.mesh, boundary_state.displacement,
                            boundary_state.ep_gp, boundary_state.rho_gp,
                            boundary_state.damage, boundary_state.elasticity_D,
                            boundary_state.material,
                            cohesive_network=boundary_state.cohesive_network,
                        )[2]
                solved_value = SolvedAcceptedState(
                    boundary_state, boundary_sigma,
                    {
                        "accepted_fem_state_sha256":
                            boundary_hash,
                        "accepted_boundary_reuse": True,
                        "adapted_same_boundary_state": boundary_hash != before_hash,
                    },
                    before_opening, before_physical_time,
                    before_mechanics_source,
                )
            else:
                solved_value = hooks.solve_accepted_fem(adapted, fraction)
            if not isinstance(solved_value, SolvedAcceptedState):
                raise StatefulProductionInterlock("solve/load hook returned a raw tuple")
            if solved_value.sigma_gp is None:
                raise StatefulProductionInterlock("solved accepted stress is unavailable")
            sigma_value = np.asarray(solved_value.sigma_gp, dtype=float)
            if (
                sigma_value.ndim != 2 or not sigma_value.size
                or not np.all(np.isfinite(sigma_value))
            ):
                raise StatefulProductionInterlock("solved accepted stress is unavailable")
            expected_end = before_physical_time + float(duration_s) * float(fraction)
            if not np.isclose(
                solved_value.physical_time_s, expected_end,
                rtol=0.0, atol=1.0e-12,
            ):
                raise StatefulProductionInterlock(
                    "accepted mechanics time differs from requested interval fraction"
                )
            fem_hash_value = accepted_fem_state_fingerprint(solved_value.fem_state)
            if solved_value.measurement.get(
                "accepted_fem_state_sha256"
            ) != fem_hash_value:
                raise StatefulProductionInterlock("stale sigma/FEM ownership sentinel failed")
            accepted_id_value = "accepted:" + canonical_hash({
                "fem_state_sha256": fem_hash_value,
                "prior_accepted_state_id": before_runtime.accepted_state_id,
                "end_time_s": solved_value.physical_time_s,
                "accepted_load_m": solved_value.accepted_load_m,
            })
            stress_id_value = stress_field_identity(
                solved_value.fem_state, sigma_value,
                mechanics_source_identity=solved_value.mechanics_source_identity,
                accepted_state_id=accepted_id_value,
            )
            runtime_value = replace(
                before_runtime, accepted_state_id=accepted_id_value,
                stress_field_state_id=stress_id_value,
            )
            context.provisional_solved_state = solved_value
            context.provisional_runtime = runtime_value
            request_value = hooks.build_exact_topology_request(
                solved_value, runtime_value
            )
            lookup = context.adapter_configuration.get("provider_lookup")
            evaluator = context.adapter_configuration.get("provider_evaluator")
            if callable(lookup):
                provider_value = context.invoke(
                    "cache_only_provider_lookup",
                    lambda: lookup(request_value, runtime_value, context), pure=True,
                )
                context.provider_lookup_count += 1
            elif callable(evaluator):
                provider_value = context.invoke(
                    "exact_provider_evaluation",
                    lambda: evaluator(request_value, runtime_value, context), pure=True,
                )
                context.provider_solve_count += 1
            else:
                from .live_topology_kernel_v12 import DynamicExactTopologyProviderV12
                provider = DynamicExactTopologyProviderV12(
                    runtime_value.resource_policy
                )
                provider_value = context.invoke(
                    "exact_provider_evaluation",
                    lambda: provider.evaluate(request_value), pure=True,
                )
                context.provider_solve_count += 1
            batch_value = hooks.extract_directional_observations(
                provider_value, runtime_value
            )
            if not isinstance(batch_value, DirectionalObservationBatch):
                raise StatefulProductionInterlock("observation hook returned a raw tuple")
            batch_value.validate(runtime_value)
            context.provisional_observation_batch = batch_value
            # Only the mechanics-consistent endpoint preview may remain
            # reachable by the subsequent exact topology trial.
            context.previewed_clock_states.clear()
            preview_value = hooks.preview_competition(
                runtime_value, batch_value.rates_by_front_and_candidate,
                batch_value.endpoints_by_front_and_candidate,
                start_time_s=before_physical_time,
                duration_s=float(duration_s) * float(fraction),
                correlation_interval_s=float(
                    context.adapter_configuration.get("correlation_interval_s", 0.0)
                ),
            )
            selected_value = select_global_topology_proposal(
                preview_value.proposals, runtime_value.scheduler,
                scheduler_policy=runtime_value.resource_policy.scheduler_policy,
            )
            return (
                solved_value, sigma_value, runtime_value, batch_value,
                preview_value, selected_value,
            )

        # A completed event can legitimately remain pending at an accepted
        # boundary while awaiting a correlated partner or an energy-admissible
        # topology decision.  It must be decided at that boundary rather than
        # searched backward to its historical clock-completion timestamp.
        has_pending_clock_event = any(
            hazard.pending_events
            for front in before_runtime.front_runtimes.values()
            for hazard in competition_state_from_dict(
                front.competition_state
            ).hazard_states
            if hazard.candidate_id in front.mechanically_active_candidate_ids
        )
        evaluated = evaluate_fraction(
            0.0 if has_pending_clock_event else 1.0
        )
        candidate = evaluated[-1]
        pending_at_boundary = (
            has_pending_clock_event and candidate is not None
        )
        if pending_at_boundary:
            candidate = replace(
                candidate, completion_time_s=before_physical_time,
            )
            evaluated = evaluated[:-1] + (candidate,)
        elif has_pending_clock_event:
            evaluated = evaluate_fraction(1.0)
            candidate = evaluated[-1]
        alignment_iterations = 0
        endpoint_clock_residual_s = 0.0
        if candidate is not None and not pending_at_boundary:
            lower = 0.0
            upper = 1.0
            event_evaluated = evaluated
            configured_fraction_tolerance = float(
                context.adapter_configuration.get(
                    "event_endpoint_fraction_tolerance", 1.0e-7
                )
            )
            absolute_time_tolerance = float(
                context.adapter_configuration.get(
                    "event_endpoint_time_tolerance_s", 1.0e-10
                )
            )
            if absolute_time_tolerance <= 0.0:
                raise StatefulProductionInterlock(
                    "event endpoint time tolerance must be positive"
                )
            if configured_fraction_tolerance <= 0.0:
                raise StatefulProductionInterlock(
                    "event endpoint fraction tolerance must be positive"
                )
            tolerance_fraction = min(
                configured_fraction_tolerance,
                absolute_time_tolerance / float(duration_s),
            )
            alignment_iteration_limit = event_endpoint_iteration_limit(
                tolerance_fraction
            )
            aligned = False
            for alignment_iterations in range(1, alignment_iteration_limit + 1):
                event_candidate = event_evaluated[-1]
                completion_fraction = (
                    float(event_candidate.completion_time_s) - before_physical_time
                ) / float(duration_s)
                if upper - lower <= tolerance_fraction:
                    aligned = True
                    break
                trial_fraction = min(upper, max(lower, completion_fraction))
                if (
                    trial_fraction <= lower + tolerance_fraction * 0.25
                    or trial_fraction >= upper - tolerance_fraction * 0.25
                ):
                    # If the event-side estimate lies at the no-event bound,
                    # expand locally instead of bisecting a nominal interval
                    # that may be many orders of magnitude longer than the
                    # physical completion time.
                    if lower > 0.0 and upper > 4.0 * lower:
                        trial_fraction = min(upper, 2.0 * lower)
                    else:
                        trial_fraction = 0.5 * (lower + upper)
                trial_evaluated = evaluate_fraction(trial_fraction)
                if trial_evaluated[-1] is None:
                    lower = trial_fraction
                else:
                    upper = trial_fraction
                    event_evaluated = trial_evaluated
            if not aligned:
                final_completion_fraction = (
                    float(event_evaluated[-1].completion_time_s)
                    - before_physical_time
                ) / float(duration_s)
                aligned = upper - lower <= tolerance_fraction
            if not aligned:
                raise StatefulProductionInterlock(
                    "event endpoint search did not converge to the absolute time tolerance: "
                    f"lower={lower:.17g}, upper={upper:.17g}, "
                    f"completion={final_completion_fraction:.17g}, "
                    f"tolerance={tolerance_fraction:.17g}, "
                    f"iterations={alignment_iterations}"
                )
            # Re-evaluate the event-side endpoint after the search so the
            # provisional FEM, provider observations, and preview reachable by
            # the exact trial all belong to the same accepted endpoint.
            evaluated = evaluate_fraction(upper)
            candidate = evaluated[-1]
            if candidate is None:
                raise StatefulProductionInterlock(
                    "event endpoint search lost the completed proposal"
                )
            endpoint_error = (
                evaluated[0].physical_time_s - float(candidate.completion_time_s)
            )
            if endpoint_error < -max(1.0e-12, absolute_time_tolerance):
                raise StatefulProductionInterlock(
                    "selected event completion lies after the accepted event-side endpoint"
                )
            # The physical endpoint is the event/no-event transition bracket,
            # not a fixed point of the endpoint-rate interpolation.  A sharp
            # mechanics-rate transition can leave the interpolated clock time
            # slightly behind the first event-side endpoint even after the
            # bracket is narrower than the configured absolute-time tolerance.
            # Relabel only that already-completed proposal at the resolved
            # transition endpoint; its threshold, ordinal, RNG state, action,
            # mechanics, and topology remain unchanged.
            endpoint_clock_residual_s = max(0.0, endpoint_error)
            candidate = replace(
                candidate, completion_time_s=float(evaluated[0].physical_time_s)
            )
            evaluated = evaluated[:-1] + (candidate,)
        solved, sigma, provisional_runtime, batch, preview, candidate = evaluated
        accepted_duration_s = float(solved.physical_time_s) - before_physical_time
        trial_by_proposal = {}
        resource_stop = bool(
            candidate is not None and candidate.action_type == "two_arm"
            and provisional_runtime.resource_policy.front_resource_limit is not None
            and len(provisional_runtime.active_front_ids) + 1
            > provisional_runtime.resource_policy.front_resource_limit
        )
        if candidate is not None and not resource_stop:
            trial_by_proposal[candidate.proposal_id] = hooks.trial_topology_proposal(
                solved.fem_state, candidate
            )
        accepted_candidate = (
            None if candidate is None or resource_stop
            else trial_by_proposal[candidate.proposal_id]
        )
        accepted = bool(
            candidate is not None and accepted_candidate is not None
            and accepted_candidate.accepted and not resource_stop
        )

        # Evolve each physical owner once, retaining isolated engines until the
        # complete transaction is publishable.
        observations_by_owner = {}
        engines = dict(provisional_runtime.process_engines)
        for owner_id, region in provisional_runtime.process_regions.items():
            pool = [x for x in batch.observations if x.owner_id == owner_id]
            controlling = min(
                pool, key=lambda x: (-x.directional_K_MPa_sqrt_m, x.identity_key)
            )
            observations_by_owner[owner_id] = controlling
            prior_engine = engines[region.process_engine_id]
            engines[region.process_engine_id] = hooks.evolve_process_engine(
                prior_engine, controlling, accepted_duration_s
            )
            hooks.validate_process_interval(
                context.interval_info_by_owner[prior_engine.engine_id], accepted_duration_s
            )
        finalized = hooks.finalize_competition(
            provisional_runtime, preview, selected_proposal=candidate,
            accepted=accepted, resource_limit_stop=resource_stop,
        )
        staged = replace(finalized.runtime, process_engines=engines)
        evolved_engines = dict(engines)
        selected_outcome = None
        if accepted:
            selected_outcome = context.trial_cache.select(provisional_runtime, candidate)
            staged = apply_exact_trial_delta(staged, selected_outcome)
            # Exact topology deltas are pre-renewal structural states.  An
            # accepted trial may introduce or retire owners, but it may not
            # impersonate the one physical moving-frame renewal owned below.
            for engine_id, prior_engine in provisional_runtime.process_engines.items():
                delta_engine = staged.process_engines.get(engine_id)
                if (
                    delta_engine is not None
                    and delta_engine.event_renewal_count
                    != prior_engine.event_renewal_count
                ):
                    raise StatefulProductionInterlock(
                        "exact trial delta performed metadata-only event renewal"
                    )
            merged_engines = dict(staged.process_engines)
            for engine_id, evolved_engine in evolved_engines.items():
                if engine_id in merged_engines:
                    merged_engines[engine_id] = evolved_engine
            staged = replace(staged, process_engines=merged_engines)
            if selected_outcome.exact_post_sigma_gp is None:
                raise StatefulProductionInterlock("accepted exact trial lacks post-event sigma_gp")
            accepted_fem_after = selected_outcome.trial_fem_state
            sigma_after = np.asarray(selected_outcome.exact_post_sigma_gp)
            renewal_distance = max(selected_outcome.exact_delta.realized_lengths_m)
        else:
            accepted_fem_after = solved.fem_state
            sigma_after = sigma
            renewal_distance = 0.0
        if failure_stage == "engine_finalization_to_renewal":
            raise StatefulProductionInterlock("injected engine/finalization interruption")
        selected_owner = None if candidate is None or not accepted else candidate.owner_id
        regions_after = dict(staged.process_regions)
        for owner_id, region in staged.process_regions.items():
            engine_id = region.process_engine_id
            engine = context.actual_engines[engine_id]
            is_selected = owner_id == selected_owner
            hooks.renew_selected_event(
                engine_id, selected=is_selected,
                distance_m=renewal_distance if is_selected else 0.0,
            )
            prior = staged.process_engines[engine_id]
            recapture = context.adapter_configuration.get("process_engine_recapture")
            staged.process_engines[engine_id] = (
                recapture(prior, engine, is_selected, context)
                if callable(recapture) else
                _recapture_process_engine(prior, engine, renewed=is_selected)
            )
            regions_after[owner_id] = replace(
                region,
                cumulative_process_advance_m=staged.process_engines[
                    engine_id
                ].local_process_coordinate_m,
            )
        staged = replace(
            staged, process_engines=dict(staged.process_engines),
            process_regions=regions_after,
        )
        connectivity = context.adapter_configuration.get("connectivity_recomputer")
        if callable(connectivity):
            staged = context.invoke(
                "process_region_connectivity",
                lambda: connectivity(staged, selected_outcome, context),
            )
        else:
            staged = context.invoke(
                "process_region_connectivity", lambda: staged, pure=True,
            )
        staged.validate()
        if failure_stage == "renewal_to_output_staging":
            raise StatefulProductionInterlock("injected renewal/output interruption")
        context.physical_time_s = float(solved.physical_time_s)
        context.accepted_opening_m = float(solved.accepted_load_m)
        next_step = before_step + 1
        accepted_id = accepted_context_state_identity(
            accepted_fem_after, staged, physical_time_s=context.physical_time_s,
            accepted_opening_m=context.accepted_opening_m, step_count=next_step,
        )
        stress_id = stress_field_identity(
            accepted_fem_after, sigma_after,
            mechanics_source_identity=solved.mechanics_source_identity,
            accepted_state_id=accepted_id,
        )
        staged = replace(staged, accepted_state_id=accepted_id, stress_field_state_id=stress_id)
        staged.validate()
        rows = {name: [] for name in TransactionalMultiFrontWriterV12.CATEGORIES}
        rows["interval"].append({
            "transaction_id": transaction_id, "accepted_state_id": accepted_id,
            "stress_field_state_id": stress_id,
            "selected_proposal_id": None if candidate is None else candidate.proposal_id,
            "accepted": accepted, "resource_stop": resource_stop,
            "accepted_duration_s": accepted_duration_s,
            "selected_completion_time_s": (
                None if candidate is None else candidate.completion_time_s
            ),
            "event_endpoint_alignment_iterations": alignment_iterations,
            "endpoint_rate_clock_residual_s": endpoint_clock_residual_s,
        })
        rows["observation"].extend(x.__dict__ for x in batch.observations)
        rows["trial"].extend({
            "proposal_id": x.proposal.proposal_id, "accepted": x.accepted,
            "reason": x.reason,
        } for x in trial_by_proposal.values())
        rows["event"].append({
            "selected_event_ids": list(finalized.selected_event_ids),
            "unselected_pending_event_ids": list(finalized.unselected_pending_event_ids),
        })
        rows["owner"].extend({
            "owner_id": owner_id, "engine_id": region.process_engine_id,
            "local_process_coordinate_m": staged.process_engines[
                region.process_engine_id
            ].local_process_coordinate_m,
        } for owner_id, region in staged.process_regions.items())
        rows["ledger"].append({
            "conserved": staged.total_conserved_ledgers(),
            "signed": staged.total_signed_system_ledgers(),
        })
        atomic.begin(transaction_id)
        context_payload = {
            "schema": SCHEMA, "accepted_state_id": accepted_id,
            "stress_field_state_id": stress_id,
            "mechanics_source_identity": solved.mechanics_source_identity,
            "physical_time_s": context.physical_time_s,
            "accepted_opening_m": context.accepted_opening_m,
            "step_count": next_step,
        }
        output_write_failure = None
        if failure_stage == "output_staging":
            output_write_failure = 1
        elif failure_stage and failure_stage.startswith("output_staging_after_"):
            output_write_failure = int(failure_stage.rsplit("_", 1)[1])
        atomic.stage_complete(
            rows_by_category=rows, runtime=staged,
            accepted_fem_state=accepted_fem_after, context_payload=context_payload,
            fail_after_write=output_write_failure,
            inject_after_outputs=failure_stage == "output_to_checkpoint_staging",
        )
        if dry_run_discard:
            atomic.discard()
            output_published = False
        else:
            atomic.publish(
                inject_before_marker=failure_stage == "checkpoint_to_commit_marker"
            )
            output_published = True
            from .multifront_checkpoint_v12 import (
                write_accepted_boundary_checkpoint_v12,
            )
            context.latest_durable_checkpoint = (
                write_accepted_boundary_checkpoint_v12(
                    staged, context.checkpoint_path,
                    accepted_fem_state=accepted_fem_after,
                    accepted_stress_field=sigma_after,
                    mechanics_source_identity=solved.mechanics_source_identity,
                    physical_time_s=float(solved.physical_time_s),
                    accepted_opening_m=float(solved.accepted_load_m),
                    step_count=next_step,
                    context_restart={
                        "output_root": str(context.output_writer.root.resolve()),
                        "output_transaction_id": transaction_id,
                        "destination_cache_root": str(
                            context.destination_cache_root.resolve()
                        ),
                    },
                )
            )
        # Only after atomic publication (or deliberate dry-run discard) does
        # the in-memory accepted owner move to the staged generation.
        context.runtime = staged
        context.accepted_fem_state = accepted_fem_after
        frozen_sigma = np.asarray(sigma_after).copy(); frozen_sigma.setflags(write=False)
        context.accepted_stress_field = frozen_sigma
        context.stress_available = True
        context.mechanics_source_identity = solved.mechanics_source_identity
        context.step_count = next_step
        context.provisional_solved_state = None
        context.provisional_runtime = None
        context.provisional_observation_batch = None
        context.transaction_phase = "accepted"
        context.trial_cache.invalidate_after_interval(accepted_id)
        context.trial_cache.require_empty()
        context.previewed_clock_states.clear()
        context.output_writer.rollback()
        context._assert_identity()
        return IntegratedAcceptedIntervalV12(
            transaction_id, (
                "accepted_event" if accepted else "resource_stop" if resource_stop
                else "rejected_trial" if candidate is not None else "no_event"
            ), before_fingerprint, context.fingerprint,
            None if candidate is None else candidate.proposal_id,
            finalized.selected_event_ids, accepted_duration_s,
            None if candidate is None else float(candidate.completion_time_s),
            alignment_iterations, output_published,
            tuple(context.lifecycle_records),
        )
    except Exception:
        atomic.discard()
        context.runtime = before_runtime
        context.accepted_fem_state = before_fem
        if before_sigma is not None:
            before_sigma.setflags(write=False)
        context.accepted_stress_field = before_sigma
        context.stress_available = before_stress_available
        context.mechanics_source_identity = before_mechanics_source
        context.physical_time_s = before_physical_time
        context.accepted_opening_m = before_opening
        context.step_count = before_step
        context.actual_engines = before_engines
        context.interval_info_by_owner = before_info
        context.previewed_clock_states = before_previews
        context.trial_cache = before_trial_cache
        context.output_writer = TransactionalMultiFrontWriterV12.restore(before_output_writer)
        (
            context.provider_lookup_count, context.provider_solve_count,
            context.mechanics_solve_count, context.pf_workers_started,
        ) = before_counts
        context.provisional_solved_state = None
        context.provisional_runtime = None
        context.provisional_observation_batch = before_observation_batch
        context.transaction_phase = "accepted"
        raise


__all__ = [
    "BOUNDARY", "SCHEMA", "AcceptedTrialKey", "AtomicIntervalTransactionWriterV12",
    "CurrentSourceMultiFrontProductionContextV12", "ExactAcceptedTrialCacheV12",
    "IntegratedAcceptedIntervalV12",
    "StatefulProductionInterlock", "TransactionalMultiFrontWriterV12",
    "accepted_context_state_identity", "accepted_fem_state_identity",
    "build_arbitrary_region_request_v12",
    "build_stateful_production_hooks", "load_cached_provider_result_for_topology",
    "restore_stateful_production_context", "run_stateful_accepted_interval_v12",
    "stress_field_identity", "unavailable_stress_identity",
]
