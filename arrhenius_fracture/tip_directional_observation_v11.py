"""Physical ownership records for v11 multi-tip directional observations.

The live topology provider returns measurements grouped by physical tip.  This
module keeps that grouping intact all the way to the shared process-state
update.  In particular, a scalar configurational drive may never be detached
from the tip coordinate and accepted FEM state that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping, Sequence


MODEL_ID = "v11.tip_directional_observation/1"


@dataclass(frozen=True)
class TipDirectionalObservation:
    tip_id: str
    parent_branch_id: str | None
    candidate_id: str
    tip_xy_m: tuple[float, float]
    branch_arclength_m: float
    projected_reach_m: float
    signed_local_J_J_per_m2: float
    kinetic_J_J_per_m2: float
    marginal_J_J_per_m2: float | None
    directional_K_Pa_sqrt_m: float
    directional_rate_per_s: float
    local_contour_valid: bool
    local_contour_invalid_reason: str | None = None
    accepted_state_id: str = ""
    stress_field_state_id: str = ""
    pre_event_topology_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.tip_id or not self.candidate_id:
            raise ValueError("tip_id and candidate_id are required")
        if len(self.tip_xy_m) != 2 or not all(math.isfinite(float(x)) for x in self.tip_xy_m):
            raise ValueError("tip_xy_m must contain two finite coordinates")
        for value in (
            self.branch_arclength_m, self.projected_reach_m,
            self.signed_local_J_J_per_m2, self.kinetic_J_J_per_m2,
            self.directional_K_Pa_sqrt_m, self.directional_rate_per_s,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("directional observations must be finite")
        if self.marginal_J_J_per_m2 is not None and not math.isfinite(
            float(self.marginal_J_J_per_m2)
        ):
            raise ValueError("marginal_J_J_per_m2 must be finite when present")
        if not self.accepted_state_id or not self.stress_field_state_id:
            raise ValueError("accepted and stress-field state identities are required")
        if not self.pre_event_topology_fingerprint:
            raise ValueError("pre-event topology fingerprint is required")

    # Backward-readable names are properties only.  The serialized/dataclass
    # contract uses the explicit V4 ownership vocabulary above.
    @property
    def signed_J_J_per_m2(self) -> float:
        return self.signed_local_J_J_per_m2

    @property
    def rate_per_s(self) -> float:
        return self.directional_rate_per_s

    @property
    def topology_fingerprint(self) -> str:
        return self.pre_event_topology_fingerprint

    def with_kinetics(
        self, *, kinetic_J_J_per_m2: float, marginal_J_J_per_m2: float | None,
        directional_K_Pa_sqrt_m: float, rate_per_s: float,
    ) -> "TipDirectionalObservation":
        return replace(
            self,
            kinetic_J_J_per_m2=float(kinetic_J_J_per_m2),
            marginal_J_J_per_m2=(
                None if marginal_J_J_per_m2 is None
                else float(marginal_J_J_per_m2)
            ),
            directional_K_Pa_sqrt_m=float(directional_K_Pa_sqrt_m),
            directional_rate_per_s=float(rate_per_s),
        )


def _branch_arclength(branch: Any) -> float:
    return float(sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(branch.path, branch.path[1:])
    ))


def candidate_tip_owners(crack_network: Any, candidate_ids: Iterable[str]) -> dict[str, str]:
    """Return the fail-closed candidate-to-active-tip ownership map.

    Before branch birth, all candidates legitimately belong to the sole active
    tip.  In a multi-tip state every active branch must carry one unique
    ``local_state.candidate_id`` and every requested candidate must have exactly
    one owner.
    """
    candidates = tuple(str(item) for item in candidate_ids)
    if len(set(candidates)) != len(candidates):
        raise RuntimeError("duplicate candidate IDs are ambiguous")
    tips = tuple(str(item) for item in crack_network.active_tip_ids)
    if not tips:
        raise RuntimeError("directional observations require an active tip")
    if len(tips) == 1:
        return {candidate: tips[0] for candidate in candidates}

    owners: dict[str, str] = {}
    for tip_id in tips:
        branch = crack_network.branch(tip_id)
        candidate_id = branch.local_state.get("candidate_id")
        if candidate_id is None:
            raise RuntimeError(
                f"active multi-tip branch {tip_id} has no candidate_id ownership"
            )
        candidate_id = str(candidate_id)
        if candidate_id in owners:
            raise RuntimeError(f"candidate {candidate_id} has duplicate tip owners")
        owners[candidate_id] = tip_id
    expected = set(candidates)
    if set(owners) != expected:
        missing = sorted(expected - set(owners))
        extra = sorted(set(owners) - expected)
        raise RuntimeError(
            f"candidate-to-tip ownership is not bijective: missing={missing}, extra={extra}"
        )
    return owners


def _provider_tip_xy(provider_tip: Mapping[str, Any]) -> tuple[float, float]:
    for key in ("tip_xy_m", "tip_xy", "tip"):
        if key in provider_tip:
            value = provider_tip[key]
            return (float(value[0]), float(value[1]))
    raise RuntimeError("provider tip row does not contain physical tip coordinates")


def observations_from_provider(
    crack_network: Any,
    provider_tips: Sequence[Mapping[str, Any]],
    candidate_ids: Iterable[str],
    *,
    accepted_state_id: str,
    topology_fingerprint: str,
    stress_field_state_id: str | None = None,
    coordinate_tolerance_m: float = 1.0e-12,
) -> tuple[TipDirectionalObservation, ...]:
    """Build one observation per candidate without flattening tip ownership."""
    candidates = tuple(str(item) for item in candidate_ids)
    owners = candidate_tip_owners(crack_network, candidates)
    rows: list[TipDirectionalObservation] = []
    seen: set[str] = set()
    for provider_tip in provider_tips:
        xy = _provider_tip_xy(provider_tip)
        possible = []
        for tip_id in crack_network.active_tip_ids:
            branch = crack_network.branch(tip_id)
            if math.dist(tuple(map(float, branch.tip)), xy) <= coordinate_tolerance_m:
                possible.append(str(tip_id))
        if len(possible) != 1:
            raise RuntimeError(
                f"provider tip coordinate {xy} has {len(possible)} physical owners"
            )
        tip_id = possible[0]
        branch = crack_network.branch(tip_id)
        for item in provider_tip.get("directional", ()):
            candidate_id = str(item["candidate_id"])
            if candidate_id not in owners:
                continue
            if owners[candidate_id] != tip_id:
                raise RuntimeError(
                    f"candidate {candidate_id} measured at {tip_id}, owned by {owners[candidate_id]}"
                )
            if candidate_id in seen:
                raise RuntimeError(f"duplicate provider observation for {candidate_id}")
            seen.add(candidate_id)
            signed = float(
                item["J_local_signed_J_per_m2"]
                if "J_local_signed_J_per_m2" in item
                else item["signed_J_J_per_m2"]
            )
            kinetic = max(signed, 0.0)
            rows.append(TipDirectionalObservation(
                tip_id=tip_id,
                parent_branch_id=(
                    None if branch.parent_branch_id is None
                    else str(branch.parent_branch_id)
                ),
                candidate_id=candidate_id,
                tip_xy_m=xy,
                branch_arclength_m=_branch_arclength(branch),
                projected_reach_m=float(branch.tip[0]) - float(branch.path[0][0]),
                signed_local_J_J_per_m2=signed,
                kinetic_J_J_per_m2=kinetic,
                marginal_J_J_per_m2=None,
                directional_K_Pa_sqrt_m=float(item.get("K_directional_Pa_sqrt_m", 0.0)),
                directional_rate_per_s=0.0,
                local_contour_valid=bool(item.get("local_J_valid", True)),
                local_contour_invalid_reason=item.get("local_J_invalid_reason"),
                accepted_state_id=str(accepted_state_id),
                stress_field_state_id=str(
                    accepted_state_id if stress_field_state_id is None
                    else stress_field_state_id
                ),
                pre_event_topology_fingerprint=str(topology_fingerprint),
            ))
    if seen != set(candidates):
        raise RuntimeError(
            f"provider observations are incomplete: missing={sorted(set(candidates)-seen)}"
        )
    return tuple(sorted(rows, key=lambda item: item.candidate_id))


def observations_from_provider_by_front_candidate(
    crack_network: Any,
    provider_tips: Sequence[Mapping[str, Any]],
    candidate_ids_by_tip: Mapping[str, Iterable[str]],
    *,
    accepted_state_id: str,
    topology_fingerprint: str,
    stress_field_state_id: str | None = None,
    coordinate_tolerance_m: float = 1.0e-12,
) -> tuple[TipDirectionalObservation, ...]:
    """Apply the V11 observation equations to explicit front/candidate pairs.

    Independent fronts may each evaluate the same crystallographic candidate.
    Attached binary daughters still pass one physically assigned candidate.
    """
    expected_tips = tuple(str(item) for item in crack_network.active_tip_ids)
    requested = {
        str(tip): tuple(sorted(str(candidate) for candidate in candidates))
        for tip, candidates in candidate_ids_by_tip.items()
    }
    if set(requested) != set(expected_tips):
        raise RuntimeError("explicit observation ownership omits an active tip")
    rows: list[TipDirectionalObservation] = []
    seen: set[tuple[str, str]] = set()
    for provider_tip in provider_tips:
        xy = _provider_tip_xy(provider_tip)
        possible = [
            tip_id for tip_id in expected_tips
            if math.dist(
                tuple(map(float, crack_network.branch(tip_id).tip)), xy
            ) <= coordinate_tolerance_m
        ]
        if len(possible) != 1:
            raise RuntimeError(
                f"provider tip coordinate {xy} has {len(possible)} physical owners"
            )
        tip_id = possible[0]
        branch = crack_network.branch(tip_id)
        allowed = set(requested[tip_id])
        for item in provider_tip.get("directional", ()):
            candidate_id = str(item["candidate_id"])
            if candidate_id not in allowed:
                continue
            key = tip_id, candidate_id
            if key in seen:
                raise RuntimeError(
                    f"duplicate provider observation for {tip_id}/{candidate_id}"
                )
            seen.add(key)
            signed = float(
                item["J_local_signed_J_per_m2"]
                if "J_local_signed_J_per_m2" in item
                else item["signed_J_J_per_m2"]
            )
            kinetic = max(signed, 0.0)
            rows.append(TipDirectionalObservation(
                tip_id=tip_id,
                parent_branch_id=(
                    None if branch.parent_branch_id is None
                    else str(branch.parent_branch_id)
                ),
                candidate_id=candidate_id, tip_xy_m=xy,
                branch_arclength_m=_branch_arclength(branch),
                projected_reach_m=float(branch.tip[0]) - float(branch.path[0][0]),
                signed_local_J_J_per_m2=signed,
                kinetic_J_J_per_m2=kinetic,
                marginal_J_J_per_m2=None,
                directional_K_Pa_sqrt_m=float(
                    item.get("K_directional_Pa_sqrt_m", 0.0)
                ),
                directional_rate_per_s=0.0,
                local_contour_valid=bool(item.get("local_J_valid", True)),
                local_contour_invalid_reason=item.get("local_J_invalid_reason"),
                accepted_state_id=str(accepted_state_id),
                stress_field_state_id=str(
                    accepted_state_id if stress_field_state_id is None
                    else stress_field_state_id
                ),
                pre_event_topology_fingerprint=str(topology_fingerprint),
            ))
    expected_pairs = {
        (tip, candidate)
        for tip, candidates in requested.items()
        for candidate in candidates
    }
    if seen != expected_pairs:
        raise RuntimeError(
            "provider observations are incomplete for explicit front/candidate ownership: "
            f"missing={sorted(expected_pairs - seen)}"
        )
    return tuple(sorted(rows, key=lambda item: (item.tip_id, item.candidate_id)))


def select_controlling_observation(
    observations: Sequence[TipDirectionalObservation],
) -> TipDirectionalObservation:
    """Select one complete tuple, invariant to active-tip and branch-ID order."""
    if not observations:
        raise RuntimeError("a controlling observation is required")
    candidates = [item.candidate_id for item in observations]
    if len(set(candidates)) != len(candidates):
        raise RuntimeError("controlling observations contain duplicate candidates")
    # Candidate identity and coordinates are physical tie-breakers.  Branch IDs
    # are deliberately excluded, so lexical branch ordering cannot affect state.
    return max(
        observations,
        key=lambda item: (
            float(item.directional_K_Pa_sqrt_m),
            item.candidate_id,
            float(item.tip_xy_m[0]),
            float(item.tip_xy_m[1]),
        ),
    )


def require_same_tip_coupling(
    controlling_scalar_K_tip_id: str, tensor_probe_tip_id: str,
) -> None:
    if str(controlling_scalar_K_tip_id) != str(tensor_probe_tip_id):
        raise RuntimeError(
            "shared process update mixed scalar-K and tensor-probe tip ownership"
        )


def require_observation_state_contract(
    observation: TipDirectionalObservation,
    *,
    accepted_state_id: str,
    stress_field_state_id: str,
    pre_event_topology_fingerprint: str,
) -> None:
    """Reject scalar/tensor observations detached from their accepted state."""
    if observation.accepted_state_id != str(accepted_state_id):
        raise RuntimeError("directional observation accepted-state identity mismatch")
    if observation.stress_field_state_id != str(stress_field_state_id):
        raise RuntimeError("directional observation stress-field identity mismatch")
    if observation.pre_event_topology_fingerprint != str(pre_event_topology_fingerprint):
        raise RuntimeError("directional observation pre-event topology mismatch")


def require_uncontaminated_replay_checkpoint(
    checkpoint_step: int,
    *,
    first_selected_event_step: int,
) -> None:
    """Refuse restart states at or after the first defect-affected event."""
    if int(checkpoint_step) >= int(first_selected_event_step):
        raise RuntimeError(
            "replay checkpoint is at or after the first selected event and "
            "contains defect-affected process-state history"
        )


def marginal_trial_origin(
    crack_network: Any, observation: TipDirectionalObservation,
) -> tuple[str, tuple[float, float]]:
    """Return the owning physical tip for an invalid-contour marginal trial."""
    branch = crack_network.branch(observation.tip_id)
    if len(crack_network.active_tip_ids) > 1:
        candidate_id = branch.local_state.get("candidate_id")
        if str(candidate_id) != observation.candidate_id:
            raise RuntimeError(
                "invalid-contour marginal trial does not follow candidate ownership"
            )
    return observation.tip_id, tuple(map(float, branch.tip))


def apply_post_interval_event_renewal(
    process_state: Any,
    interval_info: dict[str, Any],
    *,
    event_selected: bool,
    event_distance_m: float,
) -> dict[str, Any]:
    """Apply topology-owned moving-frame renewal once after interval evolution."""
    if not event_selected:
        interval_info["event_moving_frame_advance_m"] = 0.0
        interval_info["event_moving_frame_renewal_count"] = 0
        return {}
    if float(event_distance_m) <= 0.0:
        raise RuntimeError("selected topology event requires a positive renewal distance")
    if int(interval_info.get("event_moving_frame_renewal_count", 0)) != 0:
        raise RuntimeError("selected topology event renewal was already applied")
    if abs(float(interval_info.get("da", 0.0))) > 1.0e-18:
        raise RuntimeError("selected topology event would double-apply moving-frame renewal")
    if interval_info.get("interval_evolved") is not True:
        raise RuntimeError("event renewal requires completed pre-event interval evolution")
    def totals(owner: Any) -> dict[str, Any]:
        def by_system(name: str) -> list[float]:
            value = getattr(owner, name, None)
            if value is None:
                return []
            array = __import__("numpy").asarray(value, dtype=float)
            return [float(x) for x in array.reshape(array.shape[0], -1).sum(axis=1)]

        result = {
            "active_mobile_by_system": by_system("mobile"),
            "active_retained_by_system": by_system("retained"),
            "wake_mobile_by_system": by_system("wake_mobile"),
            "wake_retained_by_system": by_system("wake_retained"),
            "escaped_total": float(getattr(owner, "escaped_total", 0.0)),
            "recovered_total": float(getattr(owner, "recovered_total", 0.0)),
            "discarded_mobile_total": float(getattr(owner, "wake_discarded_mobile_total", 0.0)),
            "discarded_retained_total": float(getattr(owner, "wake_discarded_retained_total", 0.0)),
        }
        signed = {}
        for region, stem in (("active", ""), ("wake", "wake_")):
            for family in ("mobile", "retained"):
                positive = by_system(f"{stem}{family}_positive")
                negative = by_system(f"{stem}{family}_negative")
                if positive and len(positive) == len(negative):
                    signed[f"{region}_{family}_signed_by_system"] = [
                        p - n for p, n in zip(positive, negative)
                    ]
        result.update(signed)
        return result

    before = totals(process_state)
    renewal = process_state.advance(float(event_distance_m))
    after = totals(process_state)

    def delta(name: str) -> list[float]:
        return [b - a for a, b in zip(before.get(name, ()), after.get(name, ()))]

    active_mobile_change = delta("active_mobile_by_system")
    active_retained_change = delta("active_retained_by_system")
    wake_mobile_change = delta("wake_mobile_by_system")
    wake_retained_change = delta("wake_retained_by_system")
    escaped_change = after["escaped_total"] - before["escaped_total"]
    recovered_change = after["recovered_total"] - before["recovered_total"]
    discarded_mobile_change = after["discarded_mobile_total"] - before["discarded_mobile_total"]
    discarded_retained_change = after["discarded_retained_total"] - before["discarded_retained_total"]
    unsigned_residual = (
        sum(active_mobile_change) + sum(active_retained_change)
        + sum(wake_mobile_change) + sum(wake_retained_change)
        + escaped_change + recovered_change
        + discarded_mobile_change + discarded_retained_change
    )
    signed_residual = []
    signed_names = (
        "active_mobile_signed_by_system", "active_retained_signed_by_system",
        "wake_mobile_signed_by_system", "wake_retained_signed_by_system",
    )
    if all(before.get(name) and after.get(name) for name in signed_names):
        signed_residual = [sum(values) for values in zip(*(
            delta(name) for name in signed_names
        ))]
    conservation = {
        "active_mobile_change_by_system": active_mobile_change,
        "active_retained_change_by_system": active_retained_change,
        "wake_mobile_change_by_system": wake_mobile_change,
        "wake_retained_change_by_system": wake_retained_change,
        "recovered_change": recovered_change,
        "escaped_change": escaped_change,
        "discarded_mobile_change": discarded_mobile_change,
        "discarded_retained_change": discarded_retained_change,
        "active_plus_wake_plus_sinks_residual": unsigned_residual,
        "signed_residual_by_system": signed_residual,
    }
    interval_info["event_moving_frame_advance_m"] = float(event_distance_m)
    interval_info["event_moving_frame_renewal_count"] = 1
    interval_info["event_moving_frame_renewal"] = renewal
    interval_info["event_moving_frame_conservation"] = conservation
    return renewal


def realized_topology_arm_lengths(
    pre_event_network: Any,
    post_event_network: Any,
    member_candidate_ids: Sequence[str],
) -> tuple[float, ...]:
    """Measure accepted per-arm path additions from committed geometry."""
    candidates = tuple(str(item) for item in member_candidate_ids)
    if not candidates:
        raise RuntimeError("accepted topology event has no member candidates")

    def length(branch: Any) -> float:
        return float(sum(math.dist(a, b) for a, b in zip(branch.path, branch.path[1:])))

    pre_by_candidate = {
        str(branch.local_state.get("candidate_id")): branch
        for branch in pre_event_network.branches
        if branch.local_state.get("candidate_id") is not None
    }
    post_by_candidate = {
        str(branch.local_state.get("candidate_id")): branch
        for branch in post_event_network.branches
        if branch.local_state.get("candidate_id") is not None
    }
    measured: list[float] = []
    for candidate in candidates:
        post = post_by_candidate.get(candidate)
        if post is None:
            # Before branch birth, a one-arm transaction extends the sole root.
            if len(candidates) != 1 or len(pre_event_network.active_tip_ids) != 1:
                raise RuntimeError(f"accepted arm {candidate} has no post-event owner")
            pre = pre_event_network.branch(pre_event_network.active_tip_ids[0])
            post = post_event_network.branch(pre.branch_id)
            advance = length(post) - length(pre)
        else:
            pre = pre_by_candidate.get(candidate)
            advance = length(post) - (0.0 if pre is None else length(pre))
        if not math.isfinite(advance) or advance <= 0.0:
            raise RuntimeError(f"accepted arm {candidate} has nonpositive realized advance")
        measured.append(float(advance))
    return tuple(measured)


def select_shared_process_renewal_distance(
    realized_arm_lengths_m: Sequence[float],
) -> float:
    """Use maximum accepted tip advance, never total new crack length.

    A shared cluster represents one moving process-zone frame.  Two daughter
    arms therefore cause one renewal by their maximum realized tip advance;
    summing arm lengths would double-count a two-arm birth.
    """
    lengths = tuple(float(item) for item in realized_arm_lengths_m)
    if not lengths or not all(math.isfinite(item) and item > 0.0 for item in lengths):
        raise RuntimeError("realized topology arm lengths must be finite and positive")
    return max(lengths)


def selected_event_owner(
    proposal: Any | None,
    observations: Sequence[TipDirectionalObservation],
) -> tuple[str | None, str | None]:
    """Return selected-event candidate/tip independently of process control."""
    if proposal is None:
        return None, None
    by_candidate = {item.candidate_id: item for item in observations}
    member_ids = tuple(str(item) for item in proposal.member_candidate_ids)
    missing = [item for item in member_ids if item not in by_candidate]
    if missing:
        raise RuntimeError(f"selected event has unowned candidates: {missing}")
    if len(member_ids) == 1:
        item = by_candidate[member_ids[0]]
        return item.candidate_id, item.tip_id
    tips = {by_candidate[item].tip_id for item in member_ids}
    if len(tips) != 1:
        raise RuntimeError("one topology transaction spans multiple pre-event tips")
    return "+".join(sorted(member_ids)), next(iter(tips))


def serialize_directional_observation(
    observation: TipDirectionalObservation,
    *,
    controlling_scalar_K_tip_id: str,
    tensor_probe_tip_id: str,
    selected_event_tip_id: str | None,
    process_owner_id: str,
    pre_event_state_id: str,
    post_event_state_id: str,
) -> dict[str, Any]:
    require_same_tip_coupling(controlling_scalar_K_tip_id, tensor_probe_tip_id)
    return {
        "tip_id": observation.tip_id,
        "parent_branch_id": observation.parent_branch_id,
        "candidate_id": observation.candidate_id,
        "tip_xy_m": list(observation.tip_xy_m),
        "branch_arclength_m": observation.branch_arclength_m,
        "projected_reach_m": observation.projected_reach_m,
        "accepted_state_id": observation.accepted_state_id,
        "stress_field_state_id": observation.stress_field_state_id,
        "pre_event_topology_fingerprint": observation.pre_event_topology_fingerprint,
        "signed_local_J_J_per_m2": observation.signed_local_J_J_per_m2,
        "kinetic_J_J_per_m2": observation.kinetic_J_J_per_m2,
        "marginal_J_J_per_m2": observation.marginal_J_J_per_m2,
        "directional_K_Pa_sqrt_m": observation.directional_K_Pa_sqrt_m,
        "directional_rate_per_s": observation.directional_rate_per_s,
        "local_contour_valid": observation.local_contour_valid,
        "local_contour_invalid_reason": observation.local_contour_invalid_reason,
        "controlling_scalar_K_tip_id": str(controlling_scalar_K_tip_id),
        "tensor_probe_tip_id": str(tensor_probe_tip_id),
        "selected_event_tip_id": selected_event_tip_id,
        "process_owner_id": str(process_owner_id),
        "pre_event_state_id": str(pre_event_state_id),
        "post_event_state_id": str(post_event_state_id),
    }


__all__ = [
    "MODEL_ID", "TipDirectionalObservation", "candidate_tip_owners",
    "apply_post_interval_event_renewal", "observations_from_provider",
    "observations_from_provider_by_front_candidate",
    "marginal_trial_origin", "require_same_tip_coupling",
    "require_observation_state_contract",
    "require_uncontaminated_replay_checkpoint",
    "realized_topology_arm_lengths", "select_shared_process_renewal_distance",
    "select_controlling_observation", "selected_event_owner",
    "serialize_directional_observation",
]
