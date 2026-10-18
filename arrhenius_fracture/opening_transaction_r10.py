"""Accepted-time and material-cell ownership for R10 cohesive opening.

Trial hazard/action never mutates accepted state.  Thermal events consume their
actual positive waiting time; only explicitly mechanical completions may share
one physical instant.  A branch loss is a transit state, not crack advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable


@dataclass
class StochasticOwner:
    threshold: float
    action: float = 0.0
    accepted_events: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("threshold must be finite and positive")
        if not math.isfinite(self.action) or self.action < 0.0:
            raise ValueError("action must be finite and nonnegative")


@dataclass
class MaterialCellState:
    cell_id: int
    branch: str = "BONDED"
    separation_m: float = 0.0
    completed: bool = False
    accepted_event_time_s: float | None = None
    history: list[dict] = field(default_factory=list)


def locate_constant_rate_event(owner: StochasticOwner, rate_s: float,
                               duration_s: float) -> dict:
    """Return an immutable proposal for the earliest event in one interval."""
    rate = max(float(rate_s), 0.0)
    duration = max(float(duration_s), 0.0)
    available = rate * duration
    remaining = owner.threshold - owner.action
    if rate <= 0.0 or available < remaining:
        return {"event": False, "consumed_time_s": duration,
                "accepted_action": owner.action + available,
                "discarded_trial_endpoint_action": owner.action + available}
    wait = max(remaining / rate, 0.0)
    return {"event": True, "consumed_time_s": wait,
            "accepted_action": owner.threshold,
            "discarded_trial_endpoint_action": owner.action + available}


def accept_thermal_proposal(owner: StochasticOwner, proposal: dict,
                            next_threshold: float) -> None:
    if not proposal["event"]:
        owner.action = float(proposal["accepted_action"])
        return
    if next_threshold <= 0.0 or not math.isfinite(next_threshold):
        raise ValueError("next threshold must be finite and positive")
    owner.action = 0.0
    owner.threshold = float(next_threshold)
    owner.accepted_events += 1


def consume_constant_rate_successors(
    duration_s: float,
    owner: StochasticOwner,
    rate_for_cell: Callable[[int], float],
    next_threshold: Callable[[], float],
    start_cell_id: int = 0,
) -> dict:
    """Manufactured exact consumer for sequential constant-rate cell clocks."""
    elapsed = 0.0
    cell = int(start_cell_id)
    receipts = []
    while elapsed < duration_s:
        proposal = locate_constant_rate_event(
            owner, rate_for_cell(cell), duration_s - elapsed)
        elapsed += float(proposal["consumed_time_s"])
        if not proposal["event"]:
            accept_thermal_proposal(owner, proposal, owner.threshold)
            break
        receipts.append({"cell_id": cell, "accepted_time_s": elapsed,
                         "accepted_action": proposal["accepted_action"],
                         "discarded_trial_endpoint_action":
                             proposal["discarded_trial_endpoint_action"]})
        accept_thermal_proposal(owner, proposal, next_threshold())
        cell += 1
    return {"elapsed_s": elapsed, "next_cell_id": cell,
            "receipts": receipts, "remaining_s": duration_s - elapsed}


def update_cell_branch(cell: MaterialCellState, branch: str,
                       separation_m: float, *, completion: bool,
                       physical_time_s: float) -> None:
    cell.branch = str(branch)
    cell.separation_m = float(separation_m)
    cell.completed = bool(completion)
    cell.history.append({"physical_time_s": float(physical_time_s),
                         "branch": cell.branch,
                         "separation_m": cell.separation_m,
                         "completion": cell.completed})


def commit_completed_cell(cell: MaterialCellState, *, physical_time_s: float,
                          successor_branch: str,
                          successor_separation_m: float) -> tuple[dict, MaterialCellState]:
    if not cell.completed:
        raise RuntimeError("branch loss/transit cannot commit an incomplete material cell")
    cell.accepted_event_time_s = float(physical_time_s)
    receipt = {"departed_cell_id": cell.cell_id,
               "departed_branch": cell.branch,
               "committed_separation_m": cell.separation_m,
               "accepted_event_time_s": float(physical_time_s)}
    successor = MaterialCellState(
        cell_id=cell.cell_id + 1,
        branch=str(successor_branch),
        separation_m=float(successor_separation_m),
        completed=False,
    )
    return receipt, successor


__all__ = [
    "MaterialCellState", "StochasticOwner", "accept_thermal_proposal",
    "commit_completed_cell", "consume_constant_rate_successors",
    "locate_constant_rate_event", "update_cell_branch",
]
