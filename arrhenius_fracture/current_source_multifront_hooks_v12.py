"""Reviewed V12-to-V11 production hook binding and persistent factory."""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any

from .general_multifront_v12 import ProcessEngineState


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


class CurrentSourceHookBindingError(RuntimeError):
    pass


REQUIRED_SOURCE_BINDINGS = {
    "accepted_mesh_adaptation": (
        "arrhenius_fracture.adaptive_multitip_mesh_v11",
        "adapt_accepted_state_for_trials",
    ),
    "accepted_fem_solve": (
        "arrhenius_fracture.sharp_front_v11_branching",
        "solve_accepted_state_v12_hook",
    ),
    "exact_live_topology_request": (
        "arrhenius_fracture.sharp_front_v11_branching", "_request",
    ),
    "directional_local_J_observations": (
        "arrhenius_fracture.tip_directional_observation_v11",
        "observations_from_provider_by_front_candidate",
    ),
    "same_tip_tensor_conditioning": (
        "arrhenius_fracture.tip_directional_observation_v11",
        "require_same_tip_coupling",
    ),
    "accepted_tensor_binding": (
        "arrhenius_fracture.anisotropic_emission_v10174",
        "bind_explicit_accepted_tensor_drive",
    ),
    "process_engine_interval_evolution": (
        "arrhenius_fracture.sharp_front_v11_branching",
        "evolve_process_engine_v12_hook",
    ),
    "process_engine_interval_validation": (
        "arrhenius_fracture.process_update_semantics_v11",
        "require_full_accepted_interval_consumption",
    ),
    "directional_competition_preview": (
        "arrhenius_fracture.directional_competition_v11",
        "preview_directional_interval",
    ),
    "directional_competition_finalization": (
        "arrhenius_fracture.directional_competition_v11",
        "commit_directional_interval",
    ),
    "exact_topology_trial": (
        "arrhenius_fracture.topology_transaction_v11",
        "execute_topology_trial",
    ),
    "moving_frame_renewal": (
        "arrhenius_fracture.tip_directional_observation_v11",
        "apply_post_interval_event_renewal",
    ),
    "output_writer": (
        "arrhenius_fracture.stateful_multifront_production_v12",
        "TransactionalMultiFrontWriterV12",
    ),
    "checkpoint_writer": (
        "arrhenius_fracture.multifront_checkpoint_v12",
        "write_production_multifront_checkpoint",
    ),
}


def current_source_hook_map() -> dict[str, dict[str, Any]]:
    """Return symbol/file hashes and an unambiguous callable status."""
    result = {}
    for hook, (module_name, symbol_name) in REQUIRED_SOURCE_BINDINGS.items():
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        source_bytes = path.read_bytes()
        value = getattr(module, symbol_name, None)
        callable_binding = callable(value)
        diagnostic_only = "diagnostic" in module_name.lower() or "diagnostic" in symbol_name.lower()
        result[hook] = {
            "module": module_name,
            "symbol": symbol_name,
            "qualified_name": f"{module_name}.{symbol_name}",
            "source_file": str(path),
            "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "symbol_source_sha256": (
                None if not callable_binding else hashlib.sha256(
                    inspect.getsource(value).encode()
                ).hexdigest()
            ),
            "callable": callable_binding,
            "diagnostic_only": diagnostic_only,
            "status": (
                "QUALIFIED" if callable_binding and not diagnostic_only
                else "MISSING_REUSABLE_PRODUCTION_SYMBOL"
            ),
        }
    return dict(sorted(result.items()))


def build_current_source_multifront_production_hooks(context: Any):
    """Return one hook object whose closures share the reviewed context."""
    bindings = current_source_hook_map()
    failures = {
        key: value for key, value in bindings.items()
        if value["status"] != "QUALIFIED"
    }
    if failures:
        raise CurrentSourceHookBindingError(
            "current-source production hook factory is fail-closed: "
            + ", ".join(sorted(failures))
        )
    from .stateful_multifront_production_v12 import (
        CurrentSourceMultiFrontProductionContextV12,
        build_stateful_production_hooks,
    )
    if not isinstance(context, CurrentSourceMultiFrontProductionContextV12):
        raise CurrentSourceHookBindingError(
            "stateful composition requires one explicit persistent V12 context"
        )
    hooks = build_stateful_production_hooks(context)
    if hooks.context is not context:
        raise CurrentSourceHookBindingError("production hooks do not share one context")
    return hooks


def build_current_source_production_hooks(context: Any = None):
    """Compatibility name; never reconstruct hidden closure state."""
    return build_current_source_multifront_production_hooks(context)


def restore_complete_current_source_engine(state: ProcessEngineState) -> Any:
    """Allocate the pinned production classes and restore all checkpoint fields.

    Allocation invokes no initializer and therefore no stochastic draw.  The
    complete archived fields are then restored by the reviewed V11 restorer,
    which rehydrates the runtime methods from the explicit process-model registry.
    The class allow-list prevents arbitrary checkpoint-controlled imports.
    """
    if state.engine_class != "AuditedPersistentSiteStateResolvedTipEngine":
        raise CurrentSourceHookBindingError(
            f"unreviewed process-engine class: {state.engine_class}"
        )
    payload = state.complete_checkpoint_payload()
    if payload.get("mpz_type") != "UnifiedMPZState":
        raise CurrentSourceHookBindingError(
            f"unreviewed MPZ class: {payload.get('mpz_type')}"
        )
    from .persistent_site_audited_engine_v10221 import (
        AuditedPersistentSiteStateResolvedTipEngine,
    )
    from .unified_mpz import UnifiedMPZState

    engine = AuditedPersistentSiteStateResolvedTipEngine.__new__(
        AuditedPersistentSiteStateResolvedTipEngine
    )
    engine.mpz = UnifiedMPZState.__new__(UnifiedMPZState)
    return state.restore_into(engine)


__all__ = [
    "BOUNDARY", "CurrentSourceHookBindingError", "REQUIRED_SOURCE_BINDINGS",
    "build_current_source_multifront_production_hooks",
    "build_current_source_production_hooks", "current_source_hook_map",
    "restore_complete_current_source_engine",
]
