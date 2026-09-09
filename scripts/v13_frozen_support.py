"""Frozen-state fixtures using the actual persistent-source constructor."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import pickle
from pathlib import Path

import numpy as np

from arrhenius_fracture.current_source_multifront_hooks_v12 import restore_complete_current_source_engine
from arrhenius_fracture.general_multifront_v12 import ProcessEngineState
from arrhenius_fracture.multifront_checkpoint_v12 import load_accepted_boundary_checkpoint_v12
from arrhenius_fracture.persistent_site_audited_engine_v10221 import AuditedPersistentSiteStateResolvedTipEngine
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine


ATLAS = Path(__file__).resolve().parents[3] / "analysis_outputs/pf_current_source_multifront_field_atlas_300K_1000K"


def saved_owner(case="Peak_300K"):
    checkpoint = load_accepted_boundary_checkpoint_v12(
        ATLAS / "continued_1000um_runs_v2" / case / "checkpoint/latest.v12.pkl"
    )
    return checkpoint, next(iter(checkpoint.runtime.process_engines.values()))


@contextmanager
def initialized_engine(payload):
    """Run the genuine constructor with the archived physical configuration.

    Constructor draws are confined to a new fixture; accepted RNG/state is
    copied over afterwards. The A-side uses no runtime-rehydration function.
    Class-level defaults are restored even on failure.
    """
    cls = AuditedPersistentSiteStateResolvedTipEngine
    fields, mpz = payload["engine_fields"], payload["mpz_fields"]
    defaults = {
        "_default_tip_config": fields["tip_cfg"],
        "_hazard_config_default": fields["hazard_cfg"],
        "_avalanche_config_default": fields["avalanche_cfg"],
        "_anisotropic_config_default": fields["anisotropic_cfg"],
        "_persistent_site_config_default": mpz["_persistent_site_cfg"],
        "_state_family_default": fields["_state_kernel_family"],
        "_signed_kernel_default": fields["_state_kernel_family"],
        "_signed_transport_mode_default": mpz["_signed_transport_mode"],
        "_campaign_backstress_scale_default": mpz["_campaign_backstress_scale"],
        "_campaign_refresh_scale_default": mpz["_campaign_refresh_scale"],
        "_next_engine_id": int(fields["_engine_id"]),
        "_audit_records": [],
    }
    old = {key: cls.__dict__.get(key) for key in defaults}
    present = set(cls.__dict__)
    try:
        for key, value in defaults.items():
            setattr(cls, key, copy.deepcopy(value))
        engine = cls(
            *(copy.deepcopy(fields[name]) for name in ("f", "cb", "eb", "G", "nu", "b", "manifest")),
            copy.deepcopy(mpz["cfg"]),
        )
        # Transfer the identical frozen physical state while retaining the
        # constructor's actual installed methods (including bound self links).
        data = copy.deepcopy({"engine": fields, "mpz": mpz})
        for name, owner in (("engine", engine), ("mpz", engine.mpz)):
            for key in tuple(vars(owner)):
                if key != "mpz" and not callable(vars(owner)[key]) and key not in data[name]:
                    delattr(owner, key)
            for key, value in data[name].items():
                setattr(owner, key, value)
        yield engine
    finally:
        for key in defaults:
            if key in present:
                setattr(cls, key, old[key])
            else:
                delattr(cls, key)


def recapture(engine):
    return ProcessEngineState.from_v11_payload(
        engine_id="engine:parity", source_state_id="source:parity",
        payload=_capture_shared_engine(engine), active_ledgers={}, wake_ledgers={},
        signed_system_ledgers={}, update_count=0, event_renewal_count=0,
        local_process_coordinate_m=float(engine.mpz.advance_total_m),
    )


def field_differences(a, b, path="root"):
    """Recursive exact state comparison, including RNG and kernel objects."""
    if type(a) is not type(b):
        return [path + ":type"]
    if isinstance(a, np.ndarray):
        return [] if a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True) else [path]
    if isinstance(a, np.random.Generator):
        return field_differences(a.bit_generator.state, b.bit_generator.state, path + ".rng")
    if isinstance(a, dict):
        if set(a) != set(b):
            return [path + ":keys"]
        return [p for key in a for p in field_differences(a[key], b[key], path + "." + str(key))]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + ":length"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in field_differences(x, y, f"{path}[{i}]")]
    if hasattr(a, "__dict__"):
        return field_differences(vars(a), vars(b), path)
    if isinstance(a, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [path]


def rng_hash(engine):
    return hashlib.sha256(pickle.dumps(engine._hazard_rng.bit_generator.state, protocol=5)).hexdigest()


def frozen_process_step(engine, temperature, duration, K):
    """Actual production process step with directional cleavage ownership."""
    engine._directional_topology_owns_cleavage = True
    # The production hook freezes the legacy topology threshold; directional
    # clocks, not this compatibility threshold, own accepted cleavage events.
    engine.hazard_threshold_action = 1e300
    before = rng_hash(engine)
    result = engine.step(float(K), float(temperature), float(duration))
    if rng_hash(engine) != before:
        raise RuntimeError("process-only interval changed the baseline RNG")
    if abs(float(result.get("da", 0.0))) > 1e-18:
        raise RuntimeError("process-only interval advanced topology")
    return result
