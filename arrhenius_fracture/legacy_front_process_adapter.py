"""Black-box adapter for the qualified legacy emission/cleavage front process.

The adapter owns no kinetic equations.  It restores the canonical engine,
invokes its existing ``step`` implementation on an isolated instance, and
captures the resulting engine through the established checkpoint hooks.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import pickle
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA = "unified-2d.legacy-front-process-adapter/1"


def _hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


@dataclass(frozen=True)
class LegacyFrontAdvance:
    advanced_front_state: Mapping[str, Any]
    cleavage_event_occurred: bool
    cleavage_event_time_s: float | None
    updated_emission_state: Mapping[str, Any]
    updated_r_tip_m: float
    updated_shielding_Pa_sqrt_m: float
    updated_back_stress_Pa: float
    updated_mpz_state: Mapping[str, Any]
    threshold_identity: Mapping[str, Any]
    rng_identity: Mapping[str, Any]
    audit: Mapping[str, Any]
    schema: str = SCHEMA


class LegacyFrontProcessAdapter:
    """One canonical factory and one exact engine path for every front kind."""

    def __init__(self, material: Any, material_bundle: Any):
        from .voiding_production_v5 import fresh_sharp_front_engine
        self.material = material
        self.bundle = material_bundle
        probe = fresh_sharp_front_engine(material, material_bundle)
        self.factory_identity = MappingProxyType({
            "material_bundle_id": material_bundle.bundle_id,
            "material_bundle_sha256": material_bundle.bundle_hash,
            "core_model_id": getattr(probe, "_unified_core_model_id", None),
            "engine_type": type(probe).__qualname__,
            "factory": "canonical_front_engine",
        })

    def _restore(self, accepted_front_state: Mapping[str, Any]):
        from .voiding_production_v5 import restore_sharp_front_engine
        return restore_sharp_front_engine(
            self.material, self.bundle, accepted_front_state,
        )

    @staticmethod
    def _mpz_audit(engine: Any) -> Mapping[str, Any]:
        mpz = getattr(engine, "mpz", None)
        if mpz is None:
            return MappingProxyType({
                "kind": "legacy_scalar_population", "N_em": float(engine.N_em),
            })
        return MappingProxyType({
            "kind": type(mpz).__qualname__,
            "mobile": float(mpz.mobile_count),
            "retained": float(mpz.retained_count),
            "wake_mobile": float(mpz.wake_mobile_count),
            "wake_retained": float(mpz.wake_retained_count),
            "available_sites": float(__import__("numpy").sum(mpz.available_sites)),
            "emitted_total": float(mpz.emitted_total),
            "accumulated_slip": float(__import__("numpy").sum(mpz.accumulated_slip)),
            "advance_total_m": float(mpz.advance_total_m),
        })

    def advance_front_process(
        self, accepted_front_state: Mapping[str, Any],
        accepted_mechanical_drive: float | Mapping[str, Any],
        temperature: float, duration: float, *,
        topology_owned_cleavage_action_before: float | None = None,
        topology_owned_cleavage_action_after: float | None = None,
    ) -> LegacyFrontAdvance:
        """Run the actual legacy engine on an isolated restored owner.

        With no topology-owned action, this is the exact single-front
        ``FrontEngine.step`` behavior.  In qualified multi-tip mode,
        directional competition owns cleavage.  The engine still performs its
        exact emission/MPZ update, while a deliberately unreachable temporary
        compatibility value prevents it from applying topology renewal a
        second time; its accepted ``B`` is then synchronized to the one
        topology-owned action.
        """
        dt = float(duration)
        T = float(temperature)
        if not math.isfinite(dt) or dt < 0.0 or not math.isfinite(T) or T <= 0.0:
            raise ValueError("front interval and temperature must be physical")
        if isinstance(accepted_mechanical_drive, Mapping):
            K = float(accepted_mechanical_drive["K_effective_Pa_sqrt_m"])
            drive_identity = dict(accepted_mechanical_drive)
        else:
            K = float(accepted_mechanical_drive)
            drive_identity = {"K_effective_Pa_sqrt_m": K}
        engine = self._restore(accepted_front_state)
        pre_payload_hash = _hash(accepted_front_state)
        pre_engine_hash = _hash(engine.__dict__)
        pre_mpz = self._mpz_audit(engine)
        pre_B = float(engine.B)
        pre_n_adv = int(engine.n_adv)
        pre_a_adv = float(engine.a_adv)
        pre_rng = _hash(getattr(engine, "_hazard_rng", None))
        topology_owned = topology_owned_cleavage_action_before is not None
        compatibility_threshold = None
        if topology_owned:
            if topology_owned_cleavage_action_after is None:
                raise ValueError("topology-owned cleavage requires its exact end action")
            # This is the exact interlock used by
            # ``evolve_process_engine_v12_hook``: directional competition owns
            # cleavage, while the canonical engine evolves all emission/MPZ
            # variables on an isolated copy.  Raising only the compatibility
            # threshold prevents a second topology renewal without changing a
            # rate, population equation, or accepted RNG owner.
            engine.B = float(topology_owned_cleavage_action_before)
            if hasattr(engine, "hazard_action_current"):
                engine.hazard_action_current = float(topology_owned_cleavage_action_before)
            if hasattr(engine, "hazard_threshold_action"):
                compatibility_threshold = float(engine.hazard_threshold_action)
                engine.hazard_threshold_action = 1.0e300
            engine._directional_topology_owns_cleavage = True
        info = engine.step(max(K, 0.0), T, dt)
        internal_fired = bool(info.get("fired", False))
        if topology_owned:
            if internal_fired or int(engine.n_adv) != pre_n_adv or float(engine.a_adv) != pre_a_adv:
                raise RuntimeError("qualified process interval applied topology renewal")
            engine.B = float(topology_owned_cleavage_action_after)
            if hasattr(engine, "hazard_action_current"):
                engine.hazard_action_current = float(topology_owned_cleavage_action_after)
            if compatibility_threshold is not None:
                engine.hazard_threshold_action = compatibility_threshold
            info = {**info, "B": engine.B, "fired": False, "n_fire": 0,
                    "legacy_process_fired_suppressed_by_topology_owner": internal_fired}
            cleavage = False
            event_time = None
        else:
            cleavage = internal_fired
            event_time = None
            if cleavage and dt > 0.0:
                low, high = 0.0, dt
                tolerance = max(8.0*__import__("numpy").finfo(float).eps*dt,
                                min(1.0e-12, dt*1.0e-9))
                while high-low > tolerance:
                    mid = 0.5*(low+high)
                    trial = self._restore(accepted_front_state)
                    trial_info = trial.step(max(K, 0.0), T, mid)
                    if bool(trial_info.get("fired", False)):
                        high, engine, info = mid, trial, trial_info
                    else:
                        low = mid
                event_time = high
        from .voiding_production_v5 import capture_sharp_front_engine
        payload = capture_sharp_front_engine(engine)
        post_mpz = self._mpz_audit(engine)
        shielding = float(engine.K_shield()) if hasattr(engine, "K_shield") else max(
            K-float(engine.sigma_tip(K))*math.sqrt(2.0*math.pi*float(engine.r_eff())), 0.0,
        )
        audit = MappingProxyType({
            "call_graph": (
                "canonical_front_engine/restore_sharp_front_engine",
                "FrontEngine.step",
                "lambda_emit",
                "MPZ emission/mobile/retained/slip/recovery evolution",
                "lambda_cleave after emission-modified tip state",
                "cleavage action B update",
                "cleavage-only moving-frame renewal when topology owner permits",
                "capture_sharp_front_engine",
            ),
            "factory_identity": dict(self.factory_identity),
            "accepted_drive": drive_identity,
            "duration_s": dt, "temperature_K": T,
            "pre_payload_fingerprint": pre_payload_hash,
            "pre_engine_fingerprint": pre_engine_hash,
            "post_engine_fingerprint": _hash(engine.__dict__),
            "topology_owned_cleavage": topology_owned,
            "pre_B": pre_B, "post_B": float(engine.B),
            "pre_n_adv": pre_n_adv, "post_n_adv": int(engine.n_adv),
            "pre_a_adv_m": pre_a_adv, "post_a_adv_m": float(engine.a_adv),
            "step_info": copy.deepcopy(info),
        })
        return LegacyFrontAdvance(
            advanced_front_state=payload,
            cleavage_event_occurred=cleavage,
            cleavage_event_time_s=event_time,
            updated_emission_state=MappingProxyType({
                "N_em": float(engine.N_em), "W_emit_J_per_m": float(engine.W_emit),
                "dN_emit": float(info.get("dN_emit", info.get("dN_emit_raw", 0.0))),
            }),
            updated_r_tip_m=float(engine.r_eff()),
            updated_shielding_Pa_sqrt_m=shielding,
            updated_back_stress_Pa=float(engine.sigma_back()),
            updated_mpz_state=post_mpz,
            threshold_identity=MappingProxyType({
                "owner": "cleavage", "action_before": pre_B,
                "action_after": float(engine.B), "threshold": 1.0,
            }),
            rng_identity=MappingProxyType({
                "before": pre_rng,
                "after": _hash(getattr(engine, "_hazard_rng", None)),
            }),
            audit=audit,
        )


__all__ = ["LegacyFrontAdvance", "LegacyFrontProcessAdapter", "SCHEMA"]
