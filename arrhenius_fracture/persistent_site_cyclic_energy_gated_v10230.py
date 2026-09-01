"""Transactional persistent-site cyclic engine for v10.2.30.

Cleavage first passage is unchanged. During the waiting cycles the persistent-site
source/mobile/retained state evolves at a stationary geometric tip. A completed
renewal creates a pending stochastic event proposal. The continuum K^2/E' energy
comparison is diagnostic only and cannot suppress or rescale cleavage hazard. The
moving-frame state and sharp-wake geometry are translated together only after the
post-passage hazard-derived energy gate approves the committed event distance.
"""
from __future__ import annotations

import copy
from typing import Any

from .persistent_site_cyclic_coupled_audited_v10229 import (
    AuditedCoupledPersistentSiteCyclicTipEngine,
)
from .stochastic_hazard_tip import StochasticHazardDiagnosticTipEngine
from . import stochastic_avalanche_tip as _avalanche_tip
from .hazard_energy_event_gate_v10230 import (
    attach_pending_event_info,
    continuum_gate_diagnostics,
    register_engine,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state
from .persistent_site_reversible_transport_v10230 import install_reversible_transport


MODEL_ID = "v10.2.30_transactional_persistent_site_energy_gated_cyclic"


def transaction_state_snapshot(engine) -> dict[str, Any]:
    """Serialize the actual constitutive state at an event transaction boundary."""
    try:
        snapshot = serialize_active_state(engine)
    except (AttributeError, TypeError, ValueError):
        # Lightweight transaction-unit fixtures do not implement the production
        # state contract. Production engines must always take the complete path.
        return {
            "schema": "v10.2.30_event_transaction_state_snapshot_v1",
            "complete_active_state": False,
            "engine_type": type(engine).__name__,
        }
    return {
        "schema": "v10.2.30_event_transaction_state_snapshot_v1",
        "complete_active_state": True,
        "active_state_model_id": "v10.2.30_complete_high_cycle_active_state_v1",
        "vector": snapshot.vector.tolist(),
        "fields": [
            {
                "owner": field.owner,
                "name": field.name,
                "shape": list(field.shape),
                "start": int(field.start),
                "stop": int(field.stop),
                "floor": float(field.floor),
            }
            for field in snapshot.fields
        ],
        "diagnostics": dict(snapshot.diagnostics),
        "geometry_signature": list(snapshot.geometry_signature),
    }


class HazardEnergyGatedPersistentSiteCyclicTipEngine(
    AuditedCoupledPersistentSiteCyclicTipEngine
):
    """State-coupled fatigue with atomic post-passage energy-gated translation."""

    hazard_energy_gated_v10230 = True
    hazard_energy_gate_continuum_affects_hazard = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._energy_gate_provisional = False
        self._energy_gate_pending: dict[str, Any] | None = None
        self.energy_gate_last_continuum: dict[str, Any] = {}
        self.energy_gate_committed_event_count = 0
        self.energy_gate_committed_path_m = 0.0
        install_reversible_transport(self.mpz)
        register_engine(self)

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))
        result._energy_gate_provisional = True
        result._energy_gate_pending = None
        return result

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        """Integrate hazard/plasticity without translating the crack before commit."""

        self._synchronize_driver_checkpoint_length()
        proposal = float(self.avalanche_event_advance_m)
        proposal_factor = float(self.avalanche_event_length_factor)
        continuum = continuum_gate_diagnostics(
            self,
            K,
            T,
            stress_override_Pa=stress_override,
        )
        self.energy_gate_last_continuum = dict(continuum)

        # Non-negotiable invariant: the continuum K^2/E' comparison is diagnostic
        # only. Cleavage first-passage kinetics are never suppressed or rescaled by
        # the post-passage event-energy calculation.
        effective_lambda = lambda_override

        base_da = float(self.f.da)
        rng_state_before = copy.deepcopy(self._hazard_rng.bit_generator.state)
        threshold_before = float(self.hazard_threshold_action)
        action_before = float(self.hazard_action_current)
        event_index_before = int(self.hazard_event_index)
        history_len_before = len(self.hazard_threshold_history)
        n_adv_before = int(self.n_adv)

        self.f.da = 0.0
        try:
            result = StochasticHazardDiagnosticTipEngine._integrate_coupled(
                self,
                K,
                T,
                dt,
                stress_override=stress_override,
                lambda_override=effective_lambda,
            )
        finally:
            self.f.da = base_da

        fired = bool(result.get("fired", False))
        if fired:
            pre_event_state = transaction_state_snapshot(self)
            completed_threshold = float(
                result.get("hazard_threshold_completed_action", threshold_before)
            )
            completed_action = float(
                result.get("hazard_action_completed", completed_threshold)
            )
            barrier_J = max(float(result.get("Gc_J", 0.0)), 0.0)
            descriptor = {
                "event_advance_m": proposal,
                "event_length_factor": proposal_factor,
                "threshold_action": completed_threshold,
                "hazard_action_completed": completed_action,
                "hazard_seed": int(self.hazard_cfg.seed),
                "hazard_event_index": int(self.hazard_event_index - 1),
                "geometry_subsegment_fraction": float(
                    self.avalanche_cfg.geometry_subsegment_fraction
                ),
                "energy_gate_engine_id": int(self._engine_id),
                "event_K_Pa_sqrt_m": max(float(K), 0.0),
                "event_temperature_K": float(T),
                "event_sigma_tip_Pa": float(result.get("sigma_tip", 0.0)),
                "hazard_barrier_J": barrier_J,
                "hazard_cooperative_hits": float(self.f.m_hits),
                "hazard_burgers_vector_m": float(self.b),
                "energy_gate_continuum": dict(continuum),
                "hazard_energy_gate_continuum_affects_hazard": False,
                "pre_event_state": pre_event_state,
            }
            rebonding_state = getattr(self, "_rebonding_state", None)
            self._energy_gate_pending = {
                "descriptor": descriptor,
                "rng_state_before": rng_state_before,
                "threshold_before": threshold_before,
                "action_before": action_before,
                "event_index_before": event_index_before,
                "history_len_before": history_len_before,
                "n_adv_before": n_adv_before,
                "proposal_m": proposal,
                "proposal_factor": proposal_factor,
                "rebonding_state_before": (
                    rebonding_state.snapshot() if rebonding_state is not None else None
                ),
                "rebonding_block_context": getattr(self, "_rebonding_block_context", None),
            }
            if not self._energy_gate_provisional:
                _avalanche_tip._PENDING_GEOMETRY_EVENTS.append(descriptor)
            self._set_current_event_length()

        result.update(
            {
                "hazard_energy_gate_model_id": MODEL_ID,
                "hazard_energy_gate_continuum_open": bool(
                    continuum["energy_gate_continuum_open"]
                ),
                "hazard_energy_gate_continuum": dict(continuum),
                "hazard_energy_gate_continuum_affects_hazard": False,
                "stochastic_event_proposed_advance_m": proposal if fired else 0.0,
                "stochastic_event_proposed_factor": proposal_factor if fired else 0.0,
                "avalanche_event_advance_m": 0.0,
                "avalanche_event_length_factor": 0.0,
                "avalanche_current_event_advance_m": float(
                    self.avalanche_event_advance_m
                ),
                "avalanche_current_event_length_factor": float(
                    self.avalanche_event_length_factor
                ),
                "kinetic_micro_advance_step_m": 0.0,
                "kinetic_checkpoint_progress_m": float(self.B)
                * float(self.avalanche_event_advance_m),
                "transactional_event_translation_pending": bool(
                    fired and not self._energy_gate_provisional
                ),
            }
        )
        return result

    def cycle_step_waveform(
        self,
        controller,
        waveform,
        T_K: float,
        requested_cycles=None,
        force_cycles=None,
    ):
        result = super().cycle_step_waveform(
            controller,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        result["hazard_energy_gate_model_id"] = MODEL_ID
        result["hazard_energy_gate_continuum"] = dict(
            self.energy_gate_last_continuum
        )
        result["hazard_energy_gate_continuum_open"] = bool(
            self.energy_gate_last_continuum.get(
                "energy_gate_continuum_open", False
            )
        )
        result["hazard_energy_gate_continuum_affects_hazard"] = False
        result["transactional_event_translation_pending"] = bool(
            result.get("fired", False) and not self._energy_gate_provisional
        )
        if result.get("fired", False) and not self._energy_gate_provisional:
            attach_pending_event_info(self._engine_id, result)
        return result

    def _commit_rebonding_event(
        self,
        accepted_length_m: float,
        pending: dict[str, Any],
        result_ref: dict[str, Any] | None,
    ) -> None:
        """Rebonding-coupled event-time root-find and wake-transaction commit.

        Reuses the existing normalized-progress relationship
        (``dB/dt = lambda_c/threshold_action``, ``stochastic_hazard_tip.py:58-60``)
        as a pure closed-form stand-in for the stateful ``_integrate_coupled``
        (which cannot be safely re-invoked with different trial rates without
        corrupting ``self.B``/RNG state) -- the phase-resolved action is
        computed independently via ``crack_rebonding_v10230.phase_resolved_action``
        and action closure is verified at the converged ``dt_used``, per the
        event-time-coupling requirement in
        docs/v10_2_30_crack_rebonding_approved_plan.md.
        """
        from . import crack_rebonding_v10230 as _rebond

        rebonding_state = self._rebonding_state
        ctx = pending.get("rebonding_block_context")
        event_index = int(self.hazard_event_index) - 1
        active_patches = [p for p in rebonding_state.active if not p.retired]

        if ctx is None:
            # No block context was stashed (e.g. rebonding was enabled after
            # a checkpoint restore mid-run without a preceding cycle block):
            # fail closed to a length-only commit -- new patch still created
            # from the accepted length, existing patches simply translate
            # without an intervening phase-resolved state advance.
            rebonding_state.commit_event(
                accepted_length_m=accepted_length_m,
                event_index=event_index,
                pre_event_states=None,
                Eprime_Pa=_rebond.reduced_modulus_Pa(self.G, self.nu),
            )
            return

        info = result_ref if isinstance(result_ref, dict) else {}
        dt_uncoupled = max(
            float(info.get("kinetic_dt_consumed_s", info.get("dt_consumed", 0.0))), 0.0
        )
        threshold_action = max(float(pending["threshold_before"]), 1.0e-300)
        B_start = float(ctx["B_start"])
        K_signed_phase = ctx["K_signed_phase"]
        dt_phase_ctx = float(ctx["dt_phase"])
        n_phase_ctx = int(ctx["n_phase"])
        r_contact_m_ctx = float(ctx["r_contact_m"])
        Eprime_Pa_ctx = float(ctx["Eprime_Pa"])
        T_K_ctx = float(ctx["T_K"])
        K_shield_ctx = float(ctx["K_shield_Pa_sqrt_m"])
        r_eff_ctx = float(ctx["r_eff_m"])

        def K_phase_fn(idx: int) -> float:
            return float(K_signed_phase[idx % n_phase_ctx])

        def lambda_cleave_normalized(sigma: float) -> float:
            return self.lambda_cleave(float(sigma), T_K_ctx)[0] / threshold_action

        pre_event_states = {p.patch_id: p.state_vector() for p in active_patches}

        def integrate_coupled_fn(lambda_avg: float) -> dict[str, Any]:
            if lambda_avg <= 0.0:
                return {"fired": False}
            return {
                "fired": True,
                "dt_consumed": (1.0 - B_start) * threshold_action / lambda_avg,
            }

        def phase_resolved_action_fn(dt: float):
            """Wraps ``phase_resolved_action``, enforcing the S8C certified
            bulk-action bound: on an uncertified bulk result, retry once
            with an extended transient budget (bounded by
            ``cfg.bulk_action_max_transient_extensions``); if still
            uncertified, fail closed rather than let an unbounded-error
            bulk answer feed the root-finder. Strips the diagnostics dict
            before returning so ``solve_coupled_event_time`` (which expects
            exactly ``(action, end_states, end_idx)``) needs no change."""
            cfg = rebonding_state.cfg
            transient_budget = 200  # phase_resolved_action's own default
            extensions = 0
            while True:
                action, end_states, end_idx, action_diag = _rebond.phase_resolved_action(
                    active_patches=active_patches,
                    patch_states=pre_event_states,
                    k0=0,
                    t_interval=dt,
                    K_phase_fn=K_phase_fn,
                    dt_phase=dt_phase_ctx,
                    n_phase=n_phase_ctx,
                    r_contact_m=r_contact_m_ctx,
                    cfg=cfg,
                    T_K=T_K_ctx,
                    Eprime_Pa=Eprime_Pa_ctx,
                    K_shield_Pa_sqrt_m=K_shield_ctx,
                    r_eff_m=r_eff_ctx,
                    lambda_cleave_fn=lambda_cleave_normalized,
                    max_transient_cycles=transient_budget,
                )
                if action_diag.get("bulk_action_qualified", True):
                    return action, end_states, end_idx
                if extensions >= cfg.bulk_action_max_transient_extensions:
                    raise RuntimeError(
                        "crack-rebonding bulk-action certificate not qualified "
                        f"after {extensions} transient-budget extension(s) "
                        f"(tail_bound={action_diag.get('total_tail_action_error_bound')!r}, "
                        f"tol_rel={cfg.bulk_action_error_rel_tol!r}); refusing to "
                        "commit an event on an uncertified bulk-action estimate. "
                        "Increase bulk_action_max_transient_extensions or "
                        "bulk_action_error_rel_tol, or use a smaller "
                        "max_transient_cycles-compatible block, if this "
                        "configuration is expected to relax this slowly."
                    )
                transient_budget *= 2
                extensions += 1

        lambda_avg_uncoupled = (
            (1.0 - B_start) * threshold_action / dt_uncoupled if dt_uncoupled > 0.0 else 1.0
        )
        root = _rebond.solve_coupled_event_time(
            integrate_coupled_fn=integrate_coupled_fn,
            phase_resolved_action_fn=phase_resolved_action_fn,
            lambda_avg_uncoupled=lambda_avg_uncoupled,
            B_start=B_start,
            B_threshold=1.0,
            eps_B=1.0e-6,
            dt_block=dt_uncoupled if dt_uncoupled > 0.0 else None,
        )
        final_states = root["patch_states"] if root.get("fired") else pre_event_states
        rebonding_state.commit_event(
            accepted_length_m=accepted_length_m,
            event_index=event_index,
            pre_event_states=final_states,
            Eprime_Pa=Eprime_Pa_ctx,
        )
        # Advance the wake's chronological phase clock by the converged
        # (rebonding-coupled) elapsed time, not the raw uncoupled estimate --
        # this is what makes the next block/event's phase sampling correctly
        # continuous from the true event instant, not a nominal one.
        dt_for_clock = float(root["dt_used"]) if root.get("fired") else dt_uncoupled
        period_s_ctx = float(ctx.get("period_s", 0.0))
        if period_s_ctx > 0.0:
            rebonding_state.elapsed_time_s = (
                rebonding_state.elapsed_time_s + dt_for_clock
            ) % period_s_ctx

    def commit_energy_gated_event(
        self,
        committed_length_m: float,
        gate: dict[str, Any],
        result_ref: dict[str, Any] | None,
    ) -> None:
        """Commit moving-frame translation after the geometry gate succeeds."""

        length = max(float(committed_length_m), 0.0)
        if length <= 0.0:
            raise ValueError("committed energy-gated event length must be positive")
        pending = self._energy_gate_pending
        if pending is None:
            raise RuntimeError("no pending transactional event exists")

        pre_commit_state = transaction_state_snapshot(self)
        advance = self.mpz.advance(length)

        rebonding_state = getattr(self, "_rebonding_state", None)
        if rebonding_state is not None and rebonding_state.cfg.enabled:
            self._commit_rebonding_event(length, pending, result_ref)
        self._rebonding_block_context = None

        self.micro_advance_total_m += length
        self.a_adv += length
        self.checkpoint_advance_total_m += length
        self.avalanche_last_completed_advance_m = length
        base = max(float(self.avalanche_base_checkpoint_m), 1.0e-300)
        self.avalanche_last_completed_factor = length / base
        self.avalanche_event_length_history.append(length)
        self.energy_gate_committed_event_count += 1
        self.energy_gate_committed_path_m += length
        post_commit_state = transaction_state_snapshot(self)
        descriptor = pending.get("descriptor", {})
        threshold_action = float(
            descriptor.get("threshold_action", getattr(self, "hazard_threshold_action", 0.0))
        )
        transaction_audit = {
            "schema": "v10.2.30_first_passage_energy_geometry_transaction_v1",
            "threshold_action": threshold_action,
            "hazard_action_completed": float(
                descriptor.get("hazard_action_completed", threshold_action)
            ),
            "raw_proposed_advance_m": float(pending["proposal_m"]),
            "event_length_random_factor": float(pending["proposal_factor"]),
            "energy_gate_decision": str(gate.get("arrest_reason", "unknown")),
            "energy_admissible_advance_m": float(
                gate.get("energy_admissible_event_length_m", length)
            ),
            "geometry_committed_advance_m": length,
            "mpz_translated_advance_m": length,
            "pre_event_state": descriptor.get(
                "pre_event_state", pre_commit_state
            ),
            "pre_geometry_commit_state": pre_commit_state,
            "post_event_state": post_commit_state,
        }
        gate["event_transaction_audit"] = transaction_audit

        info = result_ref if isinstance(result_ref, dict) else {}
        dt_used = max(
            float(
                info.get(
                    "kinetic_dt_consumed_s",
                    info.get("dt_consumed", 0.0),
                )
            ),
            0.0,
        )
        info.update(
            {
                "hazard_energy_gate_model_id": MODEL_ID,
                "transactional_event_translation_pending": False,
                "stochastic_event_proposed_advance_m": float(
                    pending["proposal_m"]
                ),
                "stochastic_event_proposed_factor": float(
                    pending["proposal_factor"]
                ),
                "energy_admissible_event_length_m": float(
                    gate.get("energy_admissible_event_length_m", length)
                ),
                "avalanche_event_advance_m": length,
                "avalanche_event_length_factor": length / base,
                "kinetic_micro_advance_step_m": length,
                "kinetic_micro_advance_total_m": float(
                    self.micro_advance_total_m
                ),
                "kinetic_checkpoint_committed_total_m": float(
                    self.checkpoint_advance_total_m
                ),
                "v_crack": length / dt_used if dt_used > 0.0 else 0.0,
                "N_em": float(self.N_em),
                "N_em_retained": float(self.N_em),
                "N_em_shed_to_wake": float(
                    advance.get("wake_mobile", 0.0)
                    + advance.get("wake_retained", 0.0)
                ),
                "event_transaction_audit": transaction_audit,
                **{
                    key: value
                    for key, value in advance.items()
                    if isinstance(value, (int, float))
                },
                **{
                    key: value
                    for key, value in gate.items()
                    if key not in {"equilibrated_displacement", "trial_rows"}
                },
            }
        )

        records = getattr(type(self), "_audit_records", None)
        if isinstance(records, list):
            for record in reversed(records):
                if int(record.get("engine_id", -1)) == int(self._engine_id):
                    record.update(
                        {
                            "hazard_energy_gate_model_id": MODEL_ID,
                            "stochastic_event_proposed_advance_m": float(
                                pending["proposal_m"]
                            ),
                            "energy_gated_event_advance_m": length,
                            "energy_gate_arrest_reason": str(
                                gate.get("arrest_reason", "unknown")
                            ),
                            "hazard_resistance_J_per_m2": float(
                                gate.get("hazard_resistance_J_per_m2", 0.0)
                            ),
                            "orientation_gamma_relative": float(
                                gate.get("orientation_gamma_relative", 1.0)
                            ),
                            "athermal_Gc_used": False,
                        }
                    )
                    break
        self._energy_gate_pending = None

    def restore_geometry_veto(self, n_restore: int = 1) -> None:
        """Restore a completed first passage without undoing accepted plastic time."""

        pending = self._energy_gate_pending
        if pending is None:
            self.B += float(max(int(n_restore), 1))
            return
        self._hazard_rng.bit_generator.state = copy.deepcopy(
            pending["rng_state_before"]
        )
        self.hazard_threshold_action = float(pending["threshold_before"])
        self.hazard_action_current = float(
            max(pending["action_before"], pending["threshold_before"])
        )
        self.B = 1.0
        self.hazard_event_index = int(pending["event_index_before"])
        while len(self.hazard_threshold_history) > int(
            pending["history_len_before"]
        ):
            self.hazard_threshold_history.pop()
        self.n_adv = int(pending["n_adv_before"])
        rebonding_state = getattr(self, "_rebonding_state", None)
        if rebonding_state is not None:
            rebonding_state.restore(pending.get("rebonding_state_before"))
        self._rebonding_block_context = None
        self._set_current_event_length()
        self._energy_gate_pending = None


__all__ = [
    "HazardEnergyGatedPersistentSiteCyclicTipEngine",
    "MODEL_ID",
]
