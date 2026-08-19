"""Intrinsic signed reverse-glide audit for v9.14 fatigue.

This audit removes two phenomenological shortcuts from the mobile-return channel:

1. no fitted or prescribed physical-return fraction is used;
2. retained-GND shielding is not subtracted a second time from the mobile glide
   drive.  Cleavage still uses the qualified opening-only K-K_shield channel,
   while already-mobile dislocations see a signed finite-tip applied stress plus
   the existing signed local GND stress.

The Peierls mobility remains the existing forward-minus-reverse Arrhenius law,
so forward and reverse glide use the same barrier, activation volume/stress
mapping, and attempt frequency.  At zero effective shear the signed rate is
exactly zero.  Surface return remains an emergent first-passage/boundary-outflow
result of transport, storage, escape, and geometry; there is no return-fraction
parameter.

The existing persistent-site ``backstress_state`` is intentionally *not* added
as a signed kinematic reverse stress here.  It is an unsigned source-blocking
state and promoting it to a kinematic backstress would introduce an additional
model-form assumption.  Reverse glide at zero external load can still emerge
from the signed GND field.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from v914_finite_tip_shielding_state_v5 import FiniteTipFloorReversibleState
from v914_reverse_drive_utils import forward_projected_transport_stress


MODEL_ID = "v9.14_intrinsic_reverse_glide_v7_signed_applied_plus_gnd"
STATE_EXTENSION_SCHEMA = "v914_intrinsic_reverse_glide_state_extension_v7"


def intrinsic_signed_glide_stress_fields(
    K_applied_MPa_sqrt_m: float,
    tip_radius_m: float,
    drive_factors: np.ndarray,
    tau_gnd_Pa: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build the intrinsic mobile-glide stress without a closure law.

    ``K_applied`` is used only as a signed loading-coordinate supplied by the
    fatigue waveform.  Its local stress equivalent is bounded by the existing
    finite tip radius rather than a sharp-crack singularity.  The nonlocal
    cleavage shielding term is *not* subtracted from this transport channel;
    the same retained GND already acts locally through ``tau_gnd_Pa``.
    """
    drive = np.asarray(drive_factors, dtype=float)
    tau_gnd = np.asarray(tau_gnd_Pa, dtype=float)
    if drive.ndim != 1:
        raise ValueError("drive_factors must be one-dimensional")
    if tau_gnd.ndim != 2 or tau_gnd.shape[0] != drive.size:
        raise ValueError("tau_gnd_Pa must have shape (n_systems, n_bins)")

    radius = max(float(tip_radius_m), 1.0e-30)
    K_signed = float(K_applied_MPa_sqrt_m)
    sigma_applied = K_signed * 1.0e6 / math.sqrt(2.0 * math.pi * radius)
    tau_applied_column = drive[:, None] * sigma_applied
    tau_applied = np.broadcast_to(tau_applied_column, tau_gnd.shape).copy()
    tau_effective = tau_applied + tau_gnd
    return {
        "K_applied_signed_MPa_sqrt_m": K_signed,
        "sigma_applied_signed_Pa": sigma_applied,
        "tau_applied_Pa": tau_applied,
        "tau_effective_Pa": tau_effective,
    }


def reduced_competing_return_probability(
    *,
    forward_loss_rate_s: float,
    reverse_return_rate_s: float,
    reverse_loss_rate_s: float,
    forward_time_s: float,
    reverse_time_s: float,
) -> float:
    """Reduced analytical return probability for mechanism comparison only.

    This is not used by the constitutive state.  It represents survival of a
    mobile population until reversal followed by competition between return and
    irreversible loss during the reverse interval:

        P = exp(-k_f t_f) * k_r/(k_r+k_l)
            * [1-exp(-(k_r+k_l)t_r)].

    All inputs are rates/times that must come from physics or the resolved state;
    no fitted return fraction appears.
    """
    kf = max(float(forward_loss_rate_s), 0.0)
    kr = max(float(reverse_return_rate_s), 0.0)
    kl = max(float(reverse_loss_rate_s), 0.0)
    tf = max(float(forward_time_s), 0.0)
    tr = max(float(reverse_time_s), 0.0)
    survive = math.exp(-min(kf * tf, 700.0))
    total = kr + kl
    if total <= 0.0 or tr <= 0.0 or kr <= 0.0:
        return 0.0
    reverse_capture = (kr / total) * (1.0 - math.exp(-min(total * tr, 700.0)))
    return min(max(survive * reverse_capture, 0.0), 1.0)


class IntrinsicReverseGlideState(FiniteTipFloorReversibleState):
    """Finite-tip v5 state with intrinsic signed applied+GND mobile glide."""

    MODEL_ID = MODEL_ID
    STATE_EXTENSION_SCHEMA = STATE_EXTENSION_SCHEMA

    def integration_metadata(self) -> dict[str, object]:
        metadata = dict(super().integration_metadata())
        metadata.update(
            {
                "model_id": MODEL_ID,
                "intrinsic_reverse_glide_v7": True,
                "mobile_glide_external_drive": (
                    "signed_applied_load_mapped_through_existing_finite_tip_radius"
                ),
                "mobile_glide_internal_drive": "existing_signed_local_gnd_stress",
                "mobile_glide_subtracts_nonlocal_K_shield": False,
                "cleavage_uses_nonlocal_K_shield": True,
                "forward_reverse_peierls_barrier_shared": True,
                "physical_return_fraction_parameter": False,
                "physical_return_definition": (
                    "emergent_emitted_population_left_boundary_first_passage_"
                    "under_true_reverse_drive"
                ),
                "persistent_unsigned_source_backstress_promoted_to_kinematic": False,
                "crack_closure_law_added": False,
                "transport_storage_physics_changed_from_v5": True,
                "surface_return_semantics_changed_from_v5": False,
                "cleavage_barrier_changed_from_v5": False,
                "emission_barrier_changed_from_v5": False,
            }
        )
        return metadata

    def local_rates(self, K_MPa_sqrt_m: float, T_K: float) -> dict[str, np.ndarray]:
        # Parent remains authoritative for cleavage, emission, Taylor completion,
        # recovery, finite-tip shielding, and all state bookkeeping.  Replace
        # only the already-mobile Peierls transport and its distance-controlled
        # encounter/storage rate.
        rates = dict(super().local_rates(K_MPa_sqrt_m, T_K))
        drive_factors = np.asarray(self.emission_drive_factors(), dtype=float)
        tau_gnd = np.asarray(rates["tau_gnd_Pa"], dtype=float)
        fields = intrinsic_signed_glide_stress_fields(
            K_MPa_sqrt_m,
            self.tip_radius_m(),
            drive_factors,
            tau_gnd,
        )
        tau_eff = np.asarray(fields["tau_effective_Pa"], dtype=float)

        forest = self.forest_density_m2()
        spacing = 1.0 / (2.0 * np.sqrt(forest))
        jump = self.c.jump_fraction_of_forest_spacing * spacing
        p_surface = self.p.peierls.surface(self.p.emission)
        peierls = self._signed_rate(
            p_surface,
            self.p.peierls.stress_fraction * tau_eff,
            T_K,
            self.p.peierls.nu0_s,
        )
        peierls_velocity = jump * peierls
        velocity = (
            max(float(self.c.mobile_transport_velocity_scale), 0.0)
            * peierls_velocity
        )
        mfp = self.c.mean_free_path_coefficient / np.sqrt(forest)
        encounter = (
            max(float(self.c.encounter_efficiency), 0.0)
            * np.abs(peierls_velocity)
            / np.maximum(mfp, 1.0e-30)
        )

        rates["velocity_m_s"] = velocity
        rates["peierls_velocity_m_s"] = peierls_velocity
        rates["encounter_s"] = encounter
        rates["tau_external_Pa"] = np.asarray(fields["tau_applied_Pa"], dtype=float)
        rates["tau_nonlocal_shielding_Pa"] = np.zeros_like(tau_eff)
        rates["tau_eff_Pa"] = tau_eff
        rates["reversible_transport_K_signed_MPa_sqrt_m"] = np.asarray(
            float(fields["K_applied_signed_MPa_sqrt_m"])
        )
        rates["reversible_tau_transport_external_Pa"] = np.asarray(
            fields["tau_applied_Pa"], dtype=float
        )
        rates["reversible_tau_transport_eff_Pa"] = tau_eff.copy()
        projected = forward_projected_transport_stress(tau_eff, drive_factors)
        rates["reversible_forward_projected_tau_Pa"] = projected
        rates["reversible_true_reverse_drive_mask"] = projected < 0.0
        emission_sign = np.asarray(self.c.emission_signs, dtype=float)[:, None]
        rates["reversible_emitted_sign_spatial_velocity_m_s"] = emission_sign * velocity
        return rates

    def reversible_checkpoint_payload(self) -> dict[str, Any]:
        payload = dict(super().reversible_checkpoint_payload())
        payload["schema"] = STATE_EXTENSION_SCHEMA
        return payload

    def restore_reversible_checkpoint_payload(
        self,
        payload: Mapping[str, Any] | None,
    ) -> None:
        self._ensure_reversible_fields()
        if not payload:
            return
        if payload.get("schema") != STATE_EXTENSION_SCHEMA:
            raise ValueError("intrinsic reverse-glide v7 checkpoint schema mismatch")
        parent = dict(payload)
        parent["schema"] = "v914_minimal_reversible_state_extension_v5_finite_tip_floor"
        super().restore_reversible_checkpoint_payload(parent)


__all__ = [
    "MODEL_ID",
    "STATE_EXTENSION_SCHEMA",
    "IntrinsicReverseGlideState",
    "intrinsic_signed_glide_stress_fields",
    "reduced_competing_return_probability",
]
