"""Strict separation of opening, cleavage, and emission stress channels.

The FEM/J-derived opening stress is never reduced by the local Taylor back
stress.  Elastic dislocation shielding remains in the cleavage channel through
``K_shield``.  The local Taylor back stress acts only inside the continuum
source emission law.
"""
from __future__ import annotations

import math

import numpy as np

from .config import EV_TO_J
from .continuum_source_tip import (
    ContinuumSourceKineticTipEngine as _BackstressContinuumEngine,
    source_diagnostics,
)
from .kinetic_tip_cell import KineticMovingTipFrontEngine


class SeparatedSourceKineticTipEngine(_BackstressContinuumEngine):
    """Continuum source engine with three explicit tip-stress channels.

    ``sigma_opening_tip`` is obtained directly from the FEM/J-derived K and the
    current blunted radius.  ``sigma_cleave`` is inherited from ``sigma_tip``
    and therefore includes elastic K shielding.  Plastic evolution receives
    only ``sigma_opening_tip``; the local Taylor back stress is subtracted later
    and only by the continuum emission law.
    """

    separated_tip_stress_channels = True

    def sigma_opening_tip(self, K: float) -> float:
        stress = max(float(K), 0.0) / math.sqrt(
            2.0 * math.pi * max(float(self.r_eff()), 1.0e-30)
        )
        if self.f.sigma_cap > 0.0:
            stress = min(stress, float(self.f.sigma_cap))
        return float(stress)

    def _plastic_half_step(
        self, dt: float, T: float, cleavage_stress: float
    ) -> dict[str, float]:
        self.mpz._continuum_tip_radius_m = self.r_eff()
        current_K = getattr(self, "_separated_current_K_Pa_sqrt_m", None)
        if current_K is None:
            opening_stress = max(float(cleavage_stress), 0.0)
        else:
            opening_stress = self.sigma_opening_tip(float(current_K))

        # Bypass the immediate parent wrapper so that the validated kinetic
        # plastic integrator receives the unshielded opening stress.  The MPZ
        # emission method then subtracts the local Taylor back stress exactly
        # once, while cleavage continues to use the shielded stress computed by
        # the moving-tip integrator.
        result = KineticMovingTipFrontEngine._plastic_half_step(
            self, dt, T, opening_stress
        )
        result.update(source_diagnostics(self.mpz))
        result["sigma_opening_tip_Pa"] = float(opening_stress)
        result["sigma_cleave_input_Pa"] = max(float(cleavage_stress), 0.0)
        return result

    def _rewrite_monotonic_channel_diagnostics(
        self, result: dict, K: float
    ) -> dict:
        cleavage_stress = float(
            result.get("sigma_cleave_eff_Pa", result.get("sigma_tip", 0.0))
        )
        opening_stress = self.sigma_opening_tip(K)
        opening_uncapped = max(float(K), 0.0) / math.sqrt(
            2.0 * math.pi * max(float(self.r_eff()), 1.0e-30)
        )
        emit_back = float(result.get("tip_source_backstress_equivalent_Pa", 0.0))
        emit_effective = float(
            result.get("tip_source_effective_emission_stress_Pa", opening_stress)
        )

        result["sigma_tip"] = opening_stress
        result["sigma_opening_tip_Pa"] = opening_stress
        result["sigma_opening_tip_uncapped_Pa"] = opening_uncapped
        result["sigma_cleave_eff_Pa"] = cleavage_stress
        result["sigma_emission_backstress_Pa"] = emit_back
        result["sigma_emission_effective_Pa"] = emit_effective
        result["stress_channels_separated"] = True
        result["lambda_e"] = float(
            result.get("tip_source_emission_rate_s", result.get("lambda_e", 0.0))
        )
        result["G_emit_eV"] = float(
            np.asarray(self.manifest.emission.values_eV(emit_effective, 1.0))
        ) if False else result.get("G_emit_eV", 0.0)

        if type(self)._audit_records:
            record = type(self)._audit_records[-1]
            record["sigma_tip_Pa"] = opening_stress
            record["sigma_opening_tip_Pa"] = opening_stress
            record["sigma_cleave_eff_Pa"] = cleavage_stress
            record["sigma_emission_backstress_Pa"] = emit_back
            record["sigma_emission_effective_Pa"] = emit_effective
            record["stress_channels_separated"] = True
        return result

    def step(self, K, T, dt):
        self._separated_current_K_Pa_sqrt_m = max(float(K), 0.0)
        try:
            result = super().step(K, T, dt)
        finally:
            # Keep the most recent K only for diagnostics; every subsequent step
            # overwrites it before any plastic evolution occurs.
            pass

        result = self._rewrite_monotonic_channel_diagnostics(result, float(K))
        emit_effective = float(
            result.get("tip_source_effective_emission_stress_Pa", 0.0)
        )
        result["G_emit_eV"] = float(
            np.asarray(self.manifest.emission.values_eV(emit_effective, T))
        )
        return result

    def cycle_step_waveform(self, controller, waveform, T_K: float,
                            requested_cycles=None, force_cycles=None):
        # Fatigue retains the existing cycle-block approximation.  Set the
        # opening channel from Kmax so the local Taylor back stress is still
        # emission-only; a future phase-resolved source update can refine this.
        self._separated_current_K_Pa_sqrt_m = max(float(waveform.Kmax), 0.0)
        result = super().cycle_step_waveform(
            controller,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        opening = self.sigma_opening_tip(float(waveform.Kmax))
        result["sigma_opening_tip_Pa"] = opening
        result["sigma_emission_backstress_Pa"] = float(
            result.get("tip_source_backstress_equivalent_Pa", 0.0)
        )
        result["sigma_emission_effective_Pa"] = float(
            result.get("tip_source_effective_emission_stress_Pa", opening)
        )
        result["stress_channels_separated"] = True
        return result
