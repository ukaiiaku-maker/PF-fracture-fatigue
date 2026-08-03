"""Hazard-consistent plastic-flow terminal for the v10.4.3 monotonic path.

A sustained plastic-flow plateau can retain a finite yield-level reaction force,
finite positive configurational J, and finite tip stress while the cleavage
first-passage hazard is effectively inaccessible.  The v10.4.2 terminal treated
near-zero J and near-zero tip stress as independent hard gates, which rejects
that state even when the actual cleavage clock is frozen by many decades.

This outer transform preserves those J/stress quantities as audit diagnostics
but replaces their acceptance role with a conservative prospective cleavage-
action projection.  The projection uses the largest of the fitted, endpoint-
secant, and 95th-percentile positive local growth rates of log(lambda_c) over
the physical terminal window.  It integrates that exponential upper trend over
the larger of the remaining campaign horizon and a fixed prospective horizon.
Terminal acceptance requires the projected action to consume no more than the
configured fraction of the remaining first-passage budget.

No constitutive law, fracture law, hazard law, material parameter, accepted
state, directional-J convention, or event-energy criterion is changed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_physical_progress_v1043 import (
    transform_source as _physical_progress_transform,
)

MODEL_ID = "v10.4.3_plastic_flow_terminal_projected_cleavage_hazard"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_hazard_terminal"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _physical_progress_transform(source)

    text = _replace_once(
        text,
        """    p.add_argument('--plastic-flow-min-cleavage-horizon-ratio', type=float, default=100.0, dest='plastic_flow_min_cleavage_horizon_ratio')
""",
        """    p.add_argument('--plastic-flow-min-cleavage-horizon-ratio', type=float, default=100.0, dest='plastic_flow_min_cleavage_horizon_ratio')
    p.add_argument('--plastic-flow-prospective-horizon-steps', type=float, default=2000.0, dest='plastic_flow_prospective_horizon_steps',
                   help='Minimum nonzero nominal-increment horizon used to project future cleavage action even at the requested campaign endpoint.')
    p.add_argument('--plastic-flow-max-projected-hazard-fraction', type=float, default=0.01, dest='plastic_flow_max_projected_hazard_fraction',
                   help='Maximum projected fraction of the remaining cleavage first-passage budget; 0.01 preserves a factor-100 safety margin.')
    p.add_argument('--plastic-flow-hazard-growth-percentile', type=float, default=95.0, dest='plastic_flow_hazard_growth_percentile',
                   help='Percentile of positive local d(ln lambda_c)/d(nominal progress) included in the conservative hazard-growth projection.')
""",
        "v10.4.3 prospective-hazard parser options",
    )

    text = _replace_once(
        text,
        """                   help='Terminate successfully when a persistent accepted-step window demonstrates bulk-plastic accommodation with negligible positive sharp-tip J and no cleavage first passage.')
""",
        """                   help='Terminate successfully when a persistent physical-loading window demonstrates collapsed-stiffness bulk-plastic accommodation and conservatively projected cleavage first passage remains inaccessible.')
""",
        "v10.4.3 terminal help semantics",
    )

    text = _replace_once(
        text,
        """    j_tolerance = max(
        float(getattr(args, 'plastic_flow_J_abs_tol_J_per_m2', 1.0e-6) or 0.0),
        float(getattr(args, 'plastic_flow_J_rel_tol', 1.0e-6) or 0.0)
        * max(float(peak_J_positive), 0.0),
    )
    sigma_tolerance = (
        float(getattr(args, 'plastic_flow_sigma_rel_tol', 1.0e-6) or 0.0)
        * max(float(sigma_reference), 1.0)
    )
""",
        """    # Retain the former near-zero J/stress gates as explicit audit
    # diagnostics only.  They are not fracture criteria because the production
    # cleavage first-passage law already maps the accepted J/stress state to
    # lambda_c.
    j_tolerance = max(
        float(getattr(args, 'plastic_flow_J_abs_tol_J_per_m2', 1.0e-6) or 0.0),
        float(getattr(args, 'plastic_flow_J_rel_tol', 1.0e-6) or 0.0)
        * max(float(peak_J_positive), 0.0),
    )
    sigma_tolerance = (
        float(getattr(args, 'plastic_flow_sigma_rel_tol', 1.0e-6) or 0.0)
        * max(float(sigma_reference), 1.0)
    )
    legacy_negligible_positive_tip_J = j_max <= j_tolerance
    legacy_negligible_tip_stress = sigma_max <= sigma_tolerance
""",
        "v10.4.3 retain J/stress as diagnostics",
    )

    text = _replace_once(
        text,
        """    lambda_max = float(np.max(np.maximum(lambda_values, 0.0)))
    B_final = float(last['B'])
    if lambda_max <= 1.0e-300:
        remaining_cleavage_time = float('inf')
    else:
        remaining_cleavage_time = max(1.0 - B_final, 0.0) / lambda_max
    remaining_loading_horizon = max(
        float(remaining_steps), 0.0
    ) * max(float(nominal_dt_s), 0.0)
    if remaining_loading_horizon <= 0.0:
        cleavage_horizon_ratio = float('inf')
    else:
        cleavage_horizon_ratio = remaining_cleavage_time / remaining_loading_horizon
""",
        """    lambda_max = float(np.max(np.maximum(lambda_values, 0.0)))
    B_final = float(last['B'])
    if lambda_max <= 1.0e-300:
        remaining_cleavage_time = float('inf')
    else:
        remaining_cleavage_time = max(1.0 - B_final, 0.0) / lambda_max
    remaining_loading_horizon = max(
        float(remaining_steps), 0.0
    ) * max(float(nominal_dt_s), 0.0)
    if remaining_loading_horizon <= 0.0:
        cleavage_horizon_ratio = float('inf')
    else:
        cleavage_horizon_ratio = remaining_cleavage_time / remaining_loading_horizon

    # Conservative prospective first-passage projection.  Physical-progress
    # coordinates are supplied by the outer v10.4.3 adaptive-substep transform.
    nominal_progress_values = np.asarray(
        [row['nominal_progress_end'] for row in values], dtype=float
    )
    lambda_safe = np.maximum(lambda_values, 1.0e-300)
    log_lambda_values = np.log(lambda_safe)

    progress_span = float(
        nominal_progress_values[-1] - nominal_progress_values[0]
    )
    if len(values) >= 2 and progress_span > 1.0e-15:
        fitted_log_growth = max(
            float(np.polyfit(nominal_progress_values, log_lambda_values, 1)[0]),
            0.0,
        )
        secant_log_growth = max(
            float(
                (log_lambda_values[-1] - log_lambda_values[0])
                / progress_span
            ),
            0.0,
        )
        local_progress = np.diff(nominal_progress_values)
        local_growth = np.diff(log_lambda_values) / np.maximum(
            local_progress, 1.0e-300
        )
        positive_local_growth = local_growth[local_growth > 0.0]
    else:
        fitted_log_growth = 0.0
        secant_log_growth = 0.0
        positive_local_growth = np.asarray([], dtype=float)

    hazard_growth_percentile = min(
        max(
            float(
                getattr(args, 'plastic_flow_hazard_growth_percentile', 95.0)
                or 95.0
            ),
            50.0,
        ),
        100.0,
    )
    percentile_log_growth = (
        float(np.percentile(positive_local_growth, hazard_growth_percentile))
        if positive_local_growth.size
        else 0.0
    )
    conservative_log_lambda_growth = max(
        fitted_log_growth,
        secant_log_growth,
        percentile_log_growth,
        0.0,
    )

    prospective_horizon_nominal = max(
        float(
            getattr(args, 'plastic_flow_prospective_horizon_steps', 2000.0)
            or 2000.0
        ),
        1.0,
    )
    projected_horizon_nominal = max(
        prospective_horizon_nominal,
        max(float(remaining_steps), 0.0),
    )
    projected_horizon_s = projected_horizon_nominal * max(
        float(nominal_dt_s), 0.0
    )

    lambda_projection_reference = max(lambda_max, float(lambda_safe[-1]))
    if projected_horizon_nominal <= 0.0 or nominal_dt_s <= 0.0:
        log_projected_cleavage_action = float('-inf')
    elif lambda_projection_reference <= 0.0:
        log_projected_cleavage_action = float('-inf')
    elif conservative_log_lambda_growth <= 1.0e-15:
        log_projected_cleavage_action = (
            np.log(max(float(nominal_dt_s), 1.0e-300))
            + np.log(max(lambda_projection_reference, 1.0e-300))
            + np.log(max(projected_horizon_nominal, 1.0e-300))
        )
    else:
        projection_exponent = (
            conservative_log_lambda_growth * projected_horizon_nominal
        )
        if projection_exponent > 50.0:
            log_expm1_projection = projection_exponent
        else:
            log_expm1_projection = np.log(np.expm1(projection_exponent))
        log_projected_cleavage_action = (
            np.log(max(float(nominal_dt_s), 1.0e-300))
            + np.log(max(lambda_projection_reference, 1.0e-300))
            + log_expm1_projection
            - np.log(conservative_log_lambda_growth)
        )

    remaining_first_passage_budget = max(1.0 - B_final, 1.0e-300)
    log_remaining_first_passage_budget = np.log(
        remaining_first_passage_budget
    )
    max_projected_hazard_fraction = min(
        max(
            float(
                getattr(
                    args,
                    'plastic_flow_max_projected_hazard_fraction',
                    0.01,
                )
                or 0.01
            ),
            1.0e-300,
        ),
        1.0,
    )
    log_projected_hazard_fraction = (
        log_projected_cleavage_action
        - log_remaining_first_passage_budget
    )
    projected_cleavage_action_safe = (
        log_projected_hazard_fraction
        <= np.log(max_projected_hazard_fraction)
    )
    projected_cleavage_action = (
        0.0
        if not np.isfinite(log_projected_cleavage_action)
        and log_projected_cleavage_action < 0.0
        else (
            float('inf')
            if log_projected_cleavage_action > np.log(np.finfo(float).max)
            else float(np.exp(log_projected_cleavage_action))
        )
    )
    projected_hazard_fraction = (
        0.0
        if not np.isfinite(log_projected_hazard_fraction)
        and log_projected_hazard_fraction < 0.0
        else (
            float('inf')
            if log_projected_hazard_fraction > np.log(np.finfo(float).max)
            else float(np.exp(log_projected_hazard_fraction))
        )
    )
""",
        "v10.4.3 prospective cleavage-action projection",
    )

    text = _replace_once(
        text,
        """        'negligible_positive_tip_J': j_max <= j_tolerance,
        'negligible_tip_stress': sigma_max <= sigma_tolerance,
""",
        """""",
        "v10.4.3 remove redundant J/stress hard gates",
    )

    text = _replace_once(
        text,
        """        'cleavage_outside_remaining_horizon': cleavage_horizon_ratio
        >= float(getattr(args, 'plastic_flow_min_cleavage_horizon_ratio', 100.0) or 100.0),
""",
        """        'projected_cleavage_action_safe': projected_cleavage_action_safe,
""",
        "v10.4.3 projected-hazard terminal gate",
    )

    text = _replace_once(
        text,
        """        'cleavage_horizon_ratio': cleavage_horizon_ratio,
""",
        """        'cleavage_horizon_ratio': cleavage_horizon_ratio,
        'legacy_negligible_positive_tip_J': legacy_negligible_positive_tip_J,
        'legacy_negligible_tip_stress': legacy_negligible_tip_stress,
        'J_and_sigma_zero_gates_are_diagnostic_only': True,
        'prospective_horizon_nominal_increments': prospective_horizon_nominal,
        'projected_horizon_nominal_increments': projected_horizon_nominal,
        'projected_horizon_s': projected_horizon_s,
        'lambda_projection_reference_per_s': lambda_projection_reference,
        'hazard_growth_percentile': hazard_growth_percentile,
        'fitted_log_lambda_growth_per_nominal_increment': fitted_log_growth,
        'secant_log_lambda_growth_per_nominal_increment': secant_log_growth,
        'percentile_log_lambda_growth_per_nominal_increment': percentile_log_growth,
        'conservative_log_lambda_growth_per_nominal_increment': conservative_log_lambda_growth,
        'projected_cleavage_action_increment': projected_cleavage_action,
        'log_projected_cleavage_action_increment': float(log_projected_cleavage_action),
        'remaining_first_passage_budget': remaining_first_passage_budget,
        'projected_hazard_fraction_of_remaining_budget': projected_hazard_fraction,
        'log_projected_hazard_fraction_of_remaining_budget': float(log_projected_hazard_fraction),
        'max_projected_hazard_fraction': max_projected_hazard_fraction,
""",
        "v10.4.3 prospective-hazard audit fields",
    )

    text = _replace_once(
        text,
        """                        'schema': 'v10.4.2_plastic_flow_terminal_audit_v1',
""",
        """                        'schema': 'v10.4.3_projected_hazard_plastic_flow_terminal_audit_v2',
""",
        "v10.4.3 terminal audit schema",
    )

    return text


def load_transformed_sharp_front() -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    source_path = Path(__file__).with_name("sharp_front.py")
    transformed = transform_source(source_path.read_text())
    spec = importlib.util.spec_from_loader(MODULE_NAME, loader=None)
    if spec is None:
        raise RuntimeError("could not allocate v10.4.3 hazard-terminal module")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-hazard-terminal]",
                "exec",
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
