"""v10.4.3 stagger-consistent plastic-dominance censor.

The validated v10.4.2 sharp-fracture first-passage law is preserved.  This
overlay makes each monotonic stagger iteration represent the same physical time
increment, records accepted plastic work from the final constitutive iterate,
and classifies sustained plastic dominance as a model-limit censor rather than
simulated ductile fracture.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from .directional_j_positive_v1042 import transform_source as _v1042_transform

MODEL_ID = "v10.4.3_stagger_consistent_plastic_dominance_censor"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_plastic_dominance"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def terminal_metrics_v1043(
    window,
    args,
    *,
    Eprime,
    da_phys,
    sigma_reference,
    peak_J_positive,
    peak_force,
    stiffness_reference,
    remaining_steps,
    nominal_dt_s,
    cumulative_Wp,
    cumulative_Uel,
    cumulative_Wemit,
):
    """Evaluate sustained plastic dominance without requiring total collapse."""
    del Eprime, sigma_reference, peak_J_positive
    required = max(int(getattr(args, "plastic_flow_window_steps", 2000) or 2000), 2)
    minimum_step = max(
        int(getattr(args, "plastic_flow_min_step", required) or required),
        required,
    )
    if len(window) < required or int(window[-1]["step"]) < minimum_step:
        return None

    values = list(window)
    first = values[0]
    last = values[-1]
    a_values = np.asarray([row["a_tip"] for row in values], dtype=float)
    j_values = np.asarray([row["J_positive"] for row in values], dtype=float)
    sigma_values = np.asarray([row["sigma_tip"] for row in values], dtype=float)
    B_values = np.asarray([row["B"] for row in values], dtype=float)
    lambda_values = np.asarray([row["lambda_c"] for row in values], dtype=float)
    fire_values = np.asarray([row["n_fire"] for row in values], dtype=float)
    U_values = np.asarray([row["Uapp"] for row in values], dtype=float)
    F_values = np.asarray([row["Ftop"] for row in values], dtype=float)
    phi_values = np.asarray(
        [row.get("plastic_accommodation_ratio", 0.0) for row in values],
        dtype=float,
    )
    elastic_values = np.asarray(
        [row.get("elastic_accommodation_ratio", 1.0) for row in values],
        dtype=float,
    )
    active_values = np.asarray(
        [row.get("active_plastic_area_fraction", 0.0) for row in values],
        dtype=float,
    )
    stagger_values = np.asarray(
        [row.get("stagger_relative_change", 0.0) for row in values],
        dtype=float,
    )

    crack_span = float(np.max(a_values) - np.min(a_values))
    n_fire = int(np.count_nonzero(fire_values > 0.0))
    positive_dB = float(np.sum(np.maximum(np.diff(B_values), 0.0)))

    dWext = float(last["W_ext"] - first["W_ext"])
    dUel = float(last["U_el"] - first["U_el"])
    dWp = float(last["W_p"] - first["W_p"])
    dWemit = float(last["W_emit"] - first["W_emit"])
    energy_floor = max(1.0e-12, 1.0e-3 * abs(float(last["W_ext"])))
    window_residual = dWext - dUel - dWp - dWemit
    window_scale = max(
        abs(dWext),
        abs(dUel) + abs(dWp) + abs(dWemit),
        energy_floor,
    )
    window_error = abs(window_residual) / window_scale

    cumulative_Wext = float(last["W_ext"])
    cumulative_residual = (
        cumulative_Wext
        - float(cumulative_Uel)
        - float(cumulative_Wp)
        - float(cumulative_Wemit)
    )
    cumulative_scale = max(
        abs(cumulative_Wext),
        abs(float(cumulative_Uel))
        + abs(float(cumulative_Wp))
        + abs(float(cumulative_Wemit)),
        energy_floor,
    )
    cumulative_error = abs(cumulative_residual) / cumulative_scale
    energy_tolerance = float(
        getattr(args, "plastic_flow_energy_balance_tolerance", 0.01) or 0.01
    )
    negative_work_tolerance = max(
        1.0e-12,
        1.0e-15 * max(abs(float(last["W_p"])), 1.0),
    )

    cumulative_activity_scale = max(
        abs(cumulative_Wext),
        abs(float(cumulative_Uel))
        + max(float(cumulative_Wp), 0.0)
        + max(float(cumulative_Wemit), 0.0),
        energy_floor,
    )
    cumulative_plastic_fraction = (
        max(float(cumulative_Wp), 0.0) / cumulative_activity_scale
    )

    displacement_span_required = max(
        1.0e-4 * max(float(np.max(np.abs(U_values))), 1.0e-30),
        1.0e-15,
    )
    tangent_fit_valid = bool(
        len(U_values) >= 3 and np.ptp(U_values) >= displacement_span_required
    )
    if tangent_fit_valid:
        coeff = np.polyfit(U_values, F_values, 1)
        tangent = abs(float(coeff[0]))
        fit = np.polyval(coeff, U_values)
        tangent_fit_relative_rmse = float(
            np.sqrt(np.mean((F_values - fit) ** 2))
            / max(float(np.ptp(F_values)), abs(float(peak_force)), 1.0e-30)
        )
        normalized_tangent = tangent / max(
            abs(float(stiffness_reference)), 1.0e-30
        )
    else:
        tangent_fit_relative_rmse = None
        normalized_tangent = float("inf")

    tail_count = max(min(len(values), max(required // 10, 3)), 3)
    force_fraction = float(np.median(np.abs(F_values[-tail_count:]))) / max(
        abs(float(peak_force)), 1.0e-30
    )
    phi_median = float(np.median(phi_values))
    elastic_median = float(np.median(elastic_values))
    active_median = float(np.median(active_values))
    stagger_max = float(np.max(np.maximum(stagger_values, 0.0)))

    minimum_phi = float(
        getattr(args, "plastic_flow_min_plastic_fraction", 0.50) or 0.50
    )
    maximum_elastic = float(
        getattr(args, "plastic_flow_max_elastic_fraction", 0.50) or 0.50
    )
    maximum_tangent = float(
        getattr(args, "plastic_flow_max_tangent_fraction", 0.50) or 0.50
    )
    minimum_cumulative_activity = float(
        getattr(args, "plastic_flow_min_cumulative_plastic_fraction", 0.10)
        or 0.10
    )
    minimum_active_area = 0.01
    maximum_stagger_change = 0.05

    lambda_positive = np.maximum(lambda_values, 0.0)
    lambda_max = float(np.max(lambda_positive))
    B_final = float(last["B"])
    if lambda_max <= 1.0e-300:
        remaining_cleavage_time = None
        remaining_cleavage_time_infinite = True
    else:
        remaining_cleavage_time = max(1.0 - B_final, 0.0) / lambda_max
        remaining_cleavage_time_infinite = False
    remaining_horizon = (
        max(int(remaining_steps), 0) * max(float(nominal_dt_s), 0.0)
    )
    cleavage_horizon_ratio = (
        None
        if remaining_cleavage_time is None or remaining_horizon <= 0.0
        else remaining_cleavage_time / remaining_horizon
    )

    criteria = {
        "no_sharp_fracture_first_passage": n_fire == 0,
        "negligible_crack_extension": crack_span
        < float(getattr(args, "plastic_flow_max_da_fraction", 0.1) or 0.1)
        * max(float(da_phys), 1.0e-30),
        "plastic_activity_latched": (
            float(cumulative_Wp) > negative_work_tolerance
            and cumulative_plastic_fraction >= minimum_cumulative_activity
        ),
        "plastic_accommodation_dominant": phi_median >= minimum_phi,
        "elastic_or_tangent_response_subdominant": (
            elastic_median <= maximum_elastic
            or normalized_tangent <= maximum_tangent
        ),
        "spatially_resolved_plastic_activity": active_median >= minimum_active_area,
        "stagger_iteration_converged": stagger_max <= maximum_stagger_change,
        "nonnegative_plastic_dissipation": (
            dWp >= -negative_work_tolerance
            and float(cumulative_Wp) >= -negative_work_tolerance
        ),
        "energy_balance_bounded": (
            window_error <= energy_tolerance
            and cumulative_error <= energy_tolerance
        ),
    }
    thresholds = {
        "minimum_plastic_accommodation_ratio": minimum_phi,
        "maximum_elastic_accommodation_ratio": maximum_elastic,
        "maximum_normalized_tangent_stiffness": maximum_tangent,
        "minimum_cumulative_plastic_activity_fraction":
            minimum_cumulative_activity,
        "minimum_active_plastic_area_fraction": minimum_active_area,
        "maximum_stagger_relative_change": maximum_stagger_change,
        "energy_balance_relative_tolerance": energy_tolerance,
        "negative_plastic_work_absolute_tolerance_J_per_m":
            negative_work_tolerance,
    }
    margins = {
        "plastic_accommodation_dominant": phi_median - minimum_phi,
        "elastic_accommodation_subdominant": maximum_elastic - elastic_median,
        "tangent_stiffness_subdominant":
            maximum_tangent - normalized_tangent,
        "cumulative_plastic_activity":
            cumulative_plastic_fraction - minimum_cumulative_activity,
        "active_plastic_area": active_median - minimum_active_area,
        "stagger_iteration_convergence":
            maximum_stagger_change - stagger_max,
        "window_energy_balance": energy_tolerance - window_error,
        "cumulative_energy_balance": energy_tolerance - cumulative_error,
    }
    failed = [name for name, passed in criteria.items() if not passed]

    return {
        "criteria": criteria,
        "criteria_pass": all(criteria.values()),
        "failed_criteria": failed,
        "thresholds": thresholds,
        "criteria_margins": margins,
        "window_first_step": int(first["step"]),
        "window_last_step": int(last["step"]),
        "classification_window_steps": len(values),
        "crack_extension_window_m": crack_span,
        "J_tip_positive_max_window_J_per_m2":
            float(np.max(np.maximum(j_values, 0.0))),
        "sigma_tip_max_window_Pa":
            float(np.max(np.maximum(sigma_values, 0.0))),
        "cleavage_action_increment_window": positive_dB,
        "cleavage_event_count_window": n_fire,
        "W_external_increment_window_J_per_m": dWext,
        "U_elastic_change_window_J_per_m": dUel,
        "W_bulk_plastic_increment_window_J_per_m": dWp,
        "W_tip_emit_increment_window_J_per_m": dWemit,
        "window_energy_balance_residual_J_per_m": window_residual,
        "window_energy_balance_scale_J_per_m": window_scale,
        "window_energy_balance_relative_error": window_error,
        "cumulative_energy_balance_residual_J_per_m": cumulative_residual,
        "cumulative_energy_balance_scale_J_per_m": cumulative_scale,
        "cumulative_energy_balance_relative_error": cumulative_error,
        "cumulative_plastic_fraction": cumulative_plastic_fraction,
        "plastic_accommodation_ratio_median": phi_median,
        "plastic_accommodation_ratio_min": float(np.min(phi_values)),
        "plastic_accommodation_ratio_max": float(np.max(phi_values)),
        "elastic_accommodation_ratio_median": elastic_median,
        "active_plastic_area_fraction_median": active_median,
        "reaction_force_fraction_of_peak_window_median": force_fraction,
        "normalized_tangent_stiffness": (
            normalized_tangent if np.isfinite(normalized_tangent) else None
        ),
        "tangent_fit_valid": tangent_fit_valid,
        "tangent_fit_relative_rmse": tangent_fit_relative_rmse,
        "stagger_relative_change_max": stagger_max,
        "lambda_cleave_max_window_per_s": lambda_max,
        "B_final": B_final,
        "predicted_remaining_cleavage_time_s": remaining_cleavage_time,
        "predicted_remaining_cleavage_time_infinite":
            remaining_cleavage_time_infinite,
        "remaining_loading_horizon_s": remaining_horizon,
        "cleavage_horizon_ratio": cleavage_horizon_ratio,
        "plastic_terminal_is_model_limit_censor": True,
        "future_fracture_beyond_terminal_resolved": False,
    }


def contour_scan_v1043(
    compute_J_integral,
    *,
    mesh,
    u,
    sigma_gp,
    psi_gp,
    damage,
    tip_xy,
    direction,
    mat,
    base_ell_m,
    multipliers,
    crack_segments,
    exclude_radius_m,
    sign_reference,
):
    """Evaluate diagnostic contours with the production positive-J convention."""
    sign_ref = float(sign_reference)
    if not np.isclose(sign_ref, 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            "v10.4.3 contour diagnostics require J_sign_ref=1; "
            f"observed {sign_ref}"
        )
    if isinstance(multipliers, (list, tuple, np.ndarray)):
        raw = multipliers
    else:
        raw = str(multipliers).replace(",", " ").split()
    factors = sorted(
        {
            float(value)
            for value in raw
            if np.isfinite(float(value)) and float(value) > 0.0
        }
    ) or [1.0, 2.0, 4.0, 8.0]
    records = []
    for factor in factors:
        ell_m = max(float(base_ell_m) * factor, 1.0e-12)
        J_abs, K_abs, info = compute_J_integral(
            mesh,
            u,
            sigma_gp,
            psi_gp,
            damage,
            np.asarray(tip_xy, dtype=float),
            np.asarray(direction, dtype=float),
            mat,
            ell=ell_m,
            crack_segments=crack_segments,
            exclude_radius=max(float(exclude_radius_m), 0.0),
        )
        J_signed = float(info.get("J_signed", J_abs))
        records.append(
            {
                "ell_input_m": ell_m,
                "contour_multiplier": factor,
                "r_inner_m": float(info.get("r_inner", np.nan)),
                "r_outer_m": float(info.get("r_outer", np.nan)),
                "n_active_elements": int(info.get("n_active_elements", 0)),
                "J_absolute_J_per_m2": float(J_abs),
                "J_signed_J_per_m2": J_signed,
                "J_positive_root_convention_J_per_m2":
                    max(J_signed, 0.0),
                "K_absolute_MPa_sqrt_m": float(K_abs) / 1.0e6,
                "sign_reference": 1.0,
            }
        )
    records.sort(key=lambda item: item["r_outer_m"])
    return records


def transform_source(source: str) -> str:
    text = _v1042_transform(source)

    text = _replace_once(
        text,
        """        plastic_work_ledger_source_v1042 = 'constitutive_dWp_accepted_gp_all_staggers'
""",
        """        plastic_work_ledger_source_v1042 = 'constitutive_dWp_accepted_gp_final_stagger_iterate'
        plastic_dominance_phi_p_v1043 = 0.0
        plastic_dominance_elastic_ratio_v1043 = 1.0
        plastic_dominance_active_area_fraction_v1043 = 0.0
        plastic_dominance_stagger_relative_change_v1043 = 0.0
""",
        "v10.4.3 diagnostic initialization",
    )
    text = _replace_once(
        text,
        """                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = np.zeros(mesh.ne, dtype=float)
                for it in range(args.n_stagger):
""",
        """                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = np.zeros(mesh.ne, dtype=float)
                ep_gp_step0_v1043 = ep_gp.copy()
                rho_gp_step0_v1043 = rho_gp.copy()
                ep_gp_previous_iter_v1043 = None
                plastic_dominance_stagger_relative_change_v1043 = 0.0
                for it in range(args.n_stagger):
""",
        "v10.4.3 start-of-step constitutive snapshots",
    )
    text = _replace_once(
        text,
        """                        ep_gp, rho_gp, dot_ep, plastic_work_info_v1042 = update_plasticity(
                            ep_gp, rho_gp, sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations, return_info=True)
                        _v1042_dWp_stagger_gp = np.asarray(
                            plastic_work_info_v1042.get('dWp_accepted_gp', []),
                            dtype=float,
                        ).reshape(-1)
                        if _v1042_dWp_stagger_gp.size != mesh.ne:
                            raise RuntimeError(
                                'v10.4.2 accepted plastic-work ledger size mismatch: '
                                f'{_v1042_dWp_stagger_gp.size} != {mesh.ne}'
                            )
                        plastic_work_accepted_gp_v1042 += np.maximum(
                            _v1042_dWp_stagger_gp, 0.0
                        )
""",
        """                        ep_gp, rho_gp, dot_ep, plastic_work_info_v1042 = update_plasticity(
                            ep_gp_step0_v1043.copy(),
                            rho_gp_step0_v1043.copy(),
                            sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations, return_info=True)
                        _v1043_dWp_gp = np.asarray(
                            plastic_work_info_v1042.get('dWp_accepted_gp', []),
                            dtype=float,
                        ).reshape(-1)
                        if _v1043_dWp_gp.size != mesh.ne:
                            raise RuntimeError(
                                'v10.4.3 accepted plastic-work ledger size mismatch: '
                                f'{_v1043_dWp_gp.size} != {mesh.ne}'
                            )
                        _v1043_negative_tol = 1.0e-12
                        if np.any(_v1043_dWp_gp < -_v1043_negative_tol):
                            raise RuntimeError(
                                'v10.4.3 negative accepted plastic dissipation: '
                                f'min={float(np.min(_v1043_dWp_gp)):.6e}'
                            )
                        _v1043_dWp_gp = np.where(
                            np.abs(_v1043_dWp_gp) <= _v1043_negative_tol,
                            0.0,
                            _v1043_dWp_gp,
                        )
                        plastic_work_accepted_gp_v1042 = _v1043_dWp_gp.copy()
                        if ep_gp_previous_iter_v1043 is not None:
                            _v1043_iter_num = float(np.linalg.norm(
                                ep_gp - ep_gp_previous_iter_v1043
                            ))
                            _v1043_iter_den = max(
                                float(np.linalg.norm(ep_gp - ep_gp_step0_v1043)),
                                1.0e-16,
                            )
                            plastic_dominance_stagger_relative_change_v1043 = (
                                _v1043_iter_num / _v1043_iter_den
                            )
                        ep_gp_previous_iter_v1043 = ep_gp.copy()
""",
        "v10.4.3 re-based final-iterate constitutive update",
    )
    text = _replace_once(
        text,
        """                h_local = mesh.hbar_tip if mesh.hbar_tip > 0 else mesh.hbar
""",
        """                if not (fatigue_mode and cyclic_mechanics_enabled):
                    Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                        mesh, u, ep_gp, rho_gp, d, D, mat,
                        cohesive_network=cohesive_network,
                    )
                    u, Ftop = solve_dirichlet(
                        Kmat, Rint, u, bnd, Uy_top, Uy_bot
                    )
                    Kmat, Rint, sigma_gp, seq_gp, s1_gp, psi_gp = assemble_mechanics(
                        mesh, u, ep_gp, rho_gp, d, D, mat,
                        cohesive_network=cohesive_network,
                    )

                _v1043_height = max(float(np.ptp(mesh.nodes[:, 1])), 1.0e-30)
                _v1043_applied = float(dU_step) / _v1043_height
                _v1043_area = max(float(np.sum(mesh.area_e)), 1.0e-30)
                _v1043_plastic = float(
                    np.sum(
                        (ep_gp[1, :] - ep_gp_step0_v1043[1, :])
                        * mesh.area_e
                    ) / _v1043_area
                )
                plastic_dominance_phi_p_v1043 = (
                    _v1043_plastic / _v1043_applied
                    if abs(_v1043_applied) > 1.0e-30
                    else 0.0
                )
                plastic_dominance_elastic_ratio_v1043 = (
                    1.0 - plastic_dominance_phi_p_v1043
                )
                _v1043_dep_eq = np.asarray(
                    (
                        plastic_work_info_v1042
                        if isinstance(plastic_work_info_v1042, dict)
                        else {}
                    ).get('dep_eq_accepted_gp', np.zeros(mesh.ne)),
                    dtype=float,
                ).reshape(-1)
                if _v1043_dep_eq.size == mesh.ne:
                    _v1043_active_tol = max(
                        1.0e-12,
                        1.0e-6 * float(np.max(np.abs(_v1043_dep_eq))),
                    )
                    plastic_dominance_active_area_fraction_v1043 = float(
                        np.sum(
                            mesh.area_e[
                                np.abs(_v1043_dep_eq) > _v1043_active_tol
                            ]
                        ) / _v1043_area
                    )
                else:
                    plastic_dominance_active_area_fraction_v1043 = 0.0

                h_local = mesh.hbar_tip if mesh.hbar_tip > 0 else mesh.hbar
""",
        "v10.4.3 final-state equilibrium and accommodation diagnostics",
    )
    text = _replace_once(
        text,
        """            if (
                plastic_work_accepted_gp_v1042 is not None
                and np.asarray(plastic_work_accepted_gp_v1042).size == mesh.ne
                and isinstance(plastic_work_info_v1042, dict)
            ):
                dWp = float(
                    np.sum(
                        np.maximum(plastic_work_accepted_gp_v1042, 0.0)
                        * mesh.area_e
                    )
                )
                plastic_work_ledger_source_v1042 = (
                    'constitutive_dWp_accepted_gp_all_staggers'
                )
            else:
                dWp = float(
                    np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)
                ) * dt_cur
                plastic_work_ledger_source_v1042 = (
                    'post_update_sigma_dot_ep_fallback'
                )
            W_p_acc += max(dWp, 0.0)
""",
        """            if (
                plastic_work_accepted_gp_v1042 is not None
                and np.asarray(plastic_work_accepted_gp_v1042).size == mesh.ne
                and isinstance(plastic_work_info_v1042, dict)
            ):
                dWp = float(
                    np.sum(
                        np.asarray(plastic_work_accepted_gp_v1042, dtype=float)
                        * mesh.area_e
                    )
                )
                plastic_work_ledger_source_v1042 = (
                    'constitutive_dWp_accepted_gp_final_stagger_iterate'
                )
            else:
                dWp = float(
                    np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)
                ) * dt_cur
                plastic_work_ledger_source_v1042 = (
                    'post_update_sigma_dot_ep_fallback'
                )
            if dWp < -1.0e-12:
                raise RuntimeError(
                    'v10.4.3 negative integrated plastic dissipation: '
                    f'{dWp:.6e} J/m'
                )
            if abs(dWp) <= 1.0e-12:
                dWp = 0.0
            W_p_acc += dWp
""",
        "v10.4.3 final-iterate accepted-work ledger",
    )
    text = _replace_once(
        text,
        """                'W_emit': float(W_emit_tot),
            })
            if _v1042_capture_peak_load and Kc_first is None:
""",
        """                'W_emit': float(W_emit_tot),
                'plastic_accommodation_ratio': float(
                    plastic_dominance_phi_p_v1043
                ),
                'elastic_accommodation_ratio': float(
                    plastic_dominance_elastic_ratio_v1043
                ),
                'active_plastic_area_fraction': float(
                    plastic_dominance_active_area_fraction_v1043
                ),
                'stagger_relative_change': float(
                    plastic_dominance_stagger_relative_change_v1043
                ),
            })
            if _v1042_capture_peak_load and Kc_first is None:
""",
        "v10.4.3 accepted-window accommodation fields",
    )
    text = _replace_once(
        text,
        """                if _v1042_metrics is not None and _v1042_metrics['criteria_pass']:
""",
        """                if _v1042_metrics is not None:
                    _v1043_candidate = {
                        'schema': 'v10.4.3_plastic_dominance_candidate_v1',
                        'classification': 'plastic_flow_no_sharp_fracture',
                        'terminal': False,
                        'campaign_terminal': False,
                        'temperature_K': float(T),
                        'evaluation_step': int(step),
                        'plastic_terminal_is_model_limit_censor': True,
                        'interpretation': (
                            'no_sharp_fracture_before_sustained_'
                            'plastic_dominance'
                        ),
                        'future_fracture_beyond_terminal_resolved': False,
                        'plastic_work_ledger_source':
                            plastic_work_ledger_source_v1042,
                        'stagger_iterations': int(args.n_stagger),
                        **_v1042_metrics,
                    }
                    with open(
                        os.path.join(
                            args.out,
                            'plastic_flow_candidate_latest.json',
                        ),
                        'w',
                    ) as _v1043_fp:
                        json.dump(
                            _v1043_candidate,
                            _v1043_fp,
                            indent=2,
                            sort_keys=True,
                        )
                        _v1043_fp.write('\n')
                    with open(
                        os.path.join(
                            args.out,
                            'plastic_flow_candidate_history.jsonl',
                        ),
                        'a',
                    ) as _v1043_fp:
                        _v1043_fp.write(
                            json.dumps(_v1043_candidate, sort_keys=True)
                            + '\n'
                        )
                if _v1042_metrics is not None and _v1042_metrics['criteria_pass']:
""",
        "v10.4.3 unconditional candidate diagnostics",
    )
    text = _replace_once(
        text,
        "'schema': 'v10.4.2_plastic_flow_terminal_audit_v1'",
        "'schema': 'v10.4.3_plastic_dominance_terminal_audit_v1'",
        "v10.4.3 terminal schema",
    )
    text = _replace_once(
        text,
        """                        'failure_regime': 'bulk_plastic_flow',
""",
        """                        'failure_regime': 'bulk_plastic_dominance_model_limit',
                        'plastic_terminal_is_model_limit_censor': True,
                        'interpretation': (
                            'no_sharp_fracture_before_sustained_'
                            'plastic_dominance'
                        ),
                        'future_fracture_beyond_terminal_resolved': False,
                        'post_terminal_ductile_failure_modeled': False,
""",
        "v10.4.3 terminal interpretation",
    )
    text = _replace_once(
        text,
        """                    _v1042_Wp_balance = max(
                        float(W_ext_acc) - float(U_el) - float(W_emit_tot), 0.0
                    )
""",
        """                    _v1042_Wp_balance = (
                        float(W_ext_acc) - float(U_el) - float(W_emit_tot)
                    )
""",
        "v10.4.3 unclamped balance estimate",
    )
    text = _replace_once(
        text,
        """                    _v1042_energy_scale = max(
                        abs(float(W_ext_acc)),
                        abs(float(U_el)) + abs(float(W_p_acc)) + abs(float(W_emit_tot)),
                        1.0e-30,
                    )
""",
        """                    _v1042_energy_scale = max(
                        abs(float(W_ext_acc)),
                        abs(float(U_el)) + abs(float(W_p_acc)) + abs(float(W_emit_tot)),
                        1.0e-3 * abs(float(W_ext_acc)),
                        1.0e-12,
                    )
""",
        "v10.4.3 scale-aware cumulative energy denominator",
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
        raise RuntimeError("could not allocate v10.4.3 plastic-dominance module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-plastic-dominance]",
                "exec",
            ),
            module.__dict__,
        )
        module._v1042_terminal_metrics = terminal_metrics_v1043
        module._v1042_contour_scan = contour_scan_v1043
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = [
    "MODEL_ID",
    "MODULE_NAME",
    "contour_scan_v1043",
    "load_transformed_sharp_front",
    "terminal_metrics_v1043",
    "transform_source",
]
