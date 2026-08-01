"""Finalize v10.4.2 accepted plastic-work and contour-history accounting.

The underlying driver historically estimates accepted-step plastic work from
the final stress and final plastic strain rate. That expression can vanish after
a fully relaxed staggered solve even when positive plastic work was accepted
inside the constitutive updates. This overlay requests ``dWp_accepted_gp`` from
every stagger, sums the accepted contributions, and uses their area integral.

It also retains one lightweight mechanics checkpoint near the historical peak
reaction force. Terminal contour shielding is evaluated at both that peak-load
state and the fully relaxed terminal state; neither diagnostic is fed into the
cleavage hazard or energy gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_terminal_v1042 import transform_source as _terminal_transform

MODEL_ID = "v10.4.2_accepted_constitutive_plastic_work_peak_load_contours"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1042_accepted_work"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _terminal_transform(source)

    text = _replace_once(
        text,
        "        plastic_flow_stiffness_reference = 0.0\n",
        "        plastic_flow_stiffness_reference = 0.0\n"
        "        plastic_flow_peak_load_state_v1042 = None\n"
        "        plastic_work_info_v1042 = None\n"
        "        plastic_work_accepted_gp_v1042 = None\n"
        "        plastic_work_ledger_source_v1042 = "
        "'constitutive_dWp_accepted_gp_all_staggers'\n",
        "v10.4.2 accepted-work initialization",
    )

    text = _replace_once(
        text,
        """                sigma_gp = np.zeros((3, mesh.ne)); psi_gp = np.zeros(mesh.ne); Ftop = 0.0
                for it in range(args.n_stagger):
""",
        """                sigma_gp = np.zeros((3, mesh.ne)); psi_gp = np.zeros(mesh.ne); Ftop = 0.0
                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = np.zeros(mesh.ne, dtype=float)
                for it in range(args.n_stagger):
""",
        "v10.4.2 per-trial accepted-work reset",
    )

    text = _replace_once(
        text,
        """                        ep_gp, rho_gp, dot_ep = update_plasticity(
                            ep_gp, rho_gp, sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations)
""",
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
        "v10.4.2 constitutive accepted-work request",
    )

    text = _replace_once(
        text,
        """            dWp = float(np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)) * dt_cur
            W_p_acc += max(dWp, 0.0)
""",
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
        "v10.4.2 accepted-work accumulation",
    )

    text = _replace_once(
        text,
        """            plastic_flow_peak_force = max(plastic_flow_peak_force, abs(float(Ftop)))
""",
        """            _v1042_force_abs = abs(float(Ftop))
            _v1042_capture_peak_load = (
                plastic_flow_peak_load_state_v1042 is None
                or _v1042_force_abs > max(
                    plastic_flow_peak_force * 1.005,
                    plastic_flow_peak_force + 1.0e-12,
                )
            )
            plastic_flow_peak_force = max(plastic_flow_peak_force, _v1042_force_abs)
""",
        "v10.4.2 peak-load checkpoint trigger",
    )

    text = _replace_once(
        text,
        """                'W_emit': float(W_emit_tot),
            })
""",
        """                'W_emit': float(W_emit_tot),
            })
            if _v1042_capture_peak_load and Kc_first is None:
                if deflect and fronts:
                    _v1042_peak_root = fronts[0]
                    _v1042_peak_src, _v1042_peak_ell, _v1042_peak_segments = (
                        _J_params_for_front(_v1042_peak_root)
                    )
                    _v1042_peak_tip = np.asarray(
                        _v1042_peak_root['xy'], dtype=float
                    )
                    _v1042_peak_dir = np.asarray(
                        _v1042_peak_root.get(
                            't_win', _v1042_peak_root.get('fwd', [1.0, 0.0])
                        ),
                        dtype=float,
                    )
                    _v1042_peak_sign = float(
                        _v1042_peak_root.get('J_sign_ref', 1.0) or 1.0
                    )
                    _v1042_peak_exclude = 2.0 * kill_r
                else:
                    _v1042_peak_src = 'cluster'
                    _v1042_peak_ell = max(r_J_cluster_ell, 3.0 * h_local)
                    _v1042_peak_segments = _backend_crack_segments()
                    _v1042_peak_tip = np.asarray([a_tip, 0.0], dtype=float)
                    _v1042_peak_dir = np.asarray([1.0, 0.0], dtype=float)
                    _v1042_peak_sign = 1.0
                    _v1042_peak_exclude = 0.0
                plastic_flow_peak_load_state_v1042 = {
                    'step': int(step),
                    'Uapp_m': float(Uapp),
                    'reaction_force_N': float(Ftop),
                    'J_tip_positive_J_per_m2': float(J_tip_positive_v1042),
                    'J_tip_signed_J_per_m2': float(J_tip_signed_v1042),
                    'mesh': mesh,
                    'u': u.copy(),
                    'sigma_gp': sigma_gp.copy(),
                    'psi_gp': psi_gp.copy(),
                    'damage': d.copy(),
                    'tip_xy': _v1042_peak_tip.copy(),
                    'direction': _v1042_peak_dir.copy(),
                    'base_ell_m': float(_v1042_peak_ell),
                    'crack_segments': copy.deepcopy(_v1042_peak_segments),
                    'exclude_radius_m': float(_v1042_peak_exclude),
                    'sign_reference': float(_v1042_peak_sign),
                    'contour_source': _v1042_peak_src,
                }
""",
        "v10.4.2 peak-load mechanics checkpoint",
    )

    text = _replace_once(
        text,
        """                    _v1042_J_outer = (
                        float(_v1042_contours[-1]['J_positive_root_convention_J_per_m2'])
                        if _v1042_contours else float(J_tip_positive_v1042)
                    )
                    _v1042_J_shield = max(
                        _v1042_J_outer - float(J_tip_positive_v1042), 0.0
                    )
                    _v1042_ligament = max(
""",
        """                    _v1042_J_outer = (
                        float(_v1042_contours[-1]['J_positive_root_convention_J_per_m2'])
                        if _v1042_contours else float(J_tip_positive_v1042)
                    )
                    _v1042_J_shield_final = max(
                        _v1042_J_outer - float(J_tip_positive_v1042), 0.0
                    )
                    _v1042_peak_contours = []
                    _v1042_J_outer_peak = 0.0
                    _v1042_J_tip_peak = 0.0
                    _v1042_J_shield_peak = 0.0
                    if plastic_flow_peak_load_state_v1042 is not None:
                        _v1042_peak_state = plastic_flow_peak_load_state_v1042
                        _v1042_peak_contours = _v1042_contour_scan(
                            compute_J_integral,
                            mesh=_v1042_peak_state['mesh'],
                            u=_v1042_peak_state['u'],
                            sigma_gp=_v1042_peak_state['sigma_gp'],
                            psi_gp=_v1042_peak_state['psi_gp'],
                            damage=_v1042_peak_state['damage'],
                            tip_xy=_v1042_peak_state['tip_xy'],
                            direction=_v1042_peak_state['direction'],
                            mat=mat,
                            base_ell_m=_v1042_peak_state['base_ell_m'],
                            multipliers=getattr(
                                args, 'plastic_flow_contour_multipliers', '1 2 4 8'
                            ),
                            crack_segments=_v1042_peak_state['crack_segments'],
                            exclude_radius_m=_v1042_peak_state['exclude_radius_m'],
                            sign_reference=_v1042_peak_state['sign_reference'],
                        )
                        _v1042_J_tip_peak = float(
                            _v1042_peak_state['J_tip_positive_J_per_m2']
                        )
                        _v1042_J_outer_peak = (
                            float(
                                _v1042_peak_contours[-1][
                                    'J_positive_root_convention_J_per_m2'
                                ]
                            )
                            if _v1042_peak_contours else _v1042_J_tip_peak
                        )
                        _v1042_J_shield_peak = max(
                            _v1042_J_outer_peak - _v1042_J_tip_peak, 0.0
                        )
                    _v1042_J_shield = max(
                        _v1042_J_shield_final, _v1042_J_shield_peak
                    )
                    _v1042_ligament = max(
""",
        "v10.4.2 peak and terminal contour scans",
    )

    text = _replace_once(
        text,
        """                        'J_outer_positive_final_J_per_m2': _v1042_J_outer,
                        'J_contour_shielding_J_per_m2': _v1042_J_shield,
                        'contour_shielding_definition': 'max(J_outer_positive-J_tip_positive,0)',
                        'contour_shielding_is_diagnostic_only': True,
                        'contour_shielding_enters_fracture_hazard': False,
                        'contour_source': _v1042_src,
                        'contour_scan': _v1042_contours,
""",
        """                        'J_outer_positive_final_J_per_m2': _v1042_J_outer,
                        'J_contour_shielding_final_J_per_m2': _v1042_J_shield_final,
                        'J_tip_positive_peak_load_J_per_m2': _v1042_J_tip_peak,
                        'J_outer_positive_peak_load_J_per_m2': _v1042_J_outer_peak,
                        'J_contour_shielding_peak_load_J_per_m2': _v1042_J_shield_peak,
                        'J_contour_shielding_J_per_m2': _v1042_J_shield,
                        'contour_shielding_reported_value': (
                            'max_peak_load_and_terminal_outer_minus_tip_positive_J'
                        ),
                        'contour_shielding_definition': 'max(J_outer_positive-J_tip_positive,0)',
                        'contour_shielding_is_diagnostic_only': True,
                        'contour_shielding_enters_fracture_hazard': False,
                        'contour_source_final': _v1042_src,
                        'contour_scan_final': _v1042_contours,
                        'contour_peak_load_step': (
                            None if plastic_flow_peak_load_state_v1042 is None
                            else int(plastic_flow_peak_load_state_v1042['step'])
                        ),
                        'contour_peak_load_reaction_force_N': (
                            None if plastic_flow_peak_load_state_v1042 is None
                            else float(
                                plastic_flow_peak_load_state_v1042[
                                    'reaction_force_N'
                                ]
                            )
                        ),
                        'contour_source_peak_load': (
                            None if plastic_flow_peak_load_state_v1042 is None
                            else plastic_flow_peak_load_state_v1042['contour_source']
                        ),
                        'contour_scan_peak_load': _v1042_peak_contours,
""",
        "v10.4.2 contour audit provenance",
    )

    text = _replace_once(
        text,
        """                        'W_bulk_plastic_J_per_m': float(W_p_acc),
                        'W_bulk_plastic_balance_estimate_J_per_m': _v1042_Wp_balance,
""",
        """                        'W_bulk_plastic_J_per_m': float(W_p_acc),
                        'W_bulk_plastic_ledger_source': plastic_work_ledger_source_v1042,
                        'W_bulk_plastic_primary_is_constitutive_accepted_work': (
                            plastic_work_ledger_source_v1042
                            == 'constitutive_dWp_accepted_gp_all_staggers'
                        ),
                        'W_bulk_plastic_stagger_iterations_accumulated': int(args.n_stagger),
                        'W_bulk_plastic_balance_estimate_J_per_m': _v1042_Wp_balance,
""",
        "v10.4.2 accepted-work audit provenance",
    )

    text = _replace_once(
        text,
        """        'predicted_remaining_cleavage_time_s': remaining_cleavage_time,
        'remaining_loading_horizon_s': remaining_loading_horizon,
        'cleavage_horizon_ratio': cleavage_horizon_ratio,
""",
        """        'predicted_remaining_cleavage_time_s': (
            remaining_cleavage_time if np.isfinite(remaining_cleavage_time) else None
        ),
        'predicted_remaining_cleavage_time_infinite': bool(
            not np.isfinite(remaining_cleavage_time)
        ),
        'remaining_loading_horizon_s': remaining_loading_horizon,
        'cleavage_horizon_ratio': (
            cleavage_horizon_ratio if np.isfinite(cleavage_horizon_ratio) else None
        ),
        'cleavage_horizon_ratio_infinite': bool(
            not np.isfinite(cleavage_horizon_ratio)
        ),
""",
        "v10.4.2 finite JSON cleavage-horizon diagnostics",
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
        raise RuntimeError("could not allocate v10.4.2 accepted-work module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(compile(transformed, str(source_path) + "[v10.4.2]", "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
