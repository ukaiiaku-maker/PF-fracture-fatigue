"""v10.4.2 plastic-flow terminal overlay for the 2-D sharp-front solver.

The validated sharp-fracture criterion is unchanged: the cleavage first-passage
hazard consumes only the positive signed configurational J computed from the
stored elastic field.  This overlay adds a successful terminal state when a
persistent accepted-step window demonstrates that bulk plastic flow has removed
access to a sharp-tip fracture drive.

The overlay also records:
  * cumulative bulk-plastic dissipation intensity Wp/(B*b0),
  * an equivalent K scale used only for plotting,
  * multi-contour elastic configurational-J values,
  * the difference between the outer and tip positive J as a contour-shielding
    diagnostic.

No plastic-work quantity is fed back into the fracture hazard or energy gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MODEL_ID = "v10.4.2_plastic_flow_terminal_and_contour_shielding"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1042_transformed"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


_HELPERS = r'''

# ---------------------------------------------------------------------------
# v10.4.2 plastic-flow terminal helpers. These functions are diagnostics and
# termination logic only. They never modify the cleavage driving J or hazard.
# ---------------------------------------------------------------------------

def _v1042_float_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = value
    else:
        raw = str(value).replace(',', ' ').split()
    result = []
    for item in raw:
        number = float(item)
        if np.isfinite(number) and number > 0.0:
            result.append(number)
    if not result:
        result = [1.0, 2.0, 4.0, 8.0]
    return sorted(set(result))


def _v1042_terminal_metrics(window, args, *, Eprime, da_phys, sigma_reference,
                              peak_J_positive, peak_force, stiffness_reference,
                              remaining_steps, nominal_dt_s,
                              cumulative_Wp, cumulative_Uel,
                              cumulative_Wemit):
    required = max(int(getattr(args, 'plastic_flow_window_steps', 2000) or 2000), 2)
    minimum_step = max(int(getattr(args, 'plastic_flow_min_step', required) or required), required)
    if len(window) < required or int(window[-1]['step']) < minimum_step:
        return None

    values = list(window)
    first = values[0]
    last = values[-1]
    a_values = np.asarray([row['a_tip'] for row in values], dtype=float)
    j_values = np.asarray([row['J_positive'] for row in values], dtype=float)
    sigma_values = np.asarray([row['sigma_tip'] for row in values], dtype=float)
    B_values = np.asarray([row['B'] for row in values], dtype=float)
    lambda_values = np.asarray([row['lambda_c'] for row in values], dtype=float)
    fire_values = np.asarray([row['n_fire'] for row in values], dtype=float)
    U_values = np.asarray([row['Uapp'] for row in values], dtype=float)
    F_values = np.asarray([row['Ftop'] for row in values], dtype=float)

    crack_span = float(np.max(a_values) - np.min(a_values))
    j_max = float(np.max(np.maximum(j_values, 0.0)))
    sigma_max = float(np.max(np.maximum(sigma_values, 0.0)))
    positive_dB = float(np.sum(np.maximum(np.diff(B_values), 0.0)))
    n_fire = int(np.count_nonzero(fire_values > 0.0))

    dWext = max(float(last['W_ext'] - first['W_ext']), 0.0)
    dUel = abs(float(last['U_el'] - first['U_el']))
    dWp = max(float(last['W_p'] - first['W_p']), 0.0)
    dWemit = max(float(last['W_emit'] - first['W_emit']), 0.0)
    energy_scale = max(dWext, dWp + dUel + dWemit, 1.0e-30)
    plastic_fraction_window = dWp / energy_scale
    elastic_fraction_window = dUel / energy_scale

    cumulative_scale = max(
        max(float(cumulative_Wp), 0.0)
        + max(float(cumulative_Uel), 0.0)
        + max(float(cumulative_Wemit), 0.0),
        1.0e-30,
    )
    cumulative_plastic_fraction = max(float(cumulative_Wp), 0.0) / cumulative_scale

    force_fraction = abs(float(last['Ftop'])) / max(abs(float(peak_force)), 1.0e-30)
    if np.ptp(U_values) > 1.0e-30 and len(U_values) >= 3:
        tangent = abs(float(np.polyfit(U_values, F_values, 1)[0]))
    else:
        tangent = 0.0
    normalized_tangent = tangent / max(abs(float(stiffness_reference)), 1.0e-30)

    j_tolerance = max(
        float(getattr(args, 'plastic_flow_J_abs_tol_J_per_m2', 1.0e-6) or 0.0),
        float(getattr(args, 'plastic_flow_J_rel_tol', 1.0e-6) or 0.0)
        * max(float(peak_J_positive), 0.0),
    )
    sigma_tolerance = (
        float(getattr(args, 'plastic_flow_sigma_rel_tol', 1.0e-6) or 0.0)
        * max(float(sigma_reference), 1.0)
    )

    lambda_max = float(np.max(np.maximum(lambda_values, 0.0)))
    B_final = float(last['B'])
    if lambda_max <= 1.0e-300:
        remaining_cleavage_time = float('inf')
    else:
        remaining_cleavage_time = max(1.0 - B_final, 0.0) / lambda_max
    remaining_loading_horizon = max(int(remaining_steps), 0) * max(float(nominal_dt_s), 0.0)
    if remaining_loading_horizon <= 0.0:
        cleavage_horizon_ratio = float('inf')
    else:
        cleavage_horizon_ratio = remaining_cleavage_time / remaining_loading_horizon

    criteria = {
        'no_crack_event_in_window': n_fire == 0,
        'negligible_crack_extension': crack_span < (
            float(getattr(args, 'plastic_flow_max_da_fraction', 0.1) or 0.1)
            * max(float(da_phys), 1.0e-30)
        ),
        'negligible_positive_tip_J': j_max <= j_tolerance,
        'negligible_tip_stress': sigma_max <= sigma_tolerance,
        'plastic_accommodation_dominant': (
            plastic_fraction_window
            >= float(getattr(args, 'plastic_flow_min_plastic_fraction', 0.90) or 0.90)
            or cumulative_plastic_fraction
            >= float(getattr(args, 'plastic_flow_min_cumulative_plastic_fraction', 0.90) or 0.90)
        ),
        'elastic_storage_flat': elastic_fraction_window
        <= float(getattr(args, 'plastic_flow_max_elastic_fraction', 0.05) or 0.05),
        'load_carrying_capacity_collapsed': (
            force_fraction
            <= float(getattr(args, 'plastic_flow_max_force_fraction', 0.10) or 0.10)
            or normalized_tangent
            <= float(getattr(args, 'plastic_flow_max_tangent_fraction', 0.05) or 0.05)
        ),
        'cleavage_clock_stalled': positive_dB
        <= float(getattr(args, 'plastic_flow_max_dB_window', 1.0e-6) or 1.0e-6),
        'cleavage_outside_remaining_horizon': cleavage_horizon_ratio
        >= float(getattr(args, 'plastic_flow_min_cleavage_horizon_ratio', 100.0) or 100.0),
    }

    return {
        'criteria': criteria,
        'criteria_pass': all(criteria.values()),
        'window_first_step': int(first['step']),
        'window_last_step': int(last['step']),
        'classification_window_steps': len(values),
        'crack_extension_window_m': crack_span,
        'J_tip_positive_max_window_J_per_m2': j_max,
        'J_tip_positive_tolerance_J_per_m2': j_tolerance,
        'sigma_tip_max_window_Pa': sigma_max,
        'sigma_tip_tolerance_Pa': sigma_tolerance,
        'cleavage_action_increment_window': positive_dB,
        'cleavage_event_count_window': n_fire,
        'W_external_increment_window_J_per_m': dWext,
        'U_elastic_change_window_J_per_m': dUel,
        'W_bulk_plastic_increment_window_J_per_m': dWp,
        'W_tip_emit_increment_window_J_per_m': dWemit,
        'plastic_work_fraction_window': plastic_fraction_window,
        'elastic_storage_fraction_window': elastic_fraction_window,
        'cumulative_plastic_fraction': cumulative_plastic_fraction,
        'reaction_force_fraction_of_peak': force_fraction,
        'normalized_tangent_stiffness': normalized_tangent,
        'lambda_cleave_max_window_per_s': lambda_max,
        'B_final': B_final,
        'predicted_remaining_cleavage_time_s': remaining_cleavage_time,
        'remaining_loading_horizon_s': remaining_loading_horizon,
        'cleavage_horizon_ratio': cleavage_horizon_ratio,
    }


def _v1042_contour_scan(compute_J_integral, *, mesh, u, sigma_gp,
                         psi_gp, damage, tip_xy, direction, mat,
                         base_ell_m, multipliers, crack_segments,
                         exclude_radius_m, sign_reference):
    records = []
    sign_ref = float(sign_reference)
    if sign_ref == 0.0:
        sign_ref = 1.0
    for multiplier in _v1042_float_list(multipliers):
        ell_m = max(float(base_ell_m) * multiplier, 1.0e-12)
        J_abs, K_abs, info = compute_J_integral(
            mesh, u, sigma_gp, psi_gp, damage,
            np.asarray(tip_xy, dtype=float),
            np.asarray(direction, dtype=float),
            mat, ell=ell_m,
            crack_segments=crack_segments,
            exclude_radius=max(float(exclude_radius_m), 0.0),
        )
        J_signed = float(info.get('J_signed', J_abs))
        J_positive = max(sign_ref * J_signed, 0.0)
        records.append({
            'ell_input_m': ell_m,
            'contour_multiplier': float(multiplier),
            'r_inner_m': float(info.get('r_inner', np.nan)),
            'r_outer_m': float(info.get('r_outer', np.nan)),
            'n_active_elements': int(info.get('n_active_elements', 0)),
            'J_absolute_J_per_m2': float(J_abs),
            'J_signed_J_per_m2': J_signed,
            'J_positive_root_convention_J_per_m2': J_positive,
            'K_absolute_MPa_sqrt_m': float(K_abs) / 1.0e6,
            'sign_reference': sign_ref,
        })
    records.sort(key=lambda item: item['r_outer_m'])
    return records
'''


_INIT_BLOCK = r'''
        plastic_flow_terminal = False
        plastic_flow_terminal_audit = None
        plastic_flow_window_size = max(
            int(getattr(args, 'plastic_flow_window_steps', 2000) or 2000), 2
        )
        plastic_flow_window = deque(maxlen=plastic_flow_window_size)
        plastic_flow_peak_J_positive = 0.0
        plastic_flow_peak_force = 0.0
        plastic_flow_stiffness_reference = 0.0
'''


_WINDOW_APPEND = r'''
            J_tip_positive_v1042 = max(float(KJ), 0.0) ** 2 / max(float(mat.Eprime), 1.0e-30)
            if deflect and fronts:
                J_tip_signed_v1042 = float(
                    fronts[0].get('J_signed_trial', fronts[0].get('J_effective_trial', J_tip_positive_v1042))
                )
            else:
                J_tip_signed_v1042 = J_tip_positive_v1042
            plastic_flow_peak_J_positive = max(
                plastic_flow_peak_J_positive, J_tip_positive_v1042
            )
            plastic_flow_peak_force = max(plastic_flow_peak_force, abs(float(Ftop)))
            if abs(float(Uapp)) > 1.0e-30:
                plastic_flow_stiffness_reference = max(
                    plastic_flow_stiffness_reference,
                    abs(float(Ftop) / float(Uapp)),
                )
            plastic_flow_window.append({
                'step': int(step),
                'Uapp': float(Uapp),
                'Ftop': float(Ftop),
                'J_positive': float(J_tip_positive_v1042),
                'J_signed': float(J_tip_signed_v1042),
                'sigma_tip': float(info['sigma_tip']),
                'B': float(info['B']),
                'lambda_c': float(info['lambda_c']),
                'n_fire': int(info.get('n_fire', 0)),
                'a_tip': float(a_tip),
                'W_ext': float(W_ext_acc),
                'U_el': float(U_el),
                'W_p': float(W_p_acc),
                'W_emit': float(W_emit_tot),
            })
'''


_TERMINAL_BLOCK = r'''
            if (bool(getattr(args, 'plastic_flow_terminal', False))
                    and not fatigue_mode and Kc_first is None):
                _v1042_metrics = _v1042_terminal_metrics(
                    plastic_flow_window,
                    args,
                    Eprime=mat.Eprime,
                    da_phys=da_phys,
                    sigma_reference=(
                        eng.f.sigma_cap if float(getattr(eng.f, 'sigma_cap', 0.0)) > 0.0
                        else max(float(np.max(np.maximum(hist['sigma_tip'], 0.0))), 1.0)
                    ),
                    peak_J_positive=plastic_flow_peak_J_positive,
                    peak_force=plastic_flow_peak_force,
                    stiffness_reference=plastic_flow_stiffness_reference,
                    remaining_steps=max(int(args.steps) - int(step), 0),
                    nominal_dt_s=cfg.loading.dt,
                    cumulative_Wp=W_p_acc,
                    cumulative_Uel=U_el,
                    cumulative_Wemit=W_emit_tot,
                )
                if _v1042_metrics is not None and _v1042_metrics['criteria_pass']:
                    if deflect and fronts:
                        _v1042_root = fronts[0]
                        _v1042_src, _v1042_ell, _v1042_segments = _J_params_for_front(_v1042_root)
                        _v1042_tip = np.asarray(_v1042_root['xy'], dtype=float)
                        _v1042_dir = np.asarray(
                            _v1042_root.get('t_win', _v1042_root.get('fwd', [1.0, 0.0])),
                            dtype=float,
                        )
                        _v1042_sign = float(_v1042_root.get('J_sign_ref', 1.0) or 1.0)
                        _v1042_exclude = 2.0 * kill_r
                    else:
                        _v1042_src = 'cluster'
                        _v1042_ell = max(r_J_cluster_ell, 3.0 * h_local)
                        _v1042_segments = _backend_crack_segments()
                        _v1042_tip = np.asarray([a_tip, 0.0], dtype=float)
                        _v1042_dir = np.asarray([1.0, 0.0], dtype=float)
                        _v1042_sign = 1.0
                        _v1042_exclude = 0.0

                    _v1042_contours = _v1042_contour_scan(
                        compute_J_integral,
                        mesh=mesh,
                        u=u,
                        sigma_gp=sigma_gp,
                        psi_gp=psi_gp,
                        damage=d,
                        tip_xy=_v1042_tip,
                        direction=_v1042_dir,
                        mat=mat,
                        base_ell_m=_v1042_ell,
                        multipliers=getattr(
                            args, 'plastic_flow_contour_multipliers', '1 2 4 8'
                        ),
                        crack_segments=_v1042_segments,
                        exclude_radius_m=_v1042_exclude,
                        sign_reference=_v1042_sign,
                    )
                    _v1042_J_outer = (
                        float(_v1042_contours[-1]['J_positive_root_convention_J_per_m2'])
                        if _v1042_contours else float(J_tip_positive_v1042)
                    )
                    _v1042_J_shield = max(
                        _v1042_J_outer - float(J_tip_positive_v1042), 0.0
                    )
                    _v1042_ligament = max(
                        float(cfg.geometry.Lx) - float(crack_extension_start_a), 1.0e-30
                    )
                    _v1042_Jpl = max(float(W_p_acc), 0.0) / _v1042_ligament
                    _v1042_Wp_balance = max(
                        float(W_ext_acc) - float(U_el) - float(W_emit_tot), 0.0
                    )
                    _v1042_Jpl_balance = _v1042_Wp_balance / _v1042_ligament
                    _v1042_Kpl = np.sqrt(max(_v1042_Jpl, 0.0) * mat.Eprime) / 1.0e6
                    _v1042_energy_residual = (
                        float(W_ext_acc) - float(U_el) - float(W_p_acc) - float(W_emit_tot)
                    )
                    _v1042_energy_scale = max(
                        abs(float(W_ext_acc)),
                        abs(float(U_el)) + abs(float(W_p_acc)) + abs(float(W_emit_tot)),
                        1.0e-30,
                    )
                    _v1042_energy_error = abs(_v1042_energy_residual) / _v1042_energy_scale
                    plastic_flow_terminal_audit = {
                        'schema': 'v10.4.2_plastic_flow_terminal_audit_v1',
                        'classification': 'plastic_flow_no_sharp_fracture',
                        'terminal': True,
                        'campaign_terminal': True,
                        'target_extension_reached': False,
                        'sharp_fracture_occurred': False,
                        'first_passage_recorded': False,
                        'ductile_fracture_simulated': False,
                        'failure_regime': 'bulk_plastic_flow',
                        'temperature_K': float(T),
                        'terminal_step': int(step),
                        'J_fracture_definition': 'positive_signed_configurational_J_at_first_cleavage_passage',
                        'plastic_work_enters_fracture_measure': False,
                        'plastic_work_enters_cleavage_hazard': False,
                        'plastic_work_enters_energy_gate': False,
                        'J_pl_definition': 'cumulative_accepted_bulk_plastic_work_divided_by_unit_thickness_initial_ligament',
                        'J_pl_is_crack_driving_force': False,
                        'J_pl_is_fracture_toughness': False,
                        'J_pl_plot_symbol': 'open',
                        'J_fracture_plot_symbol': 'closed',
                        'unit_thickness_m': 1.0,
                        'initial_ligament_m': _v1042_ligament,
                        'W_external_J_per_m': float(W_ext_acc),
                        'U_elastic_J_per_m': float(U_el),
                        'W_bulk_plastic_J_per_m': float(W_p_acc),
                        'W_bulk_plastic_balance_estimate_J_per_m': _v1042_Wp_balance,
                        'W_tip_emission_J_per_m': float(W_emit_tot),
                        'W_fracture_J_per_m': 0.0,
                        'energy_balance_residual_J_per_m': _v1042_energy_residual,
                        'energy_balance_relative_error': _v1042_energy_error,
                        'energy_balance_tolerance': float(
                            getattr(args, 'plastic_flow_energy_balance_tolerance', 0.02) or 0.02
                        ),
                        'energy_balance_pass': _v1042_energy_error <= float(
                            getattr(args, 'plastic_flow_energy_balance_tolerance', 0.02) or 0.02
                        ),
                        'J_pl_diss_J_per_m2': _v1042_Jpl,
                        'J_pl_balance_J_per_m2': _v1042_Jpl_balance,
                        'eta_pl': 1.0,
                        'J_pl_eta_J_per_m2': _v1042_Jpl,
                        'K_pl_equivalent_MPa_sqrt_m': float(_v1042_Kpl),
                        'J_tip_positive_final_J_per_m2': float(J_tip_positive_v1042),
                        'J_tip_signed_final_J_per_m2': float(J_tip_signed_v1042),
                        'J_outer_positive_final_J_per_m2': _v1042_J_outer,
                        'J_contour_shielding_J_per_m2': _v1042_J_shield,
                        'contour_shielding_definition': 'max(J_outer_positive-J_tip_positive,0)',
                        'contour_shielding_is_diagnostic_only': True,
                        'contour_shielding_enters_fracture_hazard': False,
                        'contour_source': _v1042_src,
                        'contour_scan': _v1042_contours,
                        **_v1042_metrics,
                    }
                    with open(
                        os.path.join(args.out, 'plastic_flow_terminal_audit.json'),
                        'w',
                    ) as _v1042_fp:
                        json.dump(plastic_flow_terminal_audit, _v1042_fp, indent=2, sort_keys=True)
                        _v1042_fp.write('\n')
                    with open(os.path.join(args.out, 'PLASTIC_FLOW'), 'w') as _v1042_fp:
                        _v1042_fp.write('plastic_flow_no_sharp_fracture\n')
                    plastic_flow_terminal = True
                    print(
                        f"  [T={T:.0f}K] terminal plastic flow at step {step}: "
                        f"Jtip+={J_tip_positive_v1042:.6e} J/m^2 "
                        f"Jpl={_v1042_Jpl:.6e} J/m^2 "
                        f"Jshield(contour)={_v1042_J_shield:.6e} J/m^2"
                    )
                    break
'''


def transform_source(source: str) -> str:
    text = _replace_once(
        source,
        "import copy\nimport numpy as np\n",
        "import copy\nfrom collections import deque\nimport numpy as np\n" + _HELPERS + "\n",
        "v10.4.2 helper/import insertion",
    )

    parser_marker = """    p.add_argument('--stop-after-first-fire', action='store_true',
                   help='For diagnostic sweeps, stop a 2-D sharp-front run immediately after the first accepted crack advance. Production multifront runs should normally leave this off.')
"""
    parser_addition = parser_marker + """    p.add_argument('--plastic-flow-terminal', action=argparse.BooleanOptionalAction,
                   default=False, dest='plastic_flow_terminal',
                   help='Terminate successfully when a persistent accepted-step window demonstrates bulk-plastic accommodation with negligible positive sharp-tip J and no cleavage first passage.')
    p.add_argument('--plastic-flow-window-steps', type=int, default=2000, dest='plastic_flow_window_steps')
    p.add_argument('--plastic-flow-min-step', type=int, default=2000, dest='plastic_flow_min_step')
    p.add_argument('--plastic-flow-max-da-fraction', type=float, default=0.1, dest='plastic_flow_max_da_fraction')
    p.add_argument('--plastic-flow-J-rel-tol', type=float, default=1e-6, dest='plastic_flow_J_rel_tol')
    p.add_argument('--plastic-flow-J-abs-tol-J-per-m2', type=float, default=1e-6, dest='plastic_flow_J_abs_tol_J_per_m2')
    p.add_argument('--plastic-flow-sigma-rel-tol', type=float, default=1e-6, dest='plastic_flow_sigma_rel_tol')
    p.add_argument('--plastic-flow-min-plastic-fraction', type=float, default=0.90, dest='plastic_flow_min_plastic_fraction')
    p.add_argument('--plastic-flow-min-cumulative-plastic-fraction', type=float, default=0.90, dest='plastic_flow_min_cumulative_plastic_fraction')
    p.add_argument('--plastic-flow-max-elastic-fraction', type=float, default=0.05, dest='plastic_flow_max_elastic_fraction')
    p.add_argument('--plastic-flow-max-force-fraction', type=float, default=0.10, dest='plastic_flow_max_force_fraction')
    p.add_argument('--plastic-flow-max-tangent-fraction', type=float, default=0.05, dest='plastic_flow_max_tangent_fraction')
    p.add_argument('--plastic-flow-max-dB-window', type=float, default=1e-6, dest='plastic_flow_max_dB_window')
    p.add_argument('--plastic-flow-min-cleavage-horizon-ratio', type=float, default=100.0, dest='plastic_flow_min_cleavage_horizon_ratio')
    p.add_argument('--plastic-flow-energy-balance-tolerance', type=float, default=0.02, dest='plastic_flow_energy_balance_tolerance')
    p.add_argument('--plastic-flow-contour-multipliers', default='1 2 4 8', dest='plastic_flow_contour_multipliers',
                   help='Space/comma-separated multipliers on the production root/cluster J contour length used only for terminal contour-shielding diagnostics.')
"""
    text = _replace_once(
        text, parser_marker, parser_addition, "v10.4.2 parser insertion"
    )

    init_marker = (
        "        W_ext_acc = 0.0; W_p_acc = 0.0; Ftop_prev = 0.0; Uapp_prev = 0.0\n"
    )
    text = _replace_once(
        text,
        init_marker,
        init_marker + _INIT_BLOCK,
        "v10.4.2 terminal-state initialization",
    )

    history_marker = (
        "            hist['N_em'].append(info['N_em']); hist['a_tip'].append(a_tip)\n"
        "            hist['n_fronts'].append(n_fronts_now)\n"
    )
    text = _replace_once(
        text,
        history_marker,
        history_marker + _WINDOW_APPEND,
        "v10.4.2 accepted-window recording",
    )

    print_marker = """            if step % args.print_every == 0 or any_fired:
                nf_str = f"  nfr={n_fronts_now}" if deflect else ""
                print(f"  [T={T:.0f}K] step {step:4d}  KJ={KJ/1e6:7.3f}  "
                      f"sig_tip={info['sigma_tip']/1e9:6.2f}GPa  B={info['B']:7.3f}  "
                      f"N_em={info['N_em']:9.2f}  a={a_tip*1e3:.3f}mm{nf_str}"
                      + ("  << ADVANCE" if any_fired else ""))
"""
    text = _replace_once(
        text,
        print_marker,
        print_marker + _TERMINAL_BLOCK,
        "v10.4.2 terminal decision insertion",
    )

    mode_old = """            'mode': ('no-fracture' if Kc_first is None
                     else ('ductile' if W_emit_total > 0.1 * Kc_first ** 2 / mat.Eprime
                           else 'brittle')),
            'shelf': audit,
"""
    mode_new = """            'mode': ('plastic-flow' if plastic_flow_terminal
                     else ('no-fracture' if Kc_first is None
                           else ('ductile' if W_emit_total > 0.1 * Kc_first ** 2 / mat.Eprime
                                 else 'brittle'))),
            'terminal_status': ('plastic_flow_no_sharp_fracture'
                                if plastic_flow_terminal else None),
            'campaign_terminal': bool(plastic_flow_terminal),
            'J_pl_diss_J_per_m2': (None if plastic_flow_terminal_audit is None
                                    else plastic_flow_terminal_audit.get('J_pl_diss_J_per_m2')),
            'K_pl_equivalent_MPa_sqrt_m': (None if plastic_flow_terminal_audit is None
                                            else plastic_flow_terminal_audit.get('K_pl_equivalent_MPa_sqrt_m')),
            'J_contour_shielding_J_per_m2': (None if plastic_flow_terminal_audit is None
                                              else plastic_flow_terminal_audit.get('J_contour_shielding_J_per_m2')),
            'shelf': audit,
"""
    text = _replace_once(
        text, mode_old, mode_new, "v10.4.2 summary terminal fields"
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
        raise RuntimeError("could not allocate v10.4.2 transformed module spec")
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
