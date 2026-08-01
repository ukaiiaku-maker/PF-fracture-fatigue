"""Finalize the v10.4.2 source overlay with accepted plastic-work accounting.

The underlying driver historically estimates an accepted-step plastic-work
increment from the final stress and final plastic strain rate. That expression
can vanish after a fully relaxed staggered solve even though the constitutive
update accepted positive plastic work during the step. This overlay requests the
existing ``dWp_accepted_gp`` diagnostic from every staggered plasticity update,
sums it over all accepted stagger iterations, and uses its area integral as the
primary bulk-plastic work increment. The historical post-update expression
remains only as a compatibility fallback.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_terminal_v1042 import transform_source as _terminal_transform

MODEL_ID = "v10.4.2_accepted_constitutive_plastic_work"
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
