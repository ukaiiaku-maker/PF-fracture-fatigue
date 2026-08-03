"""Preserve nominal loading progress under adaptive v10.4.3 subdivision.

Accepted reduced substeps remain individual output rows, but ``--steps`` is a
nominal physical loading target.  The loop therefore continues until the sum of
accepted trial fractions reaches the requested target.  Terminal persistence and
remaining-horizon diagnostics use the same nominal coordinate.

No constitutive law, fracture law, hazard, or accepted substep is changed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_path_work_startup_v1043 import (
    transform_source as _startup_path_transform,
)

MODEL_ID = "v10.4.3_adaptive_substeps_preserve_nominal_loading_progress"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_physical_progress"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _startup_path_transform(source)

    text = _replace_once(
        text,
        """        step = 0
        Uapp_accepted = 0.0
        carry_frac = 1.0
""",
        """        step = 0
        Uapp_accepted = 0.0
        carry_frac = 1.0
        nominal_progress_target_v1043 = float(args.steps)
        nominal_progress_v1043 = 0.0
        nominal_progress_tol_v1043 = 128.0 * np.finfo(float).eps * max(
            nominal_progress_target_v1043, 1.0
        )
""",
        "nominal loading-progress initialization",
    )

    text = _replace_once(
        text,
        """        while step < args.steps:
""",
        """        while nominal_progress_v1043 < (
            nominal_progress_target_v1043 - nominal_progress_tol_v1043
        ):
""",
        "physical-progress loop condition",
    )

    text = _replace_once(
        text,
        """            trial_frac = min(1.0, carry_frac * adaptive_grow)
""",
        """            remaining_nominal_fraction_v1043 = max(
                nominal_progress_target_v1043 - nominal_progress_v1043,
                0.0,
            )
            trial_frac = min(
                1.0,
                carry_frac * adaptive_grow,
                remaining_nominal_fraction_v1043,
            )
""",
        "remaining nominal-fraction cap",
    )

    text = _replace_once(
        text,
        """                    _v1043_min_trial_frac = float(args.stagger_min_dt_fraction)
""",
        """                    _v1043_min_trial_frac = min(
                        float(args.stagger_min_dt_fraction),
                        max(remaining_nominal_fraction_v1043, np.finfo(float).tiny),
                    )
""",
        "final nominal-remainder retry floor",
    )

    text = _replace_once(
        text,
        """            step = step_trial
            Uapp_accepted = Uapp
            carry_frac = trial_frac
            adaptive_frac_used = trial_frac
""",
        """            step = step_trial
            Uapp_accepted = Uapp
            nominal_progress_step_start_v1043 = float(
                nominal_progress_v1043
            )
            nominal_progress_v1043 += float(trial_frac)
            if abs(
                nominal_progress_v1043 - nominal_progress_target_v1043
            ) <= nominal_progress_tol_v1043:
                nominal_progress_v1043 = nominal_progress_target_v1043
            if nominal_progress_v1043 > (
                nominal_progress_target_v1043 + nominal_progress_tol_v1043
            ):
                raise RuntimeError(
                    'v10.4.3 adaptive substep exceeded nominal loading horizon: '
                    f'progress={nominal_progress_v1043:.17e}, '
                    f'target={nominal_progress_target_v1043:.17e}'
                )
            carry_frac = trial_frac
            adaptive_frac_used = trial_frac
""",
        "accepted nominal-progress commit",
    )

    text = _replace_once(
        text,
        """        plastic_flow_window = deque(maxlen=plastic_flow_window_size)
""",
        """        plastic_flow_window = deque()
""",
        "terminal physical-span history",
    )

    text = _replace_once(
        text,
        """    required = max(int(getattr(args, 'plastic_flow_window_steps', 2000) or 2000), 2)
    minimum_step = max(int(getattr(args, 'plastic_flow_min_step', required) or required), required)
    if len(window) < required or int(window[-1]['step']) < minimum_step:
        return None

    values = list(window)
""",
        """    required = max(
        float(getattr(args, 'plastic_flow_window_steps', 2000) or 2000),
        2.0,
    )
    minimum_step = max(
        float(getattr(args, 'plastic_flow_min_step', required) or required),
        required,
    )
    if not window:
        return None
    last_progress = float(window[-1]['nominal_progress_end'])
    if last_progress < minimum_step:
        return None
    cutoff = last_progress - required
    progress_tol = 128.0 * np.finfo(float).eps * max(last_progress, required, 1.0)
    while (
        len(window) > 2
        and float(window[1]['nominal_progress_end'])
        <= cutoff + progress_tol
    ):
        window.popleft()
    values = [
        row for row in window
        if float(row['nominal_progress_end']) > cutoff + progress_tol
    ]
    if len(values) < 2:
        return None
    nominal_span = (
        float(values[-1]['nominal_progress_end'])
        - float(values[0]['nominal_progress_start'])
    )
    if nominal_span < required - progress_tol:
        return None
""",
        "terminal nominal physical-span selection",
    )

    text = _replace_once(
        text,
        """                'step': int(step),
                'Uapp': float(Uapp),
""",
        """                'step': int(step),
                'nominal_progress_start': float(
                    nominal_progress_step_start_v1043
                ),
                'nominal_progress_end': float(nominal_progress_v1043),
                'Uapp': float(Uapp),
""",
        "terminal row physical coordinates",
    )

    text = _replace_once(
        text,
        """        'classification_window_steps': len(values),
        'crack_extension_window_m': crack_span,
""",
        """        'classification_window_steps': len(values),
        'classification_window_nominal_increment_span': nominal_span,
        'window_first_nominal_progress': float(
            values[0]['nominal_progress_start']
        ),
        'window_last_nominal_progress': float(
            values[-1]['nominal_progress_end']
        ),
        'crack_extension_window_m': crack_span,
""",
        "terminal nominal-span audit",
    )

    text = _replace_once(
        text,
        """                    remaining_steps=max(int(args.steps) - int(step), 0),
""",
        """                    remaining_steps=max(
                        nominal_progress_target_v1043
                        - nominal_progress_v1043,
                        0.0,
                    ),
""",
        "terminal remaining nominal horizon",
    )

    text = _replace_once(
        text,
        """    remaining_loading_horizon = max(int(remaining_steps), 0) * max(float(nominal_dt_s), 0.0)
""",
        """    remaining_loading_horizon = max(
        float(remaining_steps), 0.0
    ) * max(float(nominal_dt_s), 0.0)
""",
        "fractional remaining loading horizon",
    )

    text = _replace_once(
        text,
        """                         float(info.get('mpz_wake_retained_total', 0.0))))
""",
        """                         float(info.get('mpz_wake_retained_total', 0.0)),
                         float(nominal_progress_step_start_v1043),
                         float(nominal_progress_v1043)))
""",
        "step-row nominal-progress columns",
    )

    text = _replace_once(
        text,
        """                          'mpz_escaped_total,mpz_recovered_total,mpz_wake_retained_total',
""",
        """                          'mpz_escaped_total,mpz_recovered_total,mpz_wake_retained_total,'
                          'nominal_progress_start,nominal_progress',
""",
        "step-header nominal-progress columns",
    )

    text = _replace_once(
        text,
        """or step == args.steps):
""",
        """or nominal_progress_v1043 >= (
                    nominal_progress_target_v1043
                    - nominal_progress_tol_v1043
                )):
""",
        "final physical-horizon snapshot",
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
        raise RuntimeError("could not allocate v10.4.3 physical-progress module")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-physical-progress]",
                "exec",
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
