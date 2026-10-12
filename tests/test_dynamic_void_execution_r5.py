from types import SimpleNamespace

import pytest

from arrhenius_fracture import dynamic_void_execution_r5 as dynamic
from arrhenius_fracture.dynamic_void_execution_r5 import execute_dynamic_void_r5
from arrhenius_fracture.current_coupled_execution_r4 import build_current_r4_engine


@pytest.mark.parametrize(
    ("horizon", "consumed", "native_scale"),
    ((1.0e-6, 0.25e-6, 1.0), (0.5, 0.125, 0.5)),
)
def test_first_event_time_reads_native_consumed_time_without_shortcuts(
    monkeypatch, horizon, consumed, native_scale,
):
    preview = SimpleNamespace(
        fired=True,
        interval_result=SimpleNamespace(
            result={"kinetic_dt_consumed_s": consumed}
        ),
    )
    monkeypatch.setattr(dynamic, "_preview", lambda *args, **kwargs: preview)
    binding = SimpleNamespace(
        native_K_scale=native_scale, source_identity="manufactured"
    )
    owner = SimpleNamespace(fingerprint="manufactured-crossing", engine=SimpleNamespace(
        mpz=SimpleNamespace(local_slip_count=lambda: -1.0),
    ))
    located = dynamic._first_event_time(
        owner, opening_m=0.0, temperature_K=900.0,
        maximum_duration_s=horizon, tolerance_s=1.0e-15,
        mechanics_binding=binding,
    )
    assert located == pytest.approx(consumed)


def test_first_event_time_preserves_no_crossing(monkeypatch):
    monkeypatch.setattr(
        dynamic, "_preview",
        lambda *args, **kwargs: SimpleNamespace(fired=False),
    )
    owner = SimpleNamespace(fingerprint="manufactured-no-crossing", engine=SimpleNamespace(
        mpz=SimpleNamespace(local_slip_count=lambda: -1.0),
    ))
    assert dynamic._first_event_time(
        owner, opening_m=0.0, temperature_K=900.0,
        maximum_duration_s=1.0, tolerance_s=1.0e-12,
    ) is None


def test_nearly_spent_low_k_mark_is_not_skipped(monkeypatch):
    preview = SimpleNamespace(
        fired=True,
        interval_result=SimpleNamespace(
            result={"kinetic_dt_consumed_s": 0.25}
        ),
    )
    monkeypatch.setattr(dynamic, "_preview", lambda *args, **kwargs: preview)
    engine = SimpleNamespace(
        mpz=SimpleNamespace(local_slip_count=lambda: 0.0),
        K_shield=lambda: 0.0,
        f=SimpleNamespace(r0=1.0e-6, sigma_cap=5.0e9),
        lambda_cleave=lambda sigma, temperature: (1.0e-3, 1.0e-3, 0.0),
        hazard_threshold_action=1.0,
        B=1.0-1.0e-6,
    )
    assert dynamic._first_event_time(
        SimpleNamespace(fingerprint="nearly-spent-low-k", engine=engine),
        opening_m=0.0, temperature_K=900.0,
        maximum_duration_s=1.0, tolerance_s=1.0e-12,
        mechanics_binding=SimpleNamespace(
            native_K_scale=0.5, source_identity="nearly-spent"
        ),
    ) == pytest.approx(0.25)


def test_actual_successor_preview_publishes_interval_relative_consumed_time():
    owner = dynamic.CurrentFractureOwner(build_current_r4_engine())
    preview = dynamic._preview(
        owner, opening_m=0.0, temperature_K=900.0,
        duration_s=1.0e-6,
        mechanics_binding=SimpleNamespace(
            native_K_scale=51.10046751415974e6 / 20.0e6,
            source_identity="actual-successor-import",
        ),
    )
    audit = preview.interval_result.accepted_update_audit
    assert preview.fired
    assert preview.interval_result.result["kinetic_dt_consumed_s"] == pytest.approx(
        audit["consumed_duration_s"]
    )
    assert preview.interval_result.result["kinetic_dt_unused_s"] == pytest.approx(
        1.0e-6 - audit["consumed_duration_s"]
    )


def test_recovered_dynamic_void_owner_uses_current_backend(tmp_path):
    result = execute_dynamic_void_r5(tmp_path / "dynamic")
    assert result["passed"]
    assert result["A_void_off"]["direct_backend_bitwise_parity"]
    assert result["B_native_incubation"]["vacancy_birth_clock_evaluated"]
    assert result["B_native_incubation"]["direct_birth_clock_evaluated"]
    assert result["C_active_dynamic"]["radius_changed"]
    assert result["C_active_dynamic"]["native_fracture_events"] >= 1
    assert result["transaction"]["rollback_exact"]
    assert not result["fixed_R4_observation"]["used_for_dynamic_run"]
    assert result["complete_dynamic_checkpoint"]["continuation_includes_native_event"]
    assert result["complete_dynamic_checkpoint"]["terminal_complete_state_exact"]
    roots = result["competing_root_ordering"]
    assert roots["selected_event"] == "FRACTURE_CLEAVAGE"
    assert roots["immediately_preceding_owner_event"] == (
        "CAVITY_CONTINUUM_GROWTH_PACKET"
    )
    assert roots["fracture_time_s"] == pytest.approx(
        roots["preceding_owner_time_s"]
    )
