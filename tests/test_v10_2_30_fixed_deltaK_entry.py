from contextlib import contextmanager

from arrhenius_fracture import sharp_front_v10_2_30_fixed_deltaK as entry


@contextmanager
def _null_context(*args, **kwargs):
    yield None


def test_energy_gate_waveform_observer_preserves_fixed_deltaK_marker():
    def fixed_factory(*args, **kwargs):
        return args, kwargs

    fixed_factory._prescribed_fixed_deltaK_control = True
    observed = entry._energy._observed_waveform_factory(fixed_factory)
    assert observed._prescribed_fixed_deltaK_control is True


def test_fixed_deltaK_entry_dispatches_to_v10230(monkeypatch, tmp_path):
    seen = {}

    def fake_main(args):
        seen["args"] = list(args)
        return "ok"

    monkeypatch.setattr(entry, "install_fixed_deltaK_waveform", _null_context)
    monkeypatch.setattr(
        entry._legacy_fixed,
        "_allow_right_censored_stochastic_summary",
        _null_context,
    )
    monkeypatch.setattr(
        entry._legacy_fixed,
        "_fixed_deltaK_console_semantics",
        _null_context,
    )
    monkeypatch.setattr(entry._energy, "main", fake_main)
    monkeypatch.setattr(
        entry,
        "_write_audit",
        lambda args, target: {
            "hazard_energy_gated_events": 0,
            "hazard_energy_zero_length_attempts": 0,
            "censor_status": "right_censored_no_event",
        },
    )

    result = entry.main(
        [
            "--target-deltaK-MPa-sqrt-m",
            "6",
            "--R",
            "0.1",
            "--out",
            str(tmp_path),
        ]
    )
    assert result == "ok"
    assert "--fatigue-cycles" in seen["args"]
    assert "--no-cyclic-mechanics" in seen["args"]
    assert "--fatigue-hold-load" in seen["args"]
    assert "--target-deltaK-MPa-sqrt-m" not in seen["args"]


def test_fixed_deltaK_entry_rejects_invalid_R(monkeypatch):
    monkeypatch.setattr(entry, "install_fixed_deltaK_waveform", _null_context)
    try:
        entry.main(
            [
                "--target-deltaK-MPa-sqrt-m",
                "6",
                "--R",
                "1",
            ]
        )
    except SystemExit as exc:
        assert "-1 <= R < 1" in str(exc)
    else:
        raise AssertionError("invalid R was not rejected")


def test_fixed_deltaK_entry_accepts_negative_R_for_reversible_solver(monkeypatch):
    seen = {}

    @contextmanager
    def context(target, **kwargs):
        seen.update(kwargs)
        yield None

    monkeypatch.setattr(entry, "install_fixed_deltaK_waveform", context)
    monkeypatch.setattr(entry._energy, "main", lambda args: "ok")
    monkeypatch.setattr(entry, "_write_audit", lambda *args: {})
    result = entry.main(["--target-deltaK-MPa-sqrt-m", "6", "--R", "-0.95"])
    assert result == "ok"
    assert seen == {"allow_negative_R": True}
