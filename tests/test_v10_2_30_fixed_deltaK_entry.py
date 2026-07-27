from contextlib import contextmanager

from arrhenius_fracture import fixed_deltaK_v10230 as entry


@contextmanager
def _null_context(*args, **kwargs):
    yield None


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
    monkeypatch.setattr(entry._fatigue, "main", fake_main)
    monkeypatch.setattr(
        entry,
        "_write_audit",
        lambda args, target: {
            "energy_gated_propagating_events": 0,
            "energy_gated_zero_length_attempts": 0,
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
        assert "0 <= R < 1" in str(exc)
    else:
        raise AssertionError("invalid R was not rejected")
