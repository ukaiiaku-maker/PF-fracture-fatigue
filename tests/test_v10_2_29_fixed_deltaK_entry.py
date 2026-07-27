from contextlib import contextmanager
import json

from arrhenius_fracture import sharp_front_v10_2_29_fixed_deltaK as entry


@contextmanager
def _null_context(*args, **kwargs):
    yield None


def test_fixed_deltaK_entry_dispatches_to_v10229(monkeypatch, tmp_path):
    seen = {}

    def fake_main(args):
        seen["args"] = list(args)
        return "ok"

    monkeypatch.setattr(entry, "install_fixed_deltaK_waveform", _null_context)
    monkeypatch.setattr(
        entry._legacy_fixed, "_allow_right_censored_stochastic_summary", _null_context
    )
    monkeypatch.setattr(
        entry._legacy_fixed, "_fixed_deltaK_console_semantics", _null_context
    )
    monkeypatch.setattr(entry._fatigue, "main", fake_main)
    monkeypatch.setattr(
        entry,
        "_write_audit",
        lambda args, target: {
            "stochastic_geometry_events": 0,
            "censor_status": "right_censored_no_event",
        },
    )

    result = entry.main([
        "--target-deltaK-MPa-sqrt-m", "6",
        "--R", "0.1",
        "--out", str(tmp_path),
    ])
    assert result == "ok"
    assert "--fatigue-cycles" in seen["args"]
    assert "--no-cyclic-mechanics" in seen["args"]
    assert "--fatigue-hold-load" in seen["args"]
    assert "--target-deltaK-MPa-sqrt-m" not in seen["args"]


def test_fixed_deltaK_entry_rejects_invalid_R(monkeypatch):
    monkeypatch.setattr(entry, "install_fixed_deltaK_waveform", _null_context)
    try:
        entry.main([
            "--target-deltaK-MPa-sqrt-m", "6",
            "--R", "1",
        ])
    except SystemExit as exc:
        assert "0 <= R < 1" in str(exc)
    else:
        raise AssertionError("invalid R was not rejected")


def test_fixed_deltaK_audit_records_coupled_hazard(monkeypatch, tmp_path):
    monkeypatch.setattr(entry._legacy_fixed, "_normalize_output_semantics", lambda *a: {})
    payload = entry._write_audit(
        [
            "--out", str(tmp_path),
            "--R", "0.1",
            "--frequency-Hz", "1000",
            "--cycles-max", "1e9",
            "--parameter-option", "v913_paper_dbtt01_0202500_persistent_sites",
        ],
        4.0,
    )
    assert payload["state_coupled_cleavage_hazard"] is True
    assert payload["cleavage_hazard_frozen_within_cycle_block"] is False
    assert payload["fatigue_engine"] == "v10.2.29_persistent_site_state_coupled_cyclic"
    stored = json.loads((tmp_path / "v10_2_29_fixed_deltaK_control.json").read_text())
    assert stored["state_coupled_cleavage_hazard"] is True
