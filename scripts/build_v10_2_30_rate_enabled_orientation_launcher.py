#!/usr/bin/env python3
"""Build a rate-aware v10.2.30 hazard-energy-gated orientation launcher.

The validated v10.2.28 launcher supplies the direct prescribed-geometry kernel,
projected-extension coordinate, four audited material options, deterministic seed
mapping, and loading-rate machinery. This builder changes only the model entry,
gate provenance/verification, and the hard-coded 1000 um restriction so the same
contract can support a short validation screen and the final 1000 um sweep.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


MODEL_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_2_30_hazard_energy_gated_audited"
)
_MODEL_TOKEN = "@@V10230_MODEL_ENTRY@@"


def _load_rate_builder():
    path = Path(__file__).with_name(
        "build_v10_2_28_rate_enabled_orientation_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("v10228_rate_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load rate builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    count: int = 1,
    label: str,
) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"{label} changed: expected {count} occurrence(s), found {actual}"
        )
    return text.replace(old, new, count if count == 1 else -1)


def _gate_scheduler_adapter() -> str:
    template = r"""replace_scheduler_exact(
    '"v10.2.28_paper_four_class_orientation_loading_rate_campaign_v1"',
    '"v10.2.30_hazard_energy_gated_orientation_rate_campaign_v1"',
    label="gate campaign manifest schema",
)
replace_scheduler_exact(
    '"v10.2.28_orientation_loading_rate_case_contract_v1"',
    '"v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1"',
    label="gate case contract schema",
)
replace_scheduler_exact(
    '    "model_entry": "@@V10230_MODEL_ENTRY@@",',
    '''    "model_entry": "@@V10230_MODEL_ENTRY@@",
    "hazard_energy_gate": True,
    "absolute_athermal_Gc": False,
    "hazard_dissipation_density": (
        "gamma_rel*m*DeltaG_cleave_eff(T,sigma)/b^2"
    ),
    "anisotropic_hazard_scaling": (
        "sigma_hazard=sigma_physical/sqrt(gamma_rel)"
    ),
    "fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2",
    "gate_resolution": "every_internal_Strang_microstep",''',
    label="gate campaign manifest fields",
)
replace_scheduler_exact(
    '''payload = {
    "schema": "v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1",''',
    '''payload = {
    "schema": "v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1",
    "model_entry": "@@V10230_MODEL_ENTRY@@",
    "hazard_energy_gate": True,
    "absolute_athermal_Gc": False,
    "gate_resolution": "every_internal_Strang_microstep",''',
    label="gate case contract fields",
)
replace_scheduler_exact(
    '''expected = {
    "option": os.environ["OPTION"],''',
    '''expected = {
    "model_entry": "@@V10230_MODEL_ENTRY@@",
    "hazard_energy_gate": True,
    "absolute_athermal_Gc": False,
    "gate_resolution": "every_internal_Strang_microstep",
    "option": os.environ["OPTION"],''',
    label="completed-case gate contract expectation",
)
replace_scheduler_exact(
    '''    root / "command.sh",
]''',
    '''    root / "command.sh",
    root / "v10_2_30_hazard_energy_gate_audit.json",
    root / "stochastic_avalanche_geometry_events.json",
]''',
    label="completed-case gate files",
)
replace_scheduler_exact(
    '''command = (root / "command.sh").read_text()''',
    '''gate_audit = json.loads(
    (root / "v10_2_30_hazard_energy_gate_audit.json").read_text()
)
if gate_audit.get("event_initiation") != "Arrhenius_first_passage_only":
    raise SystemExit(1)
if gate_audit.get("absolute_athermal_Gc") is not False:
    raise SystemExit(1)
if gate_audit.get("gate_resolution") != "every_internal_Strang_microstep":
    raise SystemExit(1)

events = json.loads(
    (root / "stochastic_avalanche_geometry_events.json").read_text()
)
if not events:
    raise SystemExit(1)
for event in events:
    gate = event.get("hazard_energy_gate")
    if event.get("hazard_energy_gate_active") is not True or not isinstance(gate, dict):
        raise SystemExit(1)
    proposed = float(gate.get("proposed_event_advance_m", float("nan")))
    accepted = float(gate.get("accepted_event_advance_m", float("nan")))
    available = float(gate.get("energy_available_integrated_J_per_m", float("nan")))
    dissipated = float(gate.get("energy_dissipated_integrated_J_per_m", float("nan")))
    if not all(math.isfinite(value) for value in (proposed, accepted, available, dissipated)):
        raise SystemExit(1)
    if proposed <= 0.0 or accepted <= 0.0 or accepted > proposed * (1.0 + 1.0e-10):
        raise SystemExit(1)
    tolerance = 1.0e-10 * max(abs(available), abs(dissipated), 1.0)
    if dissipated > available + tolerance:
        raise SystemExit(1)
    if float(gate.get("gamma_rel", 0.0)) <= 0.0:
        raise SystemExit(1)
    if float(gate.get("DeltaG_cleave_eff_eV", 0.0)) <= 0.0:
        raise SystemExit(1)

command = (root / "command.sh").read_text()''',
    label="completed-case gate audit verification",
)
replace_scheduler_exact(
    '''tokens = [
    f"--parameter-option {expected['option']}",''',
    '''tokens = [
    "-m @@V10230_MODEL_ENTRY@@",
    f"--parameter-option {expected['option']}",''',
    label="completed-case model command verification",
)
"""
    return template.replace(_MODEL_TOKEN, MODEL_ENTRY)


def transform(source: str) -> str:
    text = _load_rate_builder().transform(source)

    text = _replace_exact(
        text,
        '''if not math.isclose(target, 1000.0, rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit("ERROR: this production launcher is fixed to 1000 um crack extension")''',
        '''if not math.isfinite(target) or target <= 0.0:
    raise SystemExit("ERROR: TARGET_EXT_UM must be finite and positive")''',
        label="generic projected-target validation",
    )
    text = _replace_exact(
        text,
        '"schema": "v10.2.28_paper_four_class_orientation_1000um_campaign_lock_v3",',
        '"schema": "v10.2.30_hazard_energy_gated_orientation_rate_lock_v1",',
        label="gate campaign-lock schema",
    )
    text = _replace_exact(
        text,
        "arrhenius_fracture.sharp_front_v10_2_28_audited",
        MODEL_ENTRY,
        count=2,
        label="v10.2.30 model entry",
    )
    text = _replace_exact(
        text,
        '    "model_entry": "' + MODEL_ENTRY + '",\n',
        '    "model_entry": "' + MODEL_ENTRY + '",\n'
        '    "hazard_energy_gate": True,\n'
        '    "absolute_athermal_Gc": False,\n'
        '    "hazard_dissipation_density": (\n'
        '        "gamma_rel*m*DeltaG_cleave_eff(T,sigma)/b^2"\n'
        '    ),\n'
        '    "anisotropic_hazard_scaling": (\n'
        '        "sigma_hazard=sigma_physical/sqrt(gamma_rel)"\n'
        '    ),\n'
        '    "fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2",\n'
        '    "gate_resolution": "every_internal_Strang_microstep",\n',
        label="gate campaign-lock fields",
    )
    text = _replace_exact(
        text,
        "v10_2_28_campaign_kernel_lock.json",
        "v10_2_30_hazard_energy_gate_campaign_lock.json",
        label="gate campaign-lock filename",
    )
    text = _replace_exact(
        text,
        "v10_2_28_generated_scheduler.sh",
        "v10_2_30_hazard_energy_gate_generated_scheduler.sh",
        label="gate generated-scheduler filename",
    )

    adapter_marker = '''for old, new in replacements.items():
    if old not in scheduler:
        raise SystemExit(f"ERROR: scheduler source no longer contains expected token: {old}")
    scheduler = scheduler.replace(old, new)

plotter = source_plotter.read_text()'''
    adapter_replacement = '''for old, new in replacements.items():
    if old not in scheduler:
        raise SystemExit(f"ERROR: scheduler source no longer contains expected token: {old}")
    scheduler = scheduler.replace(old, new)

''' + _gate_scheduler_adapter() + '''
plotter = source_plotter.read_text()'''
    text = _replace_exact(
        text,
        adapter_marker,
        adapter_replacement,
        label="embedded scheduler gate adapter",
    )
    return text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
