from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_builder():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_v10_4_bulk_rate_orientation_launcher.py"
    spec = importlib.util.spec_from_file_location("v104_builder_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


def test_v104_launcher_separates_kernel_and_campaign_provenance():
    root, module = _load_builder()
    source = (
        root / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    generated = module.transform(source)

    assert generated.count('"production_physics_modified": True') == 1
    assert (
        '"production_physics_change": '
        '"full_field_bulk_peierls_taylor_coupling"'
    ) in generated
    assert '"kernel_family_production_physics_modified": False' in generated
    # The direct-family provenance contract remains unchanged and continues to
    # require the cached v10.2.28 kernel family itself to report False.
    assert generated.count('"production_physics_modified": False') == 1


def test_v104_scheduler_adapter_rewrites_single_case_contract_occurrence():
    _, module = _load_builder()
    scheduler = "\n".join(
        [
            "    --bulk-plasticity-mode tip_only",
            '"v10.2.30_hazard_energy_gated_orientation_rate_campaign_v1"',
            '"v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1"',
            '    "hazard_energy_gate": True,',
            '    "hazard_energy_gate": True,',
            '    "hazard_energy_gate": True,',
            '    root / "v10_2_30_hazard_energy_gate_audit.json",',
            'command = (root / "command.sh").read_text()',
            '    f"--parameter-option {expected[\'option\']}",',
        ]
    )

    namespace = {"scheduler": scheduler}

    def replace_scheduler_exact(old, new, expected_count=1, label="scheduler token"):
        actual = namespace["scheduler"].count(old)
        if actual != expected_count:
            raise RuntimeError(
                f"{label} changed: expected {expected_count}, found {actual}"
            )
        namespace["scheduler"] = namespace["scheduler"].replace(old, new)

    namespace["replace_scheduler_exact"] = replace_scheduler_exact
    exec(module._scheduler_adapter(), namespace)
    transformed = namespace["scheduler"]

    assert (
        transformed.count(
            '"v10.4_bulk_peierls_taylor_orientation_rate_case_contract_v1"'
        )
        == 1
    )
    assert (
        '"v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1"'
        not in transformed
    )
    assert "--bulk-plasticity-mode full_field" in transformed
