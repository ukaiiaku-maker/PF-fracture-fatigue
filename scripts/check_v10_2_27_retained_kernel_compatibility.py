#!/usr/bin/env python3
"""Reject partial-campaign reuse across incompatible mechanical kernels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tag(value: float) -> str:
    return f"{float(value):g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--seed-option-stride", type=int, required=True)
    parser.add_argument("--seed-temperature-stride", type=int, required=True)
    parser.add_argument("--option-index-offset", type=int, default=0)
    args = parser.parse_args()

    family = args.family.expanduser().resolve()
    if not family.is_file():
        raise SystemExit(f"resolved family is missing: {family}")
    family_payload = json.loads(family.read_text())
    expected = str(family_payload.get("mechanical_configuration_fingerprint", ""))
    if not expected:
        raise SystemExit(
            "resolved family lacks a mechanical-configuration fingerprint; "
            "it cannot authorize reuse of retained cases"
        )

    root = args.outroot.expanduser().resolve()
    errors: list[dict[str, object]] = []
    checked = 0
    for local_option_index, option in enumerate(args.options):
        option_index = args.option_index_offset + local_option_index
        for temperature_index, temperature in enumerate(args.temperatures):
            seed = (
                args.base_seed
                + option_index * args.seed_option_stride
                + temperature_index * args.seed_temperature_stride
            )
            case = root / option / (
                f"T{_tag(temperature)}K_th{_tag(args.theta_deg)}_seed{seed}"
            )
            transfer_path = case / "v10_2_27_paper_four_class_parameter_transfer.json"
            contract_path = case / "v10_2_27_case_contract.json"
            if not transfer_path.is_file():
                errors.append(
                    {
                        "case": str(case),
                        "reason": "missing parameter-transfer provenance",
                        "required_action": "rerun this case with the configuration-driven kernel resolver",
                    }
                )
                continue
            transfer = json.loads(transfer_path.read_text())
            observed = str(transfer.get("mechanical_configuration_fingerprint", ""))
            if observed != expected:
                errors.append(
                    {
                        "case": str(case),
                        "reason": "mechanical-configuration fingerprint mismatch",
                        "expected": expected,
                        "observed": observed or None,
                        "required_action": "do not mix this retained case with the resolved kernel",
                    }
                )
                continue
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text())
                if contract.get("option") != option:
                    errors.append(
                        {
                            "case": str(case),
                            "reason": "case-contract option mismatch",
                            "expected_option": option,
                            "observed_option": contract.get("option"),
                        }
                    )
                    continue
            checked += 1

    if errors:
        print(json.dumps({"compatible": False, "errors": errors}, indent=2))
        raise SystemExit(
            "RETAINED KERNEL COMPATIBILITY FAILED: rerun all affected material cases "
            "with one resolved mechanical kernel"
        )

    print(
        json.dumps(
            {
                "compatible": True,
                "checked_cases": checked,
                "mechanical_configuration_fingerprint": expected,
                "family": str(family),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
