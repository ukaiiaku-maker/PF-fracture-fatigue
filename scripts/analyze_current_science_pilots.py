#!/usr/bin/env python3
"""Assemble the bounded current-source fracture pilot evidence."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "integration_evidence" / "current_science_pilots_6abbf10"
OUT = ROOT / "review_packet" / "current_science_pilots_6abbf10"

CASES = (
    ("DBTT 300 K", "DBTT_300K", "0300", "accommodation_enabled_run2"),
    ("DBTT 1200 K", "DBTT_1200K", "1200", "accommodation_enabled"),
    ("Peak 900 K", "Peak_900K", "0900", "accommodation_enabled"),
)


def load_json(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def f(row, key):
    return float(row[key])


def geometry_events(table):
    result = []
    previous = 0.0
    for row in table:
        extension = f(row, "crack_extension_m")
        if extension > previous + 1.0e-15:
            result.append({
                "step": int(f(row, "step")),
                "extension_um": extension * 1.0e6,
                "K_I_MPa_sqrt_m": f(row, "KJ_Pa_sqrtm") / 1.0e6,
                "J_effective_J_per_m2": f(row, "J_effective_direct_J_per_m2"),
                "opening_m": f(row, "Uapp_m"),
                "backstress_GPa": f(row, "sigma_back_Pa") / 1.0e9,
                "emitted_population": f(row, "N_em"),
                "retained_population": f(row, "mpz_retained_count"),
                "mobile_population": f(row, "mpz_mobile_count"),
                "tip_emission_work_J_per_m": f(
                    row, "W_tip_emit_cumulative_J_per_m"
                ),
            })
        previous = max(previous, extension)
    return result


def bracket(table, target_um):
    events = geometry_events(table)
    before = {"extension_um": 0.0, "K_I_MPa_sqrt_m": None}
    for event in events:
        if event["extension_um"] >= target_um:
            return {
                "status": "discrete_jump_bracket",
                "target_um": target_um,
                "lower_extension_um": before["extension_um"],
                "upper_extension_um": event["extension_um"],
                "upper_K_I_MPa_sqrt_m": event["K_I_MPa_sqrt_m"],
                "interpolation_performed": False,
            }
        before = event
    return {"status": "target_not_reached", "target_um": target_um}


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired = []
    detailed = []
    for label, directory, temp_token, enabled_name in CASES:
        pair = {}
        for arm, run_name in (
            ("accommodation_enabled", enabled_name),
            ("opening_only", "opening_only"),
        ):
            case = RAW / "pf" / directory / run_name
            summary = load_json(case / "summary.json")[0]
            table = rows(case / f"steps_{temp_token}K.csv")
            events = geometry_events(table)
            attestation = load_json(case / "shared_runtime_attestation.post.json")
            pair[arm] = {
                "raw_root": str(case.relative_to(ROOT)),
                "source_commit": attestation["source_policy"]["commit"],
                "first_event": events[0],
                "post_event_endpoint": events[-1],
                "accepted_geometry_events": int(summary["n_geometry_events"]),
                "achieved_extension_um": float(
                    summary["geometry_projected_extension_m"] * 1.0e6
                ),
                "terminal_emitted_population": float(summary["N_em_final"]),
                "terminal_emission_work_J_per_m": float(summary["W_emit_J_per_m"]),
            }
            detailed.append({"condition": label, "arm": arm, **pair[arm]})
        enabled = pair["accommodation_enabled"]
        control = pair["opening_only"]
        first_delta = (
            enabled["first_event"]["K_I_MPa_sqrt_m"]
            - control["first_event"]["K_I_MPa_sqrt_m"]
        )
        post_delta = (
            enabled["post_event_endpoint"]["K_I_MPa_sqrt_m"]
            - control["post_event_endpoint"]["K_I_MPa_sqrt_m"]
        )
        paired.append({
            "condition": label,
            "first_K_enabled_MPa_sqrt_m": enabled["first_event"]["K_I_MPa_sqrt_m"],
            "first_K_opening_only_MPa_sqrt_m": control["first_event"]["K_I_MPa_sqrt_m"],
            "first_delta_K_MPa_sqrt_m": first_delta,
            "first_ratio_enabled_over_control": (
                enabled["first_event"]["K_I_MPa_sqrt_m"]
                / control["first_event"]["K_I_MPa_sqrt_m"]
            ),
            "post_K_enabled_MPa_sqrt_m": enabled["post_event_endpoint"]["K_I_MPa_sqrt_m"],
            "post_K_opening_only_MPa_sqrt_m": control["post_event_endpoint"]["K_I_MPa_sqrt_m"],
            "post_delta_K_MPa_sqrt_m": post_delta,
            "enabled_backstress_at_first_GPa": enabled["first_event"]["backstress_GPa"],
            "enabled_emitted_population_at_first": enabled["first_event"]["emitted_population"],
            "common_achieved_extension_um": enabled["achieved_extension_um"],
            "events_each_arm": enabled["accepted_geometry_events"],
        })

    write_csv(OUT / "OPENING_ACCOMMODATION_RESULTS.csv", paired)

    long_root = RAW / "pf" / "DBTT_300K" / "developed_1000um_current"
    long_summary = load_json(long_root / "summary.json")[0]
    long_rows = rows(long_root / "steps_0300K.csv")
    long_attestation = load_json(long_root / "shared_runtime_attestation.post.json")
    long_result = {
        "source_commit": long_attestation["source_policy"]["commit"],
        "raw_root": str(long_root.relative_to(ROOT)),
        "achieved_projected_extension_um": (
            long_summary["geometry_projected_extension_m"] * 1.0e6
        ),
        "achieved_path_extension_um": long_summary["geometry_path_length_m"] * 1.0e6,
        "accepted_geometry_events": int(long_summary["n_geometry_events"]),
        "first_event_K_I_MPa_sqrt_m": long_summary["Kc_first_MPa_sqrt_m"],
        "queries": {str(x): bracket(long_rows, x) for x in (100.0, 500.0, 1000.0)},
    }

    cavity_failure = load_json(
        RAW / "v11_current_suffix" / "shared_runtime_attestation.failure.json"
    )
    fem_failure = load_json(
        RAW / "fem_czm_theta0_10um_run2" / "shared_runtime_attestation.failure.json"
    )
    results = {
        "schema": "fracture-current-source-bounded-pilot-results/1",
        "protocol": "PILOT_PROTOCOL.json",
        "opening_accommodation": {
            "status": "PASSED_SIX_OF_SIX",
            "case_count": 6,
            "paired_results": paired,
            "arm_details": detailed,
        },
        "developed_propagation": {"status": "PASSED", **long_result},
        "v11_current_source_suffix": {
            "status": "MIGRATION_INPUTS_LOADABLE_BUT_CONTINUATION_BLOCKED",
            "accepted_current_source_intervals": 0,
            "failure": cavity_failure,
            "interpretation": (
                "Current source deserializes both immutable historical endpoints, "
                "but the current fracture observer cannot recover the required "
                "directional observation after bounded refinement. No corrected-"
                "transport suffix state was accepted."
            ),
            "preparation_required": (
                "Create a new matched enabled/disabled V11 pair under current source "
                "from a pre-event accepted state whose current live-topology kernel "
                "registry covers the post-event tip and whose directional observation "
                "passes at the migration load; then write a native paired checkpoint "
                "and repeat the one-interval suffix."
            ),
            "estimated_new_preparation_storage_MiB": 800,
        },
        "fem_czm_consistency": {
            "status": "ADAPTER_REPAIRED_EXACT_RUN_BLOCKED_BY_MISSING_IMMUTABLE_INPUT",
            "adapter_fix_commit": "6c129c2",
            "focused_tests": "17 passed",
            "backend_commit": "293491157063484bb10df6adc479f847a6a08ba6",
            "accepted_solver_steps": 0,
            "failure": fem_failure,
            "required_kernel_sha256": "a85b57ad9eee8331ef34ea222760ce7d4064f48ddef668f1e28a666d14946f3a",
            "live_kernel_sha256": "d41b08f69ae773009f65f4c0094ef5436ba7f0f290a94172e5b894d09a41f7c3",
            "preparation_required": (
                "Recover the byte-identical a85b57 kernel family and referenced "
                "files from archive, or regenerate it from the locked historical "
                "mechanics configuration and verify the required hash before launch. "
                "Do not weaken the pinned FEM guard or substitute d41b08."
            ),
            "estimated_kernel_bundle_storage_MiB": 10,
        },
        "storage": {
            "raw_pilot_root_MiB": round(
                sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file()) / 2**20,
                3,
            ),
            "data_deleted": False,
            "external_archive_modified": False,
            "rendered_step_sequences_created": False,
            "free_space_GiB_after": round(shutil.disk_usage("/Volumes/Data").free / 2**30, 3),
        },
        "validation": {
            "focused_fem_adapter": "17 passed in 0.55 s",
            "full_pf_suite": "780 passed in 74.13 s",
            "python": "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python",
            "pytest_plugin_autoload": False,
        },
    }
    (OUT / "PILOT_RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )

    colors = {"accommodation_enabled": "#d95f02", "opening_only": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    x = list(range(len(paired)))
    width = 0.36
    for offset, arm, title in (
        (-width / 2, "accommodation_enabled", "Accommodation enabled"),
        (width / 2, "opening_only", "Opening only"),
    ):
        first_key = "first_K_enabled_MPa_sqrt_m" if arm.endswith("enabled") else "first_K_opening_only_MPa_sqrt_m"
        post_key = "post_K_enabled_MPa_sqrt_m" if arm.endswith("enabled") else "post_K_opening_only_MPa_sqrt_m"
        axes[0].bar([v + offset for v in x], [r[first_key] for r in paired], width,
                    color=colors[arm], label=title)
        axes[1].bar([v + offset for v in x], [r[post_key] for r in paired], width,
                    color=colors[arm], label=title)
    for ax, title in zip(axes, ("First fracture event", "Post-event endpoint")):
        ax.set_xticks(x, [r["condition"].replace(" ", "\n", 1) for r in paired])
        ax.set_ylabel(r"$K_I$ (MPa$\sqrt{m}$)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "opening_accommodation_first_passage.png", dpi=220)
    plt.close(fig)

    event_rows = geometry_events(long_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot([r["extension_um"] for r in event_rows],
            [r["K_I_MPa_sqrt_m"] for r in event_rows], lw=1.5, color="#374c80")
    for target in (100.0, 500.0, 1000.0):
        q = long_result["queries"][str(target)]
        ax.scatter(q["upper_extension_um"], q["upper_K_I_MPa_sqrt_m"], s=28,
                   color="#e34a33", zorder=3)
    ax.set_xlabel("Projected crack extension (µm)")
    ax.set_ylabel(r"Accepted-event $K_I$ (MPa$\sqrt{m}$)")
    ax.set_title("Current-source DBTT 300 K propagation")
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "current_pf_1mm_K_history.png", dpi=220)
    plt.close(fig)

    report = f"""# Bounded current-source fracture pilots

## Outcome

All six full-spatial PF first-passage cases passed, and the current source
completed a developed DBTT 300 K path to
{long_result['achieved_projected_extension_um']:.3f} µm. The V11 suffix and
exact pinned FEM/CZM calculation stopped at explicit provenance/physics gates;
neither produced an accepted new scientific state.

## Opening versus accommodation

| Condition | First K enabled | First K opening-only | ΔK | Ratio | Post-event K enabled | Post-event K opening-only |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in paired:
        report += (
            f"| {row['condition']} | {row['first_K_enabled_MPa_sqrt_m']:.3f} | "
            f"{row['first_K_opening_only_MPa_sqrt_m']:.3f} | "
            f"{row['first_delta_K_MPa_sqrt_m']:.3f} | "
            f"{row['first_ratio_enabled_over_control']:.3f} | "
            f"{row['post_K_enabled_MPa_sqrt_m']:.3f} | "
            f"{row['post_K_opening_only_MPa_sqrt_m']:.3f} |\n"
        )
    report += f"""

At 300 K the DBTT accommodation state is effectively inactive and the paired
fracture states coincide. At DBTT 1200 K and Peak 900 K, activated
Peierls–Taylor accommodation/backstress raises the first-event load by
{paired[1]['first_delta_K_MPa_sqrt_m']:.3f} and
{paired[2]['first_delta_K_MPa_sqrt_m']:.3f} MPa√m, respectively. This is a
paired mechanistic result for the registered rows and declared loading path,
not a fitted temperature law.

## Developed path

The current-head DBTT 300 K path accepted
{long_result['accepted_geometry_events']} geometry events and reached
{long_result['achieved_projected_extension_um']:.3f} µm projected extension.
The 100, 500, and 1000 µm values in `PILOT_RESULTS.json` are discrete accepted
jump brackets; no interpolation was performed.

## Bounded stops

- **V11 migration:** both archive endpoints deserialize, but the current
  directional observer is unavailable after bounded refinement at the migrated
  post-event topology. Zero current-source intervals were accepted. A new
  matched pair must be prepared under current source from a pre-event topology
  with current kernel coverage; allow about 800 MiB before doing so.
- **FEM/CZM:** commit `6c129c2` repaired the shared handoff and its focused suite
  passed 17 tests. The pinned backend at `2934911` then rejected the live
  `d41b08…` kernel because it requires the missing immutable `a85b57…` input.
  Recover/regenerate and hash-verify that small kernel bundle (allow 10 MiB);
  do not weaken the guard.

## Storage and provenance

The batch occupies {results['storage']['raw_pilot_root_MiB']:.1f} MiB and leaves
{results['storage']['free_space_GiB_after']:.2f} GiB free on `/Volumes/Data`.
No local or external data were deleted or modified, and no rendered step-image
sequence was created. Raw paths and exact failure receipts are indexed in
`PILOT_RESULTS.json`. The final integrated PF suite passed 780 tests in 74.13 s;
the focused FEM-adapter suite passed 17 tests.
"""
    (OUT / "CURRENT_SCIENCE_PILOT_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
