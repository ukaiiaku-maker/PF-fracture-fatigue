"""Final closure C: render part_x_final_decision.json into a readable
Markdown document (part_x_final_decision.md). Pure formatting -- no new
content, no recomputation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"


def _kv_table(d: dict) -> str:
    lines = ["| key | value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def main() -> None:
    d = json.loads((ARTIFACTS_DIR / "part_x_final_decision.json").read_text())

    md = []
    md.append(f"# Part X Final Decision\n")
    md.append(f"**Status:** `{d['status']}`  ")
    md.append(f"**Primary classification:** `{d['primary_classification']}`  ")
    md.append(f"**Subordinate verification classification:** `{d['subordinate_verification_classification']}`\n")

    md.append("## Terminal provenance\n")
    md.append(_kv_table(d["terminal_provenance"]) + "\n")

    md.append("## Supersession of component decisions\n")
    for name, info in d["supersedes"].items():
        md.append(f"- **{name}**: original status preserved unchanged as "
                   f"`{info['original_status_preserved_unchanged']}`; superseded by {info['superseded_by']}.")
        md.append(f"  - {info['note']}")
    md.append("")

    md.append("## Campaign lineage\n")
    md.append(d["campaign_lineage"] + "\n")

    md.append("## Job and attempt counts\n")
    counts = d["job_and_attempt_counts"]
    md.append(f"- PX3 screen authorized jobs: **{counts['px3_screen_authorized_jobs']}**")
    md.append(f"- PX4 developed admitted trajectories: **{counts['px4_developed_admitted_trajectories']}**")
    md.append(f"- PX4 developed matched finite/zero pairs: **{counts['px4_developed_matched_finite_zero_pairs']}**")
    md.append(f"- PX5 admitted static-control trajectories: **{counts['px5_admitted_static_control_trajectories']}**")
    md.append(f"- Total unique canonical job keys across campaign: **{counts['total_unique_canonical_job_keys_across_campaign']}**")
    md.append("\n**Disposition counts:**\n")
    md.append(_kv_table(counts["disposition_counts"]) + "\n")

    md.append("## Kmax grids and seeds\n")
    md.append(_kv_table(d["kmax_grids"]) + "\n")
    md.append(_kv_table(d["seeds"]) + "\n")

    md.append("## Protocol definitions\n")
    md.append("| protocol | row_name | R | frequency_Hz | hold_s | chemistry_factor | K_target | purpose |")
    md.append("|---|---|---|---|---|---|---|---|")
    for proto, spec in d["protocol_definitions"].items():
        md.append(f"| {proto} | {spec['row_name']} | {spec['R']} | {spec['frequency_Hz']:.3f} | "
                   f"{spec['minimum_load_hold_s']} | {spec['chemistry_factor']} | "
                   f"{spec['K_rebond_max_target_Pa_sqrt_m']:.0f} | {spec['purpose']} |")
    md.append("")

    md.append("## Developed da/dN and S_h interpretation\n")
    md.append(d["developed_da_dN_and_S_h_interpretation"] + "\n")

    md.append("## Local slopes and curvature\n")
    md.append(d["local_slopes_and_curvature"] + "\n")

    for section_key, title in [
        ("load_ratio_result", "Load-ratio (R) result"),
        ("frequency_result", "Frequency result"),
        ("dwell_result", "Dwell result and invalidated duration-weighting-bug history"),
        ("passivation_chemistry_result", "Passivation / chemistry result"),
        ("reversible_vs_persistent_result", "Reversible-versus-persistent result"),
        ("cohesive_strength_result", "Cohesive-strength result"),
        ("px5_static_vs_dynamic_attribution", "PX5 static-versus-dynamic attribution"),
    ]:
        section = d[section_key]
        md.append(f"## {title}\n")
        for k, v in section.items():
            if k == "interpretation":
                md.append(f"**Interpretation:** {v}\n")
            elif isinstance(v, dict):
                md.append(f"**{k}:**\n")
                md.append(_kv_table(v) + "\n")
            elif isinstance(v, list):
                md.append(f"**{k}:** {', '.join(str(x) for x in v)}\n")
            else:
                md.append(f"**{k}:** {v}\n")

    md.append("## Seed robustness\n")
    md.append(_kv_table(d["seed_robustness"]["by_protocol_axis"]) + "\n")
    md.append(f"All 5 axes robust: **{d['seed_robustness']['all_5_axes_robust']}**\n")

    md.append("## Numerical tolerances and censor qualifications\n")
    for k, v in d["numerical_tolerances_and_censor_qualifications"].items():
        md.append(f"- **{k}**: {v}")
    md.append("")

    md.append("## Physical simulation vs analytical prediction vs post-processing\n")
    md.append(d["physical_vs_analytical_vs_postprocessing_distinction"] + "\n")

    md.append("## Producer/launch provenance closure\n")
    md.append(_kv_table(d["producer_launch_provenance"]) + "\n")

    md.append("## Component verifier results\n")
    md.append(_kv_table(d["component_verifier_results"]) + "\n")

    md.append("## Qualifiers\n")
    for q in d["qualifiers"]:
        md.append(f"- `{q}`")
    md.append("")

    md.append("## Explicit scope limitations\n")
    for s in d["explicit_scope_limitations"]:
        md.append(f"- {s}")
    md.append("")

    md.append("## Not claimed\n")
    for s in d["not_claimed"]:
        md.append(f"- {s}")
    md.append("")

    out_path = ARTIFACTS_DIR / "part_x_final_decision.md"
    out_path.write_text("\n".join(md) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
