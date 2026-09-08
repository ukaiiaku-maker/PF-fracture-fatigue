"""Independent-verifier closure (review round 4): render the verifier's own
JSON output into a human-readable matrix CSV/JSON.

This script contains NO judging logic. Every field it writes is a direct
copy of what verify_paper_evidence_v4.py already decided. This is the
concrete fix for the review's core complaint about the prior architecture:
previously, build_paper_claim_evidence_matrix_v3.py computed `pass_fail` and
`final_evidence_class` itself, and the verifier only re-checked a few
structural booleans afterward. Here the dependency is inverted: the verifier
runs first and is authoritative; this script only formats its output.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"

FIELDS = [
    "claim_id", "figure_panel", "manuscript_claim_text", "source_data_level",
    "source_repository_or_archive", "producer_script", "parameter_config_fingerprint",
    "required_inputs", "source_bundle_present", "recompute_succeeded",
    "terminal_or_censor_status", "terminal_status_present",
    "independent_recomputation_performed", "actual_value", "expected_value",
    "tolerance", "numerical_comparison_passed", "pass_fail", "final_evidence_class", "notes",
]


def main() -> None:
    verifier_output = json.loads((OUT_DIR / "paper_simulation_completion_verification_v4.json").read_text())
    rows = []
    for e in verifier_output["evaluations"]:
        row = dict(e)
        row["required_inputs"] = "; ".join(row.get("required_inputs", []))
        row["actual_value"] = json.dumps(row.get("actual_value"), default=str)
        row["expected_value"] = json.dumps(row.get("expected_value"), default=str)
        rows.append(row)

    with (OUT_DIR / "paper_claim_evidence_matrix_v4.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    (OUT_DIR / "paper_claim_evidence_matrix_v4.json").write_text(json.dumps({
        "schema": "v4_paper_claim_evidence_matrix",
        "supersedes": "paper_claim_evidence_matrix_v3.{csv,json}",
        "source_of_truth": "paper_simulation_completion_verification_v4.json (this file only formats it)",
        "n_rows": len(rows),
        "n_by_final_evidence_class": verifier_output["n_by_final_evidence_class"],
        "classification": verifier_output["classification"],
        "rows": [{k: e.get(k) for k in FIELDS} for e in verifier_output["evaluations"]],
    }, indent=2, default=str))

    unresolved_fields = ["claim_id", "figure_panel", "manuscript_claim_text", "final_evidence_class", "notes"]
    unresolved = [e for e in verifier_output["evaluations"]
                  if e["final_evidence_class"] not in ("QUALIFIED_SOURCE_RESULT_VERIFIED",
                                                        "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED")]
    with (OUT_DIR / "paper_unresolved_claims_v4.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=unresolved_fields, lineterminator="\n")
        w.writeheader()
        for r in unresolved:
            w.writerow({k: r.get(k, "") for k in unresolved_fields})

    print(f"Wrote paper_claim_evidence_matrix_v4.{{csv,json}}: {len(rows)} rows, "
          f"{len(unresolved)} unresolved, classification={verifier_output['classification']}")


if __name__ == "__main__":
    main()
