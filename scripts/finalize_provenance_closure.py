"""Paper-evidence provenance closure: final contract document and
file_hashes.json refresh. Run LAST, after every other builder/verifier
in this closure pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = {
        "schema": "v2_paper_completion_contract",
        "supersedes": "paper_completion_contract.json (v1)",
        "trigger": "Review decision requiring corrected completeness semantics, claim-level granularity, "
                   "actual figure-source forensics, individual temperature-fatigue exclusion review, a PX5 "
                   "independent verifier, campaign-branch verification, and a strict paper-evidence verifier.",
        "headline_finding": "Manuscript Figures 2, 3, and 4 were traced to their EXACT source data in "
            "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM/ (a plain, non-git-tracked directory "
            "-- confirmed distinct from the git-tracked Arrhenius_FEM_CZM_MPZ_* repositories, which host a "
            "separate theta=0-degree parity investigation that is NOT the source of these figures). Every one "
            "of the 12 K0/Kss/DeltaK values in Fig. 3 and all 4 RMS deviations in Fig. 2 were independently "
            "recomputed from raw per-seed/per-temperature CSVs and reproduce the manuscript's stated values "
            "exactly. This resolves the review's specific orientation-lineage concern: the manuscript's 45-"
            "degree comparison is a real, locatable, verified campaign, distinct from and not contradicted by "
            "the separate 0-degree parity work.",
        "honest_remaining_gaps": [
            "Figures 1, 5, 6, 7, and the SI synthetic-identifiability study remain MANUSCRIPT_RESULT_NOT_"
            "SOURCE_TRACED -- not because they are believed missing, but because their exact source data/"
            "scripts were not located in the time available this session.",
            "6 named campaign branches pass py_compile and git diff --check but could not be data-level "
            "re-verified (their strict verifiers require gitignored raw result directories not found on this "
            "filesystem) and were not pushed to origin.",
            "4 of the 7 temperature-fatigue numerical exclusions are not individually identified by (class, T, "
            "Kmax) -- only the 3 Peak-class exclusions are, via the campaign's own committed progress log.",
        ],
        "physical_simulations_launched_this_session": 0,
        "rerun_policy": "No completed campaign was rerun, and no replacement trajectory was authorized for any "
                        "of the 7 temperature-fatigue numerical exclusions -- no manuscript conclusion was "
                        "found to depend on resolving a specific excluded row.",
    }
    (OUT_DIR / "paper_completion_contract_v2.json").write_text(json.dumps(contract, indent=2, default=str))

    hashes = {}
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name != "file_hashes.json":
            hashes[str(path.relative_to(OUT_DIR))] = _sha256(path)
    (OUT_DIR / "file_hashes.json").write_text(json.dumps({"schema": "v2_file_hashes", "n_files": len(hashes), "files": hashes}, indent=2, sort_keys=True))
    print(f"Wrote paper_completion_contract_v2.json and file_hashes.json ({len(hashes)} files)")


if __name__ == "__main__":
    main()
