"""Independent-verifier closure (review round 4): reconstruct the connected-
crack S-N response across ALL available stresses and seeds for shielded and
unshielded blunt-notch cases, not just the single seed/stress pair audited
in the prior pass.

Enumerates every crack_handoff_audit_final.json under the
sn_stateful_pd_v8_5_1_reference_lives_seed{2,3,4,5} run family, and pairs it
with its sibling summary.json (present only for jobs that reached a full
handoff decision; a shielded job that never connects may lack summary.json
entirely, which is itself informative -- treated as right-censored / no
finite cycles_root_connected).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"
BUNDLE = OUT_DIR / "source_bundle_figures_2_4"
FP_ROOT = Path("/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF")
RUN_ROOT = FP_ROOT / "releases" / "stateful_pd_v8_5_standalone_v1_1" / "runs"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    audits = sorted(RUN_ROOT.glob("sn_stateful_pd_v8_5_1_reference_lives_seed*/seed_*/stress_*MPa/"
                                    "job_*/*/sigmaA_*MPa/crack_handoff_audit_final.json"))
    if not audits:
        raise SystemExit(f"No audit files found under {RUN_ROOT}")

    rows = []
    source_hashes = {}
    for audit_path in audits:
        job_dir = audit_path.parent
        parts = job_dir.parts
        seed = next((int(p.split("seed_")[1]) for p in parts if p.startswith("seed_")), None)
        stress = next((float(p.split("stress_")[1].replace("MPa", "")) for p in parts
                       if p.startswith("stress_")), None)
        condition = "shielded" if "shielded" in parts else ("no_shield" if "no_shield" in parts else "unknown")

        audit = json.loads(audit_path.read_text())
        source_hashes[str(audit_path.relative_to(FP_ROOT))] = _sha256(audit_path)

        summary_path = job_dir / "summary.json"
        cycles_root_connected = None
        status = None
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            source_hashes[str(summary_path.relative_to(FP_ROOT))] = _sha256(summary_path)
            cycles_root_connected = summary.get("cycles_root_connected")
            status = summary.get("status")

        rows.append(dict(
            seed=seed, sigma_a_MPa=stress, condition=condition,
            coverage_pass=audit.get("coverage_pass"), handoff_pass=audit.get("handoff_pass"),
            root_connected=audit.get("root_connected"),
            failure_reasons=audit.get("failure_reasons"),
            has_summary=summary_path.is_file(),
            cycles_root_connected=cycles_root_connected,
            summary_status=status,
        ))

    compact_path = BUNDLE / "fig5C_compact_sn_reconstruction.json"
    compact_path.write_text(json.dumps(dict(schema="v1_fig5c_compact_sn", n_jobs=len(rows), jobs=rows),
                                        indent=2, default=str))
    compact_hash = _sha256(compact_path)

    provenance = dict(
        schema="v1_fig5c_portable_projection_provenance",
        run_root=str(RUN_ROOT),
        n_audit_files_found=len(audits),
        n_summary_files_found=sum(1 for r in rows if r["has_summary"]),
        full_source_file_hashes=source_hashes,
        compact_table=dict(path=str(compact_path.relative_to(REPO_ROOT)), sha256=compact_hash, n_rows=len(rows)),
    )
    (OUT_DIR / "fig5c_portable_projection_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str))
    print(json.dumps(dict(n_jobs=len(rows), rows=rows), indent=2, default=str))


if __name__ == "__main__":
    main()
