#!/usr/bin/env python3
"""Publish paper text, claims/evidence, and deterministic audit provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


OUTPUT_DEFAULT = Path("analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def claims(output: Path) -> pd.DataFrame:
    rows = [
        {
            "claim": "Complete chronological PF model-native KJ trajectories were recovered for all 288 canonical cases.",
            "status": "SUPPORTED",
            "supporting_cases": "288/288",
            "supporting_artifact": "pf_canonical_full_step_trajectories.parquet; pf_canonical_full_trajectory_manifest.json",
            "scope": "accepted-step canonical single-crack histories",
            "limitation": "model-native PF driving quantity",
            "allowed_wording": "complete PF model-native KJ driving trajectories",
            "wording_to_avoid": "applied K histories; conventional R-curves",
        },
        {
            "claim": "Peak theta0 pre-initiation KJ/opening transfer is rate invariant.",
            "status": "SUPPORTED",
            "supporting_cases": "36 cases; 12 matched temperature triplets",
            "supporting_artifact": "pf_peak_rate_initial_structural_transfer.csv",
            "scope": "0.01x/1x/100x before first event",
            "limitation": "canonical geometry and production-discrete PF KJ",
            "allowed_wording": "rate separation is not caused by initial structural KJ/U",
            "wording_to_avoid": "all structural mechanics are rate independent after growth",
        },
        {
            "claim": "Peak slow-rate onset elevation is the greater opening needed after rate-dependent state evolution.",
            "status": "SUPPORTED",
            "supporting_cases": "12 matched Peak theta0 temperature triplets",
            "supporting_artifact": "pf_peak_theta0_rate_onset_decomposition.csv",
            "scope": "initial onset",
            "limitation": "does not uniquely partition nonlinear state variables",
            "allowed_wording": "greater-required-opening/local-state effect",
            "wording_to_avoid": "structural-transfer toughening",
        },
        {
            "claim": "Emission/blunting, backstress, shielding, multiplicity, and cleavage history have unique additive shares in the Peak rate effect.",
            "status": "NOT_SUPPORTED",
            "supporting_cases": "none; exact component injection unavailable",
            "supporting_artifact": "pf_peak_theta0_rate_counterfactuals.csv",
            "scope": "nonlinear evolved-state counterfactuals",
            "limitation": "complete tensor/state injection contract absent",
            "allowed_wording": "mixed local-state mechanism with strong emission/blunting and backstress changes",
            "wording_to_avoid": "percent contribution of radius/backstress/shielding/multiplicity",
        },
        {
            "claim": "Increasing theta lowers mean Peak and DBTT initial model-native onset KJ.",
            "status": "SUPPORTED",
            "supporting_cases": "96 Peak/DBTT orientation cases",
            "supporting_artifact": "pf_orientation_initial_onset_decomposition.csv",
            "scope": "theta0/15/30/45 at rate1x",
            "limitation": "theta rotates crystal orientation, not crack line",
            "allowed_wording": "mean Peak 46.59 to 21.03 and DBTT 41.14 to 21.89 MPa sqrt(m), theta0 to theta45",
            "wording_to_avoid": "crack-path rotation lowers intrinsic toughness",
        },
        {
            "claim": "The Peak/DBTT initial orientation loss is dominated by the opening/local-threshold term rather than structural KJ/opening.",
            "status": "SUPPORTED",
            "supporting_cases": "96 Peak/DBTT orientation cases plus frozen zero-history maps",
            "supporting_artifact": "pf_orientation_initial_onset_decomposition.csv; pf_orientation_frozen_swap_matrix.csv",
            "scope": "exact K=C*U decomposition and zero-history diagnostic",
            "limitation": "local tensor versus accumulated scalar-state subpartition unresolved",
            "allowed_wording": "opening/local-threshold dominated",
            "wording_to_avoid": "purely structural anisotropy",
        },
        {
            "claim": "Off-axis Peak and DBTT more frequently have finite reload-separated effective resistance candidates.",
            "status": "SUPPORTED_CONDITIONALLY",
            "supporting_cases": "Peak 48 and DBTT 48 orientation cases",
            "supporting_artifact": "pf_orientation_conditional_reinitiation_statistics.csv",
            "scope": "finite reload-separated onset incidence",
            "limitation": "conditional and temperature dependent",
            "allowed_wording": "Peak incidence rises from 0/12 at theta0 to 6/12 at theta45; DBTT is 8/12, 10/12, 8/12, 9/12",
            "wording_to_avoid": "universal off-axis rising R-curve",
        },
        {
            "claim": "Positive final Peak/DBTT Delta K reinitiation is mainly a required-opening/local-state effect, often opposed by structural wake transfer.",
            "status": "SUPPORTED_CONDITIONALLY",
            "supporting_cases": "37 positive final Peak/DBTT candidates",
            "supporting_artifact": "pf_orientation_reinitiation_decomposition.csv",
            "scope": "exact signed aggregate final-candidate decomposition",
            "limitation": "signed terms may oppose and are not material-toughness fractions",
            "allowed_wording": "required-opening contribution 300.77 versus structural contribution -59.35 MPa sqrt(m) for 241.42 aggregate Delta K",
            "wording_to_avoid": "structural wake is the primary source of the positive resistance",
        },
        {
            "claim": "The local process-zone state is unchanged from initial onset to positive reinitiation.",
            "status": "NOT_SUPPORTED",
            "supporting_cases": "37 positive final Peak/DBTT candidates",
            "supporting_artifact": "pf_orientation_reinitiation_decomposition.csv",
            "scope": "archived scalar states",
            "limitation": "individual variables change non-monotonically",
            "allowed_wording": "effective local threshold hardening with mixed scalar-state changes",
            "wording_to_avoid": "invariant local source state",
        },
        {
            "claim": "Positive PF model-native Delta K is a conventional energy-release R-curve.",
            "status": "NOT_SUPPORTED",
            "supporting_cases": "none",
            "supporting_artifact": "PF_PEAK_DBTT_ORIENTATION_REINITIATION_AUDIT.md",
            "scope": "terminology and physical interpretation",
            "limitation": "PF native domain-J/topology sensitivity remains model-native",
            "allowed_wording": "reload-separated effective resistance candidate",
            "wording_to_avoid": "conventional R-curve; intrinsic toughness increase",
        },
        {
            "claim": "Complete evolved tensor swaps establish a unique local source mechanism.",
            "status": "UNRESOLVED_MISSING_STATE",
            "supporting_cases": "zero-history orientation swap only",
            "supporting_artifact": "pf_orientation_frozen_swap_matrix.csv; pf_peak_theta0_rate_counterfactuals.csv",
            "scope": "deterministic frozen-state diagnostics",
            "limitation": "complete tensor matrices were not archived at evolved accepted states",
            "allowed_wording": "zero-history structural/tensor control; evolved-state attribution unresolved",
            "wording_to_avoid": "exact evolved-state tensor causality proved",
        },
        {
            "claim": "The analysis changed trajectories, equations, material rows, or canonical data.",
            "status": "FALSE_VERIFIED",
            "supporting_cases": "all source artifacts read-only",
            "supporting_artifact": "pf_canonical_full_trajectory_manifest.json; provenance bundle",
            "scope": "this analysis branch",
            "limitation": "new deterministic analysis outputs only",
            "allowed_wording": "analysis-only; zero new stochastic or FEM/CZM runs",
            "wording_to_avoid": "new production campaign",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pf_rate_orientation_claims_and_evidence.csv", index=False)
    return frame


def paper(output: Path) -> None:
    text = """# Paper-ready canonical PF rate and orientation mechanisms

## Methods paragraph

We recovered every accepted chronological step from the 288-case canonical PF
campaign and verified each raw trajectory SHA-256 against the final run
manifest. PF model-native J and KJ, reaction, opening, time, projected crack
extension, onset roles, physical-avalanche membership, and consolidated scalar
process-zone histories were joined without sorting by crack length, thereby
preserving fixed-extension loading and reloading. Rate comparisons used common
random numbers at fixed class and temperature. Exact differences were separated
with K=C U and the symmetric product identity. A deterministic zero-history PF
sharp-wake diagnostic rotated cubic elasticity and crystallographic sources
through theta=0/15/30/45 degrees relative to the fixed horizontal crack path;
it advanced no stochastic clock. Complete evolved-state tensor swaps were
reported unavailable where the default-off observer had not archived the
required tensors.

## Peak rate-effect results paragraph

At theta=0, the pre-initiation structural coefficient KJ/U was identical among
0.01x, 1x, and 100x at each temperature to about 10^-15 relative spread.
Consequently, Peak's large high-temperature slow-rate elevation is entirely the
greater opening required to trigger cleavage after rate-dependent local-state
evolution, not a different initial crack geometry or structural KJ/opening
transfer. In the 900–1200 K deep set, slow loading produces major, though
temperature-dependent, changes in emission history, radius, mobile/retained
populations, backstress, signed shielding, and multiplicity. The defensible
classification is time available for emission plus a mixed local-state effect;
the nonlinear archive does not support additive percentages for the individual
state variables.

## Orientation-effect results paragraph

Mean Peak initial onset decreases from 46.59 MPa sqrt(m) at theta=0 to 21.03 at
theta=45; DBTT decreases from 41.14 to 21.89. The exact onset decomposition
places almost all of these mean changes in the opening/local-threshold term,
while the structural KJ/opening term is small. The zero-history orientation
control likewise gives identical initial-geometry KJ/opening at theta=0 and
theta=45 but strongly rotates the resolved source shears. Thus the evidence
supports anisotropic local tensor/source competition and accumulated state as
the dominant origin of the onset loss, not crack-line rotation or a large
structural transfer change.

Reload-separated candidates become more common off axis for Peak: finite
incidence is 0/12, 1/12, 3/12, and 6/12 for theta=0,15,30,45. DBTT incidence is
8/12, 10/12, 8/12, and 9/12, with conditional mean Delta K values -1.86, 5.74,
7.37, and 8.56 MPa sqrt(m). For the 37 positive final Peak/DBTT candidates, the
signed aggregate exact decomposition is 300.77 MPa sqrt(m) from increased
required opening/local state and -59.35 from changed sharp-wake structural
coefficient, summing to 241.42. Structural wake transfer therefore often
opposes rather than creates the positive candidate. Individual scalar changes
are mixed, but the increased required opening is an effective local hardening
signal.

## Physical interpretation

Slow loading supplies physical time for emission, transport, retention/release,
blunting, backstress, and shielding to reorganize before the first cleavage
threshold is crossed. Orientation changes the local tensor projection and
source-channel competition relative to the fixed cleavage trace. Off-axis
trajectories more often interrupt the first avalanche and require subsequent
loading; their positive model-native Delta K is primarily associated with a
higher next-event opening after evolved local state, while geometry-dependent
KJ/opening usually contributes less and can reduce the observed Delta K.

## Limitations

The full curves are PF model-native driving trajectories, not applied-K curves
or conventional energy-release R-curves. Reload-separated pre-event points are
effective resistance candidates. The default-off archive contains resolved
shears and rich scalar histories but not complete evolved opening/channel tensor
matrices. Exact evolved-state component swaps therefore fail closed. The
zero-history swap is a deterministic diagnostic, not production physics, and
does not authorize claims of additive intrinsic-toughness contributions.

## Main-figure captions

1. **Peak theta0 rate trajectories.** Full and early accepted-step PF
   model-native KJ histories; stars mark initial onset, open circles certified
   reload-separated onsets, and triangles target-right-censored endpoints.
2. **Peak pre-initiation state.** Rate- and temperature-resolved evolution of
   radius, mobile/retained population, backstress, signed shielding, and
   multiplicity versus opening.
3. **Peak cleavage/emission competition.** Accepted-state cleavage and emission
   rates and cumulative cleavage action for the deep 900–1200 K set.
4. **Peak rate decomposition.** Initial onset and exact slow-minus-1x K=C U
   contributions, demonstrating the absence of an initial structural term.
5. **Peak and DBTT orientation response.** Initial and reload-separated onset
   candidates versus crystal orientation relative to the fixed crack path.
6. **Orientation controls.** Zero-history sharp-wake KJ/opening and resolved
   source shears at common geometry/opening.
7. **Reinitiation decomposition.** Conditional exact required-opening and
   structural-wake contributions; signed terms can oppose.
8. **Campaign mechanism summary.** Mean initial onset, reinitiation incidence,
   and conditional Delta K across all four classes and four orientations.

## SI-figure captions

The full atlas contains 16 canonical orientation and 12 canonical rate panels,
four orientation and four rate composites, all in full and 0–150 um early
versions. Each panel uses the same discrete 12-temperature color map. Four
theta45/rate0.01x supplemental panels are segregated from the primary rate
analysis; ceramic-like is explicitly incomplete at six temperatures. Every
figure has PDF, SVG, 600-dpi PNG, and source data.

## Claims and evidence

The companion `pf_rate_orientation_claims_and_evidence.csv` defines supported,
conditional, unresolved, and prohibited wording. In particular, it supports
"model-native driving trajectory," "effective reload-separated resistance
candidate," and "mixed local-state effect," while prohibiting "applied K,"
"conventional R-curve," and unique additive component percentages.
"""
    (output / "PAPER_CANONICAL_PF_RATE_AND_ORIENTATION_MECHANISMS.md").write_text(text)


def provenance(output: Path, producer_commit: str) -> None:
    artifacts = {}
    patterns = ("*.csv", "*.parquet", "*.json", "*.md")
    for pattern in patterns:
        for path in sorted(output.glob(pattern)):
            if path.name == "pf_canonical_full_trajectory_and_mechanism_provenance.json":
                continue
            artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    record = {
        "schema": "pf_canonical_full_trajectory_and_mechanism_provenance_v1",
        "analysis_branch": git("branch", "--show-current"),
        "analysis_producer_code_commit": producer_commit,
        "campaign_execution_commit": "c3f33fa7477ea44e612fa21b6b1b1fed0df73295",
        "final_publisher_source_commit": "b06e7cbcfc535081c8836f988e601eeea620892b",
        "qualified_physical_pf_source": "9e884fb0b0845da621d2612bdf1042e481b8df49",
        "campaign_lock_fingerprint": "5928e6abb7dcd59e6387d5d479128fec83c3ba4d509bae3a0e757b9e9ece5dde",
        "scientific_plan_fingerprint": "f3928476f2564a3eb10ca4737780a38578d9517a860bd77a9321dcd94fd4df99",
        "new_stochastic_pf_trajectories": 0,
        "fem_czm_runs": 0,
        "material_rows_changed": 0,
        "physical_equations_changed": 0,
        "canonical_raw_artifacts_modified": False,
        "artifacts": artifacts,
    }
    (output / "pf_canonical_full_trajectory_and_mechanism_provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--producer-code-commit", default=None)
    args = parser.parse_args()
    producer = args.producer_code_commit or git("rev-parse", "HEAD")
    table = claims(args.output)
    paper(args.output)
    provenance(args.output, producer)
    print(f"published {len(table)} claims; producer code {producer}")


if __name__ == "__main__":
    main()
