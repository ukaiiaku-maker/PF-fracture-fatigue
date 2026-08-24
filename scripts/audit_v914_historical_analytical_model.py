#!/usr/bin/env python3
"""Recover the exact low-fidelity ancestry of the v9.14 fatigue candidates.

This tool is intentionally provenance-first.  It does not reimplement either
historical model.  It fingerprints the source files and Git commits that made
the candidate population and ranking, and records the distinct roles of the
v9.13 zero-D proxy and the v9.14 exact-cycle state screen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


V913_FILES = (
    "arrhenius_fracture/zero_d_search_v913.py",
    "arrhenius_fracture/zero_d_persistent_v913.py",
    "scripts/run_v913_zero_d_large_search.py",
)
V914_FILES = (
    "arrhenius_fracture/endurance_knee_v914.py",
    "arrhenius_fracture/fatigue_v914.py",
    "scripts/generate_v914_endurance_knee_manifold.py",
    "scripts/run_v914_endurance_knee_state_screen.py",
    "scripts/analyze_v914_endurance_knee_state_screen.py",
    "scripts/rank_v914_class_a_candidates.py",
)

COORDINATES = (
    "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a",
    "cleave_exp_n", "cleave_floor_frac", "emit_G00_eV",
    "emit_sigc0_GPa", "emit_exp_a", "emit_exp_n", "emit_floor_frac",
    "peierls_H0_eV", "peierls_activation_entropy_kB", "peierls_exp_a",
    "peierls_exp_n", "taylor_H0_eV", "taylor_activation_entropy_kB",
    "taylor_exp_a", "taylor_exp_n", "rho_source0_m2",
    "taylor_corr_rho_c_m2", "taylor_corr_scale", "c_blunt",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def introducing_commit(repo: Path, relative: str) -> str:
    commits = git(repo, "log", "--diff-filter=A", "--format=%H", "--", relative).splitlines()
    if not commits:
        raise RuntimeError(f"no introducing commit for {relative}")
    return commits[-1]


def source_records(repo: Path, names: tuple[str, ...], role: str) -> list[dict[str, str]]:
    records = []
    for name in names:
        path = repo / name
        if not path.is_file():
            raise FileNotFoundError(path)
        intro = introducing_commit(repo, name)
        records.append({
            "role": role, "relative_path": name, "absolute_path": str(path.resolve()),
            "working_tree_sha256": sha256(path), "introducing_commit": intro,
            "introducing_commit_subject": git(repo, "show", "-s", "--format=%s", intro),
        })
    return records


def mapping_rows() -> list[dict[str, str]]:
    cleavage = set(COORDINATES[:5]); emission = set(COORDINATES[5:10])
    peierls = set(COORDINATES[10:14]); taylor = set(COORDINATES[14:21])
    rows = []
    for field in COORDINATES:
        if field in cleavage:
            old = "directly in cleavage ExpFloor barrier and raw cleavage rate"
            current = "directly in cleavage hazard; state screen frozen/direct and evolved response"
        elif field in emission:
            old = "directly in source-emission barrier/rate and reduced plastic estimate"
            current = "directly in emission competition and evolved mobile/retained state"
        elif field in peierls:
            old = "directly in reduced glide/transport kinetics"
            current = "directly in mobile transport and state relaxation"
        elif field in taylor:
            old = "directly in Taylor completion/correlation or completion barrier"
            current = "directly in retention/completion, backstress, and state relaxation"
        else:
            old = "directly in reduced source multiplicity or blunting"
            current = "directly in persistent-source multiplicity or tip blunting"
        rows.append({"coordinate": field, "historical_zeroD_role": old,
                     "v914_state_screen_role": current, "varied_in_global_300K_population": "True"})
    return rows


def equations_text() -> str:
    return r"""# Exact historical analytical-model equations and roles

## Scope correction

Two low-cost models exist and must not be conflated.

1. The ancestral v9.13 vectorized zero-D proxy generated/ranked the earlier
   temperature-response population. It approximated neutral cleavage loading,
   source emission, backstress-limited activation, density-dependent front
   width, and slip-driven blunting.
2. A/B/C/D came from the later v9.14 300 K, 30-coordinate Sobol population
   (22 candidate coordinates plus eight qualified CommonPhysics overrides).
   Their actual fatigue pre-screen was an exact-cycle, virgin-state,
   pre-first-passage state/hazard calculation. It sampled no stochastic
   threshold and committed no geometry event.

## Shared ExpFloor surface

For either cleavage or emission,

\[
G_0(T)=\max(G_{00}+g_T(T-T_{ref}),10^{-12}),\quad
G_f=\min(0.95G_0,\max(10^{-4},fG_0)),
\]
\[
\sigma_c(T)=\max[(\sigma_{c0}+s_T(T-T_{ref}))10^9,1],
\]
\[
G^*(\sigma,T)=G_f+(G_0-G_f)\exp[-a(\max(\sigma,0)/\sigma_c)^n].
\]

The raw Arrhenius rate is \(\Gamma_{raw}=\nu_0\exp[-G^*/(k_BT)]\).

## Ancestral v9.13 zero-D proxy

The neutral cleavage stress was obtained by inverting the production hit
completion probability at a fixed target rate. With \(h\) hits and correlation
time \(\tau_c\), \(\Gamma_{raw,target}=P^{-1}(h,r_{target}\tau_c)/\tau_c\).
The neutral stress maps to \(K=\sigma\sqrt{2\pi r_0}/10^6\).

The proxy then estimated source multiplicity, emission activation, a
backstress-limited activation count, dislocation density, front-width
contraction, and slip blunting. Its final proxy was
\(K_{proxy}=K_0\sqrt{r/r_0}\). Its historical objective concerned the
temperature location/prominence/drop/rebound of a toughness proxy—not a
developed Paris propagation slope.

## Actual v9.14 A/B/C/D fatigue pre-screen

For each candidate and load fraction, the exact existing one-cycle constitutive
integrator advanced eight cycles from a virgin state. The screen recorded the
cycle hazard actions \(H_j\), limiting action \(H_8\), and the diagnostic mean
first-passage projection \(N_q=1/H_8\). It also evaluated frozen-state hazards
at \(\Delta K(1\pm10^{-3})\).

\[
G_{eff}=-k_BT\ln(\max(H_8,10^{-300})),
\]
\[
\left.\frac{dG_{eff}}{d\Delta K}\right|_{state}
=-k_BT\frac{\ln(H_+/H_-)}{2(10^{-3})\Delta K}.
\]

Finite differences of \(G_{eff}\) across loads supplied a total derivative;
total minus frozen supplied the state-mediated derivative. Candidate ranking
rewarded dynamic hazard range, localized slope changes, coincident state or
barrier-floor crossovers, and—for Class A—a sampled bare-barrier transition.
It was not a developed crack-growth-rate or Paris-slope objective.

## New diagnostic only

The audit may compute
\(m_{raw}=d\ln\Gamma_{raw}/d\ln\Delta K\),
\(A_{3hit}=d\ln\Gamma_{prod}/d\ln\Gamma_{raw}\), and
\(m_{analytic,3hit}=A_{3hit}m_{raw}\) using current \(h=3\) and \(\tau_c\).
This changes analysis only; it does not change historical or production physics.
"""


def objective_text() -> str:
    return """# Historical analytical objective audit

## What was actually optimized

The ancestral v9.13 vectorized proxy optimized a temperature-dependent
toughness/transition shape: internal peak location, prominence, post-peak drop,
high-temperature rebound, and a broad peak-value window. It did not optimize a
developed Paris slope.

The later v9.14 campaign generated A/B/C/D from a scrambled Sobol population at
300 K. A broad monotonic-fracture gate first retained candidates with finite
50 µm response. The fatigue pre-screen then ran eight exact constitutive cycles
from a virgin state at six load fractions, without a sampled threshold or crack
advance. Ranking emphasized changing pre-first-passage hazard, localization,
barrier-floor or plastic-state crossover, and the relation between projected
first passage and plastic relaxation.

## Why A/B/C/D looked attractive

- A represented a localized direct cleavage-barrier transition.
- B represented a plastic-state-controlled transition.
- C represented a cleavage/plastic timescale crossover.
- D represented a contrasting mechanism probe retained for discrimination.

They were attractive as mechanism-resolved endurance-knee probes. They were not
historically demonstrated predictors of developed propagation rate or Paris
slope. The new analytical-v7 comparison must therefore test transfer rather
than treating the historical score as a propagation calibration target.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.historical_repo.resolve()
    if not (repo / ".git").exists():
        raise RuntimeError(f"not a Git checkout: {repo}")
    records = source_records(repo, V913_FILES, "ancestral_v913_vectorized_zeroD")
    records += source_records(repo, V914_FILES, "actual_v914_ABCD_population_and_state_screen")
    payload = {
        "schema": "v914_historical_analytical_model_provenance_v1",
        "historical_repository": str(repo), "audited_checkout_head": git(repo, "rev-parse", "HEAD"),
        "audited_checkout_branch": git(repo, "branch", "--show-current"),
        "scientific_scope": {
            "ancestral_v913": "temperature-response zero-D toughness proxy",
            "actual_ABCD": "300 K exact-cycle pre-first-passage state/hazard screen",
            "developed_paris_rate_was_historical_objective": False,
        },
        "source_files": records,
        "population_contract": {"sobol_seed": 1914, "candidate_coordinates": 22,
                                "common_physics_overrides": 8, "temperature_K": 300.0},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analytical_model_provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out / "analytical_model_equations.md").write_text(equations_text())
    (args.out / "historical_analytical_objective_audit.md").write_text(objective_text())
    rows = mapping_rows()
    with (args.out / "analytical_model_candidate_mapping.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"V914_ANALYTICAL_PROVENANCE_COMPLETE files={len(records)} coordinates={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
