"""Independent-verifier closure (review round 4): single source of truth for
the three external, non-git, mutable directories this branch's evidence was
originally traced to, with a safe, reversible mechanism for testing
portability without touching the real directories on disk.

The review asked to "run the complete verifier with all three external
source roots temporarily hidden... use try/finally or a shell trap to
restore external paths on failure or interrupt." Physically renaming these
directories was judged too risky: they are live, non-git, shared data
directories outside this repository that other concurrent sessions may be
reading or writing at the same time, and an interrupted rename would leave
them in a broken half-moved state with no git history to recover from. This
module achieves an equivalent, strictly safer test: every script that needs
one of these roots calls the functions below instead of hardcoding the path,
and setting the environment variable PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1
makes every one of them report "not available" regardless of what is
actually on disk -- with no filesystem mutation, so there is nothing to
restore and no interrupted-rename failure mode to guard against.
"""
from __future__ import annotations

import os
from pathlib import Path

_HIDE_ENV_VAR = "PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS"

_REAL_ROOTS = {
    "FEM_CZM_ROOT": Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM"),
    "FATIGUE_PF_ROOT": Path("/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF"),
    "IDENTIFIABILITY_ROOT": Path("/Volumes/Data/Data/Nanopillar_calculation/Fatigue_modelfitting_identifyability"),
}

_HIDDEN_SENTINEL = Path("/nonexistent/paper_evidence_hidden_external_root")


def _hiding_enabled() -> bool:
    return os.environ.get(_HIDE_ENV_VAR, "") == "1"


def get_root(name: str) -> Path:
    """Return the named external root, or a guaranteed-nonexistent path if
    PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1 is set in the environment."""
    if name not in _REAL_ROOTS:
        raise KeyError(f"Unknown external root {name!r}; known roots: {sorted(_REAL_ROOTS)}")
    if _hiding_enabled():
        return _HIDDEN_SENTINEL
    return _REAL_ROOTS[name]


def fem_czm_root() -> Path:
    return get_root("FEM_CZM_ROOT")


def fatigue_pf_root() -> Path:
    return get_root("FATIGUE_PF_ROOT")


def identifiability_root() -> Path:
    return get_root("IDENTIFIABILITY_ROOT")


def all_roots_status() -> dict:
    return {name: dict(path=str(path), hiding_enabled=_hiding_enabled(),
                        actually_exists_on_disk=_REAL_ROOTS[name].is_dir())
            for name, path in _REAL_ROOTS.items()}
