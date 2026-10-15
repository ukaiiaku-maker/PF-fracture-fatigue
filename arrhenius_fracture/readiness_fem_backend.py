"""Current independent FEM companion bindings for framework readiness.

The accepted v1 external pairing remains untouched.  This binding exposes the
newest independently tested source-bound preset and registry-driven entry while
preserving the still-open constraint, restart and convergence boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess


ROOT = Path("/Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion")
COMMIT = "784c57f2b48172c6bd5b22f6892faff623cbb1f3"
BRANCH = "codex/current-pf-companion"
ENTRY = Path("arrhenius_fracture/mode_i_first_passage_v10_0_5_18_4_0_current_pf_companion.py")
GENERIC_ENTRY = Path("arrhenius_fracture/generic_current_pf_companion.py")


@dataclass(frozen=True)
class CurrentCompanionBinding:
    root: str
    expected_commit_prefix: str
    observed_commit: str | None
    expected_branch: str
    observed_branch: str | None
    tracked_clean: bool
    entry: str
    generic_entry: str
    source_bound: bool
    exact_current_entry_status: str
    generic_parameter_loading_adapter_status: str
    opening_only_adapter_status: str

    def to_dict(self):
        return asdict(self)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _baseline_is_ancestor(observed: str | None) -> bool:
    if observed is None:
        return False
    try:
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", COMMIT, observed],
            cwd=ROOT, check=False, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def inspect_current_companion() -> CurrentCompanionBinding:
    observed = branch = None
    tracked_clean = False
    try:
        observed = _git("rev-parse", "HEAD")
        branch = _git("branch", "--show-current")
        tracked_clean = not bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        pass
    # COMMIT is the qualified baseline, not a reset target.  Authorized native
    # implementation commits remain source-bound only when that exact baseline
    # is an ancestor on the expected branch and the tracked tree is clean.
    bound = bool(_baseline_is_ancestor(observed) and branch == BRANCH
                 and tracked_clean and (ROOT / ENTRY).is_file()
                 and (ROOT / GENERIC_ENTRY).is_file())
    return CurrentCompanionBinding(
        str(ROOT), COMMIT, observed, BRANCH, branch, tracked_clean,
        str(ROOT / ENTRY), str(ROOT / GENERIC_ENTRY), bound,
        "SOURCE_BOUND_PEAK_1000K_CURRENT_KERNEL" if bound else "UNAVAILABLE",
        "IMPLEMENTED_REGISTRY_DRIVEN_NATIVE_PLANE_STRAIN_TIP_ONLY_ENTRY",
        "IMPLEMENTED_NATIVE_CHANNEL_OWNER_AND_EXECUTED",
    )


__all__ = ["CurrentCompanionBinding", "inspect_current_companion"]
