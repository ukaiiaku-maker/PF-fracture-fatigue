"""Source-bound current FEM companion for readiness v2.

The accepted v1 external pairing remains untouched.  This binding exposes the
newest independently tested companion and states the generic-adapter boundary
without pretending its Peak/1000 K entry is a four-family production adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess


ROOT = Path("/Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion")
COMMIT = "90cb6671523bb676cb241eb850b7f5fd41cf5461"
BRANCH = "codex/current-pf-companion"
ENTRY = Path("arrhenius_fracture/mode_i_first_passage_v10_0_5_18_4_0_current_pf_companion.py")


@dataclass(frozen=True)
class CurrentCompanionBinding:
    root: str
    expected_commit_prefix: str
    observed_commit: str | None
    expected_branch: str
    observed_branch: str | None
    tracked_clean: bool
    entry: str
    source_bound: bool
    exact_current_entry_status: str
    generic_parameter_loading_adapter_status: str
    opening_only_adapter_status: str

    def to_dict(self):
        return asdict(self)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def inspect_current_companion() -> CurrentCompanionBinding:
    observed = branch = None
    tracked_clean = False
    try:
        observed = _git("rev-parse", "HEAD")
        branch = _git("branch", "--show-current")
        tracked_clean = not bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        pass
    bound = bool(observed and observed.startswith(COMMIT) and branch == BRANCH
                 and tracked_clean and (ROOT / ENTRY).is_file())
    return CurrentCompanionBinding(
        str(ROOT), COMMIT, observed, BRANCH, branch, tracked_clean, str(ROOT / ENTRY), bound,
        "SOURCE_BOUND_PEAK_1000K_CURRENT_KERNEL" if bound else "UNAVAILABLE",
        "UNRESOLVED_NOT_EQUIVALENT_TO_LOCKED_CURRENT_ENTRY",
        "UNRESOLVED_GENERIC_CONTROL_BINDING",
    )


__all__ = ["CurrentCompanionBinding", "inspect_current_companion"]
