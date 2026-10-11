"""Execute the exact clean V3 coupled reduced fracture/voiding donor.

R2 established the donor identity but only ran its generic installed demo.
R3 runs the donor's bounded aligned driver, whose global scheduler owns the
complete V2 fracture adapter and both void-birth clocks in one state sequence.
The source is always a clean archive; a dirty donor checkout is never imported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

from .reduced_void_reference_r2 import (
    DONOR_COMMIT, DONOR_REPOSITORY, DONOR_TREE, donor_receipt,
)


MODEL_ID = "r3.clean-v3-bounded-coupled-fracture-void/1"
BOUNDED_DRIVER = "scripts/run_oneD_v3_bounded_autonomous.py"
ADAPTER_PARITY_DRIVER = "scripts/compare_complete_v2_fracture_adapter_v1.py"
BOUNDED_RESULT = "analysis_outputs/oneD_v3_bounded_autonomous/result.json"
ADAPTER_RESULT = "analysis_outputs/oneD_v3_complete_fracture_adapter/parity.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(source: Path, relative: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-r3")
    completed = subprocess.run(
        (sys.executable, relative), cwd=source, env=environment, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return {
        "command": [sys.executable, relative],
        "stdout": completed.stdout.strip().splitlines(),
        "stderr": completed.stderr.strip().splitlines(),
    }


def execute_coupled_void_r3(output_directory: str | Path) -> dict[str, Any]:
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    receipt = donor_receipt()

    with tempfile.TemporaryDirectory(prefix="r3_coupled_v3_") as raw:
        temporary = Path(raw)
        archive = temporary / "donor.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ("git", "-C", str(DONOR_REPOSITORY), "archive", DONOR_COMMIT),
                check=True, stdout=stream,
            )
        source = temporary / "source"
        source.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter="data")

        bounded_execution = _run(source, BOUNDED_DRIVER)
        adapter_execution = _run(source, ADAPTER_PARITY_DRIVER)
        bounded_path = source / BOUNDED_RESULT
        adapter_path = source / ADAPTER_RESULT
        bounded = json.loads(bounded_path.read_text())
        adapter = json.loads(adapter_path.read_text())
        shutil.copy2(bounded_path, destination / "bounded_coupled_result.json")
        shutil.copy2(adapter_path, destination / "complete_adapter_parity.json")

    no_void = bounded["outcomes"]["no_void"]
    enabled = bounded["outcomes"]["void_enabled"]
    partition = bounded["partition_convergence"]
    event_names = [str(row.get("event")) for row in enabled["events"]]
    same_start_nonvoid = {
        "initial_time_s": bounded["external_inputs"]["initial_time_s"],
        "terminal_time_s": bounded["external_inputs"]["terminal_time_s"],
        "same_external_opening_m": bounded["external_inputs"]["opening_m"],
        "same_temperature_K": bounded["external_inputs"]["temperature_K"],
        "same_material": bounded["external_inputs"]["material"],
        "no_void_fracture_intervals": no_void["fracture_adapter_intervals"],
        "void_enabled_fracture_intervals": enabled["fracture_adapter_intervals"],
        "no_void_terminal_extension_m": no_void["terminal"]["fracture_extension_m"],
        "void_enabled_terminal_extension_m": enabled["terminal"]["fracture_extension_m"],
        "divergence_is_native_cavity_history": bool(enabled["events"] and not no_void["events"]),
    }
    active = {
        "terminal_phase": enabled["terminal"]["phase"],
        "terminal_radius_m": enabled["terminal"]["radius_m"],
        "terminal_loop_count": enabled["terminal"]["loop_count"],
        "direct_birth_hits": enabled["terminal"]["direct_hits"],
        "event_count": len(enabled["events"]),
        "event_types": sorted(set(event_names)),
        "cavity_dependent_mechanics_binding": enabled["last_mechanics_binding"],
        "post_birth_fracture_intervals": enabled["fracture_adapter_intervals"],
        "mechanical_response_source": "BOUNDED_ALIGNED_HISTORY_CAVITY_RADIUS_MAP",
    }
    passive_control = {
        "ownership": partition["ownership"],
        "reporting_cadences_s": sorted(partition.keys() - {"ownership"}),
        "accumulated_plastic_exposure": {
            key: value["accumulated_plastic_exposure"]
            for key, value in partition.items() if key != "ownership"
        },
        "direct_birth_first_passage_s": {
            key: value["direct_birth_first_passage_s"]
            for key, value in partition.items() if key != "ownership"
        },
    }
    result = {
        "schema": MODEL_ID,
        "donor": receipt,
        "source_contract": {
            "commit": DONOR_COMMIT,
            "tree": DONOR_TREE,
            "bounded_driver": BOUNDED_DRIVER,
            "bounded_driver_sha256": hashlib.sha256(subprocess.run(
                ("git", "-C", str(DONOR_REPOSITORY), "show",
                 f"{DONOR_COMMIT}:{BOUNDED_DRIVER}"), check=True,
                stdout=subprocess.PIPE,
            ).stdout).hexdigest(),
            "execution": "CLEAN_GIT_ARCHIVE",
        },
        "executions": {
            "bounded_coupled": bounded_execution,
            "complete_adapter_parity": adapter_execution,
        },
        "artifact_sha256": {
            "bounded_coupled_result.json": _sha256(destination / "bounded_coupled_result.json"),
            "complete_adapter_parity.json": _sha256(destination / "complete_adapter_parity.json"),
        },
        "common_state_sequence": same_start_nonvoid,
        "active_cavity": active,
        "passive_scheduler_control": passive_control,
        "fracture_adapter": {
            "schema": adapter["schema"],
            "candidate_id": adapter["candidate_id"],
            "actual_renewals": adapter["actual_renewals"],
            "signed_waveform_exercised": adapter["signed_waveform_exercised"],
            "complete_state_parity": adapter["all_complete_states_equal"],
            "history": adapter["history"],
        },
        "qualification": {
            "combined_run_is_executable": True,
            "one_global_event_scheduler": True,
            "second_fracture_clock_in_void_adapter": False,
            "both_birth_clocks_owned": True,
            "remaining_interval_rebuilt_after_event": True,
            "natural_nucleation_claim": False,
            "aligned_bounded_map_only": True,
            "combined_run_fracture_event_count": int(
                enabled["terminal"]["fracture_event_count"]
            ),
            "accepted_fracture_advance_is_separately_exercised_through_same_adapter": bool(
                adapter["actual_renewals"] > 0
            ),
            "global_status": "PARTIAL_RESTORATION",
        },
    }
    result["passed"] = bool(
        active["terminal_radius_m"] is not None
        and active["event_count"] > 0
        and no_void["fracture_adapter_intervals"] > 0
        and enabled["fracture_adapter_intervals"] > 0
        and adapter["actual_renewals"] > 0
        and adapter["all_complete_states_equal"]
    )
    (destination / "coupled_void_r3.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


__all__ = ["MODEL_ID", "execute_coupled_void_r3"]

