"""Construction and execution adapter for qualified shared-option paths.

Resolution is intentionally distinct from execution.  A plan names independent
roles; :func:`execute_shared` imports the selected entry, records what was
actually loaded in that process, applies the seed/output controls, and invokes
the native driver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from .fem_backend_adapter import ENTRY_MODULE, require_fem_backend
from .shared_options import (
    CavityMode, CrackPathMode, IntegrationStrategy, KineticsStrategy,
    LoadingMode, MechanicsBackend, OutputProfile, PlasticityMode, SharedOptions,
    SourceLifecycle, SourceLocation,
)

@dataclass(frozen=True)
class RoleBinding:
    role: str
    requested: str
    module: str
    attribute: str
    consumer: str

@dataclass(frozen=True)
class ExecutionPlan:
    schema: str
    options_sha256: str
    bindings: tuple[RoleBinding, ...]
    executor_module: str | None
    executor_attribute: str | None
    restart_family: str
    executable: bool
    limitation: str | None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "bindings": [asdict(item) for item in self.bindings]}

def _binding(role: str, requested: Any, module: str, attribute: str, consumer: str) -> RoleBinding:
    return RoleBinding(role, getattr(requested, "value", str(requested)), module, attribute, consumer)

def build_execution_plan(options: SharedOptions) -> ExecutionPlan:
    """Compose independent roles and state an honest native execution boundary."""
    options.validate()
    if options.mechanics is MechanicsBackend.FULL_FEM_CZM:
        receipt = require_fem_backend()
        bindings = (_binding("mechanics", options.mechanics, ENTRY_MODULE, "main",
                             f"verified external entry {receipt.entry_file}"),)
        exact = (options == __import__("arrhenius_fracture.shared_options", fromlist=["PRESETS"]).PRESETS["fem_czm_theta0"])
        return ExecutionPlan("shared-execution-plan/1", options.fingerprint, bindings,
                             ENTRY_MODULE if exact else None, "main" if exact else None,
                             "external_fem_native", exact,
                             None if exact else "only the exact fem_czm_theta0 preset is mapped")

    bindings = [
        _binding("mechanics", options.mechanics, "arrhenius_fracture.fem", "assemble_mechanics",
                 "full spatial finite-element mechanics assembly"),
    ]
    loading_map = {
        LoadingMode.MONOTONIC: ("arrhenius_fracture.sharp_front", "run_2d", "native displacement-step controller"),
        LoadingMode.EXPLICIT_CYCLIC: ("arrhenius_fracture.fatigue_v1", "FatigueWaveform", "phase-resolved waveform controller"),
        LoadingMode.FIXED_DELTA_K: ("arrhenius_fracture.fixed_deltaK_v1021", "install_fixed_deltaK_waveform", "prescribed local range controller"),
        LoadingMode.HOLD_DWELL: ("arrhenius_fracture.global_physical_scheduler_v1", "advance_event_aware_coupled_interval", "hold/ramp load_history consumer"),
    }
    bindings.append(_binding("loading", options.loading, *loading_map[options.loading]))
    location_map = {
        SourceLocation.TIP_SURFACE: ("arrhenius_fracture.sharp_front", "run_2d", "tip/surface activation support"),
        SourceLocation.BULK_UNIFORM: ("arrhenius_fracture.bulk_plasticity_v104", "accepted_bulk_load_step", "material-measure bulk activation"),
        SourceLocation.TIP_PLUS_BULK: ("arrhenius_fracture.global_physical_scheduler_v1", "advance_event_aware_coupled_interval", "composed tip and bulk scheduler"),
    }
    bindings.append(_binding("source_location", options.source_location, *location_map[options.source_location]))
    lifecycle_map = {
        SourceLifecycle.PERSISTENT_REUSABLE: ("arrhenius_fracture.persistent_site_source_v10221", "install_persistent_site_source", "persistent site ledger installer"),
        SourceLifecycle.FINITE_INVENTORY: ("arrhenius_fracture.sharp_front_v10_2_0", "main", "finite inventory native entry"),
        SourceLifecycle.CURRENT_SOURCE: ("arrhenius_fracture.current_source_runtime_bindings", "MODEL_ID", "current-source state binding"),
    }
    bindings.append(_binding("source_lifecycle", options.source_lifecycle, *lifecycle_map[options.source_lifecycle]))
    if options.plasticity is not PlasticityMode.TIP_PROCESS_ONLY:
        bindings.append(_binding("plasticity", options.plasticity,
                                 "arrhenius_fracture.bulk_plasticity_v104", "accepted_bulk_load_step",
                                 "accepted full-field plastic update"))
    kinetics_map = {
        KineticsStrategy.NATIVE_SIGNED: ("arrhenius_fracture.sharp_front", "run_2d", "native signed state update"),
        KineticsStrategy.REVERSIBLE_SIGNED: ("arrhenius_fracture.persistent_site_reversible_transport_v10230", "install_reversible_transport", "local-drive reversible transport installer"),
        KineticsStrategy.OPERATOR_CONSISTENT_V3_6: ("arrhenius_fracture.operator_consistent_projection_v10230", "OperatorConsistentProjection", "V3.6 projection strategy"),
    }
    bindings.append(_binding("kinetics", options.kinetics, *kinetics_map[options.kinetics]))
    topology_map = {
        CrackPathMode.SINGLE_FRONT: ("arrhenius_fracture.sharp_front", "run_2d", "single-front topology"),
        CrackPathMode.BRANCHING_MULTIFRONT: ("arrhenius_fracture.sharp_front_v11_branching", "main", "multifront ownership graph"),
    }
    bindings.append(_binding("crack_path", options.crack_path, *topology_map[options.crack_path]))
    if options.cavity is CavityMode.FULL_ORDER_V5:
        bindings.append(_binding("cavity", options.cavity, "arrhenius_fracture.voiding_production_v5",
                                 "build_production_void_state", "full-order cavity state; branch permission remains separate"))
    integration_map = {
        IntegrationStrategy.ADAPTIVE_TRANSACTIONAL: ("arrhenius_fracture.sharp_front", "run_2d", "adaptive accepted-state transaction"),
        IntegrationStrategy.EXPLICIT_EVENT: ("arrhenius_fracture.global_physical_scheduler_v1", "advance_event_aware_coupled_interval", "explicit global event scheduler"),
        IntegrationStrategy.COUPLED_FULL_HALF: ("arrhenius_fracture.persistent_site_cyclic_coupled_v10229", "integrate_state_coupled_waveform", "full/half coupled waveform update"),
    }
    bindings.append(_binding("integration", options.integration, *integration_map[options.integration]))
    bindings.append(_binding("output", options.output, "arrhenius_fracture.sharp_front", "run_2d",
                             "native --save-snapshots/--no-plots retention flags"))
    bindings.append(_binding("random_seed", options.random_seed, "arrhenius_fracture.sharp_front_v10_2_17",
                             "_require_stochastic_seed", "CLEAVAGE_HAZARD_SEED before provider import"))

    executor = limitation = None
    restart = "unmapped"
    simple = (options.source_location is SourceLocation.TIP_SURFACE
              and options.plasticity is PlasticityMode.TIP_PROCESS_ONLY
              and options.cavity is CavityMode.DISABLED
              and options.integration is IntegrationStrategy.ADAPTIVE_TRANSACTIONAL)
    if simple and options.crack_path is CrackPathMode.SINGLE_FRONT and options.source_lifecycle is SourceLifecycle.PERSISTENT_REUSABLE:
        if options.loading is LoadingMode.MONOTONIC and options.kinetics in {KineticsStrategy.NATIVE_SIGNED, KineticsStrategy.REVERSIBLE_SIGNED}:
            executor = "arrhenius_fracture.sharp_front_v10_2_27_audited"
            restart = "single_front_outer_checkpoint"
        elif options.loading is LoadingMode.EXPLICIT_CYCLIC and options.kinetics is KineticsStrategy.REVERSIBLE_SIGNED:
            executor = "arrhenius_fracture.sharp_front_v10_2_30_energy_gated_fatigue"
            restart = "combined_outer_and_high_cycle"
        elif options.loading is LoadingMode.FIXED_DELTA_K and options.kinetics is KineticsStrategy.REVERSIBLE_SIGNED:
            executor = "arrhenius_fracture.sharp_front_v10_2_30_fixed_deltaK"
            restart = "combined_outer_and_high_cycle"
    elif (simple and options.loading is LoadingMode.MONOTONIC
          and options.crack_path is CrackPathMode.BRANCHING_MULTIFRONT):
        if options.source_lifecycle is SourceLifecycle.CURRENT_SOURCE:
            executor = "arrhenius_fracture.sharp_front_current_source_multifront_physical_v12"
            restart = "v12_accepted_multifront"
        elif options.source_lifecycle is SourceLifecycle.PERSISTENT_REUSABLE:
            executor = "arrhenius_fracture.sharp_front_v11_branching_audited"
            restart = "v11_native_multifront"
    if executor is None:
        limitation = ("accepted components resolve, but this composed combination has no qualified "
                      "shared native executor; use its preserved legacy launcher until an adapter is completed")
    return ExecutionPlan("shared-execution-plan/1", options.fingerprint, tuple(bindings),
                         executor, "main" if executor else None, restart,
                         executor is not None, limitation)

def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(args, cwd=root, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True)
        return result.stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "tree": run("git", "rev-parse", "HEAD^{tree}"),
            "dirty": bool(run("git", "status", "--porcelain"))}

def construct_and_attest(plan: ExecutionPlan, options: SharedOptions) -> tuple[Any, dict[str, Any]]:
    if not plan.executable or not plan.executor_module:
        raise RuntimeError(plan.limitation or "execution plan is not executable")
    observed = []
    for binding in plan.bindings:
        module = importlib.import_module(binding.module)
        target = getattr(module, binding.attribute)
        path = Path(module.__file__).resolve()
        observed.append({**asdict(binding), "observed_module_file": str(path),
                         "module_sha256": _file_sha256(path),
                         "observed_attribute_type": type(target).__qualname__})
    executor_module = importlib.import_module(plan.executor_module)
    executor = getattr(executor_module, plan.executor_attribute or "main")
    root = Path(__file__).resolve().parents[1]
    inputs = {}
    for label, raw in (("material_manifest", options.material_manifest),
                       ("kernel_manifest", options.kernel_manifest)):
        if raw is not None:
            path = Path(raw).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
            inputs[label] = {"path": str(path), "sha256": _file_sha256(path)}
    receipt = {"schema": "shared-runtime-attestation/1", "status": "constructed",
               "execution_verified": False, "options": options.to_dict(),
               "options_sha256": options.fingerprint, "plan": plan.to_dict(),
               "python": {"executable": sys.executable, "version": sys.version},
               "source": _git_identity(root), "observed_bindings": observed,
               "executor": {"module": plan.executor_module,
                            "module_file": str(Path(executor_module.__file__).resolve()),
                            "module_sha256": _file_sha256(executor_module.__file__),
                            "callable": getattr(executor, "__qualname__", str(executor))},
               "inputs": inputs, "seed": {"hazard": options.random_seed,
                                           "mode": os.environ.get("CLEAVAGE_HAZARD_MODE", "native_default")}}
    return executor, receipt

def execute_shared(options: SharedOptions, native_args: Sequence[str], output_dir: str | Path) -> dict[str, Any]:
    """Execute one qualified native path and write pre/post in-process receipts."""
    if options.material_manifest or options.kernel_manifest:
        raise RuntimeError("shared manifest-to-native-argument mapping is not yet qualified")
    plan = build_execution_plan(options)
    output = Path(output_dir).resolve()
    if any(arg == "--out" or arg.startswith("--out=") for arg in native_args):
        raise ValueError("--out is owned by execute_shared")
    output.mkdir(parents=True, exist_ok=True)
    if options.mechanics is MechanicsBackend.FULL_FEM_CZM:
        backend = require_fem_backend()
        argv = [*native_args, "--out", str(output)]
        wrapper = (
            "import hashlib,json,pathlib,sys; "
            f"import {ENTRY_MODULE} as backend; "
            "out=pathlib.Path(sys.argv[2]); args=json.loads(sys.argv[1]); "
            "p=pathlib.Path(backend.__file__).resolve(); "
            "r={'schema':'shared-runtime-attestation/1','status':'constructed',"
            "'execution_verified':False,'python':{'executable':sys.executable,'version':sys.version},"
            "'executor':{'module':backend.__name__,'module_file':str(p),"
            "'module_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),"
            "'callable':backend.main.__qualname__},'native_argv':args}; "
            "(out/'shared_runtime_attestation.pre.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\\n'); "
            "value=backend.main(args); r.update(status='executed',execution_verified=True,return_value=value); "
            "(out/'shared_runtime_attestation.post.json').write_text(json.dumps(r,indent=2,sort_keys=True,default=str)+'\\n')"
        )
        environment = dict(os.environ)
        environment.update({"CLEAVAGE_HAZARD_SEED": str(options.random_seed),
                            "CLEAVAGE_HAZARD_MODE": "exponential",
                            "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled"})
        result = subprocess.run([backend.interpreter, "-c", wrapper, json.dumps(argv), str(output)],
                                cwd=backend.root, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(f"external FEM/CZM execution failed: {result.stderr.strip()}")
        return json.loads((output / "shared_runtime_attestation.post.json").read_text())
    old_seed = os.environ.get("CLEAVAGE_HAZARD_SEED")
    os.environ["CLEAVAGE_HAZARD_SEED"] = str(options.random_seed)
    try:
        executor, receipt = construct_and_attest(plan, options)
        output_flags = {
            OutputProfile.COMPACT_SMOKE: ["--save-snapshots", "0", "--no-plots"],
            OutputProfile.SELECTED_FIELDS: ["--save-snapshots", "4"],
            OutputProfile.PUBLICATION_RESTART: ["--save-snapshots", "4"],
        }[options.output]
        argv = [*native_args, "--out", str(output), *output_flags]
        (output / "shared_runtime_attestation.pre.json").write_text(
            json.dumps({**receipt, "native_argv": argv}, indent=2, sort_keys=True) + "\n")
        result = executor(argv)
        final = {**receipt, "status": "executed", "execution_verified": True,
                 "native_argv": argv, "return_value": result}
        (output / "shared_runtime_attestation.post.json").write_text(
            json.dumps(final, indent=2, sort_keys=True, default=str) + "\n")
        return final
    finally:
        if old_seed is None:
            os.environ.pop("CLEAVAGE_HAZARD_SEED", None)
        else:
            os.environ["CLEAVAGE_HAZARD_SEED"] = old_seed

__all__ = ["RoleBinding", "ExecutionPlan", "build_execution_plan",
           "construct_and_attest", "execute_shared"]
