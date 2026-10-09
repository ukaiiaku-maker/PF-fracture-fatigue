"""Resolve :mod:`shared_options` into auditable, existing runtime providers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

from .shared_options import (
    CavityMode, CrackPathMode, KineticsStrategy, LoadingMode,
    MechanicsBackend, PlasticityMode, SharedOptions, SourceLifecycle,
)


PF_BASE_COMMIT = "9e884fb0b0845da621d2612bdf1042e481b8df49"
UNIFIED_DONOR_COMMIT = "9e2c43ebfb42d6694c27f01ea9092e79da7e3a06"
V36_DONOR_COMMIT = "0a0e522291882bd13e0d03c7e231fd8a6627aa4f"
FEM_REPOSITORY = "https://github.com/ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ.git"
FEM_COMMIT = "293491157063484bb10df6adc479f847a6a08ba6"
FEM_BRANCH = "v10.0.5.18.4.0-theta0-pf-parity"


@dataclass(frozen=True)
class ProviderBinding:
    role: str
    module: str
    identity: str
    required_attribute: str | None = None

    def verify(self) -> None:
        _verify_provider_isolated(self.module, self.required_attribute, self.role)

    def to_dict(self) -> dict[str, Any]:
        spec = importlib.util.find_spec(self.module)
        path = None if spec is None else spec.origin
        digest = None
        if path and Path(path).is_file():
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return {**asdict(self), "observed_module_file": path,
                "observed_module_sha256": digest}


@lru_cache(maxsize=None)
def _verify_provider_isolated(module: str, required_attribute: str | None,
                              role: str) -> None:
    # Legacy provider modules install versioned adapters at import time.
    # Check each provider in a fresh interpreter so resolving one option
    # cannot select physics for another through import order.
    expression = f"import {module} as provider"
    if required_attribute:
        expression += (
            f"; assert hasattr(provider, {required_attribute!r}), "
            f"{required_attribute!r}"
        )
    result = subprocess.run(
        [sys.executable, "-c", expression], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            f"provider verification failed for {role} ({module}): "
            f"{result.stderr.strip()}"
        )


@dataclass(frozen=True)
class RuntimeReceipt:
    schema: str
    options: dict[str, Any]
    options_sha256: str
    providers: tuple[ProviderBinding, ...]
    external_pair: dict[str, Any] | None
    units: dict[str, str]
    verification_level: str = "isolated_import_checked_resolution"
    execution_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "options": self.options,
            "options_sha256": self.options_sha256,
            "providers": [provider.to_dict() for provider in self.providers],
            "external_pair": self.external_pair,
            "units": self.units,
            "verification_level": self.verification_level,
            "execution_verified": self.execution_verified,
        }

    def verify(self) -> None:
        for provider in self.providers:
            provider.verify()

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _pf_driver(options: SharedOptions) -> ProviderBinding:
    if options.crack_path is CrackPathMode.BRANCHING_MULTIFRONT:
        if options.source_lifecycle is SourceLifecycle.CURRENT_SOURCE:
            return ProviderBinding(
                "driver", "arrhenius_fracture.sharp_front_current_source_multifront_physical_v12",
                "current_source_multifront_physical_v12", "run_v12_2d",
            )
        return ProviderBinding("driver", "arrhenius_fracture.sharp_front_v11_branching_audited",
                               "persistent_multifront_v11", "main")
    if options.loading is LoadingMode.EXPLICIT_CYCLIC:
        return ProviderBinding(
            "driver", "arrhenius_fracture.sharp_front_v10_2_30_energy_gated_fatigue",
            "v10.2.30_hazard_energy_gated_fatigue", "main",
        )
    if options.loading is LoadingMode.FIXED_DELTA_K:
        return ProviderBinding("driver", "arrhenius_fracture.sharp_front_v10_2_30_fixed_deltaK",
                               "v10.2.30_fixed_deltaK", "main")
    return ProviderBinding("driver", "arrhenius_fracture.sharp_front_v10_2_27_audited",
                           "v10.2.27_full_spatial_pf", "main")


def _loading_controller(options: SharedOptions) -> ProviderBinding:
    return {
        LoadingMode.MONOTONIC: ProviderBinding(
            "loading_controller", "arrhenius_fracture.sharp_front",
            "native_displacement_step", "run_2d"),
        LoadingMode.EXPLICIT_CYCLIC: ProviderBinding(
            "loading_controller", "arrhenius_fracture.fatigue_v1",
            "phase_resolved_explicit_cyclic", "FatigueWaveform"),
        LoadingMode.FIXED_DELTA_K: ProviderBinding(
            "loading_controller", "arrhenius_fracture.fixed_deltaK_v1021",
            "prescribed_local_deltaK", "install_fixed_deltaK_waveform"),
        LoadingMode.HOLD_DWELL: ProviderBinding(
            "loading_controller", "arrhenius_fracture.global_physical_scheduler_v1",
            "hold_ramp_load_history", "advance_event_aware_coupled_interval"),
    }[options.loading]


def resolve_runtime(options: SharedOptions) -> RuntimeReceipt:
    """Return and import-check the exact providers selected by *options*."""
    options.validate()
    if options.mechanics is MechanicsBackend.FULL_FEM_CZM:
        from .fem_backend_adapter import inspect_fem_backend
        external = inspect_fem_backend().to_dict()
        external.update({
            "commit": FEM_COMMIT,
            "branch": FEM_BRANCH,
            "configuration": "theta0 full anisotropic adaptive-CZM parity",
            "relationship": "comparison backend; not source-merged into PF",
        })
        receipt = RuntimeReceipt(
            schema="shared-fracture-fatigue-runtime-receipt/1",
            options=options.to_dict(), options_sha256=options.fingerprint,
            providers=(),
            external_pair=external,
            units=_units(),
        )
        return receipt

    providers = [_pf_driver(options), _loading_controller(options)]
    if (options.cavity is CavityMode.FULL_ORDER_V5
            or options.plasticity is not PlasticityMode.TIP_PROCESS_ONLY
            or options.loading is LoadingMode.HOLD_DWELL):
        providers.append(ProviderBinding(
            "event_scheduler", "arrhenius_fracture.global_physical_scheduler_v1",
            "unified_2d_event_aware_coupled_interval_v1",
            "advance_event_aware_coupled_interval"))
    lifecycle = {
        SourceLifecycle.PERSISTENT_REUSABLE: ProviderBinding(
            "source_lifecycle", "arrhenius_fracture.persistent_site_source_v10221",
            "v10.2.21_persistent_site_backstress_blunting", "install_persistent_site_source"),
        SourceLifecycle.FINITE_INVENTORY: ProviderBinding(
            "source_lifecycle", "arrhenius_fracture.sharp_front_v10_2_0",
            "legacy_finite_inventory_refresh", "main"),
        SourceLifecycle.CURRENT_SOURCE: ProviderBinding(
            "source_lifecycle", "arrhenius_fracture.current_source_runtime_bindings",
            "current_source_v12_runtime", None),
    }[options.source_lifecycle]
    providers.append(lifecycle)
    if options.plasticity is not PlasticityMode.TIP_PROCESS_ONLY:
        providers.append(ProviderBinding(
            "bulk_plasticity", "arrhenius_fracture.bulk_plasticity_v104",
            options.plasticity.value, "accepted_bulk_load_step"))
    if options.cavity is CavityMode.FULL_ORDER_V5:
        providers.append(ProviderBinding(
            "cavity", "arrhenius_fracture.voiding_production_v5",
            "full_order_v5", "build_production_void_state"))
    if options.kinetics is KineticsStrategy.REVERSIBLE_SIGNED:
        providers.append(ProviderBinding(
            "kinetics", "arrhenius_fracture.persistent_site_reversible_transport_v10230",
            "v10.2.30_intrinsic_signed_mobile_transport_v1",
            "install_reversible_transport"))
    elif options.kinetics is KineticsStrategy.OPERATOR_CONSISTENT_V3_6:
        providers.extend((
            ProviderBinding("mechanics_projection",
                            "arrhenius_fracture.operator_consistent_projection_v10230",
                            "v3.6_operator_consistent", "OperatorConsistentProjection"),
            ProviderBinding("mechanism_feedback",
                            "arrhenius_fracture.mechanism_specific_feedback_v10230",
                            "v3.6_mechanism_specific", "MechanismSpecificFeedback"),
        ))
    receipt = RuntimeReceipt(
        schema="shared-fracture-fatigue-runtime-receipt/1",
        options=options.to_dict(), options_sha256=options.fingerprint,
        providers=tuple(providers), external_pair=None, units=_units(),
    )
    receipt.verify()
    return receipt


def _units() -> dict[str, str]:
    return {
        "length": "m", "time": "s", "temperature": "K", "stress": "Pa",
        "K": "Pa sqrt(m) internally; MPa sqrt(m) only where named",
        "J_G": "J/m^2", "density": "m^-2",
    }


def require_restart_compatible(saved_receipt: dict[str, Any], options: SharedOptions) -> None:
    """Reject restart under a changed option or provider identity."""
    current = resolve_runtime(options).to_dict()
    if saved_receipt.get("schema") != current["schema"]:
        raise RuntimeError("restart runtime-receipt schema mismatch")
    if saved_receipt.get("options_sha256") != current["options_sha256"]:
        raise RuntimeError("restart option fingerprint mismatch")
    saved_providers = [
        (item.get("role"), item.get("module"), item.get("identity"),
         item.get("observed_module_sha256"))
        for item in saved_receipt.get("providers", [])
    ]
    current_providers = [
        (item["role"], item["module"], item["identity"],
         item.get("observed_module_sha256"))
        for item in current["providers"]
    ]
    if saved_providers != current_providers:
        raise RuntimeError("restart provider identity mismatch")
    if saved_receipt.get("external_pair") != current["external_pair"]:
        raise RuntimeError("restart external mechanics-pair mismatch")


__all__ = [
    "PF_BASE_COMMIT", "UNIFIED_DONOR_COMMIT", "V36_DONOR_COMMIT",
    "FEM_REPOSITORY", "FEM_COMMIT", "FEM_BRANCH", "ProviderBinding",
    "RuntimeReceipt", "resolve_runtime",
    "require_restart_compatible",
]
