"""Unambiguous physical and numerical length identities for v11 branching."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import os


@dataclass(frozen=True)
class BranchScaleIdentity:
    physical_process_zone_length_m: float
    branch_handoff_length_m: float
    local_J_contour_radius_m: float
    interaction_integral_length_m: float
    tip_h_fine_m: float
    actual_local_hbar_m: float
    event_length_da_phys_m: float
    physical_process_zone_source: str
    branch_handoff_source: str
    local_J_contour_source: str
    interaction_integral_source: str
    tip_h_fine_source: str
    event_length_source: str

    def __post_init__(self) -> None:
        for name in (
            "physical_process_zone_length_m", "branch_handoff_length_m",
            "local_J_contour_radius_m", "interaction_integral_length_m",
            "tip_h_fine_m", "actual_local_hbar_m", "event_length_da_phys_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.branch_handoff_source != self.physical_process_zone_source:
            raise ValueError("branch handoff must explicitly source the physical process-zone identity")
        if not math.isclose(
            self.branch_handoff_length_m, self.physical_process_zone_length_m,
            rel_tol=0.0, abs_tol=1.0e-18,
        ):
            raise ValueError("branch handoff length must equal the physical process-zone length")

    def with_local_measurements(self, *, J_contour_radius_m: float, hbar_m: float) -> "BranchScaleIdentity":
        return replace(
            self, local_J_contour_radius_m=float(J_contour_radius_m),
            actual_local_hbar_m=float(hbar_m),
        )

    def to_dict(self) -> dict:
        return {"schema": "v11.branch-scale-identity/1", **asdict(self)}


def resolve_branch_scale_identity(args, mesh) -> BranchScaleIdentity:
    """Resolve physics from the promoted mechanical configuration, never mesh aliases."""
    configuration_path = os.environ.get("MECHANICAL_CONFIG", "").strip()
    if configuration_path:
        from .kernel_configuration_v10227 import load_configuration
        configuration = load_configuration(configuration_path)
        physical = float(configuration.process_zone_length_m)
        interaction = float(configuration.interaction_length_m)
        physical_source = "MECHANICAL_CONFIG.process_zone_length_m"
        interaction_source = "MECHANICAL_CONFIG.interaction_length_m"
    else:
        physical = float(args.mpz_length_um) * 1.0e-6
        interaction = float(getattr(args, "rJ", None) or max(float(args.L_pz), 1.0e-6))
        physical_source = "runtime_args.mpz_length_um_promoted_from_material_registry"
        interaction_source = "runtime_args.rJ_or_legacy_interaction_default"
    requested_contour = float(getattr(args, "rJ", None) or max(float(args.L_pz), 1.0e-6))
    tip_h = float(getattr(args, "tip_h_fine", 0.0) or 1.0e-6)
    hbar = float(getattr(mesh, "hbar_tip", 0.0) or mesh.hbar)
    da = float(args.da_phys)
    return BranchScaleIdentity(
        physical_process_zone_length_m=physical,
        branch_handoff_length_m=physical,
        local_J_contour_radius_m=requested_contour,
        interaction_integral_length_m=interaction,
        tip_h_fine_m=tip_h,
        actual_local_hbar_m=hbar,
        event_length_da_phys_m=da,
        physical_process_zone_source=physical_source,
        branch_handoff_source=physical_source,
        local_J_contour_source="live_FEM_selected_nested_contour;initial=request.contour_radius_m",
        interaction_integral_source=interaction_source,
        tip_h_fine_source="runtime_args.tip_h_fine",
        event_length_source="runtime_args.da_phys",
    )


def selected_local_J_contour_radius(live_result, fallback_m: float) -> float:
    radii = [
        float(row["J_contour_radius_m"])
        for tip in (live_result or {}).get("tips", ())
        for row in tip.get("directional", ())
        if row.get("J_contour_radius_m") is not None
    ]
    return max(radii, default=float(fallback_m))


__all__ = [
    "BranchScaleIdentity", "resolve_branch_scale_identity",
    "selected_local_J_contour_radius",
]
