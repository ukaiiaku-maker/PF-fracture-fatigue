"""Unchanged selected material with full-section plasticity on linked specimens.

Diagnostic replay, not a fit: the saved multi-temperature material, geometry,
event convention and loading are reused; only the plasticity mode changes and
the imposed-displacement horizon returns to the AstAssumptions default
(45 MPa sqrt(m) elastic equivalent) so that yielding specimens are not cut off.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

from .config import AstAssumptions
from .mechanics import observe_like_ast
from .multi_temperature_bending_fit import (ROOT as FIT_ROOT, SHAPE, base_material,
                                            score, target)
from .refine_multi_temperature_bending import shifted_material
from .section_plasticity import SectionPlasticityConfig
from .single_temp_bending_fit import _jsonable
from .spatial import predict_spatial

OUTPUT = Path("runs/ast2018_section_plasticity_v1")
LINKED = (("F3b_T21_r0.2", "Tp21C_rate0p2"),
          ("F3b_T100_r0.2", "Tp100C_rate0p2"),
          ("F3b_T250_r0.2", "Tp250C_rate0p2"))


def selected_material(primary_base):
    selected = json.loads((FIT_ROOT / "selected_material.json").read_text())
    material = shifted_material(primary_base, list(selected["coordinates"].values()),
                                selected["delta_floor_eV"], selected["delta_emission_eV"])
    if json.loads(json.dumps(asdict(material))) != selected["material"]:
        raise ArithmeticError("selected material reconstruction differs")
    return material


def run_case(curve_id, condition_id, mode, output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{mode}_{curve_id}.json"
    if path.is_file():
        return json.loads(path.read_text())["record"]
    primary, _, base = base_material()
    material = selected_material(base)
    assumptions = AstAssumptions(target_extension_m=.51e-6,
        nominal_K_step_MPa_sqrt_m=.1, maximum_nominal_K_MPa_sqrt_m=45.,
        loading_mode="nominal_bending_shared_temperature",
        bending_opening_shape_path=str(SHAPE))
    # v1 run: bulk nucleation from zero carriers (superseded default).
    section = SectionPlasticityConfig(mode=mode, bulk_nucleation=True,
                                      initial_mobile="zero")
    condition = primary.ev.data.condition(condition_id)
    specimen = assumptions.specimen(condition)
    physics = assumptions.physics(primary.ev.base_physics, specimen)
    T = float(condition["temperature_K"])
    rate = float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"])
    start = time.monotonic()
    pred = predict_spatial(primary.ev.spatial, material, physics, specimen, T, rate,
                           assumptions, section=section)
    observed = observe_like_ast(pred.history, specimen)
    x, y, w = target(primary.ev.data, curve_id)
    record = {"curve_id": curve_id, "condition_id": condition_id, "mode": mode,
              "status": pred.status, "numerical_failure": pred.numerical_failure,
              "events": len(pred.events), "load_steps": pred.load_steps,
              "final_extension_um": (pred.history[-1].a_m - specimen.a0_m) * 1e6,
              "final_u_um": pred.history[-1].imposed_u_m * 1e6,
              "wall_seconds": time.monotonic() - start, **score(observed, x, y, w)}
    payload = {"record": record, "assumptions": assumptions.as_dict(),
               "section": section.as_dict(),
               "material_sha256": hashlib.sha256(
                   (FIT_ROOT / "selected_material.json").read_bytes()).hexdigest(),
               "native_commit": primary.ev.receipt["commit"],
               "observations": [asdict(o) for o in observed],
               "events": pred.events,
               "final_section": (pred.final_state or {}).get("section")}
    path.write_text(json.dumps(payload, indent=1, default=_jsonable) + "\n")
    return record


if __name__ == "__main__":
    curve, condition, mode = sys.argv[1:4]
    print(json.dumps(run_case(curve, condition, mode), default=_jsonable))
