"""Pinned native v9.13 model loader; fails closed on a different source tree."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

PIN = "ce2bf9d4e868468b6979f0fc8398de95ce983367"
REPOSITORY = "https://github.com/ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ.git"
FIRST_CANDIDATE = "v912_targeted_local_peak_013476_0368"


def load_native(source: Path):
    source = source.resolve()
    head = subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"], text=True).strip()
    if head != PIN:
        raise RuntimeError(f"native source HEAD {head} differs from pinned {PIN}")
    files = ["arrhenius_fracture/zero_d_persistent_v913.py",
             "arrhenius_fracture/emergent_gnd_rcurve_v913.py",
             "arrhenius_fracture/emergent_gnd_state_v913.py",
             "arrhenius_fracture/emergent_gnd_types_v912.py",
             "arrhenius_fracture/emergent_gnd_types_v913.py"]
    receipt = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
    already = sys.modules.get("arrhenius_fracture")
    if already is not None and not Path(already.__file__).resolve().is_relative_to(source):
        raise RuntimeError("another arrhenius_fracture package was imported first")
    sys.path.insert(0,str(source))
    zero = importlib.import_module("arrhenius_fracture.zero_d_persistent_v913")
    spatial = importlib.import_module("arrhenius_fracture.emergent_gnd_rcurve_v913")
    campaign = importlib.import_module("arrhenius_fracture.emergent_gnd_campaign_v913")
    types = importlib.import_module("arrhenius_fracture.emergent_gnd_types_v913")
    for module in (zero,spatial,campaign,types):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"native import escaped selected checkout: {module.__file__}")
    return (zero,spatial,campaign,types),{"repository":REPOSITORY,"commit":head,
                                          "source":str(source),"sha256":receipt}


def historical_start(source: Path, campaign, physics_type):
    registry = source/"candidates/v9_13_persistent_sites_top5_registry.csv"
    with registry.open(newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if r["candidate_id"] == FIRST_CANDIDATE)
    candidate = campaign.candidate_from_registry_row(row)
    raw = json.loads((source/"mpz_v9_13_persistent_sites_common_physics.json").read_text())
    common = raw["common_physics"]
    for key in ("emission_signs","emission_schmid_factors","shielding_orientation_factors",
                "activation_to_line_content_per_system"):
        common[key] = tuple(common[key])
    for key in ("forest_interaction_matrix","gnd_stress_projection_matrix"):
        common[key] = tuple(tuple(row) for row in common[key])
    physics = physics_type(**common)
    physics.validate()
    return candidate,physics,{"candidate_source":str(registry),
                              "candidate_id":FIRST_CANDIDATE,"fixture_metadata":
                              {k:v for k,v in raw.items() if k != "common_physics"}}


def ast_condition_physics(base, specimen):
    physics = replace(base,G_Pa=specimen.shear_modulus_Pa,nu=specimen.nu)
    physics.validate()
    return physics
