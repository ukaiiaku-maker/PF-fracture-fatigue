"""Exact v9.14 endurance-knee A--D overlay on qualified v10.2.30 fatigue."""
from __future__ import annotations
import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_22 as _registry_entry
from . import sharp_front_v10_2_27 as _paper
from . import sharp_front_v10_2_30_energy_gated_fatigue as _production
from .persistent_site_source_v10221 import PersistentSiteConfig

MODEL_ID="v10.2.31_endurance_knee_ABCD_sparse_2D_validation"
VALID={f"v914_endurance_knee_{suffix}":f"v914_endurance_knee_{suffix}" for suffix in ("0462","0658","0554","0133")}

def _number(row,key):
    try: return float(row[key])
    except Exception as exc: raise SystemExit(f"unrepresentable active 1-D field {key!r}") from exc

def _mapped_prepare(args):
    selected,manifest,audit=_ORIGINAL_PREPARE(args)
    row=selected.row
    cfg=PersistentSiteConfig(
        rho_site0_m2=_number(row,"rho_source0_m2"),
        reference_source_area_m2=1e-12*_number(row,"reference_source_area_um2"),
        reference_front_width_m=1e-6*_number(row,"reference_front_width_um"),
        reference_density_m2=_number(row,"rho_forest_floor_m2"),
        source_zone_length_m=1e-6*_number(row,"source_zone_length_um"),
        maximum_front_width_m=1e-6*_number(row,"L_pz_um_recommended"),
        backstress_scale=_number(row,"physics__persistent_backstress_scale"),
    ).validate()
    _registry_entry.PersistentSiteStateResolvedTipEngine.configure_persistent_sites(cfg)
    setters=(("--multihit-m","physics__cleavage_hits"),("--multihit-tau","physics__cleavage_correlation_time_s"),
             ("--mpz-blunting-length-um","physics__blunting_length_m"),("--mpz-blunting-slip-fraction","physics__blunting_slip_fraction"),
             ("--pt-taylor-phi-max","physics__taylor_phi_max"),("--mpz-mobile-transport-velocity-scale","physics__mobile_transport_velocity_scale"))
    for option,key in setters:
        value=_number(row,key); value=1e6*value if key=="physics__blunting_length_m" else value
        _registry_entry._base._set_value_option(args,option,f"{value:.17g}")
    Path(audit).write_text(json.dumps({**json.loads(Path(audit).read_text()),"v10_2_31_mapping":{
        "model_id":MODEL_ID,"persistent_site_config":cfg.__dict__,"active_physics_fields":{k:_number(row,k) for _,k in setters},
        "parameter_refit":False}},indent=2,sort_keys=True)+"\n")
    return selected,manifest,audit

def main(argv=None):
    global _ORIGINAL_PREPARE
    args=list(sys.argv[1:] if argv is None else argv)
    registry=_registry_entry._option_value(args,"--parameter-registry")
    if not registry: raise SystemExit("v10.2.31 requires an explicit transferred --parameter-registry")
    _paper.DEFAULT_REGISTRY=Path(registry).resolve(); _paper.VALID_OPTIONS=dict(VALID)
    _ORIGINAL_PREPARE=_registry_entry._prepare_option
    _registry_entry._prepare_option=_mapped_prepare
    try: return _production.main(args)
    finally: _registry_entry._prepare_option=_ORIGINAL_PREPARE

if __name__=="__main__": main()
