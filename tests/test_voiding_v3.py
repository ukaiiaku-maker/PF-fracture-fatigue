from copy import deepcopy
from arrhenius_fracture.voiding_v3 import *

def test_default_off_instantiates_nothing_and_does_not_touch_rng():
 config=VoidingV3Config(); rng={"state":[3,1,4]}; before=deepcopy(rng)
 assert initialize_voiding(config,rng_state=rng) is None
 assert rng==before

def test_disabled_metadata_is_separate_from_physical_artifact():
 physical={"events":[1,2],"graph":"abc","energy":3.0}; before=deepcopy(physical)
 result,capability=attach_disabled_voiding_manifest(physical)
 assert result==before==physical and capability["voiding_enabled"] is False

def test_contract_has_all_required_typed_states():
 assert {item.value for item in VoidingState}=={"AVAILABLE_SITE","EMBRYO","HEALED_SITE","CONSUMED_SITE","STABLE_SUBGRID_VOID","RESOLVED_VOID","CONNECTED_VOID","DOWNSTREAM_FRONT_ACTIVE","MERGED_OR_CONSUMED"}
