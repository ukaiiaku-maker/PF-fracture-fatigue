"""Default-off, backend-neutral V12 one-void state contract."""
from __future__ import annotations
from dataclasses import dataclass,field
from enum import Enum
from typing import Any,Mapping

SCHEMA="v12.stateful-voiding/1"
class VoidingState(str,Enum):
 AVAILABLE_SITE="AVAILABLE_SITE"; EMBRYO="EMBRYO"; HEALED_SITE="HEALED_SITE"
 CONSUMED_SITE="CONSUMED_SITE"; STABLE_SUBGRID_VOID="STABLE_SUBGRID_VOID"
 RESOLVED_VOID="RESOLVED_VOID"; CONNECTED_VOID="CONNECTED_VOID"
 DOWNSTREAM_FRONT_ACTIVE="DOWNSTREAM_FRONT_ACTIVE"; MERGED_OR_CONSUMED="MERGED_OR_CONSUMED"

@dataclass(frozen=True)
class VoidingV3Config:
 voiding_enabled: bool=False
 schema: str=SCHEMA

@dataclass(frozen=True)
class VoidingV3State:
 sites: tuple[Any,...]=()
 cavities: tuple[Any,...]=()
 rng_state: Mapping[str,Any]=field(default_factory=dict)
 thresholds: Mapping[str,float]=field(default_factory=dict)
 growth_state: Mapping[str,Any]=field(default_factory=dict)
 geometry_lineage: Mapping[str,Any]=field(default_factory=dict)
 length_ledgers: Mapping[str,float]=field(default_factory=dict)
 schema: str=SCHEMA

def initialize_voiding(config: VoidingV3Config, *, rng_state=None):
 """Disabled capability creates and advances no physical or stochastic state."""
 if not config.voiding_enabled: return None
 return VoidingV3State(rng_state={} if rng_state is None else rng_state)

def attach_disabled_voiding_manifest(physical_manifest: Mapping[str,Any]):
 """Capability metadata is separate and leaves the physical manifest untouched."""
 return dict(physical_manifest),{"voiding_schema":SCHEMA,"voiding_enabled":False}

__all__=["SCHEMA","VoidingState","VoidingV3Config","VoidingV3State","attach_disabled_voiding_manifest","initialize_voiding"]
