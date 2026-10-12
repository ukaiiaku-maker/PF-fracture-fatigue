"""V2 readiness wrapper; leaves accepted SharedOptions/1 identities untouched."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .mechanics_state_contract import MechanicsStateOptions
from .shared_options import SharedOptions

SCHEMA = "common-fracture-readiness-options/2"


@dataclass(frozen=True)
class ReadinessOptions:
    base: SharedOptions
    mechanics_state: MechanicsStateOptions
    production_execution_authorized: bool = False
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported readiness schema: {self.schema}")
        if self.production_execution_authorized:
            raise ValueError("readiness-stage options cannot authorize production")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "base": self.base.to_dict(),
                "mechanics_state": self.mechanics_state.to_dict(),
                "production_execution_authorized": False}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReadinessOptions":
        unknown = set(payload) - {"schema", "base", "mechanics_state",
                                  "production_execution_authorized"}
        if unknown:
            raise ValueError(f"unknown readiness option fields: {sorted(unknown)}")
        return cls(base=SharedOptions.from_dict(payload["base"]),
                   mechanics_state=MechanicsStateOptions.from_dict(payload["mechanics_state"]),
                   production_execution_authorized=payload.get("production_execution_authorized", False),
                   schema=payload.get("schema", SCHEMA))

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


__all__ = ["ReadinessOptions", "SCHEMA"]
