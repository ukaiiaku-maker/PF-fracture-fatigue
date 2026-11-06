"""Fail-closed nominal elastic field-shape input for the Ast 1-D evaluator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BendingOpeningShape:
    extension_m: tuple[float, ...]
    radius_m: tuple[float, ...]
    opening_shape: tuple[tuple[float, ...], ...]
    height_m: float
    thickness_m: float
    span_m: float

    @classmethod
    def from_file(cls, path: str | Path):
        data=json.loads(Path(path).read_text())
        if data.get("plane") != "strain" or data.get("status") != "nominal diagnostic":
            raise ValueError("unsupported elastic field provenance")
        beam=data["beam"]
        if data.get("mesh_spacing_m",0)<=0 or "intercept" not in data.get("normalization",""):
            raise ValueError("elastic field lacks mesh/normalization provenance")
        result=cls(tuple(map(float,data["extension_m"])),
                   tuple(map(float,data["radius_m"])),
                   tuple(tuple(map(float,row)) for row in data["opening_shape"]),
                   float(beam["height_m"]),float(beam["thickness_m"]),
                   float(beam["load_x_m"])-float(beam["notch_x_m"]))
        result.validate()
        return result

    def validate(self):
        a=np.asarray(self.extension_m);r=np.asarray(self.radius_m)
        q=np.asarray(self.opening_shape)
        if (len(a)<2 or len(r)<2 or q.shape!=(len(a),len(r)) or
            not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)) or
            not np.all(np.isfinite(q)) or a[0]!=0 or r[0]!=0 or
            np.any(np.diff(a)<=0) or np.any(np.diff(r)<=0) or
            not np.allclose(q[:,0],1,atol=1e-12)):
            raise ValueError("invalid elastic opening shape table")
        if np.any(q[:,1]<=0):
            raise ValueError("nonpositive near-tip opening shape")
        if min(self.height_m,self.thickness_m,self.span_m)<=0:
            raise ValueError("invalid source beam geometry")

    def factor(self, extension_m: float, radius_m: float) -> float:
        a=float(extension_m);r=float(radius_m)
        aa=np.asarray(self.extension_m);rr=np.asarray(self.radius_m)
        if not (np.isfinite(a) and np.isfinite(r) and
                aa[0]-1e-15<=a<=aa[-1]+1e-15 and rr[0]<=r<=rr[-1]):
            raise ValueError("elastic field requested outside computed crack/radius support")
        q=np.asarray(self.opening_shape)
        along=np.array([np.interp(r,rr,row) for row in q])
        return float(np.interp(a,aa,along))
