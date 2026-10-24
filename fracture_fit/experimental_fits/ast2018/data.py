"""Lossless loading and scenario selection for the supplied Ast extraction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


SCENARIOS = ("TABLE_PRIMARY", "FIGURE5_ALTERNATIVE")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True)
class AstData:
    root: Path
    conditions: tuple[dict[str, str], ...]
    toughness: tuple[dict[str, str], ...]
    J_curves: tuple[dict[str, str], ...]
    K_displacement: tuple[dict[str, str], ...]
    stiffness: tuple[dict[str, str], ...]
    no_cleavage: tuple[dict[str, str], ...]
    discrepancies: tuple[dict[str, str], ...]

    @classmethod
    def load(cls, root: Path) -> "AstData":
        data = root / "data"
        obj = cls(root,
            tuple(rows(data/"table1_conditions.csv")),
            tuple(rows(data/"table1_toughness_long.csv")),
            tuple(rows(data/"J_R_curves.csv")),
            tuple(rows(data/"K_displacement_curves.csv")),
            tuple(rows(data/"relative_stiffness.csv")),
            tuple(rows(data/"no_cleavage_endpoints.csv")),
            tuple(rows(data/"source_discrepancies.csv")))
        obj.validate()
        return obj

    def validate(self) -> None:
        if tuple(map(len,(self.conditions,self.toughness,self.J_curves,
                          self.K_displacement,self.stiffness,self.no_cleavage,
                          self.discrepancies))) != (31,42,629,1796,356,8,2):
            raise ValueError("supplied Ast extraction coverage differs from accepted contract")
        ids = {r["condition_id"] for r in self.conditions}
        if len(ids) != 31 or any(r["condition_id"] not in ids for r in self.toughness):
            raise ValueError("invalid condition IDs")
        if {r["table1_field"] for r in self.discrepancies} != {"Kq_Jcontinuous_onset_MPa_sqrt_m"}:
            raise ValueError("unexpected conflicting observable")

    def toughness_scenario(self, scenario: str) -> tuple[dict[str, str], ...]:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown Ast source scenario: {scenario}")
        selected = [dict(r) for r in self.toughness]
        if scenario == "FIGURE5_ALTERNATIVE":
            replacement = {r["condition_id"]: r["K_digitized_MPa_sqrt_m"]
                           for r in self.discrepancies}
            matched = set()
            for r in selected:
                if r["observable"] == "Kq_Jcontinuous_onset" and r["condition_id"] in replacement:
                    r["value"] = replacement[r["condition_id"]]
                    r["scenario_source"] = "Figure5b digitized center"
                    matched.add(r["condition_id"])
            if matched != set(replacement):
                raise ValueError("Figure 5 alternatives failed to match table onset rows")
        return tuple(selected)

    def condition(self, condition_id: str) -> dict[str, str]:
        return next(r for r in self.conditions if r["condition_id"] == condition_id)
