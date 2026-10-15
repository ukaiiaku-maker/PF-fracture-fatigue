#!/usr/bin/env python3
"""Plot a chronological K history while retaining repeated crack lengths."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from arrhenius_fracture.fracture_observation_history import (
    LengthCoordinate, ObservationHistory, ObservationStage, ValidityStatus,
    from_legacy_steps_csv,
)

K_QUANTITIES = {
    "K_J": ("K_J_Pa_sqrt_m", r"Native $K_J$ (MPa $\sqrt{\mathrm{m}}$)"),
    "K_I": ("K_I_Pa_sqrt_m", r"Native $K_I$ (MPa $\sqrt{\mathrm{m}}$)"),
    "K_II": ("K_II_Pa_sqrt_m", r"Native $K_{II}$ (MPa $\sqrt{\mathrm{m}}$)"),
    "K_effective": ("K_effective_Pa_sqrt_m", r"Effective $K$ (MPa $\sqrt{\mathrm{m}}$)"),
    "K_probe": ("K_probe_Pa_sqrt_m", r"Probe $K$ (MPa $\sqrt{\mathrm{m}}$)"),
    "K_applied": ("K_applied_Pa_sqrt_m", r"Applied $K$ (MPa $\sqrt{\mathrm{m}}$)"),
}


def plot_history(history: ObservationHistory, output: Path, *, front_id: str | None = None,
                 segment_id: str | None = None,
                 coordinate: LengthCoordinate | str = LengthCoordinate.EXTENSION,
                 quantity: str,
                 stage: ObservationStage | str | None = None,
                 validity: ValidityStatus | str | None = None) -> tuple:
    coordinate = LengthCoordinate(coordinate)
    if quantity not in K_QUANTITIES:
        raise ValueError(f"unknown K quantity: {quantity}")
    field, ylabel = K_QUANTITIES[quantity]
    rows = tuple(
        row for row in history.filtered(
            front_id=front_id, segment_id=segment_id,
            stage=stage, validity=validity,
        ) if math.isfinite(getattr(row, field))
    )
    if not rows:
        raise ValueError(f"no finite {quantity} observations selected")
    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    groups: dict[tuple[str, str, str], list] = {}
    for row in rows:
        groups.setdefault((row.segment_id, row.stage.value, row.validity_status.value), []).append(row)
    for (segment, stage, validity), selected in groups.items():
        axis.plot(
            [row.length_m(coordinate) * 1e6 for row in selected],
            [getattr(row, field) * 1e-6 for row in selected],
            marker="o", markersize=2.5, linewidth=1.0,
            label=f"{segment}: {stage}; {validity}",
        )
    axis.set_xlabel({
        LengthCoordinate.EXTENSION: "Projected crack extension (µm)",
        LengthCoordinate.ABSOLUTE_PROJECTED: "Absolute projected crack length (µm)",
        LengthCoordinate.PATH_ARCLENGTH: "Crack-path arclength (µm)",
    }[coordinate])
    axis.set_ylabel(ylabel)
    if len(groups) > 1:
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-steps", action="store_true")
    parser.add_argument("--front-id")
    parser.add_argument("--segment-id")
    parser.add_argument("--coordinate", choices=[item.value for item in LengthCoordinate],
                        default=LengthCoordinate.EXTENSION.value)
    parser.add_argument("--quantity", choices=sorted(K_QUANTITIES), required=True)
    parser.add_argument("--stage", choices=[item.value for item in ObservationStage])
    parser.add_argument("--validity", choices=[item.value for item in ValidityStatus])
    args = parser.parse_args(argv)
    history = (from_legacy_steps_csv(args.history)
               if args.legacy_steps else ObservationHistory.load(args.history))
    plot_history(history, args.output, front_id=args.front_id, segment_id=args.segment_id,
                 coordinate=args.coordinate, quantity=args.quantity,
                 stage=args.stage, validity=args.validity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
