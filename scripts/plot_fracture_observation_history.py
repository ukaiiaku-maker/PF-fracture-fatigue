#!/usr/bin/env python3
"""Plot a chronological K history while retaining repeated crack lengths."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from arrhenius_fracture.fracture_observation_history import (
    ObservationHistory, from_legacy_steps_csv,
)


def plot_history(history: ObservationHistory, output: Path, *, front_id: str | None = None,
                 segment_id: str | None = None) -> None:
    rows = history.filtered(front_id=front_id, segment_id=segment_id)
    if not rows:
        raise ValueError("no observations selected")
    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    groups: dict[tuple[str, str, str], list] = {}
    for row in rows:
        groups.setdefault((row.segment_id, row.stage.value, row.validity_status.value), []).append(row)
    for (segment, stage, validity), selected in groups.items():
        axis.plot(
            [row.stage_length_m * 1e6 for row in selected],
            [(row.K_J_Pa_sqrt_m if math.isfinite(row.K_J_Pa_sqrt_m)
              else row.K_probe_Pa_sqrt_m if math.isfinite(row.K_probe_Pa_sqrt_m)
              else row.K_I_Pa_sqrt_m) * 1e-6 for row in selected],
            marker="o", markersize=2.5, linewidth=1.0,
            label=f"{segment}: {stage}; {validity}",
        )
    axis.set_xlabel("Crack length or extension (µm)")
    axis.set_ylabel(r"Native $K$ observer (MPa $\sqrt{\mathrm{m}}$)")
    if len(groups) > 1:
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-steps", action="store_true")
    parser.add_argument("--front-id")
    parser.add_argument("--segment-id")
    args = parser.parse_args(argv)
    history = (from_legacy_steps_csv(args.history)
               if args.legacy_steps else ObservationHistory.load(args.history))
    plot_history(history, args.output, front_id=args.front_id, segment_id=args.segment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
