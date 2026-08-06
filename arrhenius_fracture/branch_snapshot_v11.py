"""Automatic visual evidence for accepted v11 production topology transitions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_topology_snapshot(
    root: str | Path, state, *, step: int, reason: str,
    physical_extension_m: float, branch_birth_count: int,
    latest_action: str | None, mechanics: Mapping[str, Any] | None = None,
    final: bool = False,
) -> tuple[Path, Path]:
    out = Path(root); directory = out / "snapshots"; directory.mkdir(parents=True, exist_ok=True)
    stem = "v11_final_topology_and_energy" if final else f"topology_step{int(step):07d}_{reason}"
    image = out / f"{stem}.png" if final else directory / f"{stem}.png"
    metadata = out / "v11_visual_snapshot.json" if final else directory / f"{stem}.json"
    figure, (overview, axis) = plt.subplots(1, 2, figsize=(14, 5.8))
    nodes = state.mesh.nodes
    specimen_x = [nodes[:, 0].min() * 1e6, nodes[:, 0].max() * 1e6, nodes[:, 0].max() * 1e6,
                  nodes[:, 0].min() * 1e6, nodes[:, 0].min() * 1e6]
    specimen_y = [nodes[:, 1].min() * 1e6, nodes[:, 1].min() * 1e6, nodes[:, 1].max() * 1e6,
                  nodes[:, 1].max() * 1e6, nodes[:, 1].min() * 1e6]
    for plot_axis in (overview, axis):
        plot_axis.plot(specimen_x, specimen_y, color="0.75", lw=1.0)
    junctions = []
    crack_x: list[float] = []
    crack_y: list[float] = []
    branch_key = []
    for branch_index, branch in enumerate(state.crack_network.branches):
        x = [point[0] * 1e6 for point in branch.path]; y = [point[1] * 1e6 for point in branch.path]
        crack_x.extend(x); crack_y.extend(y)
        active = branch.status == "active"
        for plot_axis in (overview, axis):
            plot_axis.plot(x, y, "-" if active else "--", lw=2.2, marker="o", ms=3,
                           color="tab:red" if active else "0.35")
        branch_key.append(f"B{branch_index}: {branch.branch_id} [{branch.status}]")
        midpoint = len(x) // 2
        axis.annotate(
            f"B{branch_index}", (x[midpoint], y[midpoint]), fontsize=7,
            xytext=(3, 4 + 6 * (branch_index % 3)), textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        if branch.parent_branch_id is not None:
            junctions.append(branch.root)
    if junctions:
        unique = sorted(set(junctions))
        for plot_axis in (overview, axis):
            plot_axis.scatter([p[0] * 1e6 for p in unique], [p[1] * 1e6 for p in unique],
                              marker="s", s=34, color="tab:blue", label="junction")
    active_tips = [state.crack_network.branch(item).tip for item in state.crack_network.active_tip_ids]
    if active_tips:
        for plot_axis in (overview, axis):
            plot_axis.scatter([p[0] * 1e6 for p in active_tips], [p[1] * 1e6 for p in active_tips],
                              marker="*", s=90, color="gold", edgecolor="black", label="active tip")
    info = dict(mechanics or {})
    annotation = (
        f"reason: {reason}\nextension: {physical_extension_m * 1e6:.3f} µm\n"
        f"branch births: {branch_birth_count}\nactive tips: {len(active_tips)}\n"
        f"latest action: {latest_action or 'none'}\n"
        f"J/K: {info.get('J_K_summary', 'recorded in action diagnostics')}\n"
        f"release/cost/margin: {info.get('energy_summary', 'no topology action')}"
    )
    overview.text(0.01, 0.99, annotation, transform=overview.transAxes, va="top", fontsize=8,
              bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
    overview.set_title("Specimen overview")
    axis.set_title("Crack-network detail")
    detail_x = [p[0] * 1e6 for p in junctions] or crack_x
    detail_y = [p[1] * 1e6 for p in junctions] or crack_y
    if detail_x and detail_y:
        detail_x = detail_x + [p[0] * 1e6 for p in active_tips]
        detail_y = detail_y + [p[1] * 1e6 for p in active_tips]
        x_span = max(detail_x) - min(detail_x)
        y_span = max(detail_y) - min(detail_y)
        x_margin = max(10.0, 0.12 * max(x_span, 1.0))
        y_margin = max(10.0, 0.18 * max(y_span, 1.0))
        axis.set_xlim(min(detail_x) - x_margin, max(detail_x) + x_margin)
        axis.set_ylim(min(detail_y) - y_margin, max(detail_y) + y_margin)
    for plot_axis in (overview, axis):
        plot_axis.set_xlabel("x (µm)"); plot_axis.set_ylabel("y (µm)")
        plot_axis.set_aspect("equal", adjustable="box"); plot_axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize=8)
    figure.suptitle("v11 exact production topology and energy state", fontsize=12)
    figure.text(0.515, 0.02, "Branch key: " + "   ".join(branch_key), ha="center", va="bottom", fontsize=6, wrap=True)
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    figure.savefig(image, dpi=180); plt.close(figure)
    payload = {
        "schema": "v11.production-topology-snapshot/1", "image": str(image.relative_to(out)),
        "step": int(step), "reason": reason, "physical_extension_m": float(physical_extension_m),
        "branch_birth_count": int(branch_birth_count), "active_tip_ids": list(state.crack_network.active_tip_ids),
        "branch_count": len(state.crack_network.branches), "latest_action": latest_action,
        "mechanics": info,
    }
    temporary = metadata.with_name(metadata.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, metadata)
    return image, metadata


__all__ = ["write_topology_snapshot"]
