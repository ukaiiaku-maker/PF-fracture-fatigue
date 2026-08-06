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
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    nodes = state.mesh.nodes
    axis.plot(
        [nodes[:, 0].min() * 1e6, nodes[:, 0].max() * 1e6, nodes[:, 0].max() * 1e6, nodes[:, 0].min() * 1e6, nodes[:, 0].min() * 1e6],
        [nodes[:, 1].min() * 1e6, nodes[:, 1].min() * 1e6, nodes[:, 1].max() * 1e6, nodes[:, 1].max() * 1e6, nodes[:, 1].min() * 1e6],
        color="0.75", lw=1.0,
    )
    junctions = []
    for branch in state.crack_network.branches:
        x = [point[0] * 1e6 for point in branch.path]; y = [point[1] * 1e6 for point in branch.path]
        active = branch.status == "active"
        axis.plot(x, y, "-" if active else "--", lw=2.2, marker="o", ms=3,
                  color="tab:red" if active else "0.35")
        axis.annotate(branch.branch_id, (x[-1], y[-1]), fontsize=6)
        if branch.parent_branch_id is not None:
            junctions.append(branch.root)
    if junctions:
        unique = sorted(set(junctions))
        axis.scatter([p[0] * 1e6 for p in unique], [p[1] * 1e6 for p in unique], marker="s", s=34, color="tab:blue", label="junction")
    active_tips = [state.crack_network.branch(item).tip for item in state.crack_network.active_tip_ids]
    if active_tips:
        axis.scatter([p[0] * 1e6 for p in active_tips], [p[1] * 1e6 for p in active_tips], marker="*", s=90, color="gold", edgecolor="black", label="active tip")
    info = dict(mechanics or {})
    annotation = (
        f"reason: {reason}\nextension: {physical_extension_m * 1e6:.3f} µm\n"
        f"branch births: {branch_birth_count}\nactive tips: {len(active_tips)}\n"
        f"latest action: {latest_action or 'none'}\n"
        f"J/K: {info.get('J_K_summary', 'recorded in action diagnostics')}\n"
        f"release/cost/margin: {info.get('energy_summary', 'no topology action')}"
    )
    axis.text(0.01, 0.99, annotation, transform=axis.transAxes, va="top", fontsize=8,
              bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
    axis.set_title("v11 exact production crack topology")
    axis.set_xlabel("x (µm)"); axis.set_ylabel("y (µm)"); axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2); axis.legend(loc="best", fontsize=8)
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
