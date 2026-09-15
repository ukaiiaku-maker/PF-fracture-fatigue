"""Sparse portable exports of atomic accepted 2-D checkpoint generations."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .run_state_checkpoint_v10230 import load_combined_checkpoint

SCHEMA = "v10.2.30_sparse_accepted_fields_v1"
MILESTONES_M = (250e-6, 500e-6, 750e-6, 1000e-6)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        tmp = Path(f.name)
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _stress_derived(sigma: np.ndarray) -> dict[str, np.ndarray]:
    xx, yy, xy = sigma
    radius = np.sqrt((0.5 * (xx - yy)) ** 2 + xy ** 2)
    mean = 0.5 * (xx + yy)
    return {
        "sigma_xx_gp": xx, "sigma_yy_gp": yy, "sigma_xy_gp": xy,
        "sigma_max_principal_gp": mean + radius,
        "sigma_min_principal_gp": mean - radius,
        "hydrostatic_plane_gp": (xx + yy) / 3.0,
        "von_mises_plane_gp": np.sqrt(np.maximum(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy, 0.0)),
    }


def _portable(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {k: np.asarray(v) for k, v in arrays.items() if k != "kinetic_active_vector"}
    u = result["displacement"]
    result["u_x"] = u[0::2]
    result["u_y"] = u[1::2]
    result["u_magnitude"] = np.hypot(result["u_x"], result["u_y"])
    ep = result["ep_gp"]
    result["equivalent_plastic_strain_gp"] = np.sqrt(
        2.0 / 3.0 * (ep[0] ** 2 + ep[1] ** 2 + 2.0 * ep[2] ** 2)
    )
    if "sigma_gp" in result:
        result.update(_stress_derived(result["sigma_gp"]))
    return result


def _write_vtu(path: Path, arrays: dict[str, np.ndarray]) -> None:
    nodes = arrays["mesh_nodes"]
    elems = arrays["mesh_elems"].astype(int)
    point = {"damage": arrays["damage"], "u_x": arrays["u_x"],
             "u_y": arrays["u_y"], "u_magnitude": arrays["u_magnitude"]}
    cell = {k: v for k, v in arrays.items() if np.asarray(v).shape == (len(elems),)}
    def data(name, values, components=1, dtype="Float64"):
        flat = np.asarray(values).reshape(-1)
        body = " ".join(f"{float(x):.17g}" for x in flat)
        return f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="ascii">{body}</DataArray>'
    connectivity = " ".join(str(int(x)) for x in elems.reshape(-1))
    offsets = " ".join(str(3 * (i + 1)) for i in range(len(elems)))
    types = " ".join("5" for _ in elems)
    xyz = np.column_stack((nodes, np.zeros(len(nodes))))
    text = ['<?xml version="1.0"?>', '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
            '<UnstructuredGrid>', f'<Piece NumberOfPoints="{len(nodes)}" NumberOfCells="{len(elems)}">',
            '<PointData>', *(data(k, v) for k, v in point.items()), '</PointData>', '<CellData>',
            *(data(k, v) for k, v in sorted(cell.items())), '</CellData>', '<Points>',
            data("coordinates", xyz, 3), '</Points>', '<Cells>',
            f'<DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>',
            f'<DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>',
            f'<DataArray type="UInt8" Name="types" format="ascii">{types}</DataArray>',
            '</Cells>', '</Piece>', '</UnstructuredGrid>', '</VTKFile>']
    path.write_text("\n".join(text) + "\n")


def _write_path(root: Path, stem: str, outer: dict) -> list[Path]:
    paths = outer["geometry"]["front_paths"]
    payload = {"schema": SCHEMA, "front_paths_m": paths,
               "crack_tip_m": outer["geometry"]["crack_tip_m"]}
    json_path = root / f"{stem}_crack_path.json"
    _atomic_json(json_path, payload)
    pts = np.asarray(paths[0], float)
    xyz = np.column_stack((pts, np.zeros(len(pts))))
    conn = " ".join(str(i) for i in range(len(pts)))
    text = ('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
            f'<PolyData><Piece NumberOfPoints="{len(pts)}" NumberOfLines="1">'
            '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
            + " ".join(f"{x:.17g}" for x in xyz.reshape(-1)) + '</DataArray></Points><Lines>'
            f'<DataArray type="Int64" Name="connectivity" format="ascii">{conn}</DataArray>'
            f'<DataArray type="Int64" Name="offsets" format="ascii">{len(pts)}</DataArray>'
            '</Lines></Piece></PolyData></VTKFile>\n')
    vtp = root / f"{stem}_crack_path.vtp"
    vtp.write_text(text)
    return [json_path, vtp]


def _write_package(root: Path, stem: str, roles: list[str], outer: dict,
                   arrays: dict[str, np.ndarray], generation: str, reason: str) -> None:
    prior_meta = root / f"{stem}.json"
    if prior_meta.is_file():
        prior = json.loads(prior_meta.read_text())
        if prior.get("checkpoint_generation") == generation:
            roles = sorted(set(roles) | set(prior.get("roles", [])))
    portable = _portable(arrays)
    npz = root / f"{stem}.npz"
    vtu = root / f"{stem}.vtu"
    _atomic_npz(npz, portable)
    _write_vtu(vtu, portable)
    paths = _write_path(root, stem, outer)
    inventory = []
    for name, value in sorted(portable.items()):
        a = np.asarray(value)
        inventory.append({"name": name, "shape": list(a.shape), "dtype": str(a.dtype),
                          "location": "nodal" if a.shape[:1] == (len(portable["mesh_nodes"]),) else
                                      "cell_or_native" if a.shape[-1:] == (len(portable["mesh_elems"]),) else "metadata",
                          "identically_zero": bool(np.all(a == 0))})
    meta = {"schema": SCHEMA, "roles": sorted(set(roles)), "checkpoint_generation": generation,
            "reason": reason, "accepted_step": int(outer["driver"]["step"]),
            "projected_extension_m": float(outer["driver"]["a_tip"] - outer["driver"]["crack_extension_start_a"]),
            "committed_event_count": int(outer["geometry"]["committed_event_count"]),
            "files": {}, "field_inventory": inventory}
    for p in [npz, vtu, *paths]:
        meta["files"][p.name] = _sha(p)
    _atomic_json(root / f"{stem}.json", meta)


def export_latest_checkpoint(export_root: str | Path, *, reason: str) -> None:
    root = Path(export_root).resolve()
    outer, _kinetic, raw = load_combined_checkpoint(root)
    manifest = json.loads((root / "run_state_checkpoint.json").read_text())
    sparse = root / "portable_fields"
    sparse.mkdir(parents=True, exist_ok=True)
    step = int(outer["driver"]["step"])
    events = int(outer["geometry"]["committed_event_count"])
    extension = float(outer["driver"]["a_tip"] - outer["driver"]["crack_extension_start_a"])
    roles: list[tuple[str, list[str]]] = []
    if step == 0 and events == 0:
        roles.append(("initial", ["INITIAL_ACCEPTED_STATE"]))
    if events == 0:
        roles.append(("last_pre_first_event", ["LAST_ACCEPTED_PRE_FIRST_EVENT"]))
    state = sparse / "export_state.json"
    done = json.loads(state.read_text()) if state.is_file() else {"first_post": False, "milestones": []}
    current_labels: list[str] = []
    current_stem = ""
    if events > 0 and not done["first_post"]:
        current_labels.append("FIRST_ACCEPTED_POST_EVENT")
        current_stem = "first_post_event"
        done["first_post"] = True
    for value in MILESTONES_M:
        label = str(int(round(value * 1e6)))
        if extension + 1e-15 >= value and label not in done["milestones"]:
            current_labels.append(f"FIRST_ACCEPTED_AT_OR_BEYOND_{label}UM")
            current_stem = current_stem or f"milestone_{label}um"
            done["milestones"].append(label)
    terminal = (reason in {"outer_driver_target_reached", "outer_driver_ligament_severed", "outer_driver_final_committed_state"}
                or reason.startswith("terminal_"))
    if terminal:
        current_labels.append("TERMINAL_ACCEPTED_STATE")
        current_stem = "terminal"
    if current_labels:
        roles.append((current_stem, current_labels))
    for stem, labels in roles:
        _write_package(sparse, stem, labels, outer, raw, manifest["generation"], reason)
    _atomic_json(state, done)


__all__ = ["SCHEMA", "export_latest_checkpoint"]
