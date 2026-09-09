"""Explicit, fail-closed loader for optional historical test products.

Bundled bytes are checked both against the SHA-256 manifest and the archived
Git blobs. External packs require a separately supplied published manifest hash;
a missing configured pack, malformed manifest, missing member or tampering is
an error, never an unavailable-fixture skip. No mutable output-tree fallback.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "tests/fixtures/historical_products"
MANIFEST_NAME = "historical_product_fixture_manifest.json"


class HistoricalProductUnavailable(Exception):
    """Only the intentionally unconfigured optional package is unavailable."""


@dataclass(frozen=True)
class HistoricalProduct:
    root: Path
    bindings: dict
    source_root: Path | None = None


def contained(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("historical fixture path escapes package: " + name)
    return path


def verify_package(root, manifest):
    """Verify every member; an assertion failure must propagate to pytest."""
    members = manifest["members"]
    if not members:
        raise ValueError("empty historical fixture manifest")
    for name, record in members.items():
        path = contained(root, name)
        data = path.read_bytes()
        if len(data) != record["size_bytes"]:
            raise ValueError("historical fixture size mismatch: " + name)
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("historical fixture SHA-256 mismatch: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(members) | {MANIFEST_NAME}:
        raise ValueError("historical fixture member set mismatch")


def load_product(node_id, *, env=None, bundled=BUNDLED):
    env = os.environ if env is None else env
    catalog = json.loads((bundled / MANIFEST_NAME).read_text())
    spec = catalog["tests"][node_id]  # Unknown tests are errors, not skipped.
    if spec["availability"] == "BUNDLED_COMMIT_VERIFIED":
        root, manifest = bundled, catalog
        verify_package(root, manifest)
        for name, record in manifest["members"].items():
            blob = subprocess.check_output(
                ["git", "show", manifest["source_record_commit"] + ":" + record["source_git_path"]],
                cwd=ROOT,
            )
            if hashlib.sha256(blob).hexdigest() != record["sha256"]:
                raise ValueError("historical fixture differs from archived Git blob: " + name)
    else:
        location = env.get("PF_HISTORICAL_PRODUCT_FIXTURE_ROOT")
        expected = env.get("PF_HISTORICAL_PRODUCT_MANIFEST_SHA256")
        if not location and not expected:
            raise HistoricalProductUnavailable(
                "HISTORICAL_PRODUCT_FIXTURE_UNAVAILABLE: optional verified external "
                "V5/V6 product package is intentionally not installed; " + node_id
            )
        if not location or not expected or len(expected) != 64:
            raise ValueError("external historical fixture needs root AND published manifest SHA-256")
        root = Path(location).resolve()
        data = (root / MANIFEST_NAME).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("external historical fixture manifest SHA-256 mismatch")
        manifest = json.loads(data)
        verify_package(root, manifest)
        spec = manifest["tests"][node_id]  # Partial configured packs fail closed.
    bindings = spec.get("bindings", {})
    if set(bindings) != set(catalog["tests"][node_id]["binding_names"]):
        raise ValueError("historical fixture bindings differ from explicit test contract")
    paths = {key: contained(root, value) for key, value in bindings.items()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    source = spec.get("source_root_relative")
    if "runtime_parity_v12" in node_id and source is None:
        raise ValueError("prefix parity needs a manifested portable source root")
    return HistoricalProduct(root, paths, contained(root, source) if source else None)
