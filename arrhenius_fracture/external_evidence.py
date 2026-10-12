"""Content-addressed contracts for raw evidence kept outside Git.

The directory digest commits to every regular file's relative path, byte size,
and SHA-256.  It is deliberately independent of mtimes and directory order.
Full qualification treats a missing mandatory item as INCOMPLETE, never as a
pytest skip or an implicit pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if root.is_file():
        return {
            "kind": "file", "bytes": root.stat().st_size,
            "file_count": 1, "sha256": sha256_file(root),
            "digest_algorithm": "sha256(file-bytes)",
        }
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    aggregate = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        digest = sha256_file(item)
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(str(size).encode("ascii") + b"\0")
        aggregate.update(digest.encode("ascii") + b"\n")
        total += size
    return {
        "kind": "directory", "bytes": total, "file_count": len(files),
        "sha256": aggregate.hexdigest(),
        "digest_algorithm": "sha256(sorted(relative_path\\0bytes\\0file_sha256\\n))",
    }


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    path: str
    mandatory: bool
    producer: str
    dependencies: tuple[str, ...]
    permitted_locations: tuple[str, ...]
    restore_command: str
    fingerprint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id, "path": self.path,
            "mandatory": self.mandatory, "producer": self.producer,
            "dependencies": list(self.dependencies),
            "permitted_locations": list(self.permitted_locations),
            "restore_command": self.restore_command,
            "fingerprint": self.fingerprint,
        }


def make_item(*, evidence_id: str, path: str | Path, mandatory: bool,
              producer: str, dependencies: Iterable[str] = (),
              permitted_locations: Iterable[str], restore_command: str) -> EvidenceItem:
    resolved = Path(path).expanduser().resolve()
    return EvidenceItem(
        evidence_id=evidence_id, path=str(resolved), mandatory=bool(mandatory),
        producer=producer, dependencies=tuple(dependencies),
        permitted_locations=tuple(permitted_locations),
        restore_command=restore_command, fingerprint=fingerprint_path(resolved),
    )


def verify_manifest(payload: dict[str, Any], *, base_directory: str | Path | None = None) -> dict[str, Any]:
    base = Path.cwd() if base_directory is None else Path(base_directory)
    results = []
    for item in payload.get("items", []):
        path = Path(item["path"]).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        if not path.exists():
            status = "INCOMPLETE" if item.get("mandatory", False) else "OPTIONAL_ABSENT"
            results.append({"evidence_id": item["evidence_id"], "path": str(path), "status": status})
            continue
        observed = fingerprint_path(path)
        expected = item["fingerprint"]
        match = all(observed.get(key) == expected.get(key) for key in ("kind", "bytes", "file_count", "sha256"))
        results.append({
            "evidence_id": item["evidence_id"], "path": str(path),
            "status": "PASS" if match else "FAIL_HASH_MISMATCH",
            "expected": expected, "observed": observed,
        })
    mandatory_bad = [row for row in results if row["status"] != "PASS" and next(
        item.get("mandatory", False) for item in payload["items"] if item["evidence_id"] == row["evidence_id"]
    )]
    return {
        "schema": "common-framework-external-evidence-verification/1",
        "status": "PASS" if not mandatory_bad else "INCOMPLETE",
        "mandatory_incomplete_count": len(mandatory_bad), "items": results,
    }


def load_and_verify(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    return verify_manifest(json.loads(source.read_text()), base_directory=source.parent)


__all__ = ["EvidenceItem", "fingerprint_path", "load_and_verify", "make_item", "sha256_file", "verify_manifest"]
