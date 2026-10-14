"""Content-addressed contracts for raw evidence kept outside Git.

The directory digest commits to every regular file's relative path, byte size,
and SHA-256. It is independent of mtimes and directory order. Full
qualification treats missing mandatory evidence as INCOMPLETE.
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
        return {"kind": "file", "bytes": root.stat().st_size, "file_count": 1,
                "sha256": sha256_file(root), "digest_algorithm": "sha256(file-bytes)"}
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
    return {"kind": "directory", "bytes": total, "file_count": len(files),
            "sha256": aggregate.hexdigest(),
            "digest_algorithm": "sha256(sorted(relative_path\\0bytes\\0file_sha256\\n))"}


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
        return {"evidence_id": self.evidence_id, "path": self.path,
                "mandatory": self.mandatory, "producer": self.producer,
                "dependencies": list(self.dependencies),
                "permitted_locations": list(self.permitted_locations),
                "restore_command": self.restore_command, "fingerprint": self.fingerprint}


def make_item(*, evidence_id: str, path: str | Path, mandatory: bool,
              producer: str, dependencies: Iterable[str] = (),
              permitted_locations: Iterable[str], restore_command: str) -> EvidenceItem:
    resolved = Path(path).expanduser().resolve()
    return EvidenceItem(evidence_id=evidence_id, path=str(resolved), mandatory=bool(mandatory),
                        producer=producer, dependencies=tuple(dependencies),
                        permitted_locations=tuple(permitted_locations),
                        restore_command=restore_command, fingerprint=fingerprint_path(resolved))


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
        match = all(observed.get(key) == expected.get(key)
                    for key in ("kind", "bytes", "file_count", "sha256"))
        results.append({"evidence_id": item["evidence_id"], "path": str(path),
                        "status": "PASS" if match else "FAIL_HASH_MISMATCH",
                        "expected": expected, "observed": observed})
    mandatory_bad = [row for row in results if row["status"] != "PASS" and next(
        item.get("mandatory", False) for item in payload["items"]
        if item["evidence_id"] == row["evidence_id"]
    )]
    return {"schema": "common-framework-external-evidence-verification/1",
            "status": "PASS" if not mandatory_bad else "INCOMPLETE",
            "mandatory_incomplete_count": len(mandatory_bad), "items": results}


def load_and_verify(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    return verify_manifest(json.loads(source.read_text()), base_directory=source.parent)


def directory_fingerprint(path: str | Path) -> dict[str, Any]:
    result = fingerprint_path(path)
    if result["kind"] != "directory":
        raise ValueError(f"expected directory evidence: {path}")
    return result


class EvidenceManifest:
    """Selective audit facade that preserves the original verification API."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.items = tuple(json.loads(self.path.read_text())["items"])
        ids = [item["evidence_id"] for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("external evidence manifest has duplicate IDs")

    def selected(self, evidence_ids: Iterable[str] | None = None) -> tuple[dict[str, Any], ...]:
        if evidence_ids is None:
            return self.items
        requested = set(evidence_ids)
        rows = tuple(item for item in self.items if item["evidence_id"] in requested)
        missing_ids = requested - {item["evidence_id"] for item in rows}
        if missing_ids:
            raise KeyError(f"unknown evidence IDs: {sorted(missing_ids)}")
        return rows

    def resolve(self, item: dict[str, Any]) -> Path:
        raw = Path(item["path"]).expanduser()
        return raw.resolve() if raw.is_absolute() else (self.path.parent / raw).resolve()

    def audit(self, evidence_ids: Iterable[str] | None = None, *, verify_hashes: bool = True) -> dict[str, Any]:
        records, incomplete = [], []
        for item in self.selected(evidence_ids):
            path = self.resolve(item)
            expected = item["fingerprint"]
            if not path.exists():
                status, observed = "MISSING", None
            elif verify_hashes:
                observed = fingerprint_path(path)
                status = "VERIFIED" if all(
                    observed.get(key) == expected.get(key)
                    for key in ("kind", "bytes", "file_count", "sha256") if key in expected
                ) else "FINGERPRINT_MISMATCH"
            else:
                observed = {"kind": "directory" if path.is_dir() else "file"}
                status = "PRESENT_NOT_HASHED"
            record = {"evidence_id": item["evidence_id"],
                      "mandatory": bool(item.get("mandatory")), "path": str(path),
                      "status": status, "expected": expected, "observed": observed,
                      "restore_command": item.get("restore_command"),
                      "backup_archive": item.get("backup_archive")}
            records.append(record)
            if record["mandatory"] and status not in {"VERIFIED", "PRESENT_NOT_HASHED"}:
                incomplete.append(record)
        return {"schema": "external-evidence-audit/1", "manifest": str(self.path),
                "manifest_sha256": sha256_file(self.path), "verify_hashes": verify_hashes,
                "decision": "COMPLETE" if not incomplete else "INCOMPLETE",
                "records": records, "mandatory_incomplete": incomplete,
                "retrieval_instructions": [record["restore_command"] or
                    f"restore {record['evidence_id']} from {record['backup_archive']} to {record['path']}"
                    for record in incomplete]}


file_sha256 = sha256_file

__all__ = ["EvidenceItem", "EvidenceManifest", "directory_fingerprint", "file_sha256",
           "fingerprint_path", "load_and_verify", "make_item", "sha256_file", "verify_manifest"]
