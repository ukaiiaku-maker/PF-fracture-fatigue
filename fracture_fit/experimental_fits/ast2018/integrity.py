"""Verify the supplied experimental extraction against its own pinned hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def verify_archive(archive: Path, run_dir: Path):
    prefix="Ast2018_joint_fit_data/"
    with ZipFile(archive) as bundle:
        checks=bundle.read(prefix+"SHA256SUMS.txt").decode("utf-8").splitlines()
        verified=[]
        for line in checks:
            digest,relative=line.split(maxsplit=1)
            relative=relative.lstrip("*")
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError("unsafe data member in SHA256SUMS")
            member=prefix+relative
            actual=hashlib.sha256(bundle.read(member)).hexdigest()
            if digest!=actual:
                raise RuntimeError(f"experimental input hash mismatch: {relative}")
            verified.append(relative)
    output={"archive":str(archive.resolve()),
            "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "verified_file_count":len(verified),"verified_members":verified,
            "checks_file":"Ast2018_joint_fit_data/SHA256SUMS.txt"}
    run_dir.mkdir(parents=True,exist_ok=True)
    path=run_dir/"input_integrity.json"
    if path.exists() and json.loads(path.read_text())!=output:
        raise RuntimeError("experimental archive integrity receipt changed")
    path.write_text(json.dumps(output,indent=2)+"\n")
    return output


def verify_extracted(run_dir: Path):
    receipt=json.loads((run_dir/"input_integrity.json").read_text())
    root=run_dir/"input/Ast2018_joint_fit_data"
    if not root.is_dir():
        raise RuntimeError("experimental data extraction is missing")
    with ZipFile(receipt["archive"]) as bundle:
        prefix="Ast2018_joint_fit_data/"
        for relative in receipt["verified_members"]:
            disk=root/relative
            if not disk.exists() or hashlib.sha256(disk.read_bytes()).digest()!=hashlib.sha256(
                    bundle.read(prefix+relative)).digest():
                raise RuntimeError(f"extracted experimental file changed: {relative}")
    output={"verified_extracted_file_count":len(receipt["verified_members"]),
            "archive_sha256":receipt["archive_sha256"],"directory":str(root.resolve())}
    (run_dir/"extracted_integrity.json").write_text(json.dumps(output,indent=2)+"\n")
    return output
