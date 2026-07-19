"""Prevent reintroduction of the removed variational fracture subsystem."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_variational_fracture_modules_are_absent():
    package = ROOT / "arrhenius_fracture"
    forbidden = {
        "a" + "t1.py",
        "a" + "t2_overlay.py",
        "phase_" + "field.py",
    }
    assert forbidden.isdisjoint({path.name for path in package.iterdir()})


def test_removed_model_labels_are_absent_from_repository_text():
    token = "a" + "t"
    pattern = re.compile(
        rf"(?i)(?<![A-Za-z0-9_]){token}[-_ ]?[12](?![A-Za-z0-9_])"
    )
    binary_suffixes = {
        ".7z", ".gif", ".gz", ".jpeg", ".jpg", ".npz", ".pdf", ".png",
        ".pyc", ".tar", ".tif", ".tiff", ".webp", ".xz", ".zip",
    }
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in binary_suffixes or path.stat().st_size > 2_000_000:
            continue
        relative = path.relative_to(ROOT)
        if pattern.search(relative.as_posix()):
            findings.append(f"path:{relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                findings.append(f"text:{relative}:{line_number}")
    assert not findings, "removed model labels found: " + ", ".join(findings)
