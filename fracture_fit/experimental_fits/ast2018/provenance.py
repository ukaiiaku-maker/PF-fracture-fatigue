"""Hash the paired forward wrappers independently of the frozen search core."""
from __future__ import annotations

import hashlib
from pathlib import Path


def forward_adapter_sha256():
    root=Path(__file__).parent
    names=("mechanics.py","reduced.py","spatial.py")
    return hashlib.sha256(b"".join(name.encode()+b"\0"+(root/name).read_bytes()
                                   for name in names)).hexdigest()
