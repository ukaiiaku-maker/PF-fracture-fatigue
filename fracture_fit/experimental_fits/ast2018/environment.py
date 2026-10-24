"""Small immutable software/hardware receipt for runtime interpretation."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import matplotlib
import numpy
import scipy


def write_receipt(run_dir: Path):
    path=run_dir/"environment_receipt.json"
    receipt={"python":sys.version,"platform":platform.platform(),
             "machine":platform.machine(),"logical_cpu_count":os.cpu_count(),
             "numpy":numpy.__version__,"scipy":scipy.__version__,
             "matplotlib":matplotlib.__version__,
             "OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),
             "OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS"),
             "VECLIB_MAXIMUM_THREADS":os.environ.get("VECLIB_MAXIMUM_THREADS")}
    if path.exists() and json.loads(path.read_text())!=receipt:
        raise RuntimeError("execution environment receipt changed on resume")
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
