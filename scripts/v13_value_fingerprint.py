"""Deterministic value fingerprints, not pickle memo/alias-layout hashes."""
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np


def physical_state_fingerprint(value):
    def canonical(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                return ["object_array", x.shape, canonical(x.tolist())]
            return ["array", x.dtype.str, x.shape, hashlib.sha256(x.tobytes(order="C")).hexdigest()]
        if isinstance(x, np.random.Generator):
            return ["rng", canonical(x.bit_generator.state)]
        if isinstance(x, np.generic):
            return canonical(x.item())
        if isinstance(x, Mapping):
            return ["mapping", sorted([[canonical(k), canonical(v)] for k, v in x.items()], key=lambda item: json.dumps(item[0], sort_keys=True))]
        if isinstance(x, (tuple, list)):
            return [type(x).__name__, [canonical(v) for v in x]]
        if isinstance(x, (set, frozenset)):
            return [type(x).__name__, sorted([canonical(v) for v in x], key=lambda v: json.dumps(v, sort_keys=True))]
        if isinstance(x, float):
            return ["float", x.hex()]
        if x is None or isinstance(x, (bool, int, str)):
            return [type(x).__name__, x]
        if isinstance(x, Path):
            return ["path", str(x)]
        if callable(x):
            raise TypeError("callable in serialized physical state")
        if hasattr(x, "__dict__"):
            return [type(x).__module__ + "." + type(x).__qualname__, canonical(vars(x))]
        raise TypeError(f"unsupported physical state field: {type(x)}")
    return hashlib.sha256(json.dumps(canonical(value), separators=(",", ":"), allow_nan=False).encode()).hexdigest()
