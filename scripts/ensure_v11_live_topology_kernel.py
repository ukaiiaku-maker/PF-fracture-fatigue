#!/usr/bin/env python3
"""Evaluate or resolve one exact v11 live-topology FEM request."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.kernel_resolver_v11 import resolve_pickled_request


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--accepted", action="store_true")
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args(argv)
    result, cache_hit = resolve_pickled_request(
        args.request, cache_root=args.cache_root, accepted=args.accepted
    )
    manifest = {
        key: value for key, value in result.items()
        if key not in {"base_equilibrium", "tips", "signed_shared_cluster_response"}
    }
    manifest["cache_hit"] = cache_hit
    manifest["accepted_state"] = bool(args.accepted)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.manifest_out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
