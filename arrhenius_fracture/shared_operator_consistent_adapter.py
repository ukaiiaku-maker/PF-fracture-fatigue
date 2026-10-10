"""Thin public adapter for the preserved V3.6 operator-consistent campaign."""
from __future__ import annotations

from pathlib import Path
import os
import sys


def _value(args: list[str], name: str) -> str | None:
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _remove(args: list[str], *names: str) -> list[str]:
    names = set(names)
    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        name = token.split("=", 1)[0]
        if name in names:
            index += 1 if "=" in token else 2
            continue
        result.append(token)
        index += 1
    return result


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    output = _value(values, "--out")
    if output is None:
        raise SystemExit("V3.6 shared execution requires --out")
    values = _remove(values, "--out", "--save-snapshots")
    values = [item for item in values if item != "--no-plots"]
    restart = os.environ.get("V10230_RESTART_CHECKPOINT_DIR", "").strip()
    if restart and _value(values, "--restart-checkpoint") is None:
        values.extend(["--restart-checkpoint", restart])
    checkpoint = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR", "").strip()
    if checkpoint and _value(values, "--checkpoint-dir") is None:
        values.extend(["--checkpoint-dir", checkpoint])
    from scripts.run_v3_6_operator_consistent_developed_fatigue_v10230 import (
        main as native_main,
    )
    return native_main([*values, "--out-root", str(Path(output).resolve())])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
