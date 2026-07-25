#!/usr/bin/env python3
"""Compatibility entry point for the direct-J and energy postprocessor.

The former implementation reconstructed J from K using a user-supplied elastic
modulus.  That path has been retired because it could be misinterpreted as a
total elastic-plastic absorbed-energy calculation.  The replacement reads the
direct configurational FEM J and separately reports tip-emission, bulk-plastic,
and energy-balance contributions.
"""
from __future__ import annotations

import sys

from plot_v10_2_27_paper_four_class_J_energy_vs_temperature import main


def _remove_deprecated_modulus_options(args: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    deprecated = {"--youngs-modulus-pa", "--poisson-ratio"}
    while index < len(args):
        token = args[index]
        if any(token.startswith(name + "=") for name in deprecated):
            index += 1
            continue
        if token in deprecated:
            index += 2
            continue
        output.append(token)
        index += 1
    return output


if __name__ == "__main__":
    sys.argv[1:] = _remove_deprecated_modulus_options(sys.argv[1:])
    raise SystemExit(main())
