#!/usr/bin/env python3
import argparse

from arrhenius_fracture.fatigue_convergence_r5 import execute_fatigue_convergence_r5


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
execute_fatigue_convergence_r5(args.output)
