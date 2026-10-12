#!/usr/bin/env python3
import argparse

from arrhenius_fracture.matched_execution_r6 import execute_matched_r6


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
execute_matched_r6(args.output)
