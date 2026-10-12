#!/usr/bin/env python3
import argparse

from arrhenius_fracture.dynamic_void_execution_r5 import execute_dynamic_void_r5


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
execute_dynamic_void_r5(args.output)
