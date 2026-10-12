#!/usr/bin/env python3
import argparse

from arrhenius_fracture.mechanical_correspondence_r5 import (
    execute_mechanical_correspondence_r5,
)


parser = argparse.ArgumentParser()
parser.add_argument("--spatial", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
execute_mechanical_correspondence_r5(args.spatial, args.output)
