#!/usr/bin/env python3
import argparse

from arrhenius_fracture.state_correspondence_r5 import qualify_no_event_partition


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
qualify_no_event_partition(args.output)
