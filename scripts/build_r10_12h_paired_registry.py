#!/usr/bin/env python3
"""Material-identical four-arm registry for selected autonomous-search vectors."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR, FAMILY, OPENING_ALTERNATIVES_FAMILY, SMOOTH_EXPONENTIAL,
)
from scripts.run_r10_global_mechanism_search import canonical_hash


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count",type=int)
    args=parser.parse_args(argv)
    with args.registry.open(newline="") as stream: registry=list(csv.DictReader(stream))
    with args.selection.open(newline="") as stream: selected=list(csv.DictReader(stream))
    if args.count:selected=selected[:args.count]
    by_source={}
    for row in registry:by_source.setdefault(row["source_material_candidate_id"],row)
    output=[]
    for selected_row in selected:
        source=selected_row["source_material_candidate_id"]
        original=by_source[source]
        for shape in (SMOOTH_EXPONENTIAL,BILINEAR):
            for mode in ("cohesive_athermal","cohesive_mixed"):
                row=dict(original)
                code=("A" if mode=="cohesive_athermal" else "M")+("S" if shape==SMOOTH_EXPONENTIAL else "B")
                row["candidate_id"]=f"{source}__12H_{code}"
                row["source_material_candidate_id"]=source
                row["selected_shape"]=shape;row["selected_mode"]=mode
                row["source_complete_row_sha256"]=original["complete_row_sha256"]
                row["complete_row_sha256"]=canonical_hash({
                    "family":FAMILY,"additive_family":OPENING_ALTERNATIVES_FAMILY,
                    "source_complete_row_sha256":original["complete_row_sha256"],
                    "shape":shape,"mode":mode,"autonomous_campaign":"R10_12H"})
                output.append(row)
    fields=[]
    for row in output:
        for key in row:
            if key not in fields:fields.append(key)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n")
        writer.writeheader();writer.writerows(output)
    print({"physical_vectors":len(selected),"execution_identities":len(output)})


if __name__=="__main__":raise SystemExit(main())
