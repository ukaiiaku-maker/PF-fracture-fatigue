#!/usr/bin/env python3
"""Tolerance-aware cross-platform comparator for V12 scientific evidence."""
from __future__ import annotations
import argparse, ast, csv, json, math
from pathlib import Path

RTOL=1e-10; ATOL=1e-12
CSV_FILES=("straight_primal_matrix.csv","straight_3p125um_prescreen.csv","centered_G_matrix.csv","angle_primal_matrix.csv")
EXACT_JSON_KEYS=("schema","base_git_sha","implementation_git_sha","evidence_generation_parent_sha","conforming_oracle_source_commit","conforming_oracle_source_sha256","thresholds_predeclared","gates")
EXACT_COLUMNS={"representation","parent_geometry_fingerprint","parent_connectivity_fingerprint","support_fingerprint","minus_graph_fingerprint","plus_graph_fingerprint","minus_support_fingerprint","plus_support_fingerprint","minus_mechanical_fingerprint","plus_mechanical_fingerprint"}

def number(value):
    if value in ("",None): return None
    try: return float(value)
    except ValueError: return None

def compare_nested(left,right,path):
    if isinstance(left,dict):
        if left.keys()!=right.keys(): raise AssertionError(f"{path}: nested keys mismatch")
        for key in left: compare_nested(left[key],right[key],f"{path}.{key}")
    elif isinstance(left,list):
        if len(left)!=len(right): raise AssertionError(f"{path}: nested length mismatch")
        for index,(a,b) in enumerate(zip(left,right)): compare_nested(a,b,f"{path}[{index}]")
    elif isinstance(left,(float,int)) and not isinstance(left,bool):
        if not math.isclose(float(left),float(right),rel_tol=RTOL,abs_tol=ATOL): raise AssertionError(f"{path}: {left} != {right}")
    elif left!=right: raise AssertionError(f"{path}: exact mismatch {left!r} != {right!r}")

def compare_csv(expected,actual,name):
    a=list(csv.DictReader((expected/name).open())); b=list(csv.DictReader((actual/name).open()))
    if len(a)!=len(b) or (a and a[0].keys()!=b[0].keys()): raise AssertionError(f"{name}: row/schema mismatch")
    for index,(left,right) in enumerate(zip(a,b)):
        for key in left:
            x,y=left[key],right[key]
            if key=="cod_samples" and x and y: compare_nested(ast.literal_eval(x),ast.literal_eval(y),f"{name}:{index}:{key}")
            elif key in EXACT_COLUMNS or number(x) is None or number(y) is None:
                if x!=y: raise AssertionError(f"{name}:{index}:{key}: exact mismatch {x!r} != {y!r}")
            else:
                xf,yf=number(x),number(y)
                if math.isnan(xf) and math.isnan(yf): continue
                column_atol=1e-8 if key.endswith("_force_N_per_m") else ATOL
                if not math.isclose(xf,yf,rel_tol=RTOL,abs_tol=column_atol): raise AssertionError(f"{name}:{index}:{key}: {xf} != {yf}")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("expected",type=Path); parser.add_argument("actual",type=Path); args=parser.parse_args()
    for name in CSV_FILES: compare_csv(args.expected,args.actual,name)
    left=json.loads((args.expected/"qualification.json").read_text()); right=json.loads((args.actual/"qualification.json").read_text())
    for key in EXACT_JSON_KEYS:
        if left[key]!=right[key]: raise AssertionError(f"qualification:{key}: exact mismatch")
    for key,value in left["checks"].items():
        other=right["checks"][key]
        if isinstance(value,bool):
            if value is not other: raise AssertionError(f"qualification:checks:{key}: exact mismatch")
        elif not math.isclose(value,other,rel_tol=RTOL,abs_tol=ATOL): raise AssertionError(f"qualification:checks:{key}: {value} != {other}")
    print("V12 primal scientific evidence matches across platforms")

if __name__=="__main__": main()
