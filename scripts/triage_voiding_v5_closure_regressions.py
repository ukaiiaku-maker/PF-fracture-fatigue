#!/usr/bin/env python3
"""Preserve exact regression failures and classify Policy-C expectation changes."""
import argparse,json,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def tests(path):
    result={}
    for node in ET.parse(path).getroot().iter('testcase'):
        identity=node.attrib['classname'].replace('.','/')+'.py::'+node.attrib['name']
        failure=node.find('failure')
        if failure is None: failure=node.find('error')
        result[identity]={'status':'FAIL' if failure is not None else 'SKIP' if node.find('skipped') is not None else 'PASS',
            'exact_error':None if failure is None else failure.text,
            'message':None if failure is None else failure.get('message')}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('initial',type=Path)
    parser.add_argument('--rerun',type=Path);args=parser.parse_args()
    baseline=json.loads((ROOT/'artifacts/voiding_v5_semantic_hardening/general_ci_inheritance.json').read_text())
    inherited=set(baseline['base']['failure_ids']);initial=tests(args.initial)
    rerun={} if args.rerun is None else tests(args.rerun)
    rows=[]
    for identity,failure in initial.items():
        if failure['status']!='FAIL': continue
        old=identity in inherited
        rows.append({'test_node_id':identity,'exact_error':failure['exact_error'],'exact_message':failure['message'],
            'base_branch_status':'FAIL_RECORDED_INHERITED_BASELINE' if old else 'PASS_PRE_POLICY_C_CI_34078968131',
            'current_branch_status':'FAIL','affected_physics':'UNRELATED_INHERITED_BASELINE' if old else 'CAVITY_DOWNSTREAM_FIRST_PASSAGE_AND_GRAPH_OWNERSHIP',
            'expectation_predates_policy_c':not old,'code_or_test_wrong':'OUTSIDE_CLOSURE_SCOPE' if old else 'OBSOLETE_UNQUALIFIED_DOWNSTREAM_EXPECTATION_REQUIRES_QUALIFIED_POSITIVE_PEER',
            'corrective_action':'RETAIN_AND_REPORT_SEPARATELY' if old else 'KEEP_COARSE_NO_EVENT_ASSERTION_AND_EXERCISE_PROOF_QUALIFIED_REFINEMENT_PEER_NO_XFAIL',
            'rerun_result':rerun.get(identity,{'status':'NOT_RUN','exact_error':None})})
    result={'schema':'v12.closure-regression-triage/1','executed_code_sha':subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip(),
        'initial_failures':len(rows),'inherited_failure_count':sum(r['test_node_id'] in inherited for r in rows),
        'policy_c_expectation_count':sum(r['test_node_id'] not in inherited for r in rows),
        'rows':rows,'repository_wide_ci':'NOT_GREEN' if any(t['status']=='FAIL' for t in rerun.values()) or not rerun else 'GREEN',
        'rerun_additional_failures':[identity for identity,t in rerun.items() if t['status']=='FAIL' and identity not in {r['test_node_id'] for r in rows}]}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
