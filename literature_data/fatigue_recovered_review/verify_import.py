#!/usr/bin/env python3
"""Verify the normalized CSVs against source-cell exports, without Excel/RAR tools.

The original workbooks and their hashes are included. raw_sheet_csv was produced
from their actual values using artifact_tool. This program checks normalization,
identity, selected source rows and conversions; it does not validate a fatigue model.
"""
from pathlib import Path
import csv, hashlib, json, math, sys
ROOT = Path(__file__).resolve().parent

def load_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def column_index(cell):
    v = 0
    for c in cell:
        if not c.isalpha(): break
        v = 26*v + ord(c.upper()) - ord('A') + 1
    return v-1

def main():
    checks=0
    sources=load_csv(ROOT/'audit/source_inventory.csv')
    for rec in sources:
        p=ROOT/rec['relative_path']
        assert p.stat().st_size==int(rec['bytes']),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],p
        checks+=1
    raw_tables={}
    for p in (ROOT/'raw_sheet_csv').glob('*.csv'):
        with p.open(newline='',encoding='utf-8') as f:raw_tables[p.stem]=list(csv.reader(f))
    specifications=[
        ('qste340tm_crack_histories.csv','lu_fig3','cycles','reported_crack_length_mm','reported_crack_length_m',1e-3),
        ('qste340tm_growth_rates.csv','lu_fig4','reported_DeltaK_MPa_sqrt_m','reported_dadn_mm_per_cycle','dadn_m_per_cycle',1e-3),
        ('qste340tm_closure_loads.csv','lu_fig5','reported_DeltaK_MPa_sqrt_m','reported_closure_load_kN','closure_load_N',1e3),
        ('steel316ln_crack_histories.csv',None,'cycles','reported_crack_length_mm','reported_crack_length_m',1e-3)
    ]
    output_counts={}
    for name,table_id,xkey,ykey,converted,factor in specifications:
        rows=load_csv(ROOT/'normalized'/name)
        output_counts[name]=len(rows)
        for rec in rows:
            key=table_id or ('316ln_'+rec['source_sheet'])
            row=raw_tables[key][int(rec['source_row'])-1]
            assert float(row[column_index(rec['source_x_cell'])])==float(rec[xkey]),rec
            assert float(row[column_index(rec['source_y_cell'])])==float(rec[ykey]),rec
            assert math.isclose(float(rec[ykey])*factor,float(rec[converted]),rel_tol=2e-15,abs_tol=1e-30),rec
            if table_id=='lu_fig4':
                assert rec['data_kind']=='author_reduced_experimental_rate'
                assert float(rec['dadn_m_per_cycle'])>0
            checks+=3
    expected={'qste340tm_crack_histories.csv':2417,'qste340tm_growth_rates.csv':2396,
              'qste340tm_closure_loads.csv':1380,'steel316ln_crack_histories.csv':1539}
    assert output_counts==expected,output_counts
    ca=load_csv(ROOT/'normalized/qste340tm_constant_amplitude.csv')
    ol=load_csv(ROOT/'normalized/qste340tm_overload.csv')
    assert len(ca)==760 and len(ol)==1636
    assert {float(r['R']) for r in ca}=={.1,.3,.5,.7}
    assert all(r['protocol']=='CONSTANT_AMPLITUDE' for r in ca)
    assert all(r['protocol']=='SINGLE_OVERLOAD' for r in ol)
    assert len(load_csv(ROOT/'audit/unpaired_cells.csv'))==6
    assert len(load_csv(ROOT/'normalized/qste340tm_additional_fig15_points_REVIEW.csv'))==14
    main316=load_csv(ROOT/'normalized/steel316ln_unirradiated_primary.csv')
    assert len(main316)==799
    assert {r['specimen_id'] for r in main316}=={'A15','B3','B7'}
    assert all(r['irradiation_dpa']=='0' and r['author_labeled_outlier']=='False' for r in main316)
    metadata=json.loads((ROOT/'dataset_metadata.json').read_text())
    assert metadata['qste340tm']['temperature_K'] is None
    assert metadata['qste340tm']['frequency_Hz'] is None
    assert metadata['steel316ln']['width_W_mm'] is None
    integrity=json.loads((ROOT/'audit/archive_integrity.json').read_text())
    assert all(r['expected_crc32']==r['computed_crc32'] for r in integrity['rar_file_crc_checks'])
    manifest=ROOT/'MANIFEST_SHA256.json'
    members=0
    if manifest.exists():
        for name,rec in json.loads(manifest.read_text())['files'].items():
            p=ROOT/name
            assert p.stat().st_size==rec['bytes'],name
            assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],name
            members+=1
    receipt={'status':'PASS','source_files_checked':len(sources),'paired_records_cell_checked':sum(output_counts.values()),
             'cell_and_conversion_assertions':checks,'primary_table_counts':output_counts,
             'manifest_files_checked':members,'native_or_barrier_fitting_executed':False}
    print(json.dumps(receipt,indent=2))
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (AssertionError,ValueError,OSError,KeyError) as exc:
        print('VALIDATION FAILED:',repr(exc),file=sys.stderr);sys.exit(1)
