"""Regression for the zero-physics Tref rejection and frozen-row isolation."""
import dataclasses
import json
import subprocess

import pytest

from arrhenius_fracture import parameter_registry_v9111 as generic
from arrhenius_fracture import prospective_paris_candidate_engine_v10230 as scoped
from arrhenius_fracture import sharp_front_v10_2_30_prospective_candidate_fixed_deltaK as entry
from arrhenius_fracture.material_manifest import MaterialManifest


def test_exact_row_and_five_cleavage_values():
    row = scoped.select_frozen_option(scoped.P40).row
    assert scoped.digest(row) == scoped.P40_SHA
    m, audit = scoped.build_frozen_manifest()
    assert m.cleavage.G00_eV == float(row['cleave_G00_eV'])
    assert m.cleavage.sigc0_Pa == float(row['cleave_sigc0_GPa']) * 1e9
    assert m.cleavage.alpha == float(row['cleave_exp_a'])
    assert m.cleavage.exponent == float(row['cleave_exp_n'])
    assert m.cleavage.floor_fraction == float(row['cleave_floor_frac'])
    assert audit['candidate_row_sha256'] == scoped.P40_SHA
    native = scoped.select_frozen_option('A_NATIVE').row
    assert {k for k in row if row[k] != native[k]} == scoped.CLEAVAGE_FIELDS | scoped.IDENTITY_FIELDS


def test_native_matches_accepted_builder(tmp_path):
    # Execute the source-qualified manifest builder verbatim, omitting unrelated
    # engine imports from the other branch (including its rebonding installer).
    import ast
    source = subprocess.check_output(['git', 'show', '30db009ff7172728a6bdc885886f21b94cbec225:arrhenius_fracture/a_native_engine_v10230.py'], text=True)
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_a_native_manifest')
    import tempfile
    from pathlib import Path
    namespace = dict(MaterialManifest=MaterialManifest, SelectedResponseOption=generic.SelectedResponseOption,
        write_compatibility_manifest=generic.write_compatibility_manifest, sha256_file=generic.sha256_file,
        Path=Path, tempfile=tempfile, MPZ_N_BINS=80, Any=object,
        load_a_native_provenance=lambda: json.loads((scoped.ARTIFACTS / 'A_native_provenance.json').read_text()))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<accepted-builder>', 'exec'), namespace)
    expected, _ = namespace['build_a_native_manifest']()
    actual, _ = scoped.build_frozen_manifest('A_NATIVE')
    assert scoped.canonical_bytes(actual.as_dict()) == scoped.canonical_bytes(expected.as_dict())


@pytest.mark.parametrize('option', ['A_NATIVE', scoped.P40])
def test_tref_metadata_manifest_parity(option, tmp_path):
    selected = scoped.select_frozen_option(option)
    manifests = []
    for tref in ('300.0', '481.33'):
        row = dict(selected.row, Tref_K=tref)
        path = generic.write_compatibility_manifest(dataclasses.replace(selected, row=row), tmp_path / f'{tref}.csv')
        manifests.append(scoped.canonical_bytes(MaterialManifest.from_csv(path).as_dict()))
    assert manifests[0] == manifests[1]


def test_generic_gate_is_unchanged():
    with pytest.raises(ValueError, match='Tref_K'):
        generic.select_option(scoped.P40, scoped.REGISTRY, canonical_stage3_only=False)
    path = 'arrhenius_fracture/parameter_registry_v9111.py'
    expected = subprocess.check_output(['git', 'show', f'68121e27:{path}'])
    assert (scoped.ROOT / path).read_bytes() == expected


@pytest.mark.parametrize('option', ['P40_RAW_B1_ENVELOPE_V1', 'dbtt_primary', 'unknown'])
def test_ineligible_options_refused(option):
    with pytest.raises(ValueError, match='not eligible'):
        scoped.select_frozen_option(option)


def test_entry_reaches_production_without_generic_selector_and_restores(monkeypatch, tmp_path):
    manifest_path = tmp_path / 'high_cycle_run_manifest.json'
    manifest_path.write_text('{}')
    original = entry.paper._SOURCE_SELECT_OPTION
    def physical(args):
        selected = entry.paper._select_option_four_class(scoped.P40, scoped.REGISTRY)
        assert scoped.digest(selected.row) == scoped.P40_SHA
        audit = json.loads(manifest_path.read_text())['prospective_candidate']
        assert audit['material_manifest_sha256'] == scoped.build_frozen_manifest()[1]['material_manifest_sha256']
        assert audit['construction_source_sha256']
        return 17
    monkeypatch.setattr(entry.fixed, 'main', physical)
    assert entry.main(['--out', str(tmp_path), '--parameter-option', scoped.P40]) == 17
    assert entry.paper._SOURCE_SELECT_OPTION is original


def test_entry_never_creates_unclaimed_output(tmp_path):
    out = tmp_path / 'virgin'
    with pytest.raises(ValueError, match='claim virgin'):
        entry.main(['--out', str(out), '--parameter-option', scoped.P40])
    assert not out.exists()


def test_canonical_disabled_path_parity():
    for option in generic.CANONICAL_STAGE3_OPTIONS:
        before = generic.select_option(option)
        scoped.build_frozen_manifest()
        assert generic.select_option(option) == before


def test_controller_files_not_inherited_capture_pipes(monkeypatch, tmp_path):
    import importlib.util
    path = scoped.ARTIFACTS / 'scripts/run_p40_pilots.py'
    spec = importlib.util.spec_from_file_location('pilot_controller', path)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    monkeypatch.setattr(controller, 'free_gib', lambda: 10)
    def launch(command, **kwargs):
        assert 'capture_output' not in kwargs
        assert kwargs['stdout'].fileno() >= 0
        assert kwargs['stderr'].fileno() >= 0
        out = __import__('pathlib').Path(kwargs['env']['OUTROOT'])
        assert not out.exists()
        assert (out.parent / f'{out.name}__launch.json').is_file()
        assert kwargs['env']['V10230_ENTRY_MODULE'].endswith('prospective_candidate_fixed_deltaK')
        return subprocess.CompletedProcess(command, 2)
    monkeypatch.setattr(controller.subprocess, 'run', launch)
    job = dict(result_path=str(tmp_path/'job'), seed=1720, parameter_option=scoped.P40,
               deltaK_MPa_sqrt_m=16.2, R=0.1, composite_id='test')
    result = controller.run_one(job, 'launch_head')
    assert result['exit_code'] == 2
    assert result['launch_head'] == 'launch_head'
    assert result['result_path_virgin_at_launch']
    assert not (tmp_path/'job__attempt1').exists()
    assert controller.next_virgin_path(tmp_path/'job').name == 'job__attempt2'
