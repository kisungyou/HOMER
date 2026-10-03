"""Check archival integrity and public workflow behavior independently of fitting."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'code/reproduce.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_archived_artifacts_are_intact():
    runner.verify_saved_inputs(ROOT)


def test_full_counts_match_recorded_manifests():
    config = json.loads((ROOT / 'code/configs/full.json').read_text())
    for study in ('scalar', 'functional', 'covariance'):
        original = json.loads((ROOT / f'results/{study}_manifest.json').read_text())
        assert config[f'{study}_replications'] == original['reps_per_cell' if study == 'scalar' else 'replications']
    original = json.loads((ROOT / 'results/covariance_manifest.json').read_text())
    assert config['bootstrap_replications'] == original['bootstrap_replications']


def test_plan_from_another_directory_is_side_effect_free(tmp_path):
    output = tmp_path / 'must-not-exist'
    result = subprocess.run([sys.executable, str(ROOT / 'code/reproduce.py'), '--mode', 'full',
                             '--study', 'sim', '--dry-run'], cwd=tmp_path, text=True,
                            capture_output=True, check=True,
                            env={**os.environ, 'HOMER_OUTPUT_ROOT': str(output)})
    plan = json.loads(result.stdout)
    assert plan['studies'] == ['scalar', 'functional', 'covariance']
    assert plan['mode'] == 'full'
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_resume_rejects_missing_changed_and_incompatible_results(tmp_path):
    artifact = tmp_path / 'results.csv'
    artifact.write_text('estimate\n1.0\n')
    receipt = tmp_path / 'receipt.json'
    fingerprint = {'seed': 12, 'replications': 3}
    receipt.write_text(json.dumps({'fingerprint': fingerprint,
        'artifacts': {'results.csv': runner.sha256(artifact)}}))
    assert runner.reusable(receipt, fingerprint, tmp_path)
    assert not runner.reusable(receipt, {'seed': 13, 'replications': 3}, tmp_path)
    artifact.write_text('estimate\n2.0\n')
    assert not runner.reusable(receipt, fingerprint, tmp_path)
    artifact.unlink()
    assert not runner.reusable(receipt, fingerprint, tmp_path)


def test_resume_rejects_artifact_outside_output(tmp_path):
    output = tmp_path / 'outputs'
    output.mkdir()
    artifact = tmp_path / 'outside.csv'
    artifact.write_text('1')
    receipt = output / 'receipt.json'
    receipt.write_text(json.dumps({'fingerprint': {},
        'artifacts': {'../outside.csv': runner.sha256(artifact)}}))
    assert not runner.reusable(receipt, {}, output)


@pytest.mark.parametrize('protected', ['results', 'data', 'figures', 'code', 'docs'])
def test_settings_refuse_frozen_output_destinations(protected, tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'code'),
           'HOMER_OUTPUT_ROOT': str(ROOT / protected), 'HOMER_MODE': 'smoke'}
    result = subprocess.run([sys.executable, '-c', 'import settings'], env=env,
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'must not overwrite' in result.stderr


def test_partial_figure_request_explains_required_studies(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'code/reproduce.py'), '--study',
                             'scalar', '--figures', '--dry-run'], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'scalar + functional' in result.stderr
    assert runner.figure_plan(['scalar', 'functional', 'covariance']) == ['exp_overview', 'exp_covariance']


def test_efficiency_files_belong_to_functional_resume_receipt():
    assert {'efficiency_ratio_concentrated.csv', 'efficiency_ratio_diffuse.csv'} <= set(runner.STUDY_ARTIFACTS['functional'])
    assert 'scalar_target_precision_audit.csv' not in runner.STUDY_ARTIFACTS['scalar']


def test_settings_rejects_redirected_result_directory(tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'results').symlink_to(ROOT / 'results', target_is_directory=True)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'code'),
           'HOMER_OUTPUT_ROOT': str(output), 'HOMER_MODE': 'smoke'}
    result = subprocess.run([sys.executable, '-c', 'import settings'], env=env,
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'symbolic links' in result.stderr


def test_failed_replacement_updates_run_report(tmp_path):
    import shutil
    checkout = tmp_path / 'checkout'
    (checkout / 'code/configs').mkdir(parents=True)
    (checkout / 'docs').mkdir()
    shutil.copy2(ROOT / 'code/reproduce.py', checkout / 'code/reproduce.py')
    shutil.copy2(ROOT / 'code/configs/smoke.json', checkout / 'code/configs/smoke.json')
    (checkout / 'docs/reference_sha256.json').write_text('{}')
    (checkout / 'requirements.txt').write_text('')
    (checkout / 'code/run_scalar.py').write_text('raise RuntimeError("fixture failure")\n')
    output = checkout / 'runs/smoke'
    output.mkdir(parents=True)
    (output / 'run_report.json').write_text('{"status":"completed"}')
    result = subprocess.run([sys.executable, str(checkout / 'code/reproduce.py'), '--smoke',
                             '--study', 'scalar'], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((output / 'run_report.json').read_text())
    assert report['status'] == 'failed'
    assert 'fixture failure' in report['error']


@pytest.mark.parametrize('flags,mode', [([], 'full'), (['--smoke'], 'smoke'),
                                      (['--mode', 'smoke'], 'smoke')])
def test_driver_mode_plans_are_side_effect_free(flags, mode, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'code/reproduce.py'), *flags,
                             '--dry-run'], cwd=tmp_path, text=True,
                            capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['mode'] == mode
    assert plan['output'] == f'runs/{mode}'
    expected = json.loads((ROOT / f'code/configs/{mode}.json').read_text())
    assert plan['settings'] == expected
    assert not list(tmp_path.iterdir())


def test_managed_outputs_cannot_escape_runs(tmp_path):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    with pytest.raises(ValueError, match='runs/'):
        runner.validate_output_tree(checkout / 'results', checkout)
    outside = tmp_path / 'external'
    outside.mkdir()
    (checkout / 'runs').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic links'):
        runner.validate_output_tree(checkout / 'runs/full', checkout)
