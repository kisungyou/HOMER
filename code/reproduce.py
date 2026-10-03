#!/usr/bin/env python3
"""Run HOMER's small offline checks or recorded manuscript experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ('scalar', 'functional', 'wearable', 'covariance')
PACKAGES = ('numpy', 'scipy', 'pandas', 'matplotlib', 'mpmath')
STUDY_ARTIFACTS = {
    'scalar': ['scalar_replicates.csv.gz', 'scalar_summary.csv', 'scalar_targets.csv',
               'scalar_timing.csv', 'scalar_manifest.json'],
    'functional': ['functional_inference_replicates.csv.gz', 'functional_inference_summary.csv',
                   'functional_point_replicates.csv.gz', 'functional_point_summary.csv',
                   'functional_paired_differences.csv', 'functional_timing.csv', 'functional_manifest.json']
        + [f'functional_pairing_{spectrum}_{law}.npz'
           for spectrum in ('concentrated', 'diffuse') for law in ('gaussian', 'student_t4')]
        + [f'efficiency_ratio_{spectrum}.csv' for spectrum in ('concentrated', 'diffuse')],
    'wearable': ['wearable_results.csv', 'wearable_manifest.json'],
    'covariance': ['covariance_replicates.csv.gz', 'covariance_summary.csv',
                   'covariance_paired_differences.csv', 'covariance_estimates.npz',
                   'covariance_inputs.npz', 'covariance_timing.csv', 'covariance_manifest.json'],
}


def figure_plan(studies):
    dependencies = {'exp_overview': {'scalar', 'functional'},
                    'exp_functional_wearable': {'functional', 'wearable'},
                    'exp_covariance': {'covariance'}}
    return [name for name, required in dependencies.items() if required <= set(studies)]


def validate_output_tree(output, root=ROOT):
    # Managed outputs cannot be redirected through symlinks, including files.
    if (root / 'runs').is_symlink() or output.is_symlink():
        raise ValueError('Managed output directories must not be symbolic links')
    if not output.resolve().is_relative_to((root / 'runs').resolve()):
        raise ValueError('Managed outputs must remain under runs/')
    if output.exists() and any(path.is_symlink() for path in output.rglob('*')):
        raise ValueError('Managed output files and subdirectories must not be symbolic links')



def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_studies(names):
    requested = set(names)
    if 'all' in requested:
        return list(STUDIES)
    if 'sim' in requested:
        requested.update(('scalar', 'functional', 'covariance'))
    return [name for name in STUDIES if name in requested]


def verify_saved_inputs(root=ROOT):
    """Check the committed inputs, reference results, and manuscript displays."""
    manifest = json.loads((root / 'docs' / 'reference_sha256.json').read_text())
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Saved input differs from the release: {relative}')


def fingerprint(root, config, study):
    sources = [root / 'requirements.txt']
    sources += sorted((root / 'code').glob('*.py'))
    sources += sorted((root / 'data' / 'derived').glob('*'))
    return {
        'study': study, 'config': config,
        'python': '.'.join(map(str, sys.version_info[:3])),
        'packages': {name: version(name) for name in PACKAGES},
        'source_and_input_sha256': {
            str(path.relative_to(root)): sha256(path) for path in sources if path.is_file()
        },
    }


def reusable(receipt, expected, output):
    """Reuse only a completed run with matching code, settings, and intact files."""
    try:
        record = json.loads(receipt.read_text())
        if record['fingerprint'] != expected or not record['artifacts']:
            return False
        for relative, digest in record['artifacts'].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file() or sha256(path) != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_study(output, study, config):
    """Check row counts and numerical diagnostics; keep all failed fits visible."""
    import pandas as pd
    counts = {
        'scalar': [('scalar_replicates.csv.gz', 26 * config['scalar_replications'])],
        'functional': [
            ('functional_inference_replicates.csv.gz', 36 * config['functional_replications']),
            ('functional_point_replicates.csv.gz', 140 * config['functional_replications']),
        ],
        'wearable': [('wearable_results.csv', 210)],
        'covariance': [('covariance_replicates.csv.gz', 9 * config['covariance_replications'])],
    }
    diagnostics = {}
    for name, expected in counts[study]:
        frame = pd.read_csv(output / 'results' / name)
        if len(frame) != expected:
            raise ValueError(f'{name}: expected {expected} records, found {len(frame)}')
        flags = [column for column in (
            'solver_converged', 'converged', 'preliminary_converged',
            'covariance_valid', 'certificate_pass', 'psd_valid',
        ) if column in frame]
        failures = {}
        for column in flags:
            valid = frame[column].isin([True, 'True', 'true', 1])
            failures[column] = int((~valid).sum())
        diagnostics[name] = {'rows': len(frame), 'failures': failures}
        if any(failures.values()):
            raise ValueError(f'{name}: numerical checks failed: {failures}; all records retained')
    return diagnostics


def run_script(script, arguments, output, environment, label):
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f'{label}.log'
    command = [sys.executable, str(ROOT / 'code' / script), *arguments]
    print(f'Running {label} ...', flush=True)
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-25:])
        raise RuntimeError(f'{label} failed (exit {result.returncode}).\n{tail}\nLog: {log}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_const', dest='mode', const='smoke', help='quick execution check with reduced replication counts')
    mode.add_argument('--mode', choices=('smoke', 'full'), help='choose smoke or full (default: full)')
    parser.set_defaults(mode='full')
    parser.add_argument('--study', nargs='+', choices=('all', 'sim', *STUDIES), default=['all'])
    parser.add_argument('--resume', action='store_true', help='reuse verified completed studies')
    parser.add_argument('--dry-run', action='store_true', help='print settings without creating outputs')
    parser.add_argument('--figures', action='store_true', help='render PGF, PDF, and PNG figures (requires TeX and Poppler)')
    parser.add_argument('--audit-targets', action='store_true', help='independently audit scalar targets at high precision')
    args = parser.parse_args(argv)
    selected = selected_studies(args.study)
    if args.audit_targets and 'scalar' not in selected:
        parser.error('--audit-targets requires selecting scalar (or sim/all)')
    figures = figure_plan(selected) if args.figures else []
    if args.figures:
        covered = set()
        if 'exp_overview' in figures:
            covered.update(('scalar', 'functional'))
        if 'exp_functional_wearable' in figures:
            covered.update(('functional', 'wearable'))
        if 'exp_covariance' in figures:
            covered.add('covariance')
        if set(selected) - covered:
            parser.error('--figures requires scalar + functional for the overview, functional + wearable for the combined figure, or covariance; select the corresponding studies together')
    config = json.loads((ROOT / 'code' / 'configs' / f'{args.mode}.json').read_text())
    output = ROOT / 'runs' / args.mode
    plan = {'mode': args.mode, 'studies': selected, 'settings': config,
            'output': f'runs/{args.mode}', 'resume': args.resume,
            'figures': args.figures, 'figure_stems': figures, 'audit_targets': args.audit_targets}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.figures:
        missing = [tool for tool in ('pdflatex', 'pdftoppm') if shutil.which(tool) is None]
        if missing:
            parser.error('Figure rendering requires ' + ', '.join(missing) + ' on PATH; omit --figures to run numerical studies')
    environment = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        environment[name] = '1'
    environment.update(HOMER_MODE=args.mode, HOMER_OUTPUT_ROOT=str(output),
                       PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(output / '.matplotlib'))
    started = time.monotonic()
    records = {}
    report_started = False
    try:
        validate_output_tree(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'run_report.json').write_text(json.dumps({**plan, 'status': 'running'}, indent=2) + '\n')
        report_started = True
        verify_saved_inputs()
        print(f'HOMER {args.mode}: {", ".join(selected)}', flush=True)
        if args.mode == 'smoke':
            print('Reduced Monte Carlo counts; these results check the pipeline, not the paper claims.', flush=True)
        for study in selected:
            receipt = output / 'receipts' / f'{study}.json'
            expected = fingerprint(ROOT, config, study)
            if args.resume and reusable(receipt, expected, output):
                print(f'Reusing verified {study} results.', flush=True)
                records[study] = {'status': 'reused', 'checks': validate_study(output, study, config)}
                continue
            # An interrupted replacement must never retain a valid completion marker.
            receipt.unlink(missing_ok=True)
            run_script(f'run_{study}.py', [], output, environment, study)
            checks = validate_study(output, study, config)
            artifacts = [output / 'results' / name for name in STUDY_ARTIFACTS[study]]
            if study == 'wearable':
                artifacts.append(output / 'data' / 'subject_partition.csv')
            missing = [path.name for path in artifacts if not path.is_file()]
            if missing:
                raise ValueError(f'{study}: missing required output files: {missing}')
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({'fingerprint': expected, 'checks': checks,
                'artifacts': {str(path.relative_to(output)): sha256(path)
                              for path in artifacts if path.is_file()}}, indent=2) + '\n')
            records[study] = {'status': 'computed', 'checks': checks}
        run_script('make_tables.py', ['--study', *selected], output, environment, 'tables')
        if args.audit_targets:
            run_script('check_targets.py', [], output, environment, 'target_audit_60digits')
            run_script('verify_population_targets.py', ['--scipy-zero-mask'], output, environment, 'target_audit_decimal')
        if args.figures:
            run_script('make_figures.py', ['--study', *selected], output, environment, 'figures')
            for stem in figures:
                for suffix in ('.pgf', '.pdf', '.png'):
                    if not (output / 'figures' / (stem + suffix)).is_file():
                        raise ValueError(f'Missing generated figure: {stem}{suffix}')
            print('Figures: ' + ', '.join(figures), flush=True)
        run_script('write_provenance.py', [], output, environment, 'provenance')
        verify_saved_inputs()
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        if report_started:
            (output / 'run_report.json').write_text(json.dumps({**plan, 'status': 'failed',
                'studies': records, 'error': str(error)}, indent=2) + '\n')
        print(f'Reproduction stopped: {error}', file=sys.stderr)
        return 1
    report = {**plan, 'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.monotonic() - started, 'studies': records}
    (output / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Completed in {report["elapsed_seconds"]:.1f} seconds. Results: runs/{args.mode}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
