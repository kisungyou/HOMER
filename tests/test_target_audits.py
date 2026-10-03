"""The independent audits must reject corrupted production targets."""
import csv
from decimal import Decimal
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core import exact_binomial_target
from verify_population_targets import assert_production_agreement, TARGET_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def test_production_audit_allows_float64_roundoff():
    reference = {name: Decimal('0.01') for name in TARGET_FIELDS}
    production = {name: reference[name] + Decimal('2e-15') for name in TARGET_FIELDS}
    assert_production_agreement(production, reference)


@pytest.mark.parametrize('field', TARGET_FIELDS)
def test_production_audit_rejects_each_corrupted_quantity(field):
    reference = {name: Decimal('0.01') for name in TARGET_FIELDS}
    production = dict(reference)
    production[field] += Decimal('0.00001')
    with pytest.raises(ArithmeticError, match=field):
        assert_production_agreement(production, reference)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_production_audit_rejects_nonfinite_values(value):
    reference = {name: Decimal('0.01') for name in TARGET_FIELDS}
    production = dict(reference, V=value)
    with pytest.raises(ArithmeticError, match='nonfinite'):
        assert_production_agreement(production, reference)


@pytest.mark.parametrize('script', ['check_targets.py', 'verify_population_targets.py'])
@pytest.mark.parametrize('corrupt', [False, True])
def test_audit_command_reports_and_rejects_inconsistent_production(script, corrupt, tmp_path):
    row = dict(exact_binomial_target(16, .1), distribution='bernoulli')
    if corrupt:
        row['V'] += .01
    source = tmp_path/'targets.csv'
    report = tmp_path/'audit.csv'
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    result = subprocess.run([sys.executable, str(ROOT/'code'/script), '--input', str(source),
                             '--output', str(report)], capture_output=True, text=True,
                            cwd=tmp_path, env={**os.environ, 'HOMER_MODE':'smoke',
                                              'HOMER_OUTPUT_ROOT':str(tmp_path/'generated')})
    assert report.is_file(), result.stdout + result.stderr
    with report.open(newline='') as stream:
        diagnostic = list(csv.DictReader(stream))[0]
    assert diagnostic['production_agreement'] == str(not corrupt)
    if corrupt:
        assert result.returncode != 0
        assert 'Production target audit failed' in result.stderr
        assert 'V:' in diagnostic['production_audit_error']
    else:
        assert result.returncode == 0, result.stdout + result.stderr
