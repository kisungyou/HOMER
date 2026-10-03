"""Shared, read-only study settings and isolated output locations.

Set HOMER_MODE to ``smoke`` (default), ``full``, or ``paper`` before importing
study modules. ``paper`` reads the archived results and only regenerates reports.
HOMER_OUTPUT_ROOT may specify an absolute destination for generated artifacts.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get('HOMER_MODE', 'smoke').lower()
if MODE not in ('smoke', 'full', 'paper'):
    raise ValueError('HOMER_MODE must be smoke, full, or paper')
CONFIG_PATH = ROOT / 'code' / 'configs' / ('full.json' if MODE == 'paper' else f'{MODE}.json')
CONFIG = json.loads(CONFIG_PATH.read_text())
for key in ('scalar_replications', 'functional_replications', 'covariance_replications', 'bootstrap_replications'):
    if not isinstance(CONFIG[key], int) or CONFIG[key] < 2:
        raise ValueError(f'{key} must be an integer of at least two')
DERIVED = ROOT / 'data' / 'derived'
PAPER_RESULTS = ROOT / 'results'
custom = os.environ.get('HOMER_OUTPUT_ROOT')
if custom and not Path(custom).is_absolute():
    raise ValueError('HOMER_OUTPUT_ROOT must be an absolute path')
OUTPUT_ROOT = Path(custom).resolve() if custom else (ROOT / 'runs' / MODE).resolve()
protected = [ROOT / name for name in ('data', 'results', 'figures', 'code', 'docs', 'tests', '.github', 'notebooks')]
if (OUTPUT_ROOT == ROOT or ROOT.is_relative_to(OUTPUT_ROOT)
        or any(OUTPUT_ROOT.is_relative_to(p.resolve()) for p in protected)
        or (OUTPUT_ROOT.is_relative_to(ROOT) and not OUTPUT_ROOT.is_relative_to((ROOT / 'runs').resolve()))):
    raise ValueError('Output destination must not overwrite repository sources or frozen artifacts')
GENERATED_RESULTS = OUTPUT_ROOT / 'results'
RESULTS = PAPER_RESULTS if MODE == 'paper' else GENERATED_RESULTS
DATA = OUTPUT_ROOT / 'data'
FIGURES = OUTPUT_ROOT / 'figures'
TABLES = OUTPUT_ROOT / 'tables'
# Refuse redirected output trees before creating or modifying any artifacts.
raw_output = Path(custom) if custom else ROOT / 'runs' / MODE
if raw_output.is_symlink() or (ROOT / 'runs').is_symlink():
    raise ValueError('Output directories must not be symbolic links')
if OUTPUT_ROOT.exists() and any(path.is_symlink() for path in OUTPUT_ROOT.rglob('*')):
    raise ValueError('Output files and subdirectories must not be symbolic links')
for directory in (GENERATED_RESULTS, DATA, FIGURES, TABLES):
    directory.mkdir(parents=True, exist_ok=True)


def require_run_mode():
    """Prevent simulation scripts from writing into the frozen paper results."""
    if MODE == 'paper':
        raise ValueError('paper mode is read-only; use smoke or full to rerun experiments')


def manifest_settings():
    return {'mode': MODE, 'configuration': CONFIG, 'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()}
