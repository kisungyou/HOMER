#!/usr/bin/env python3
"""Execute notebooks and verify inline figures without exporting image files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'scalar.ipynb': 1, 'functional.ipynb': 2, 'covariance_wearable.ipynb': 2}


def external_figures():
    """Include existing exports so overwriting a file also fails the check."""
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ROOT.rglob('*')
        if path.is_file() and path.suffix.lower() in ('.png', '.pdf', '.pgf', '.svg')
        and not any(part in ('.git', '.venv', 'venv') for part in path.relative_to(ROOT).parts)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', nargs='+', choices=('paper', 'smoke', 'full'), default=['smoke'])
    parser.add_argument('--write-paper', action='store_true',
                        help='embed executed paper-mode outputs in the notebook files for GitHub previews')
    args = parser.parse_args()
    if args.write_paper and 'paper' not in args.mode:
        parser.error('--write-paper requires --mode paper')
    before = external_figures()
    for mode in dict.fromkeys(args.mode):
        for name, expected in EXPECTED.items():
            path = ROOT / 'notebooks' / name
            notebook = nbformat.read(path, as_version=4)
            configured = 0
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop('execution', None)
                if re.match(r'^MODE\s*=', cell.source):
                    cell.source, count = re.subn(r'^MODE\s*=.*$',
                                                 f'MODE = "{mode}"  # "paper", "smoke", or "full"',
                                                 cell.source, count=1, flags=re.M)
                    configured += count
            if configured != 1:
                raise AssertionError(f'{name}: expected one mode selector')
            print(f'Executing {name}: {mode}', flush=True)
            NotebookClient(notebook, timeout=300, kernel_name='python3',
                           resources={'metadata': {'path': str(ROOT)}}).execute()
            figure_cells = [cell for cell in notebook.cells
                            if 'inline-figure' in cell.metadata.get('tags', [])]
            if len(figure_cells) != expected:
                raise AssertionError(f'{name}: expected {expected} tagged figure cells')
            for cell in figure_cells:
                images = [output.get('data', {}).get('image/png') for output in cell.outputs
                          if output.output_type in ('display_data', 'execute_result')]
                if not any(images):
                    raise AssertionError(f'{name} ({mode}): inline figure cell produced no PNG output')
            if external_figures() != before:
                raise AssertionError(f'{name} ({mode}): wrote or changed an external figure file')
            if args.write_paper and mode == 'paper':
                for cell in notebook.cells:
                    cell.metadata.pop('execution', None)
                nbformat.write(notebook, path)
            print(f'Passed: {expected} inline figures; no external figure files', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
