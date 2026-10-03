#!/usr/bin/env python3
"""Regenerate the three manuscript figures from the compact saved results."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('paper', 'smoke', 'full'), default='paper')
    parser.add_argument('--study', nargs='+', choices=('all', 'scalar', 'functional', 'wearable', 'covariance'), default=['all'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(HOMER_MODE=args.mode, HOMER_OUTPUT_ROOT=str(root / 'runs' / args.mode),
               MPLCONFIGDIR=str(root / 'runs' / args.mode / '.matplotlib'))
    if args.mode == 'paper':
        from reproduce import verify_saved_inputs
        verify_saved_inputs(root)
    return subprocess.call([sys.executable, str(root / 'code' / 'make_figures.py'),
                            '--study', *args.study], cwd=root, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
