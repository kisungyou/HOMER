"""Record the environment, inputs, code, and outputs without personal paths."""
from pathlib import Path
import hashlib
import json
import platform
import numpy
import pandas
import scipy
import matplotlib
import mpmath
from settings import ROOT, OUTPUT_ROOT, RESULTS, DERIVED, FIGURES, CONFIG, MODE


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory, pattern='*'):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob(pattern))
            if path.is_file() and '__pycache__' not in path.parts}


def main():
    sources = {str(path.relative_to(ROOT)): sha256(path)
               for folder in ('code',)
               for path in sorted((ROOT / folder).rglob('*'))
               if path.is_file() and path.suffix in ('.py', '.json')}
    record = {
        'mode': MODE, 'configuration': CONFIG,
        'python': platform.python_version(), 'operating_system': platform.system(),
        'os_release': platform.release(), 'machine': platform.machine(),
        'numpy': numpy.__version__, 'pandas': pandas.__version__,
        'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__,
        'mpmath': mpmath.__version__, 'floating_dtype': 'float64',
        'source_sha256': sources,
        'derived_input_sha256': hashes(DERIVED),
        'results_source': 'results' if MODE == 'paper' else 'generated results',
        'simulation_manifests': {p.name: json.loads(p.read_text())
                                 for p in sorted(RESULTS.glob('*_manifest.json'))},
        'numerical_tolerances': {'scalar_score': 1e-13, 'pseudo_huber_score': 1e-12,
                                 'geometric_subgradient': 1e-10, 'certificate_fraction_of_se': .001},
        'scope': 'Smoke runs validate execution only. Adaptive runs illustrate point stability; fixed-threshold block-mean experiments assess inference.'
    }
    if (FIGURES / 'figure_style.json').exists():
        record['figure_rendering'] = json.loads((FIGURES / 'figure_style.json').read_text())
    (OUTPUT_ROOT / 'environment.json').write_text(json.dumps(record, indent=2) + '\n')
    generated = {str(path.relative_to(OUTPUT_ROOT)): sha256(path)
                 for folder in ('results', 'data', 'figures', 'tables')
                 for path in sorted((OUTPUT_ROOT / folder).rglob('*')) if path.is_file()}
    (OUTPUT_ROOT / 'result_hashes.json').write_text(json.dumps(generated, indent=2) + '\n')
    print(f'Recorded {len(sources)} source hashes and {len(generated)} generated artifacts.')


if __name__ == '__main__':
    main()
