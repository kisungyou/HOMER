"""The committed summaries reproduce displays without archived replicates."""
import importlib.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['MOM', 'PH(0.25)', 'PH(1)', 'PH(4)', 'PH(16)', 'Mean']


@pytest.fixture
def paper_figures(monkeypatch, tmp_path):
    # Isolate plotting tests from simulation output and from external TeX tools.
    result_dir = tmp_path / 'results'
    result_dir.mkdir()
    figure_dir = tmp_path / 'figures'
    figure_dir.mkdir()
    settings = SimpleNamespace(RESULTS=result_dir, FIGURES=figure_dir,
                               CONFIG={'bootstrap_replications': 2000}, MODE='paper')
    monkeypatch.setitem(sys.modules, 'settings', settings)
    spec = importlib.util.spec_from_file_location('compact_paper_figures', ROOT / 'code/make_figures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.plt.close('all')


def gaussian_summary(spectrum):
    summary = pd.read_csv(ROOT / 'results/functional_point_summary.csv')
    return summary[(summary.spectrum == spectrum) & (summary.distribution == 'gaussian')
                   & (summary.mechanism == 'clean')].set_index('method').loc[ORDER]


@pytest.mark.parametrize('spectrum', ['concentrated', 'diffuse'])
def test_paper_efficiency_uses_saved_intervals_without_replicates(paper_figures, spectrum):
    source = ROOT / f'results/efficiency_ratio_{spectrum}.csv'
    shutil.copy2(source, paper_figures.RES / source.name)
    summary = gaussian_summary(spectrum)
    # No replicate records are present in the temporary result directory.
    value, lower, upper = paper_figures.efficiency_intervals(spectrum, summary)
    saved = pd.read_csv(source).set_index('method').loc[ORDER]
    np.testing.assert_allclose(value, summary.mse / summary.loc['Mean', 'mse'], rtol=1e-12)
    np.testing.assert_allclose(value, saved.mse_ratio, rtol=1e-12)
    np.testing.assert_array_equal(lower, saved.mc_low)
    np.testing.assert_array_equal(upper, saved.mc_high)


def test_paper_efficiency_rejects_inconsistent_summary(paper_figures):
    source = ROOT / 'results/efficiency_ratio_concentrated.csv'
    shutil.copy2(source, paper_figures.RES / source.name)
    summary = gaussian_summary('concentrated')
    summary.loc['MOM', 'mse'] *= 2
    with pytest.raises(AssertionError):
        paper_figures.efficiency_intervals('concentrated', summary)


def test_paper_covariance_display_needs_only_prespecified_replication(paper_figures, monkeypatch):
    source = ROOT / 'results/covariance_display.npz'
    shutil.copy2(source, paper_figures.RES / source.name)
    with np.load(source) as saved:
        assert saved['estimates'].shape[0] == 1
        replication = int(saved['display_replicate'])
        assert replication == 0
        scenario = int(np.flatnonzero(saved['delta'] == saved['display_delta'])[0])
        methods = list(saved['methods'])
        estimates = saved['estimates']
        expected = [saved['truth'], estimates[0, 0, methods.index('Mean')]]
        expected.extend(estimates[0, scenario, methods.index(name)]
                        for name in ('Mean', 'GMed', 'Adaptive PH'))
        expected = np.array(expected)
    displayed = []

    def capture_export(figure, stem):
        assert stem == 'exp_covariance'
        displayed.extend(np.asarray(axis.collections[0].get_array()).reshape(12, 12)
                         for axis in figure.axes[:5])
        paper_figures.LAYOUT[stem] = {}

    monkeypatch.setattr(paper_figures, 'export', capture_export)
    paper_figures.covariance_illustration()
    np.testing.assert_array_equal(displayed, expected)
    layout = paper_figures.LAYOUT['exp_covariance']
    assert layout['display_replicate'] == 0
    assert layout['display_delta'] == 32
    low, high = layout['shared_color_limits']
    assert low <= expected.min() <= expected.max() <= high
