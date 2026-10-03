"""Notebook figures must render the supplied results without file exports or TeX."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess

import matplotlib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['MOM', 'PH(0.25)', 'PH(1)', 'PH(4)', 'PH(16)', 'Mean']


def csv(name):
    return pd.read_csv(ROOT / 'results' / name)


def covariance_data():
    with np.load(ROOT / 'results/covariance_display.npz', allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


@pytest.fixture
def plots(monkeypatch, tmp_path):
    # Other tests load the publication exporter, which selects the PGF backend.
    # These helpers must also work without that backend or external TeX programs.
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def forbidden(*args, **kwargs):
        pytest.fail('Inline plotting must not save figures or start external programs')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Figure, 'savefig', forbidden)
    monkeypatch.setattr(plt, 'savefig', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    spec = importlib.util.spec_from_file_location('tested_notebook_plots', ROOT / 'code/notebook_plots.py')
    module = importlib.util.module_from_spec(spec)
    with matplotlib.rc_context():
        # The manuscript exporter uses TeX. The inline helpers must override this
        # locally and restore the caller's setting after constructing each plot.
        plt.rcParams['text.usetex'] = True
        spec.loader.exec_module(module)
        yield module
        assert plt.rcParams['text.usetex'] is True
    plt.close('all')
    assert list(tmp_path.iterdir()) == [], 'Inline plotting wrote an external file'


def errorbar_series(fig):
    """Read artist data, independently of labels, colours, or panel placement."""
    from matplotlib.container import ErrorbarContainer
    result = []
    for axis in fig.axes:
        for container in axis.containers:
            if isinstance(container, ErrorbarContainer):
                central, _, bars = container.lines
                y = np.asarray(central.get_ydata(), dtype=float)
                # Each interval is a vertical segment with endpoints (x, lo/hi).
                segments = np.asarray(bars[0].get_segments())
                result.append((y, segments[:, 0, 1], segments[:, 1, 1]))
    return result


def assert_series_present(actual, value, low, high):
    value, low, high = map(np.asarray, (value, low, high))
    matches = [entry for entry in actual if entry[0].shape == value.shape
               and np.allclose(entry[0], value, rtol=1e-12, atol=1e-14)]
    assert matches, f'Supplied observations are absent from plot: {value}'
    assert any(np.allclose(entry[1], low, rtol=1e-12, atol=1e-14)
               and np.allclose(entry[2], high, rtol=1e-12, atol=1e-14)
               for entry in matches), 'Uncertainty intervals differ from supplied results'


@pytest.mark.parametrize('kind', ['scalar', 'functional', 'stability', 'covariance', 'wearable'])
def test_figures_render_in_memory_without_tex_or_exports(plots, kind):
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    arguments = {
        'scalar': ('scalar_coverage', [csv('scalar_summary.csv')]),
        'functional': ('functional_coverage', [csv('functional_inference_summary.csv')]),
        'stability': ('functional_stability', [csv('functional_point_summary.csv'),
                        csv('efficiency_ratio_concentrated.csv'), csv('efficiency_ratio_diffuse.csv')]),
        'covariance': ('covariance_heatmaps', [covariance_data()]),
        'wearable': ('wearable_stability', [csv('wearable_results.csv')]),
    }
    name, args = arguments[kind]
    before = deepcopy(args)
    figure = getattr(plots, name)(*args)
    assert isinstance(figure, Figure)
    assert figure.axes
    assert all(not artist.get_usetex() for artist in figure.findobj(Text))
    figure.canvas.draw()
    assert np.asarray(figure.canvas.buffer_rgba()).std() > 0
    for original, supplied in zip(before, args):
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(original, supplied)
        else:
            for key in original:
                np.testing.assert_array_equal(original[key], supplied[key])


@pytest.mark.parametrize('name,file,grouping,sort', [
    ('scalar_coverage', 'scalar_summary.csv', ['schedule', 'metric'], 'n'),
    ('functional_coverage', 'functional_inference_summary.csv',
     ['contrast', 'spectrum', 'distribution'], 'k'),
])
def test_coverage_uses_current_results_and_intervals_in_order(plots, name, file, grouping, sort):
    summary = csv(file)
    # A fresh smoke/full result can differ sharply from the committed manuscript
    # preview. Perturb all estimates and intervals, then scramble row order.
    summary[['coverage', 'mc_low', 'mc_high']] *= 0.7
    shuffled = summary.sample(frac=1, random_state=42).reset_index(drop=True)
    actual = errorbar_series(getattr(plots, name)(shuffled))
    if name == 'scalar_coverage':
        summary = summary[summary.schedule.isin(['slow', 'boundary', 'many'])
                          & summary.metric.isin(['plugin_mean_covered', 'plugin_target_covered',
                                                 'oracle_mean_covered', 'oracle_target_covered'])]
    else:
        summary = summary[summary.contrast.isin(['interval_average', 'evening_minus_morning'])]
    for _, group in summary.groupby(grouping):
        q = group.sort_values(sort)
        assert_series_present(actual, q.coverage, q.mc_low, q.mc_high)


def test_stability_uses_supplied_efficiency_and_contamination_results(plots):
    point = csv('functional_point_summary.csv')
    efficiency = [csv(f'efficiency_ratio_{spectrum}.csv')
                  for spectrum in ('concentrated', 'diffuse')]
    # Scale all norm-error summaries together, retaining coherent error bars.
    point[['q95', 'q95_mc_low', 'q95_mc_high']] *= 1.5
    for saved in efficiency:
        changed = saved.method == 'MOM'
        saved.loc[changed, ['mse_ratio', 'mc_low', 'mc_high']] *= 1.2
        point.loc[(point.method == 'MOM') & (point.spectrum == saved.spectrum.iloc[0])
                  & (point.distribution == 'gaussian') & (point.mechanism == 'clean'), 'mse'] *= 1.2
    figure = plots.functional_stability(point.sample(frac=1, random_state=1), *efficiency)
    actual = errorbar_series(figure)
    for saved in efficiency:
        q = saved.set_index('method').loc[METHODS]
        assert_series_present(actual, q.mse_ratio, q.mc_low, q.mc_high)
    order = ['MOM', 'PH(1)', 'Adaptive(2)', 'PH(16)', 'Mean']
    for mechanism in ('whole_block', 'raw_dispersed'):
        for magnitude in (10, 100):
            q = point[(point.spectrum == 'concentrated') & (point.distribution == 'gaussian')
                      & (point.mechanism == mechanism) & (point.magnitude == magnitude)]
            q = q.set_index('method').loc[order]
            assert_series_present(actual, q.q95, q.q95_mc_low, q.q95_mc_high)


def test_covariance_selects_requested_replication_and_methods_without_clipping(plots):
    saved = covariance_data()
    # Full simulations contain multiple replications; select a visibly different
    # one and permute the methods to catch stale previews or positional lookups.
    original = saved['estimates']
    saved['estimates'] = np.concatenate([original, original + 20], axis=0)
    saved['display_replicate'] = np.array(1)
    permutation = np.arange(len(saved['methods']))[::-1]
    saved['methods'] = saved['methods'][permutation]
    saved['estimates'] = saved['estimates'][:, :, permutation]
    scenarios = np.arange(len(saved['delta']))[::-1]
    saved['delta'] = saved['delta'][scenarios]
    saved['estimates'] = saved['estimates'][:, scenarios]
    methods = list(saved['methods'])
    scenario = int(np.flatnonzero(saved['delta'] == saved['display_delta'])[0])
    clean = int(np.flatnonzero(saved['delta'] == 0)[0])
    estimates = saved['estimates'][1]
    expected = [saved['truth'], estimates[clean, methods.index('Mean')]]
    expected += [estimates[scenario, methods.index(name)]
                 for name in ('Mean', 'GMed', 'Adaptive PH')]
    figure = plots.covariance_heatmaps(saved)
    limits = []
    for axis, matrix in zip(figure.axes[:5], expected):
        # Support either vector mesh or image artists without prescribing style.
        artist = axis.images[0] if axis.images else axis.collections[0]
        np.testing.assert_array_equal(np.asarray(artist.get_array()).reshape(matrix.shape), matrix)
        limits.append(artist.get_clim())
    assert len(limits) == 5
    assert all(limit == limits[0] for limit in limits)
    assert limits[0][0] <= min(matrix.min() for matrix in expected)
    assert limits[0][1] >= max(matrix.max() for matrix in expected)


def test_wearable_uses_current_results_and_activity_order(plots):
    summary = csv('wearable_results.csv')
    summary['shift_relative_clean_mean'] *= 0.4
    figure = plots.wearable_stability(summary.sample(frac=1, random_state=12))
    actual = [(np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
              for axis in figure.axes for line in axis.lines]
    for method in ('Mean', 'MOM', 'Adaptive(2)'):
        q = summary[(summary.mechanism == 'one_block')
                    & (summary.amplitude_multiplier == 10) & (summary.method == method)]
        q = q.sort_values('activity')
        assert any(x.shape == q.activity.shape and np.array_equal(x, q.activity)
                   and np.allclose(y, q.shift_relative_clean_mean, rtol=1e-12, atol=1e-14)
                   for x, y in actual), f'Current wearable results are not plotted for {method}'
