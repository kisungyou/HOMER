"""Public input contracts and mathematical invariants of the estimators."""
import numpy as np
import pytest

from core import (
    Fit, adaptive, canonical_huber, exact_binomial_target, geometric_median,
    projected_sandwich, pseudo_batch, pseudo_huber,
)


@pytest.mark.parametrize('fit', [geometric_median, canonical_huber, pseudo_huber, adaptive])
@pytest.mark.parametrize('x', [[], np.empty((4, 0)), np.empty((0, 2)), [[np.inf]], [[np.nan]], np.zeros((2, 2, 2))])
def test_invalid_point_arrays(fit, x):
    with pytest.raises(ValueError):
        fit(x)


@pytest.mark.parametrize('x', [np.empty((0, 3, 2)), np.empty((2, 0, 3)), np.empty((2, 3, 0)), np.zeros((2, 3))])
def test_invalid_batch_arrays(x):
    with pytest.raises(ValueError):
        pseudo_batch(x)


@pytest.mark.parametrize('fit', [canonical_huber, pseudo_huber])
@pytest.mark.parametrize('tau', [0., -1., np.inf, np.nan, [1.]])
def test_invalid_threshold(fit, tau):
    with pytest.raises(ValueError):
        fit([1., 2., 3.], tau=tau)


@pytest.mark.parametrize('c', [0., -1., np.inf, np.nan])
def test_adaptive_validates_multiplier_even_on_zero_scale(c):
    with pytest.raises(ValueError):
        adaptive(np.zeros(5), c=c)


def test_adaptive_rejects_unknown_loss():
    with pytest.raises(ValueError, match='loss'):
        adaptive([1., 2., 3.], loss='psuedo')


@pytest.mark.parametrize('fit', [geometric_median, canonical_huber, pseudo_huber])
def test_invalid_initial_shape_and_controls(fit):
    x = np.array([[0., 0.], [1., 2.], [4., 3.]])
    for initial in ([0.], [0., np.nan], [[0., 0.]]):
        with pytest.raises(ValueError, match='initial'):
            fit(x, initial=initial)
    for kwargs in ({'tol': 0.}, {'tol': np.inf}, {'max_iter': -1}, {'max_iter': 1.5}):
        with pytest.raises(ValueError):
            fit(x, **kwargs)


def test_failed_preliminary_fit_is_not_silently_accepted():
    preliminary = Fit(np.zeros(2), 0, 1., False)
    with pytest.raises(RuntimeError, match='did not converge'):
        adaptive([[0., 0.], [1., 2.], [4., 3.]], preliminary=preliminary)


@pytest.mark.parametrize('fit', [canonical_huber, pseudo_huber])
def test_iteration_exhaustion_is_reported(fit):
    x = np.array([[0., 0.], [1., 2.], [10., 3.]])
    result = fit(x, initial=np.array([20., -20.]), max_iter=0)
    assert not result.converged
    assert result.iterations == 0
    assert result.score_norm > 1e-4


def test_batch_and_individual_fits_agree():
    x = np.random.default_rng(21).normal(size=(6, 13, 3))
    centers, diagnostics = pseudo_batch(x, tau=.8)
    individual = [pseudo_huber(row, tau=.8) for row in x]
    assert diagnostics['converged'].all()
    np.testing.assert_allclose(centers, [fit.center for fit in individual], atol=1e-10)
    np.testing.assert_allclose(diagnostics['certificate'], [fit.certificate for fit in individual], atol=1e-9)


@pytest.mark.parametrize('fit', [canonical_huber, pseudo_huber, adaptive, geometric_median])
def test_permuting_subsets_preserves_the_aggregate(fit):
    x = np.random.default_rng(34).normal(size=(31, 4))
    a, b = fit(x), fit(x[::-1])
    assert a.converged and b.converged
    np.testing.assert_allclose(a.center, b.center, atol=1e-8)


def test_fixed_threshold_scales_with_estimates():
    x = np.random.default_rng(28).normal(size=(23, 3))
    shift = np.array([1., -2., 5.])
    a, b = pseudo_huber(x, tau=.6), pseudo_huber(3 * x + shift, tau=1.8)
    np.testing.assert_allclose(b.center, 3 * a.center + shift, atol=1e-9)
    np.testing.assert_allclose(b.certificate, 3 * a.certificate, atol=1e-10)


def test_contrast_covariance_matches_linear_projection():
    x = np.random.default_rng(51).normal(size=(45, 3))
    fit = pseudo_huber(x, tau=.9)
    H = np.array([[1., -1., 0.], [0., 1., 2.]])
    V, diagnostics = projected_sandwich(x, fit.center, .9, np.eye(3))
    projected, projected_diagnostics = projected_sandwich(x, fit.center, .9, H)
    np.testing.assert_allclose(projected, H @ V @ H.T, atol=1e-12)
    assert not diagnostics['singular']
    assert not projected_diagnostics['singular']
    assert projected_diagnostics['relative_solve_residual'] < 1e-14


def test_zero_contrast_remains_singular_without_ridge():
    x = np.array([[-1., 0.], [0., 1.], [1., 0.]])
    fit = pseudo_huber(x)
    V, diagnostics = projected_sandwich(x, fit.center, 1., np.zeros((1, 2)))
    np.testing.assert_array_equal(V, [[0.]])
    assert diagnostics['singular']
    assert diagnostics['relative_solve_residual'] == 0.


@pytest.mark.parametrize('H', [[], [[1.]], [[1., np.nan]], np.zeros((1, 1, 2))])
def test_invalid_contrast(H):
    with pytest.raises(ValueError, match='H'):
        projected_sandwich([[0., 1.], [1., 0.]], [0., 0.], 1., H)


@pytest.mark.parametrize('arguments', [(0, .1), (3.5, .1), (3, 0.), (3, 1.), (3, np.nan)])
def test_binomial_target_domain(arguments):
    with pytest.raises(ValueError):
        exact_binomial_target(*arguments)


def test_symmetric_binomial_target_and_reflection():
    for m in (7, 32):
        symmetric = exact_binomial_target(m, .5)
        assert abs(symmetric['u']) < 1e-12
        left = exact_binomial_target(m, .1)
        right = exact_binomial_target(m, .9)
        np.testing.assert_allclose(right['u'], -left['u'], atol=1e-12)
        np.testing.assert_allclose(right['V'], left['V'], atol=1e-12)
