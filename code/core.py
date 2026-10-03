"""Finite-dimensional reference implementation of HOMER.

Inputs are subset estimates, one per row. Statistical thresholds are distinct
from numerical stopping tolerances. Callers must inspect ``Fit.converged``;
failed iterative fits are returned as failures rather than silently discarded.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


@dataclass
class Fit:
    """An aggregate and its numerical diagnostics.

    ``certificate`` bounds distance to the exact pseudo-Huber minimizer.
    ``gap_bound`` instead bounds the geometric-median objective gap; it is not
    a distance bound. Unavailable diagnostics are NaN. ``threshold`` is in the
    same units as the input estimates.
    """

    center: np.ndarray
    iterations: int
    score_norm: float
    converged: bool
    certificate: float = np.nan
    branch: str = 'positive_scale'
    threshold: float = np.nan
    gap_bound: float = np.nan


def points(x):
    """Validate a nonempty finite (subsets, coordinates) array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite point matrix')
    return x


def _positive(value, name):
    if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite scalar')


def _controls(tol, max_iter):
    _positive(tol, 'tol')
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer')


def _initial(initial, shape):
    if initial is None:
        return None
    value = np.asarray(initial, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'initial must be finite with shape {shape}')
    return value.copy()


def geometric_median(x, initial=None, tol=1e-10, max_iter=20000):
    """Compute a geometric median by modified Weiszfeld iteration.

    Scalar ties use their midpoint. Coincident points use the exact subgradient
    condition, without a numerical distance floor. The reported objective-gap
    bound is the maximum residual distance times the subgradient norm.
    """
    x = points(x)
    k, d = x.shape
    _controls(tol, max_iter)
    initial = _initial(initial, (d,))
    if d == 1:
        v = np.median(x, axis=0)
        r = x - v
        nz = r[:, 0] != 0
        g = max(0., abs(np.sign(r[nz, 0]).sum()) - (~nz).sum()) / k
        return Fit(v, 0, g, True, gap_bound=0.)
    # A strict repeated-point majority has a unique exact median.
    u, c = np.unique(x, axis=0, return_counts=True)
    if c.max() > k / 2:
        return Fit(u[c.argmax()].copy(), 0, 0., True, branch='exact_majority', gap_bound=0.)
    # Verify a vertex optimum before iterating to avoid arithmetic stagnation.
    if k <= 512:
        diff = x[:, None, :] - x[None, :, :]
        dd = np.linalg.norm(diff, axis=2)
        unit = np.divide(diff, dd[:, :, None], out=np.zeros_like(diff), where=dd[:, :, None] > 0)
        vertex_sub = np.linalg.norm(unit.sum(1), axis=1) - (dd == 0).sum(1)
        candidates = np.flatnonzero(vertex_sub <= 0)
        if len(candidates):
            return Fit(x[candidates[0]].copy(), 0, 0., True, branch='exact_vertex', gap_bound=0.)
    v = x.mean(0) if initial is None else initial
    for it in range(max_iter):
        r = x - v
        dist = np.linalg.norm(r, axis=1)
        nz = dist > 0
        multiplicity = (~nz).sum()
        R = (r[nz] / dist[nz, None]).sum(0)
        Rn = np.linalg.norm(R)
        subgrad = max(0., Rn - multiplicity) / k
        if subgrad <= tol:
            return Fit(v, it, subgrad, True, gap_bound=float(dist.max()) * subgrad)
        T = (x[nz] / dist[nz, None]).sum(0) / (1 / dist[nz]).sum()
        alpha = min(1., multiplicity / Rn) if Rn else 1.
        v = (1 - alpha) * T + alpha * v
    r = x - v
    dist = np.linalg.norm(r, axis=1)
    nz = dist > 0
    subgrad = max(0., np.linalg.norm((r[nz] / dist[nz, None]).sum(0)) - (~nz).sum()) / k
    return Fit(v, max_iter, subgrad, False, gap_bound=float(dist.max()) * subgrad)


def pseudo_components(x, v, tau):
    """Internal vectorized score, Hessian, objective, residuals, and weights."""
    r = x - v
    norm = np.linalg.norm(r, axis=-1)
    h = np.hypot(1., norm / tau)
    w = 1 / h
    if not np.isfinite(h).all() or np.any(w == 0):
        raise FloatingPointError('Nonfinite residual or underflowed weight')
    score = (r * w[..., None]).mean(axis=-2)
    d = x.shape[-1]
    A = w.mean(axis=-1)[..., None, None] * np.eye(d) - np.einsum('...ki,...kj,...k->...ij', r, r, w**3) / (x.shape[-2] * tau**2)
    objective = (tau**2 * (h - 1)).mean(axis=-1)
    return score, A, objective, r, w


def pseudo_batch(x, tau=1., tol=1e-12, max_iter=100, initial=None):
    """Fit arrays shaped (replications, subsets, coordinates).

    Damped Newton iteration uses backtracking and no diagonal ridge. Returns
    ``(centers, diagnostics)`` with a convergence flag and distance certificate
    for every replication. ``tau`` is a shared positive scalar threshold.
    """
    x = np.asarray(x, float)
    if x.ndim != 3 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite three-dimensional array')
    _positive(tau, 'tau')
    _controls(tol, max_iter)
    b, k, d = x.shape
    initial = _initial(initial, (b, d))
    v = x.mean(1) if initial is None else initial
    counts = np.zeros(b, int)
    for it in range(max_iter):
        score, A, obj, _, _ = pseudo_components(x, v[:, None, :], tau)
        norms = np.linalg.norm(score, axis=1)
        active = norms > tol * max(1., tau)
        if not active.any():
            break
        counts[active] += 1
        step = np.linalg.solve(A, score[..., None])[..., 0]
        rate = np.ones(b)
        candidate = v + step
        for _ in range(45):
            newobj = pseudo_components(x, candidate[:, None, :], tau)[2]
            bad = (newobj > obj - 1e-4 * rate * np.sum(score * step, axis=1)) & active
            # At roundoff objective resolution, accept a score-improving step.
            newscore = np.linalg.norm(pseudo_components(x, candidate[:, None, :], tau)[0], axis=1)
            bad &= newscore >= norms
            if not bad.any():
                break
            rate[bad] *= .5
            candidate[bad] = v[bad] + rate[bad, None] * step[bad]
        v[active] = candidate[active]
    score, A, obj, r, w = pseudo_components(x, v[:, None, :], tau)
    norms = np.linalg.norm(score, axis=1)
    hullD = 2 * np.linalg.norm(x - x.mean(1)[:, None, :], axis=2).max(1)
    D = np.maximum(hullD, np.linalg.norm(r, axis=2).max(1))
    floor = (1 + (D / tau)**2)**(-1.5)
    cert = norms / floor
    ok = norms <= tol * max(1., tau)
    return v, {'iterations': counts, 'score': norms, 'certificate': cert, 'converged': ok, 'hessian_condition': np.linalg.cond(A)}


def pseudo_huber(x, tau=1., tol=1e-12, max_iter=100, initial=None):
    """Aggregate subset estimates using a fixed pseudo-Huber threshold.

    ``x`` has shape (subsets, coordinates), or is a scalar vector. Scalar fits
    use a bracketed root; ``max_iter`` controls the multivariate Newton solver.
    The threshold is in input units. Robust aggregation alone does not ensure
    valid inference for a population mean; see the manuscript's conditions.
    """
    x = points(x)
    _positive(tau, 'tau')
    _controls(tol, max_iter)
    initial = _initial(initial, (x.shape[1],))
    if x.shape[1] == 1:
        lo = float(x.min())
        hi = float(x.max())
        if lo == hi:
            v = lo
        else:
            v = brentq(lambda z: np.mean((x[:, 0] - z) / np.hypot(1., (x[:, 0] - z) / tau)), lo, hi, xtol=1e-14, rtol=1e-14)
        r = x[:, 0] - v
        s = abs(np.mean(r / np.hypot(1, r / tau)))
        D = hi - lo
        return Fit(np.array([v]), 0, s, bool(s <= tol * max(1., tau)), s * (1 + (D / tau)**2)**1.5, threshold=tau)
    v, diagnostics = pseudo_batch(x[None], tau, tol, max_iter, None if initial is None else initial[None])
    return Fit(v[0], int(diagnostics['iterations'][0]), float(diagnostics['score'][0]), bool(diagnostics['converged'][0]), float(diagnostics['certificate'][0]), threshold=tau)


def canonical_huber(x, tau=1., tol=1e-12, max_iter=20000, initial=None):
    """Aggregate subset estimates using the radial canonical Huber loss."""
    x = points(x)
    _positive(tau, 'tau')
    _controls(tol, max_iter)
    initial = _initial(initial, (x.shape[1],))
    v = x.mean(0) if initial is None else initial
    for it in range(max_iter):
        r = x - v
        dist = np.linalg.norm(r, axis=1)
        w = np.ones(len(x))
        large = dist > tau
        w[large] = tau / dist[large]
        score = (w[:, None] * r).mean(0)
        sn = np.linalg.norm(score)
        if sn <= tol * max(1., tau):
            return Fit(v, it, sn, True, threshold=tau)
        v = np.sum(w[:, None] * x, axis=0) / w.sum()
    r = x - v
    dist = np.linalg.norm(r, axis=1)
    w = np.ones(len(x))
    large = dist > tau
    w[large] = tau / dist[large]
    sn = np.linalg.norm((w[:, None] * r).mean(0))
    return Fit(v, max_iter, sn, False, threshold=tau)


def adaptive(x, c=2., loss='pseudo', preliminary=None):
    """Fit HOMER using a threshold of ``c`` times median radial deviation.

    A geometric median supplies the preliminary center. ``loss`` is ``pseudo``
    or ``canonical``. A zero median radius returns the preliminary center
    exactly. A supplied ``preliminary`` must be a converged ``Fit`` computed
    from the same input estimates. This adaptive fit is for point estimation;
    the fixed-threshold mean-inference theorem does not cover adaptive tuning.
    """
    x = points(x)
    _positive(c, 'c')
    if loss not in ('pseudo', 'canonical'):
        raise ValueError("loss must be 'pseudo' or 'canonical'")
    g = geometric_median(x) if preliminary is None else preliminary
    if not isinstance(g, Fit):
        raise TypeError('preliminary must be a Fit')
    _initial(g.center, (x.shape[1],))
    if not g.converged:
        raise RuntimeError('Preliminary geometric median did not converge')
    scale = float(np.median(np.linalg.norm(x - g.center, axis=1)))
    if scale == 0:
        return Fit(g.center.copy(), g.iterations, g.score_norm, g.converged, branch='zero_scale', threshold=0., gap_bound=g.gap_bound)
    f = (pseudo_huber if loss == 'pseudo' else canonical_huber)(x, c * scale, initial=g.center)
    f.branch = 'positive_scale'
    return f


def projected_sandwich(x, v, tau, H):
    """Return ``(V, diagnostics)`` for fixed-threshold pseudo-Huber inference.

    Rows of ``H`` are linear contrasts, and ``v`` is the fitted center. With
    ``k = len(x)`` independent subset estimates, the estimated covariance of
    ``H @ v`` is ``V / k``, in the input units. This does not remove the bias
    between the block M-target and the population mean. Singular projections
    remain singular and are flagged; no ridge or eigenvalue floor is applied.
    """
    x = points(x)
    _positive(tau, 'tau')
    v = _initial(v, (x.shape[1],))
    if v is None:
        raise ValueError('v must be a finite fitted center')
    H = np.atleast_2d(np.asarray(H, float))
    if H.ndim != 2 or 0 in H.shape or H.shape[1] != x.shape[1] or not np.isfinite(H).all():
        raise ValueError('H must be a nonempty finite contrast matrix with one column per coordinate')
    r = x - v
    w = 1 / np.hypot(1., np.linalg.norm(r, axis=1) / tau)
    A = w.mean() * np.eye(x.shape[1]) - np.einsum('ki,kj,k->ij', r, r, w**3) / (len(x) * tau**2)
    condition = float(np.linalg.cond(A))
    if not np.isfinite(condition) or condition > 1e12:
        raise np.linalg.LinAlgError('Ill-conditioned Hessian')
    q = np.linalg.solve(A, H.T)
    s = (r * w[:, None]) @ q
    denominator = np.linalg.norm(A) * np.linalg.norm(q) + np.linalg.norm(H)
    solve_residual = np.linalg.norm(A @ q - H.T) / denominator if denominator else 0.
    V = s.T @ s / len(x)
    e = np.linalg.eigvalsh(V)
    singular = e.min() <= 1e-12 * max(1., e.max())
    return V, {'hessian_condition': condition, 'projected_eigen_min': float(e.min()), 'singular': bool(singular), 'relative_solve_residual': float(solve_residual)}


def exact_binomial_target(m, p, lam=1.):
    """Evaluate the standardized Bernoulli block M-target by finite summation.

    Here ``m`` is subset size, ``p`` the Bernoulli success probability, and
    ``lam`` the fixed threshold for standardized block means. Every one of the
    ``m + 1`` support points is retained, subject to floating-point underflow.
    """
    if isinstance(m, bool) or not isinstance(m, Integral) or m < 1:
        raise ValueError('m must be a positive integer')
    if np.ndim(p) != 0 or not np.isfinite(p) or not 0 < p < 1:
        raise ValueError('p must lie strictly between zero and one')
    _positive(lam, 'lam')
    j = np.arange(m + 1)
    y = (j - m * p) / np.sqrt(m * p * (1 - p))
    prob = binom.pmf(j, m, p)
    mass = prob.sum()
    prob = prob / mass
    f = lambda u: np.dot(prob, (y - u) / np.hypot(1., (y - u) / lam))
    u = brentq(f, y[0], y[-1], xtol=2e-14, rtol=2e-14)
    r = y - u
    w = 1 / np.hypot(1., r / lam)
    A = np.dot(prob, w**3)
    B = np.dot(prob, (r * w)**2)
    return {'m': m, 'p': p, 'lambda': lam, 'u': u, 'theta': u / np.sqrt(m), 'A': A, 'B': B, 'V': B / A**2, 'score_residual': abs(f(u)), 'pmf_mass_error': abs(mass - 1), 'underflowed_terms': int((prob == 0).sum()), 'target_type': 'finite_sum_all_m_plus_1_terms'}
