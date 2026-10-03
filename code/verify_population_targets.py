"""Independent high-precision verification of the reported Bernoulli targets.

The reference calculation needs only Python's standard library. Binomial
probabilities are represented by exact integers over a common denominator;
all m + 1 support points enter every score and sandwich sum. Decimal Newton
iterations at two precisions are checked against a directed-rounding root
bracket. SciPy is optional and used only to identify production PMF zeros,
never to evaluate the reference PMF, root, or sandwich.

Run from any directory:
    python3 verify_population_targets.py
    python3 verify_population_targets.py --scipy-zero-mask
"""

from __future__ import annotations

import argparse
import csv
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from pathlib import Path
import sys


def exact_binomial_numerators(m: int, p: Decimal) -> tuple[list[int], int]:
    """Return exact PMF numerators and their common denominator."""
    rational = Fraction(p)
    a, d = rational.numerator, rational.denominator
    b = d - a
    if not (0 < a < d):
        raise ValueError("The Bernoulli probability must belong to (0, 1).")
    denominator = d**m
    numerators = [b**m]
    for j in range(m):
        following, remainder = divmod(numerators[-1] * (m - j) * a, (j + 1) * b)
        if remainder:
            raise ArithmeticError("The exact binomial recurrence was not integral.")
        numerators.append(following)
    if sum(numerators) != denominator:
        raise ArithmeticError("The exact full-support probabilities do not sum to one.")
    return numerators, denominator


def decimal_reference(m, p, lam, numerators, denominator, precision):
    with localcontext() as context:
        context.prec = precision
        root_m = Decimal(m).sqrt()
        scale = (Decimal(m) * p * (1 - p)).sqrt()
        support = [(Decimal(j) - m * p) / scale for j in range(m + 1)]
        probabilities = [Decimal(z) / Decimal(denominator) for z in numerators]

        def evaluate(u):
            score = A = B = Decimal(0)
            for probability, y in zip(probabilities, support):
                residual = y - u
                weight = 1 / (1 + (residual / lam) ** 2).sqrt()
                phi = residual * weight
                score += probability * phi
                A += probability * weight**3
                B += probability * phi**2
            return score, A, B

        lower, upper, u = support[0], support[-1], Decimal(0)
        tolerance = Decimal(10) ** (-(precision - 15))
        for iteration in range(100):
            score, A, B = evaluate(u)
            if score > 0:
                lower = u
            else:
                upper = u
            candidate = u + score / A
            if abs(candidate - u) < tolerance:
                u = candidate
                break
            if not lower < candidate < upper:
                candidate = (lower + upper) / 2
            u = candidate
        else:
            raise ArithmeticError("Safeguarded Decimal Newton iterations did not converge.")
        score, A, B = evaluate(u)
        return {
            "u": +u, "theta": +(u / root_m), "A": +A, "B": +B,
            "V": +(B / A**2), "score_residual": abs(score),
            "iterations": iteration + 1,
            "decimal_pmf_mass_error": abs(sum(probabilities) - 1),
        }


class IntervalArithmetic:
    """Outward-rounded Decimal interval operations for root verification."""

    def __init__(self, precision):
        self.down = Context(prec=precision, rounding=ROUND_FLOOR)
        self.up = Context(prec=precision, rounding=ROUND_CEILING)

    @staticmethod
    def point(x):
        value = Decimal(x)
        return value, value

    def add(self, x, y):
        return self.down.add(x[0], y[0]), self.up.add(x[1], y[1])

    def subtract(self, x, y):
        return self.down.subtract(x[0], y[1]), self.up.subtract(x[1], y[0])

    def multiply(self, x, y):
        lower = min(self.down.multiply(a, b) for a in x for b in y)
        upper = max(self.up.multiply(a, b) for a in x for b in y)
        return lower, upper

    def divide(self, x, y):
        if y[0] <= 0 <= y[1]:
            raise ZeroDivisionError("An interval denominator contains zero.")
        reciprocal = self.down.divide(1, y[1]), self.up.divide(1, y[0])
        return self.multiply(x, reciprocal)

    def square(self, x):
        lower = Decimal(0) if x[0] <= 0 <= x[1] else min(
            self.down.multiply(z, z) for z in x
        )
        upper = max(self.up.multiply(z, z) for z in x)
        return lower, upper

    def sqrt(self, x):
        if x[0] < 0:
            raise ValueError("A square-root interval has a negative endpoint.")
        # Decimal sqrt is correctly rounded to nearest. Expanding each endpoint
        # by one representable number encloses its exact real square root.
        lower = max(Decimal(0), self.down.next_minus(self.down.sqrt(x[0])))
        upper = self.up.next_plus(self.up.sqrt(x[1]))
        return lower, upper


def certified_root_bracket(m, p, lam, numerators, denominator, u, precision):
    interval = IntervalArithmetic(precision)
    point = interval.point
    p_interval = point(p)
    scale = interval.sqrt(interval.multiply(
        point(m), interval.multiply(p_interval, interval.subtract(point(1), p_interval))
    ))
    mp = interval.multiply(point(m), p_interval)
    support = [interval.divide(interval.subtract(point(j), mp), scale) for j in range(m + 1)]
    probabilities = [interval.divide(point(z), point(denominator)) for z in numerators]
    lam_interval = point(lam)
    lam_squared = interval.square(lam_interval)

    def score_at(candidate, with_sandwich=False):
        score = point(0)
        A = B = point(0)
        for probability, y in zip(probabilities, support):
            residual = interval.subtract(y, point(candidate))
            denom = interval.sqrt(interval.add(lam_squared, interval.square(residual)))
            phi = interval.divide(interval.multiply(lam_interval, residual), denom)
            score = interval.add(score, interval.multiply(probability, phi))
            if with_sandwich:
                weight = interval.divide(lam_interval, denom)
                cube = interval.multiply(interval.square(weight), weight)
                A = interval.add(A, interval.multiply(probability, cube))
                B = interval.add(B, interval.multiply(probability, interval.square(phi)))
        return (score, A, B) if with_sandwich else score

    with localcontext() as context:
        context.prec = precision
        radius = Decimal(10) ** (-(precision - 20))
        lower, upper = u - radius, u + radius
    left_score, right_score = score_at(lower), score_at(upper)
    if not (left_score[0] > 0 and right_score[1] < 0):
        raise ArithmeticError("Directed-rounding scores did not certify the root bracket.")
    root_error = max(interval.up.subtract(u, lower), interval.up.subtract(upper, u))
    theta_error = interval.up.divide(root_error, interval.sqrt(point(m))[0])
    _, A_at_u, B_at_u = score_at(u, with_sandwich=True)
    # The scalar Hessian is 6/lambda Lipschitz. Squared scores are
    # 2 lambda Lipschitz because |phi| <= lambda and |phi'| <= 1.
    A_error = interval.up.multiply(interval.up.divide(6, lam), root_error)
    B_error = interval.up.multiply(interval.up.multiply(2, lam), root_error)
    A_true = interval.add(A_at_u, (A_error.copy_negate(), A_error))
    B_true = interval.add(B_at_u, (B_error.copy_negate(), B_error))
    V_true = interval.divide(B_true, interval.square(A_true))
    return {
        "root_bracket_lower": lower, "root_bracket_upper": upper,
        "left_score_lower": left_score[0], "right_score_upper": right_score[1],
        "certified_u_error_bound": root_error,
        "certified_theta_error_bound": theta_error,
        "certified_A_lower": A_true[0], "certified_A_upper": A_true[1],
        "certified_B_lower": B_true[0], "certified_B_upper": B_true[1],
        "certified_V_lower": V_true[0], "certified_V_upper": V_true[1],
    }


def upper_mass(numerators, denominator, indices, precision):
    numerator = sum(numerators[j] for j in indices)
    return Context(prec=precision, rounding=ROUND_CEILING).divide(
        Decimal(numerator), Decimal(denominator)
    )


# Production uses float64 sums and a root tolerance of 2e-14. These tolerances
# allow conservative cross-platform rounding accumulation while detecting
# discrepancies many orders smaller than the simulation's Monte Carlo error.
# The archived differences are below 2e-15; no tolerance is fitted per target.
PRODUCTION_ATOL = Decimal("1e-12")
PRODUCTION_RTOL = Decimal("1e-10")
TARGET_FIELDS = ("u", "theta", "A", "B", "V")


def assert_production_agreement(production, reference):
    """Raise when any reported float64 target disagrees with an independent value.

    This helper only compares numbers; it does not construct either reference.
    Shared use by the Decimal and mpmath audits preserves their independent
    numerical calculations.
    """
    failures = []
    with localcontext() as context:
        context.prec = 100
        for name in TARGET_FIELDS:
            actual = Decimal(str(production[name]))
            expected = Decimal(str(reference[name]))
            if not actual.is_finite() or not expected.is_finite():
                failures.append(f"{name}: nonfinite production or reference value")
                continue
            error = abs(actual - expected)
            limit = PRODUCTION_ATOL + PRODUCTION_RTOL * abs(expected)
            if error > limit:
                failures.append(f"{name}: absolute difference {error:.6E} exceeds {limit:.6E}")
    if failures:
        raise ArithmeticError("Production target audit failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from settings import RESULTS, GENERATED_RESULTS
    parser.add_argument("--input", type=Path, default=RESULTS / "scalar_targets.csv")
    parser.add_argument("--output", type=Path, default=GENERATED_RESULTS / "population_target_crosscheck.csv")
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--scipy-zero-mask", action="store_true")
    args = parser.parse_args()
    if args.precision < 60:
        parser.error("Use at least 60 digits for the independent reference.")
    scipy_binom = None
    if args.scipy_zero_mask:
        from scipy.stats import binom as scipy_binom

    with args.input.open(newline="") as stream:
        original = list(csv.DictReader(stream))
    distinct = {}
    for row in original:
        if row["distribution"] == "bernoulli":
            key = int(row["m"]), Decimal(row["p"]), Decimal(row["lambda"])
            if key in distinct:
                for name in ("u", "theta", "A", "B", "V"):
                    if row[name] != distinct[key][name]:
                        raise ArithmeticError("Repeated target rows disagree.")
            else:
                distinct[key] = row

    if not distinct:
        raise ValueError("No Bernoulli targets were found to audit.")
    outputs = []
    failures = []
    for (m, p, lam), production in sorted(distinct.items()):
        numerators, denominator = exact_binomial_numerators(m, p)
        ref = decimal_reference(m, p, lam, numerators, denominator, args.precision)
        ref_higher = decimal_reference(m, p, lam, numerators, denominator, args.precision + 20)
        certificate = certified_root_bracket(m, p, lam, numerators, denominator, ref["u"], args.precision)
        result = {"m": m, "p": p, "lambda": lam, "precision": args.precision,
                  "higher_precision": args.precision + 20, "full_support_terms": m + 1,
                  "exact_pmf_mass_error": 0, "reported_underflowed_terms": production["underflowed_terms"]}
        with localcontext() as context:
            context.prec = args.precision + 20
            for name in ("u", "theta", "A", "B", "V"):
                result[f"reported_{name}"] = production[name]
                result[f"decimal_{name}"] = ref[name]
                result[f"abs_difference_{name}"] = abs(Decimal(production[name]) - ref[name])
                result[f"precision_difference_{name}"] = abs(ref[name] - ref_higher[name])
        result.update({name: value for name, value in ref.items() if name not in ("u", "theta", "A", "B", "V")})
        result.update(certificate)
        result["production_atol"] = PRODUCTION_ATOL
        result["production_rtol"] = PRODUCTION_RTOL
        try:
            assert_production_agreement(production, ref)
            result["production_agreement"] = True
            result["production_audit_error"] = ""
        except ArithmeticError as error:
            result["production_agreement"] = False
            result["production_audit_error"] = str(error)
            failures.append(f"m={m}, p={p}, lambda={lam}: {error}")
        with localcontext() as context:
            context.prec = args.precision
            rounded_zero = [j for j, value in enumerate(numerators)
                            if float(Decimal(value) / Decimal(denominator)) == 0.0]
        result["correct_rounding_zero_terms"] = len(rounded_zero)
        result["correct_rounding_zero_mass_upper"] = upper_mass(numerators, denominator, rounded_zero, args.precision)
        normal_threshold = Fraction(Decimal.from_float(sys.float_info.min))
        subnormal = [j for j, value in enumerate(numerators)
                     if value * normal_threshold.denominator < denominator * normal_threshold.numerator]
        result["below_smallest_normal_mass_upper"] = upper_mass(numerators, denominator, subnormal, args.precision)
        if scipy_binom is not None:
            pmf = scipy_binom.pmf(list(range(m + 1)), m, float(p))
            zeros = [j for j, value in enumerate(pmf) if value == 0.0]
            if len(zeros) != int(production["underflowed_terms"]):
                raise ArithmeticError("The reproduced production PMF zero count differs from the archived run.")
            result["production_zero_mask_source"] = "scipy.stats.binom.pmf; mask only"
            result["production_zero_terms"] = len(zeros)
            result["production_zero_mass_upper"] = upper_mass(numerators, denominator, zeros, args.precision)
        else:
            result["production_zero_mask_source"] = "not_requested"
            result["production_zero_terms"] = ""
            result["production_zero_mass_upper"] = ""
        outputs.append(result)
        print(f"m={m} p={p} lambda={lam}: |du|={result['abs_difference_u']:.3E}, "
              f"|dV|={result['abs_difference_V']:.3E}; root bracket certified", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
    for name in ("u", "theta", "A", "B", "V"):
        print(f"max |difference {name}| = {max(row[f'abs_difference_{name}'] for row in outputs):.9E}")
    if failures:
        raise ArithmeticError("\n".join(failures) + f"\nFull diagnostics saved to {args.output}")
    print(f"Verified {len(outputs)} distinct Bernoulli targets and production agreement; saved {args.output}")


if __name__ == "__main__":
    main()
