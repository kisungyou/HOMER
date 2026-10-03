"""Independent 60-digit finite-sum audit with enforced production agreement."""
import argparse
from pathlib import Path
import mpmath as mp
import pandas as pd
from verify_population_targets import (
    assert_production_agreement, PRODUCTION_ATOL, PRODUCTION_RTOL, TARGET_FIELDS,
)


def audit(input_path, output_path):
    rows = []
    failures = []
    original = pd.read_csv(input_path).query("distribution=='bernoulli'")
    if original.empty:
        raise ValueError('No Bernoulli targets were found to audit.')
    for _, group in original.groupby(['m', 'p', 'lambda']):
        if any(group[name].nunique(dropna=False) != 1 for name in TARGET_FIELDS):
            raise ArithmeticError('Repeated target rows disagree.')
    with mp.workdps(60):
        for item in original.drop_duplicates(['m', 'p', 'lambda']).to_dict('records'):
            m = int(item['m'])
            p = mp.mpf(str(item['p']))
            lam = mp.mpf(str(item['lambda']))
            weights = [(1-p)**m]
            for j in range(m):
                weights.append(weights[-1]*(m-j)*p/((j+1)*(1-p)))
            y = [(j-m*p)/mp.sqrt(m*p*(1-p)) for j in range(m+1)]

            def score(u):
                return mp.fsum(w*(t-u)/mp.sqrt(1+((t-u)/lam)**2) for w,t in zip(weights,y))

            def hessian(u):
                return mp.fsum(w*(1+((t-u)/lam)**2)**(-mp.mpf('1.5')) for w,t in zip(weights,y))

            u = mp.mpf(str(item['u']))
            for _ in range(4):
                u += score(u)/hessian(u)
            A = hessian(u)
            B = mp.fsum(w*(t-u)**2/(1+((t-u)/lam)**2) for w,t in zip(weights,y))
            V = B/A**2
            reference = {'u': u, 'theta': u/mp.sqrt(m), 'A': A, 'B': B, 'V': V}
            row = {'m':m, 'p':float(p), 'lambda':float(lam),
                   'u_60digits':str(u), 'V_60digits':str(V),
                   'absolute_u_difference':abs(float(u)-item['u']),
                   'absolute_V_difference':abs(float(V)-item['V']),
                   'high_precision_score_residual':str(abs(score(u))),
                   'high_precision_probability_sum_error':str(abs(mp.fsum(weights)-1)),
                   'digits':60}
            for name in ('theta', 'A', 'B'):
                row[f'{name}_60digits'] = str(reference[name])
                row[f'absolute_{name}_difference'] = abs(float(reference[name])-item[name])
            row.update(production_atol=str(PRODUCTION_ATOL), production_rtol=str(PRODUCTION_RTOL))
            try:
                assert_production_agreement(item, {name:str(value) for name,value in reference.items()})
                row.update(production_agreement=True, production_audit_error='')
            except ArithmeticError as error:
                row.update(production_agreement=False, production_audit_error=str(error))
                failures.append(f'm={m}, p={p}, lambda={lam}: {error}')
            rows.append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(result[[f'absolute_{name}_difference' for name in TARGET_FIELDS]].max().to_string())
    if failures:
        raise ArithmeticError('\n'.join(failures) + f'\nFull diagnostics saved to {output_path}')
    print(f'Verified production agreement for {len(rows)} distinct Bernoulli targets.')
    return result


def main():
    from settings import RESULTS, GENERATED_RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=RESULTS/'scalar_targets.csv')
    parser.add_argument('--output', type=Path, default=GENERATED_RESULTS/'scalar_target_precision_audit.csv')
    args = parser.parse_args()
    audit(args.input, args.output)


if __name__ == '__main__':
    main()
