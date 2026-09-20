"""The reference applicant grid.

A stand-in for the applicant population the app will actually serve, shared
by 06_bands.py and 07_population_shift.py so the band cutoffs and the
population shift are derived from the same distribution.

Replace this with the real marginal distribution as soon as you have one — a
pull of accepted education-loan applications is enough. Until then it is an
assumption, and it is labelled as one wherever it is used.
"""

from __future__ import annotations

import math
from itertools import product

# Comments on the individual rows explain the India-specific choices.
REFERENCE_GRID = {
    "fico_avg": [680, 710, 740, 770],          # CIBIL 700-800 rescaled
    "log_annual_inc": [math.log(v) for v in (45_000, 65_000, 95_000)],
    "dti": [8, 18, 30],
    "emp_length_years": [2, 5, 9],
    "home_rent": [0.0, 1.0],
    "home_other": [0.0],
    "revol_util": [15, 40, 70],
    "open_acc": [3, 6, 10],
    "delinq_2yrs": [0.0],
    "inq_last_6mths": [0.0, 2.0],
    "pub_rec": [0.0],
    "log_loan_amnt": [math.log(v) for v in (25_000, 50_000, 80_000)],
    "term_60": [1.0],                           # always, in this population
    "purpose_high_risk": [0.0],
    "purpose_low_risk": [0.0],
    "loan_to_income": [0.3, 0.6, 1.0],
}


def grid_rows():
    names = list(REFERENCE_GRID)
    for values in product(*(REFERENCE_GRID[n] for n in names)):
        yield dict(zip(names, values))


