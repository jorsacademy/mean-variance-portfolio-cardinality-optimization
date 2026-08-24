# Mean-Variance Portfolio Cardinality Optimization

A small quadratic portfolio optimization example implemented in Python with CVXPY.

The project solves two related models:

1. A continuous mean-variance portfolio problem.
2. A cardinality-constrained portfolio problem in which at most three assets may be selected.

The cardinality-constrained model is solved exactly by complete enumeration of feasible asset subsets. Each subset produces a convex quadratic program, and the feasible solution with the minimum portfolio variance is selected.

## Mathematical Model

Let:

- `x_i` be the fraction of the portfolio invested in asset `i`.
- `mu_i` be the expected return of asset `i`.
- `V` be the covariance matrix.
- `R` be the required portfolio return.

The continuous model is:

```text
minimize    x' V x

subject to  sum_i x_i = 1
            sum_i mu_i x_i = R
            x_i >= 0
```

The cardinality-constrained version additionally requires that no more than a specified number of assets receive positive allocation.

For this small four-asset instance, complete enumeration is simple and provides an exact solution without requiring a mixed-integer nonlinear solver.

## Data

The example uses four generic asset classes:

| Asset | Mean Return |
| --- | ---: |
| hardware | 8% |
| software | 9% |
| entertainment | 12% |
| treasury_bills | 7% |

The target portfolio return is 10%.

The covariance matrix is:

```text
                 hardware  software  entertainment  treasury_bills
hardware              4         3            -1               0
software              3         6             1               0
entertainment         -1         1            10               0
treasury_bills         0         0             0               0
```

## Expected Results

The unconstrained continuous solution is approximately:

```text
hardware        0.302885
software        0.086538
entertainment   0.504808
treasury_bills  0.105769

Expected return  10.000000
Variance          2.899038
```

With a maximum of three active assets, the optimal solution is approximately:

```text
hardware        0.375000
software        0.000000
entertainment   0.525000
treasury_bills  0.100000

Expected return  10.000000
Variance          2.925000
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run:

```bash
python portfolio_optimization.py
```

The script prints the continuous and cardinality-constrained solutions and writes the subset enumeration table to:

```text
results/enumeration_results.csv
```

## Implementation Notes

The portfolio variance is genuinely quadratic:

```text
x' V x
```

For this reason, the implementation uses CVXPY rather than a linear-programming-only modeling library. CVXPY represents the variance directly with `quad_form` and solves the resulting convex quadratic programs with Clarabel.

The covariance matrix is checked for symmetry and positive semidefiniteness before optimization.

## Project Structure

```text
.
├── portfolio_optimization.py
├── requirements.txt
├── results/
│   └── enumeration_results.csv
├── LICENSE.md
├── .gitignore
└── README.md
```

## License

This repository is source-available for educational, academic, research, and other non-commercial purposes only.

Commercial use is prohibited. See [LICENSE.md](LICENSE.md) for the full terms.
