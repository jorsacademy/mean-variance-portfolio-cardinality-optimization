"""Mean-variance portfolio optimization with a cardinality constraint.

This module reproduces the structure of a small quadratic portfolio model.
The continuous mean-variance problem is solved with CVXPY, while the
cardinality-constrained problem is solved exactly by enumerating feasible
asset subsets and solving one convex quadratic program per subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd


ASSETS = ("hardware", "software", "entertainment", "treasury_bills")
TARGET_RETURN = 10.0
MAX_ASSETS = 3

MEAN_RETURNS = np.array([8.0, 9.0, 12.0, 7.0], dtype=float)
COVARIANCE = np.array(
    [
        [4.0, 3.0, -1.0, 0.0],
        [3.0, 6.0, 1.0, 0.0],
        [-1.0, 1.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PortfolioSolution:
    """Container for one portfolio optimization result."""

    status: str
    variance: float
    expected_return: float
    weights: dict[str, float]
    active_assets: tuple[str, ...]


def validate_covariance_matrix(matrix: np.ndarray, tolerance: float = 1e-10) -> None:
    """Validate symmetry and positive semidefiniteness of a covariance matrix."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -tolerance:
        raise ValueError(
            "Covariance matrix must be positive semidefinite for this convex model."
        )


def solve_mean_variance(
    allowed_assets: Iterable[int] | None = None,
    target_return: float = TARGET_RETURN,
) -> PortfolioSolution:
    """Solve the convex mean-variance problem for a specified asset subset.

    Parameters
    ----------
    allowed_assets:
        Indices of assets that may receive positive weight. If omitted, all
        assets are allowed.
    target_return:
        Required portfolio expected return, expressed in the same percentage
        units as ``MEAN_RETURNS``.
    """
    validate_covariance_matrix(COVARIANCE)

    n_assets = len(ASSETS)
    allowed = set(range(n_assets)) if allowed_assets is None else set(allowed_assets)

    weights = cp.Variable(n_assets, nonneg=True)
    objective = cp.Minimize(cp.quad_form(weights, COVARIANCE))

    constraints = [
        cp.sum(weights) == 1.0,
        MEAN_RETURNS @ weights == target_return,
    ]

    for index in range(n_assets):
        if index not in allowed:
            constraints.append(weights[index] == 0.0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(f"No feasible portfolio found. Solver status: {problem.status}")

    raw_weights = np.asarray(weights.value, dtype=float).reshape(-1)
    raw_weights[np.abs(raw_weights) < 1e-9] = 0.0

    expected_return = float(MEAN_RETURNS @ raw_weights)
    variance = float(raw_weights @ COVARIANCE @ raw_weights)
    active_assets = tuple(
        ASSETS[index] for index, value in enumerate(raw_weights) if value > 1e-8
    )

    return PortfolioSolution(
        status=problem.status,
        variance=variance,
        expected_return=expected_return,
        weights={asset: float(raw_weights[i]) for i, asset in enumerate(ASSETS)},
        active_assets=active_assets,
    )


def enumerate_cardinality_constrained(
    max_assets: int = MAX_ASSETS,
    target_return: float = TARGET_RETURN,
) -> tuple[PortfolioSolution, pd.DataFrame]:
    """Solve the cardinality-constrained model by complete enumeration.

    Every non-empty asset subset with size at most ``max_assets`` is tested.
    For each subset, a convex quadratic program is solved. The feasible
    solution with minimum variance is returned together with a detailed table.
    """
    if max_assets < 1 or max_assets > len(ASSETS):
        raise ValueError("max_assets must be between 1 and the number of assets.")

    rows: list[dict[str, object]] = []
    best_solution: PortfolioSolution | None = None

    for subset_size in range(1, max_assets + 1):
        for subset in combinations(range(len(ASSETS)), subset_size):
            subset_names = tuple(ASSETS[index] for index in subset)

            try:
                solution = solve_mean_variance(subset, target_return)
                feasible = True
                variance = solution.variance

                if best_solution is None or variance < best_solution.variance - 1e-10:
                    best_solution = solution
            except ValueError:
                solution = None
                feasible = False
                variance = np.nan

            row: dict[str, object] = {
                "subset": ", ".join(subset_names),
                "asset_count": subset_size,
                "feasible": feasible,
                "variance": variance,
            }

            for asset in ASSETS:
                row[f"weight_{asset}"] = (
                    solution.weights[asset] if solution is not None else np.nan
                )

            rows.append(row)

    if best_solution is None:
        raise ValueError("No feasible portfolio satisfies the cardinality constraint.")

    results = pd.DataFrame(rows).sort_values(
        by=["feasible", "variance"], ascending=[False, True], na_position="last"
    )
    return best_solution, results.reset_index(drop=True)


def print_solution(title: str, solution: PortfolioSolution) -> None:
    """Print a compact human-readable solution summary."""
    print(title)
    print("-" * len(title))
    for asset, weight in solution.weights.items():
        print(f"{asset:16s}: {weight:.6f}")
    print(f"Expected return : {solution.expected_return:.6f}")
    print(f"Variance        : {solution.variance:.6f}")
    print(f"Active assets   : {', '.join(solution.active_assets)}")
    print()


def main() -> None:
    """Run both the continuous and cardinality-constrained portfolio models."""
    continuous_solution = solve_mean_variance()
    constrained_solution, enumeration_results = enumerate_cardinality_constrained()

    print_solution("Continuous mean-variance solution", continuous_solution)
    print_solution(
        f"Cardinality-constrained solution (max {MAX_ASSETS} assets)",
        constrained_solution,
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "enumeration_results.csv"
    enumeration_results.to_csv(output_path, index=False)
    print(f"Enumeration results written to: {output_path}")


if __name__ == "__main__":
    main()
