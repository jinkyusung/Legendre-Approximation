#!/usr/bin/env python3
"""Legendre approximation experiment for a d-dimensional Wiener process.

For each coordinate j, this script projects white noise onto shifted Legendre
polynomials,

    c[k, j] = integral_0^T P_k(2t/T - 1) dW_j(t),

and reconstructs the Wiener path by analytically integrating the truncated
white-noise expansion.  It checks four properties:

1. the empirical path RMSE agrees with the exact truncation RMSE;
2. the RMSE decays at the expected O(K^{-1/2}) rate;
3. every reconstructed coordinate preserves the terminal increment;
4. standardized coefficients have unit variance and remain independent across
   Wiener dimensions.

Run the default experiment with:

    python test.py

or choose a different Wiener dimension:

    python test.py --dim 16 --degrees 1 2 4 8 16 32 64 128
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


DEFAULT_DEGREES = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class DegreeMetrics:
    """Approximation diagnostics for one truncation degree."""

    degree: int
    rmse: float
    theoretical_rmse: float
    relative_rmse: float
    endpoint_max_error: float
    per_dimension_rmse: torch.Tensor


@dataclass(frozen=True)
class ExperimentResult:
    """All numerical results needed for reporting, checking, and plotting."""

    dimension: int
    num_paths: int
    num_steps: int
    horizon: float
    metrics: tuple[DegreeMetrics, ...]
    fitted_slope: float
    theoretical_slope: float
    coefficient_mean: float
    coefficient_std: float
    normalized_coefficient_variance: torch.Tensor
    dimension_correlation: torch.Tensor
    sample_time: torch.Tensor
    sample_path: torch.Tensor
    sample_approximations: dict[int, torch.Tensor]


def evaluate_legendre(x: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Evaluate P_0, ..., P_max_degree at x using Bonnet's recurrence."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    values = torch.empty(
        (*x.shape, max_degree + 1),
        dtype=x.dtype,
        device=x.device,
    )
    values[..., 0] = 1.0

    if max_degree >= 1:
        values[..., 1] = x

    for k in range(1, max_degree):
        values[..., k + 1] = (
            (2 * k + 1) * x * values[..., k] - k * values[..., k - 1]
        ) / (k + 1)

    return values


def simulate_wiener_paths(
    *,
    num_paths: int,
    num_steps: int,
    dimension: int,
    horizon: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate independent d-dimensional Wiener paths on a uniform grid."""
    if num_paths <= 0 or num_steps <= 0 or dimension <= 0:
        raise ValueError("num_paths, num_steps, and dimension must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    dtype = torch.float64
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    time = torch.linspace(
        0.0,
        horizon,
        num_steps + 1,
        dtype=dtype,
        device=device,
    )
    dt = horizon / num_steps
    increments = torch.randn(
        num_paths,
        num_steps,
        dimension,
        dtype=dtype,
        device=device,
        generator=generator,
    ) * (dt**0.5)

    initial_value = torch.zeros(
        num_paths,
        1,
        dimension,
        dtype=dtype,
        device=device,
    )
    paths = torch.cat((initial_value, increments.cumsum(dim=1)), dim=1)
    return time, increments, paths


def project_increments(
    increments: torch.Tensor,
    time: torch.Tensor,
    max_degree: int,
) -> torch.Tensor:
    """Approximate all Legendre-Ito coefficients with deterministic integrands."""
    num_paths, num_steps, dimension = increments.shape
    horizon = time[-1] - time[0]
    z_left = 2.0 * (time[:-1] - time[0]) / horizon - 1.0
    basis = evaluate_legendre(z_left, max_degree - 1)

    # One matrix multiplication projects every path and every Wiener coordinate.
    flat_increments = increments.permute(1, 0, 2).reshape(
        num_steps, num_paths * dimension
    )
    flat_coefficients = basis.transpose(0, 1) @ flat_increments
    return (
        flat_coefficients.reshape(max_degree, num_paths, dimension)
        .permute(1, 0, 2)
        .contiguous()
    )


def integrated_legendre_basis(
    time: torch.Tensor,
    max_degree: int,
) -> torch.Tensor:
    """Return the analytically integrated shifted-Legendre basis."""
    horizon = time[-1] - time[0]
    z = 2.0 * (time - time[0]) / horizon - 1.0
    polynomials = evaluate_legendre(z, max_degree)

    integrated = torch.empty(
        time.numel(),
        max_degree,
        dtype=time.dtype,
        device=time.device,
    )
    integrated[:, 0] = (z + 1.0) / 2.0
    if max_degree > 1:
        integrated[:, 1:] = (
            polynomials[:, 2 : max_degree + 1]
            - polynomials[:, : max_degree - 1]
        ) / 2.0
    return integrated


def exact_truncation_rmse(horizon: float, degree: int) -> float:
    """Exact time-averaged, per-coordinate RMSE of the K-term expansion."""
    # E[1/T integral_0^T |W_t - W_t^(K)|^2 dt]
    #     = T K / (2 (4 K^2 - 1)).
    return (horizon * degree / (2.0 * (4.0 * degree**2 - 1.0))) ** 0.5


def log_log_slope(degrees: Sequence[int], values: Sequence[float]) -> float:
    """Fit a slope to the final four points in log-log coordinates."""
    fit_count = min(4, len(degrees))
    x = torch.log(torch.tensor(degrees[-fit_count:], dtype=torch.float64))
    y = torch.log(torch.tensor(values[-fit_count:], dtype=torch.float64))
    x_centered = x - x.mean()
    return float((x_centered * (y - y.mean())).sum() / x_centered.square().sum())


def run_experiment(
    *,
    dimension: int,
    num_paths: int,
    num_steps: int,
    horizon: float,
    degrees: Sequence[int],
    seed: int,
    device: torch.device,
) -> ExperimentResult:
    """Simulate paths, reconstruct them, and collect convergence statistics."""
    unique_degrees = tuple(sorted(set(degrees)))
    if len(unique_degrees) < 2:
        raise ValueError("provide at least two distinct approximation degrees")
    if unique_degrees[0] < 1:
        raise ValueError("all approximation degrees must be positive")

    max_degree = unique_degrees[-1]
    if max_degree > num_steps:
        raise ValueError("the largest degree cannot exceed the number of time steps")
    if num_steps < 16 * max_degree:
        print(
            "Warning: fewer than 16 time steps per maximum degree; "
            "quadrature error may obscure the theoretical convergence rate."
        )

    time, increments, paths = simulate_wiener_paths(
        num_paths=num_paths,
        num_steps=num_steps,
        dimension=dimension,
        horizon=horizon,
        seed=seed,
        device=device,
    )
    coefficients = project_increments(increments, time, max_degree)
    integrated_basis = integrated_legendre_basis(time, max_degree)

    metrics: list[DegreeMetrics] = []
    approximation = torch.zeros_like(paths)
    previous_degree = 0
    signal_rms = float(paths.square().mean().sqrt())

    plot_degrees = {unique_degrees[0], unique_degrees[-1]}
    sample_approximations: dict[int, torch.Tensor] = {}

    for degree in unique_degrees:
        # Add only the newly included modes. Across all checkpoints, each mode
        # is therefore reconstructed exactly once.
        approximation += torch.einsum(
            "tk,bkd->btd",
            integrated_basis[:, previous_degree:degree],
            coefficients[:, previous_degree:degree, :],
        )
        error = approximation - paths
        rmse = float(error.square().mean().sqrt())
        per_dimension_rmse = (
            error.square().mean(dim=(0, 1)).sqrt().detach().cpu()
        )
        endpoint_max_error = float(error[:, -1, :].abs().max())

        metrics.append(
            DegreeMetrics(
                degree=degree,
                rmse=rmse,
                theoretical_rmse=exact_truncation_rmse(horizon, degree),
                relative_rmse=rmse / signal_rms,
                endpoint_max_error=endpoint_max_error,
                per_dimension_rmse=per_dimension_rmse,
            )
        )
        if degree in plot_degrees:
            sample_approximations[degree] = approximation[0].detach().cpu().clone()
        previous_degree = degree

    coefficient_indices = torch.arange(
        max_degree,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    standardization = torch.sqrt((2.0 * coefficient_indices + 1.0) / horizon)
    standardized = coefficients * standardization[None, :, None]

    normalized_variance = (
        standardized.var(dim=(0, 2), correction=1).detach().cpu()
    )
    coefficient_mean = float(standardized.mean())
    coefficient_std = float(standardized.std(correction=1))

    coefficient_samples = standardized.reshape(-1, dimension)
    centered_samples = coefficient_samples - coefficient_samples.mean(dim=0)
    covariance = (
        centered_samples.transpose(0, 1) @ centered_samples
    ) / (coefficient_samples.shape[0] - 1)
    marginal_std = covariance.diag().clamp_min(0.0).sqrt()
    correlation = covariance / (
        marginal_std[:, None] * marginal_std[None, :]
    ).clamp_min(torch.finfo(covariance.dtype).eps)

    empirical_values = [metric.rmse for metric in metrics]
    theoretical_values = [metric.theoretical_rmse for metric in metrics]
    return ExperimentResult(
        dimension=dimension,
        num_paths=num_paths,
        num_steps=num_steps,
        horizon=horizon,
        metrics=tuple(metrics),
        fitted_slope=log_log_slope(unique_degrees, empirical_values),
        theoretical_slope=log_log_slope(unique_degrees, theoretical_values),
        coefficient_mean=coefficient_mean,
        coefficient_std=coefficient_std,
        normalized_coefficient_variance=normalized_variance,
        dimension_correlation=correlation.detach().cpu(),
        sample_time=time.detach().cpu(),
        sample_path=paths[0].detach().cpu(),
        sample_approximations=sample_approximations,
    )


def max_off_diagonal(matrix: torch.Tensor) -> float:
    """Maximum absolute off-diagonal entry of a square matrix."""
    if matrix.shape[0] == 1:
        return 0.0
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return float(matrix[mask].abs().max())


def print_report(result: ExperimentResult) -> None:
    """Print a compact numerical report."""
    print(
        f"\nD-dimensional Wiener experiment: D={result.dimension}, "
        f"paths={result.num_paths}, steps={result.num_steps}, "
        f"T={result.horizon:g}"
    )
    print(
        "\n"
        "    K | empirical RMSE | exact RMSE | ratio | relative RMSE | endpoint max\n"
        "------+----------------+------------+-------+---------------+-------------"
    )
    for metric in result.metrics:
        print(
            f"{metric.degree:5d} | {metric.rmse:14.6e} | "
            f"{metric.theoretical_rmse:10.6e} | "
            f"{metric.rmse / metric.theoretical_rmse:5.3f} | "
            f"{metric.relative_rmse:13.6e} | "
            f"{metric.endpoint_max_error:11.3e}"
        )

    final_per_dimension = result.metrics[-1].per_dimension_rmse
    if result.dimension <= 16:
        coordinate_report = ", ".join(
            f"d{index}={value:.3e}"
            for index, value in enumerate(final_per_dimension.tolist())
        )
    else:
        coordinate_report = (
            f"min={float(final_per_dimension.min()):.3e}, "
            f"median={float(final_per_dimension.median()):.3e}, "
            f"max={float(final_per_dimension.max()):.3e}"
        )

    print(
        f"\nlog-log slope: empirical={result.fitted_slope:.4f}, "
        f"exact={result.theoretical_slope:.4f} "
        "(asymptotic target: -0.5)"
    )
    print(
        "standardized coefficients: "
        f"mean={result.coefficient_mean:+.4f}, "
        f"std={result.coefficient_std:.4f}"
    )
    print(
        "maximum cross-dimension coefficient correlation: "
        f"{max_off_diagonal(result.dimension_correlation):.4f}"
    )
    print(f"final RMSE by coordinate: {coordinate_report}")


def check_experiment(result: ExperimentResult) -> None:
    """Fail loudly if the experiment does not exhibit the predicted behavior."""
    empirical = torch.tensor(
        [metric.rmse for metric in result.metrics], dtype=torch.float64
    )
    theoretical = torch.tensor(
        [metric.theoretical_rmse for metric in result.metrics],
        dtype=torch.float64,
    )
    ratios = empirical / theoretical

    trajectory_count = result.num_paths * result.dimension
    theory_tolerance = max(0.25, 4.0 / (2.0 * trajectory_count) ** 0.5)
    coefficient_count = (
        result.num_paths
        * result.dimension
        * result.normalized_coefficient_variance.numel()
    )
    mean_tolerance = max(0.05, 4.0 / coefficient_count**0.5)
    std_tolerance = max(0.08, 4.0 / (2.0 * (coefficient_count - 1)) ** 0.5)
    correlation_sample_count = (
        result.num_paths * result.normalized_coefficient_variance.numel()
    )
    correlation_tolerance = max(
        0.10, 4.0 / correlation_sample_count**0.5
    )

    max_endpoint_error = max(
        metric.endpoint_max_error for metric in result.metrics
    )
    observed_reduction = float(empirical[-1] / empirical[0])
    predicted_reduction = float(theoretical[-1] / theoretical[0])

    checks = {
        "RMSE agrees with exact theory": bool(
            torch.all((ratios - 1.0).abs() <= theory_tolerance)
        ),
        "higher degree reduces RMSE as predicted": (
            observed_reduction < 1.0
            and abs(observed_reduction / predicted_reduction - 1.0)
            <= theory_tolerance
        ),
        "empirical convergence slope agrees with theory": (
            abs(result.fitted_slope - result.theoretical_slope) <= 0.15
        ),
        "terminal Wiener increment is preserved": (
            max_endpoint_error <= 1.0e-10 * max(1.0, result.horizon**0.5)
        ),
        "standardized coefficients are centered": (
            abs(result.coefficient_mean) <= mean_tolerance
        ),
        "standardized coefficients have unit variance": (
            abs(result.coefficient_std - 1.0) <= std_tolerance
        ),
        "different Wiener dimensions remain independent": (
            max_off_diagonal(result.dimension_correlation)
            <= correlation_tolerance
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if failed:
        raise AssertionError("experiment checks failed: " + "; ".join(failed))


def save_figure(result: ExperimentResult, output: Path) -> None:
    """Save path, convergence, coefficient, and independence diagnostics."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    time = result.sample_time.numpy()
    low_degree = result.metrics[0].degree
    high_degree = result.metrics[-1].degree
    visible_dimensions = min(3, result.dimension)

    for coordinate, axis in enumerate(axes[0]):
        if coordinate >= visible_dimensions:
            axis.set_visible(False)
            continue

        axis.plot(
            time,
            result.sample_path[:, coordinate].numpy(),
            color="black",
            linewidth=1.1,
            alpha=0.75,
            label="Wiener path",
        )
        axis.plot(
            time,
            result.sample_approximations[low_degree][:, coordinate].numpy(),
            color="#1f77b4",
            linewidth=1.0,
            linestyle="--",
            label=f"K={low_degree}",
        )
        axis.plot(
            time,
            result.sample_approximations[high_degree][:, coordinate].numpy(),
            color="#d62728",
            linewidth=1.1,
            label=f"K={high_degree}",
        )
        axis.set_title(f"sample coordinate {coordinate}")
        axis.set_xlabel("time")
        axis.set_ylabel(f"W[{coordinate}](t)")
        axis.grid(alpha=0.25)
        if coordinate == 0:
            axis.legend(loc="best")

    convergence_axis = axes[1, 0]
    degrees = [metric.degree for metric in result.metrics]
    empirical_rmse = [metric.rmse for metric in result.metrics]
    theoretical_rmse = [metric.theoretical_rmse for metric in result.metrics]
    convergence_axis.loglog(
        degrees,
        empirical_rmse,
        "o-",
        color="#d62728",
        label="empirical (all dimensions)",
    )
    convergence_axis.loglog(
        degrees,
        theoretical_rmse,
        "k--",
        label="exact truncation RMSE",
    )
    convergence_axis.set_xlabel("number of modes K")
    convergence_axis.set_ylabel("time-averaged RMSE")
    convergence_axis.set_title(
        f"convergence slope = {result.fitted_slope:.3f}"
    )
    convergence_axis.grid(which="both", alpha=0.25)
    convergence_axis.legend(loc="best")

    variance_axis = axes[1, 1]
    coefficient_indices = torch.arange(
        result.normalized_coefficient_variance.numel()
    )
    variance_axis.plot(
        coefficient_indices.numpy(),
        result.normalized_coefficient_variance.numpy(),
        color="#1f77b4",
        linewidth=1.0,
    )
    variance_axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="theory",
    )
    variance_axis.set_xlabel("Legendre coefficient index k")
    variance_axis.set_ylabel("Var(sqrt((2k+1)/T) c_k)")
    variance_axis.set_title("coefficient variance")
    variance_axis.grid(alpha=0.25)
    variance_axis.legend(loc="best")

    correlation_axis = axes[1, 2]
    shown_dimensions = min(12, result.dimension)
    shown_correlation = result.dimension_correlation[
        :shown_dimensions, :shown_dimensions
    ]
    image = correlation_axis.imshow(
        shown_correlation.numpy(),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    correlation_axis.set_title(
        f"cross-dimension correlation (first {shown_dimensions})"
    )
    correlation_axis.set_xlabel("coordinate")
    correlation_axis.set_ylabel("coordinate")
    correlation_axis.set_xticks(range(shown_dimensions))
    correlation_axis.set_yticks(range(shown_dimensions))
    figure.colorbar(image, ax=correlation_axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        f"Legendre approximation of a {result.dimension}-dimensional "
        "Wiener process",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"\nSaved diagnostic figure to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test Legendre approximation on a d-dimensional Wiener process."
        )
    )
    parser.add_argument("--dim", type=int, default=8, help="Wiener dimension")
    parser.add_argument(
        "--num-paths",
        type=int,
        default=128,
        help="number of independent Monte Carlo paths",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4096,
        help="uniform time steps per path",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=1.0,
        help="terminal time T",
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEGREES),
        help="increasing Legendre truncation degrees",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legendre_wiener_d_dim.png"),
        help="diagnostic figure path",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="run numerical checks without creating a figure",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="report diagnostics without enforcing pass/fail thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        dimension=args.dim,
        num_paths=args.num_paths,
        num_steps=args.num_steps,
        horizon=args.horizon,
        degrees=args.degrees,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print_report(result)
    if not args.skip_checks:
        print()
        check_experiment(result)
    if not args.no_plot:
        save_figure(result, args.output)


if __name__ == "__main__":
    main()
