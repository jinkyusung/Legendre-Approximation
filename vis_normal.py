"""Visualize Brownian motion using a normalized Legendre basis.

The projection and reconstruction functions support both a scalar path with
shape ``[N]`` and a d-dimensional path with shape ``[N, d]``. Running this
file creates an animation analogous to ``vis.py``:

    python vis_normal.py
    python vis_normal.py --dim 3 --max-degree 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# 1. Normalized Legendre operations
# ---------------------------------------------------------------------------
def eval_legendre(z: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Evaluate standard Legendre polynomials through ``L_max_degree``."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    values = [torch.ones_like(z)]
    if max_degree >= 1:
        values.append(z)

    for k in range(1, max_degree):
        # (k + 1) L_{k+1} = (2k + 1) z L_k - k L_{k-1}
        next_value = (
            (2 * k + 1) * z * values[k] - k * values[k - 1]
        ) / (k + 1)
        values.append(next_value)

    return torch.stack(values, dim=-1)


def eval_normalized_legendre_basis(
    tau: torch.Tensor,
    s: float | torch.Tensor,
    t: float | torch.Tensor,
    num_modes: int,
) -> torch.Tensor:
    r"""Evaluate the orthonormal shifted Legendre basis on ``[s, t]``.

    .. math::
        P_k(\tau)
        = \sqrt{\frac{2k+1}{t-s}}\,
          L_k\left(\frac{2(\tau-s)}{t-s}-1\right).

    Returns:
        Tensor with shape ``(*tau.shape, num_modes)``.
    """
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    s = torch.as_tensor(s, dtype=tau.dtype, device=tau.device)
    t = torch.as_tensor(t, dtype=tau.dtype, device=tau.device)
    interval = t - s
    if torch.any(interval <= 0):
        raise ValueError("Require t > s")

    z = 2.0 * (tau - s) / interval - 1.0
    standard_basis = eval_legendre(z, num_modes - 1)
    indices = torch.arange(
        num_modes,
        dtype=tau.dtype,
        device=tau.device,
    )
    normalization = torch.sqrt((2.0 * indices + 1.0) / interval)
    return standard_basis * normalization


def eval_integrated_normalized_legendre_basis(
    tau: torch.Tensor,
    s: float | torch.Tensor,
    t: float | torch.Tensor,
    num_modes: int,
) -> torch.Tensor:
    r"""Evaluate ``A_k(tau) = integral_s^tau P_k(u) du`` analytically.

    For ``k = 0``,

    .. math::
        A_0(\tau) = \sqrt{t-s}\,\frac{z(\tau)+1}{2},

    and for ``k >= 1``,

    .. math::
        A_k(\tau)
        = \frac{1}{2}\sqrt{\frac{t-s}{2k+1}}
          \left[L_{k+1}(z(\tau))-L_{k-1}(z(\tau))\right].

    Returns:
        Tensor with shape ``(*tau.shape, num_modes)``.
    """
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    s = torch.as_tensor(s, dtype=tau.dtype, device=tau.device)
    t = torch.as_tensor(t, dtype=tau.dtype, device=tau.device)
    interval = t - s
    if torch.any(interval <= 0):
        raise ValueError("Require t > s")

    z = 2.0 * (tau - s) / interval - 1.0
    standard_basis = eval_legendre(z, num_modes)
    integrated = torch.empty(
        (*tau.shape, num_modes),
        dtype=tau.dtype,
        device=tau.device,
    )
    integrated[..., 0] = torch.sqrt(interval) * (z + 1.0) / 2.0

    if num_modes > 1:
        indices = torch.arange(
            1,
            num_modes,
            dtype=tau.dtype,
            device=tau.device,
        )
        scale = 0.5 * torch.sqrt(interval / (2.0 * indices + 1.0))
        integrated[..., 1:] = scale * (
            standard_basis[..., 2 : num_modes + 1]
            - standard_basis[..., : num_modes - 1]
        )

    return integrated


def project_path_to_coeffs(
    brownian_path: torch.Tensor,
    time: torch.Tensor,
    num_modes: int,
) -> torch.Tensor:
    r"""Project a sampled Brownian path onto normalized Legendre modes.

    The deterministic-integrand Ito integrals are approximated by

    .. math::
        \xi_k \approx \sum_n P_k(t_n)
        \left(B_{t_{n+1}} - B_{t_n}\right).

    ``brownian_path`` may have shape ``[N]`` or ``[N, *path_shape]``. The
    returned shape is ``[num_modes, *path_shape]``.
    """
    if time.ndim != 1 or time.numel() < 2:
        raise ValueError("time must be a one-dimensional grid with at least 2 points")
    if brownian_path.shape[0] != time.numel():
        raise ValueError("brownian_path and time must share their first dimension")

    increments = brownian_path[1:] - brownian_path[:-1]
    basis = eval_normalized_legendre_basis(
        time[:-1],
        time[0],
        time[-1],
        num_modes,
    )

    trailing_shape = brownian_path.shape[1:]
    flat_increments = increments.reshape(increments.shape[0], -1)
    flat_coefficients = basis.transpose(0, 1) @ flat_increments
    return flat_coefficients.reshape(num_modes, *trailing_shape)


def reconstruct_path_from_coeffs(
    coefficients: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    r"""Reconstruct ``B_tau^(K) = sum_k xi_k integral_s^tau P_k(u)du``.

    ``coefficients`` may have shape ``[K]`` or ``[K, *path_shape]``. The
    returned tensor has shape ``[N, *path_shape]``.
    """
    if coefficients.ndim == 0 or coefficients.shape[0] == 0:
        raise ValueError("coefficients must contain at least one mode")
    if time.ndim != 1 or time.numel() < 2:
        raise ValueError("time must be a one-dimensional grid with at least 2 points")

    num_modes = coefficients.shape[0]
    integrated_basis = eval_integrated_normalized_legendre_basis(
        time,
        time[0],
        time[-1],
        num_modes,
    )
    trailing_shape = coefficients.shape[1:]
    flat_coefficients = coefficients.reshape(num_modes, -1)
    reconstruction = integrated_basis @ flat_coefficients
    return reconstruction.reshape(time.numel(), *trailing_shape)


# ---------------------------------------------------------------------------
# 2. Brownian path generation and animation
# ---------------------------------------------------------------------------
def simulate_brownian_path(
    time: torch.Tensor,
    dim: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Simulate a d-dimensional Brownian path on a uniform time grid."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    increments = torch.randn(
        time.numel() - 1,
        dim,
        dtype=time.dtype,
        device=time.device,
        generator=generator,
    ) * torch.sqrt(time[1] - time[0])
    return torch.cat(
        (
            torch.zeros(1, dim, dtype=time.dtype, device=time.device),
            increments.cumsum(dim=0),
        ),
        dim=0,
    )


def degree_frames(max_degree: int) -> list[int]:
    """Create a concise increasing list of mode counts for the animation."""
    if max_degree <= 0:
        raise ValueError("max_degree must be positive")

    candidates = (
        list(range(1, min(20, max_degree) + 1))
        + list(range(20, min(100, max_degree) + 1, 5))
        + list(range(100, max_degree + 1, 10))
        + [max_degree]
    )
    return sorted({degree for degree in candidates if degree <= max_degree})


def create_animation(
    *,
    dim: int,
    num_points: int,
    start: float,
    end: float,
    max_degree: int,
    seed: int,
    output: Path,
    fps: int,
) -> None:
    """Generate and save the normalized-Legendre Brownian animation."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if end <= start:
        raise ValueError("end must be greater than start")
    if max_degree >= num_points:
        raise ValueError("max_degree must be smaller than num_points")

    generator = torch.Generator().manual_seed(seed)
    time = torch.linspace(start, end, num_points, dtype=torch.float64)
    brownian_path = simulate_brownian_path(time, dim, generator=generator)

    # Project and integrate once at the maximum degree. Each animation frame
    # only slices these precomputed tensors.
    coefficients = project_path_to_coeffs(
        brownian_path,
        time,
        max_degree,
    )
    integrated_basis = eval_integrated_normalized_legendre_basis(
        time,
        time[0],
        time[-1],
        max_degree,
    )
    frames = degree_frames(max_degree)

    visible_dimensions = min(dim, 3)
    figure, axes = plt.subplots(
        visible_dimensions,
        1,
        figsize=(10, 3.2 * visible_dimensions + 1.0),
        squeeze=False,
        sharex=True,
    )
    axes = axes[:, 0]
    approximation_lines = []

    time_numpy = time.numpy()
    for coordinate, axis in enumerate(axes):
        axis.plot(
            time_numpy,
            brownian_path[:, coordinate].numpy(),
            color="black",
            alpha=0.35,
            linewidth=1.4,
            label="True Brownian motion",
        )
        (line,) = axis.plot(
            [],
            [],
            color="#d62728",
            linewidth=1.5,
            label="Normalized Legendre approximation",
        )
        approximation_lines.append(line)
        axis.set_ylabel(f"B[{coordinate}](t)")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left")
        axis.set_xlim(start, end)
    axes[-1].set_xlabel("Time (t)")
    figure.subplots_adjust(top=0.91, bottom=0.1, hspace=0.22)

    def update(frame_index: int):
        num_modes = frames[frame_index]
        approximation = (
            integrated_basis[:, :num_modes] @ coefficients[:num_modes]
        )

        for coordinate, (axis, line) in enumerate(
            zip(axes, approximation_lines)
        ):
            line.set_data(time_numpy, approximation[:, coordinate].numpy())
            lower = min(
                brownian_path[:, coordinate].min().item(),
                approximation[:, coordinate].min().item(),
            )
            upper = max(
                brownian_path[:, coordinate].max().item(),
                approximation[:, coordinate].max().item(),
            )
            margin = max(0.1, 0.08 * (upper - lower))
            axis.set_ylim(lower - margin, upper + margin)

        figure.suptitle(
            "Normalized Legendre expansion "
            f"({dim}D Wiener process, K={num_modes} modes)",
            fontsize=14,
        )
        return tuple(approximation_lines)

    brownian_animation = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation: {output}")
    brownian_animation.save(output, writer="pillow", fps=fps)
    plt.close(figure)
    print(f"Successfully saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a normalized-Legendre Brownian approximation."
    )
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=10_000)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=10.0)
    parser.add_argument("--max-degree", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legendre_normalized_brownian_approximation.gif"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_animation(
        dim=args.dim,
        num_points=args.num_points,
        start=args.start,
        end=args.end,
        max_degree=args.max_degree,
        seed=args.seed,
        output=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
