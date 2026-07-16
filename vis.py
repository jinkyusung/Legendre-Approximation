import torch
import matplotlib.pyplot as plt


def save_legendre_brownian_path(
    coeffs: torch.Tensor,
    s: float,
    t: float,
    save_path: str = "legendre_brownian_path.png",
    num_grid: int = 1000,
    title: str | None = None,
) -> str:
    """
    Input:
        coeffs:
            Output of sample_legendre_brownian_input.
            Expected shape:
                (n,) or (1, n)

        s, t:
            Interval endpoints.

        save_path:
            Path to save the figure.

    Output:
        save_path
    """
    if t <= s:
        raise ValueError("Require t > s.")

    coeffs = coeffs.detach()

    if coeffs.ndim == 2:
        if coeffs.shape[0] != 1:
            raise ValueError("This function expects batch size 1.")
        coeffs = coeffs[0]

    if coeffs.ndim != 1:
        raise ValueError("coeffs must have shape (n,) or (1, n).")

    device = coeffs.device
    dtype = coeffs.dtype
    n = coeffs.shape[0]

    times = torch.linspace(s, t, num_grid, device=device, dtype=dtype)
    x = (times - s) / (t - s)
    y = 2.0 * x - 1.0

    # Compute ordinary Legendre polynomials P_0(y), ..., P_n(y)
    P = torch.empty(n + 1, num_grid, device=device, dtype=dtype)
    P[0] = 1.0

    if n >= 1:
        P[1] = y

    for k in range(1, n):
        P[k + 1] = ((2 * k + 1) * y * P[k] - k * P[k - 1]) / (k + 1)

    # Compute A_k(x) = integral_0^x Pe_k(r) dr
    A = torch.empty(n, num_grid, device=device, dtype=dtype)
    A[0] = x

    for k in range(1, n):
        A[k] = (P[k + 1] - P[k - 1]) / (2.0 * (2 * k + 1))

    # Reconstruct W_N(u) - W_s
    k = torch.arange(n, device=device, dtype=dtype)
    weights = 2.0 * k + 1.0

    W = torch.sum(weights[:, None] * coeffs[:, None] * A, dim=0)

    times_np = times.cpu().numpy()
    W_np = W.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(times_np, W_np, linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.scatter([s, t], [W_np[0], W_np[-1]], zorder=3)

    plt.xlabel("time")
    plt.ylabel("W_N(u) - W_s")

    if title is None:
        title = f"Shifted Legendre Brownian approximation, N={n}"
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

    return save_path
