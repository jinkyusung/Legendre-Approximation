import torch
import vis


def sample_brownian_legendre_coeffs(
    s: float | torch.Tensor,
    t: float | torch.Tensor,
    degree: int,
    dim: int,
    *,
    batch_shape=(),
    device=None,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Samples the coefficients for the Legendre polynomial expansion of a Brownian motion increment.

    This function computes the stochastic coefficients obtained by projecting the white noise
    (the formal derivative of a Wiener process) over the time interval :math:`[s, t]` onto a set
    of shifted Legendre polynomials.

    By Itô isometry, the projection of the Brownian motion increments yields independent
    normally distributed coefficients. Specifically, the :math:`k`-th degree coefficient
    :math:`c_k` has mean zero and variance strictly derived from the L2-norm of the shifted
    Legendre polynomial:

    .. math::
        c_k \sim \mathcal{N}\left(0, \frac{t - s}{2k + 1}\right)

    Args:
        s (float or Tensor): The start time(s) :math:`s`.
        t (float or Tensor): The terminal time(s) :math:`t`. Must satisfy :math:`t \ge s`.
        degree (int): The number of Legendre polynomial coefficients to sample (denoted as :math:`K`).
        dim (int): The dimension of the Brownian motion (denoted as :math:`D`).

    Keyword Args:
        batch_shape (tuple, optional): The shape of the batch for independent paths. Default: ``()``.
        device (torch.device, optional): The desired device of returned tensor. Default: ``None``.
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: ``torch.float32``.
        generator (torch.Generator, optional): A pseudorandom number generator for sampling. Default: ``None``.

    Returns:
        Tensor: A tensor containing the sampled Legendre coefficients scaled by their exact standard deviations.

    Shape:
        - Output: :math:`(*\text{batch\_shape}, \text{degree}, \text{dim})`

    Raises:
        ValueError: If any element in the time increment :math:`h = t - s` is strictly negative.
    """
    s = torch.as_tensor(s, device=device, dtype=dtype)
    t = torch.as_tensor(t, device=device, dtype=dtype)

    h = t - s
    if torch.any(h < 0):
        raise ValueError("Require t >= s. Backward integration is not supported in this function.")

    k = torch.arange(degree, device=device, dtype=dtype)

    h_safe = torch.where(h > 0, h, torch.ones_like(h)) # 0으로 나누기 및 sqrt(0) 방지
    std = torch.where(
        h[..., None] > 0,
        torch.sqrt(h_safe[..., None] / (2.0 * k + 1.0)),
        torch.zeros_like(h[..., None]) # h=0 이면 분산을 0으로 (Degenerate distribution)
    )

    z = torch.randn(
        (*batch_shape, degree, dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    return std.unsqueeze(-1) * z


if __name__ == '__main__':

    # Legendre Polynomial Degree.
    K = 500

    # General case.
    D = 16    # D-dimensional Wiener process.
    B = 1024  # Batch Size.
    s = torch.full((B,), 0)  # Start time.
    t = torch.full((B,), 10)  # Terminal time.

    # coeffs shape: [B, K, D]
    coeffs = sample_brownian_legendre_coeffs(s, t, degree=K, dim=D, batch_shape=(B,))
    print(f"Sampled coefficients shape: {coeffs.shape}")
