"""Normalized Legendre coefficients for Brownian motion.

On an interval ``[s, t]`` with ``t > s``, define

    P_k(tau) = sqrt((2k + 1) / (t - s))
               L_k(2 (tau - s) / (t - s) - 1),

where ``L_k`` is the standard Legendre polynomial.  The functions ``P_k`` are
orthonormal in ``L^2([s, t])``.  Consequently, the Brownian coefficients

    xi_k = integral_s^t P_k(tau) dB_tau

are independent standard normal vectors.
"""

from __future__ import annotations

import torch


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
    r"""Sample normalized Legendre coefficients of Brownian motion.

    Unlike the coefficients associated with unnormalized shifted Legendre
    polynomials, these coefficients have no degree- or interval-dependent
    scale.  Ito isometry and orthonormality give

    .. math::
        \mathbb{E}[\xi_{k,i}\xi_{\ell,j}]
        = \delta_{k,\ell}\delta_{i,j},
        \qquad
        \xi_k \sim \mathcal{N}(0, I_{\mathrm{dim}}).

    Args:
        s: Start time of the expansion interval.
        t: End time of the expansion interval. Every element must satisfy
            ``t > s``.
        degree: Number of Legendre modes. The returned indices are
            ``k = 0, ..., degree - 1``.
        dim: Dimension of the Brownian motion.

    Keyword Args:
        batch_shape: Shape of a batch of independent coefficient samples.
        device: Device of the returned tensor.
        dtype: Floating-point dtype of the returned tensor.
        generator: Optional PyTorch random-number generator.

    Returns:
        A standard normal tensor with shape
        ``(*batch_shape, degree, dim)``.

    Raises:
        ValueError: If ``degree`` is negative, ``dim`` is not positive, or the
            interval is not strictly increasing.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")

    s_tensor = torch.as_tensor(s, device=device, dtype=dtype)
    t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
    if torch.any(t_tensor <= s_tensor):
        raise ValueError(
            "Require t > s because a normalized L2 basis is undefined "
            "on a zero-length or backward interval."
        )

    return torch.randn(
        (*batch_shape, degree, dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )


if __name__ == "__main__":
    # Number of normalized Legendre modes.
    K = 500

    # General d-dimensional case.
    D = 16
    B = 1024
    s = torch.zeros(B)
    t = torch.full((B,), 10.0)

    # xi shape: [B, K, D], with xi[k] ~ N(0, I_D).
    xi = sample_brownian_legendre_coeffs(
        s,
        t,
        degree=K,
        dim=D,
        batch_shape=(B,),
    )
    print(f"Sampled normalized coefficients shape: {xi.shape}")
    print(f"Empirical mean: {xi.mean().item():+.4f}")
    print(f"Empirical standard deviation: {xi.std().item():.4f}")
