import diffrax
import jax
import jax.numpy as jnp


def sample_diffrax_coeffs(
    s: float,
    t: float,
    batch_size: int,
    dim: int,
    tol: float = 1e-3,
    seed: int = 0,
):
    key = jax.random.PRNGKey(seed)

    brownian = diffrax.VirtualBrownianTree(
        t0=s,
        t1=t,
        tol=tol,
        shape=(batch_size, dim),
        key=key,
        levy_area=diffrax.SpaceTimeTimeLevyArea,
    )

    levy = brownian.evaluate(s, t, use_levy=True)

    return levy.W, levy.H, levy.K


if __name__ == "__main__":
    W, H, K = sample_diffrax_coeffs(
        s=0.0,
        t=10.0,
        batch_size=1024,
        dim=16,
    )

    print(W.shape)  # (1024, 16)
    print(H.shape)  # (1024, 16)
    print(K.shape)  # (1024, 16)
    # I0 = W
    # I1 = -2.0 * H
    # I2 = 12.0 * K
